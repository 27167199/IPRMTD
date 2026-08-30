import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pair_features(drug_z, disease_z, extra=None):
    base = torch.cat([drug_z, disease_z, drug_z * disease_z, torch.abs(drug_z - disease_z)], dim=-1)
    if extra is None or extra.numel() == 0:
        return base
    return torch.cat([base, extra], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiModalEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dim, dropout=0.15):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for dim in input_dims
            if dim > 0
        ])
        self.attn = nn.Linear(hidden_dim, 1)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs):
        projected = []
        for projector, x in zip(self.projectors, inputs):
            projected.append(projector(x))
        if not projected:
            raise ValueError("At least one modality is required.")
        if len(projected) == 1:
            return self.out_norm(projected[0]), projected, torch.ones(projected[0].shape[0], 1, device=projected[0].device)

        stacked = torch.stack(projected, dim=1)
        weights = torch.softmax(self.attn(stacked).squeeze(-1), dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return self.out_norm(fused), projected, weights


class BipartiteViewEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout=0.15):
        super().__init__()
        self.drug_sem = nn.Linear(hidden_dim, hidden_dim)
        self.disease_sem = nn.Linear(hidden_dim, hidden_dim)
        self.drug_inter = nn.Linear(hidden_dim, hidden_dim)
        self.disease_inter = nn.Linear(hidden_dim, hidden_dim)
        self.drug_norm = nn.LayerNorm(hidden_dim)
        self.disease_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_base, disease_base, graphs):
        drug_sem_msg = torch.sparse.mm(graphs["drug_sem"], drug_base)
        disease_sem_msg = torch.sparse.mm(graphs["disease_sem"], disease_base)
        drug_inter_msg = torch.sparse.mm(graphs["drug_to_disease"], disease_base)
        disease_inter_msg = torch.sparse.mm(graphs["disease_to_drug"], drug_base)

        drug_sem = self.drug_norm(drug_base + self.dropout(F.relu(self.drug_sem(drug_sem_msg))))
        disease_sem = self.disease_norm(disease_base + self.dropout(F.relu(self.disease_sem(disease_sem_msg))))
        drug_inter = self.drug_norm(drug_base + self.dropout(F.relu(self.drug_inter(drug_inter_msg))))
        disease_inter = self.disease_norm(disease_base + self.dropout(F.relu(self.disease_inter(disease_inter_msg))))
        return drug_sem, disease_sem, drug_inter, disease_inter


class ViewFusion(nn.Module):
    def __init__(self, hidden_dim, method="gated"):
        super().__init__()
        self.method = method
        if method == "attention":
            self.attn = nn.Linear(hidden_dim, 1)
        else:
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, semantic_z, interaction_z):
        if self.method == "attention":
            stacked = torch.stack([semantic_z, interaction_z], dim=1)
            weights = torch.softmax(self.attn(stacked).squeeze(-1), dim=1)
            fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        else:
            gate = self.gate(torch.cat([semantic_z, interaction_z], dim=-1))
            fused = gate * semantic_z + (1.0 - gate) * interaction_z
        return self.norm(fused)


class HybridDDA(nn.Module):
    def __init__(
        self,
        drug_input_dims,
        disease_input_dims,
        hidden_dim=256,
        dropout=0.2,
        fusion_method="gated",
        pair_extra_dim=0,
    ):
        super().__init__()
        self.pair_extra_dim = pair_extra_dim
        self.drug_encoder = MultiModalEncoder(drug_input_dims, hidden_dim, dropout=dropout)
        self.disease_encoder = MultiModalEncoder(disease_input_dims, hidden_dim, dropout=dropout)
        self.graph_encoder = BipartiteViewEncoder(hidden_dim, dropout=dropout)
        self.drug_view_fusion = ViewFusion(hidden_dim, method=fusion_method)
        self.disease_view_fusion = ViewFusion(hidden_dim, method=fusion_method)

        pair_dim = hidden_dim * 4 + pair_extra_dim
        self.main_predictor = MLP(pair_dim, hidden_dim, 1, dropout=dropout)
        self.semantic_predictor = MLP(pair_dim, hidden_dim, 1, dropout=dropout)
        self.interaction_predictor = MLP(pair_dim, hidden_dim, 1, dropout=dropout)
        self.single_predictor = MLP(pair_dim, hidden_dim, 1, dropout=dropout)

    def shared_parameters_for_balance(self):
        modules = [self.drug_encoder, self.disease_encoder, self.graph_encoder, self.drug_view_fusion, self.disease_view_fusion]
        for module in modules:
            yield from module.parameters()

    def encode_nodes(self, features, graphs):
        drug_inputs = list(features["drug"])
        drug_base, drug_modalities, drug_weights = self.drug_encoder(drug_inputs)
        disease_base, disease_modalities, disease_weights = self.disease_encoder(features["disease"])
        drug_sem, disease_sem, drug_inter, disease_inter = self.graph_encoder(drug_base, disease_base, graphs)
        drug_fused = self.drug_view_fusion(drug_sem, drug_inter)
        disease_fused = self.disease_view_fusion(disease_sem, disease_inter)
        return {
            "drug_base": drug_base,
            "disease_base": disease_base,
            "drug_sem": drug_sem,
            "disease_sem": disease_sem,
            "drug_inter": drug_inter,
            "disease_inter": disease_inter,
            "drug_fused": drug_fused,
            "disease_fused": disease_fused,
            "drug_modalities": drug_modalities,
            "disease_modalities": disease_modalities,
            "drug_weights": drug_weights,
            "disease_weights": disease_weights,
        }

    def pair_extra(self, drug_idx, disease_idx, features):
        if self.pair_extra_dim <= 0:
            return None
        path_features = features.get("path_features")
        if path_features is None or path_features.numel() == 0:
            device = drug_idx.device
            return torch.zeros(drug_idx.shape[0], self.pair_extra_dim, device=device)
        return path_features[drug_idx, disease_idx]

    def forward(self, drug_idx, disease_idx, features, graphs):
        views = self.encode_nodes(features, graphs)
        d_fused = views["drug_fused"][drug_idx]
        s_fused = views["disease_fused"][disease_idx]
        d_sem = views["drug_sem"][drug_idx]
        s_sem = views["disease_sem"][disease_idx]
        d_inter = views["drug_inter"][drug_idx]
        s_inter = views["disease_inter"][disease_idx]
        extra = self.pair_extra(drug_idx, disease_idx, features)
        main_pair = pair_features(d_fused, s_fused, extra)

        single_logits = []
        for d_mod, s_mod in zip(views["drug_modalities"], views["disease_modalities"]):
            single_logits.append(self.single_predictor(pair_features(d_mod[drug_idx], s_mod[disease_idx], extra)).squeeze(-1))

        fused_logit = self.main_predictor(main_pair).squeeze(-1)
        semantic_logit = self.semantic_predictor(pair_features(d_sem, s_sem, extra)).squeeze(-1)
        interaction_logit = self.interaction_predictor(pair_features(d_inter, s_inter, extra)).squeeze(-1)

        return {
            "main": fused_logit,
            "fused": fused_logit,
            "semantic": semantic_logit,
            "interaction": interaction_logit,
            "single": single_logits,
            "views": views,
        }


def info_nce(anchor, positive, temperature=0.2):
    if anchor.shape[0] <= 1:
        return anchor.new_tensor(0.0)
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def cross_view_contrastive_loss(views, max_nodes=512, temperature=0.2):
    device = views["drug_fused"].device
    losses = []
    for prefix in ["drug", "disease"]:
        n_nodes = views[f"{prefix}_fused"].shape[0]
        sample_size = min(max_nodes, n_nodes)
        idx = torch.randperm(n_nodes, device=device)[:sample_size]
        fused = views[f"{prefix}_fused"][idx]
        semantic = views[f"{prefix}_sem"][idx]
        interaction = views[f"{prefix}_inter"][idx]
        losses.append(info_nce(fused, semantic, temperature=temperature))
        losses.append(info_nce(fused, interaction, temperature=temperature))
    return torch.stack(losses).mean()


def _sample_sparse_edges(adj, max_edges):
    adj = adj.coalesce()
    indices = adj.indices()
    if indices.shape[1] == 0:
        return None, None
    sample_size = min(max_edges, indices.shape[1])
    perm = torch.randperm(indices.shape[1], device=indices.device)[:sample_size]
    return indices[0, perm], indices[1, perm]


def edge_info_nce(left_z, right_z, row_idx, col_idx, temperature=0.2):
    if row_idx is None or row_idx.shape[0] <= 1:
        return left_z.new_tensor(0.0)
    return info_nce(left_z[row_idx], right_z[col_idx], temperature=temperature)


def intra_view_contrastive_loss(views, graphs, max_edges=1024, temperature=0.2):
    losses = []

    row_idx, col_idx = _sample_sparse_edges(graphs["drug_sem"], max_edges)
    losses.append(edge_info_nce(views["drug_sem"], views["drug_sem"], row_idx, col_idx, temperature=temperature))

    row_idx, col_idx = _sample_sparse_edges(graphs["disease_sem"], max_edges)
    losses.append(edge_info_nce(views["disease_sem"], views["disease_sem"], row_idx, col_idx, temperature=temperature))

    row_idx, col_idx = _sample_sparse_edges(graphs["drug_to_disease"], max_edges)
    losses.append(edge_info_nce(views["drug_inter"], views["disease_inter"], row_idx, col_idx, temperature=temperature))

    active_losses = [loss for loss in losses if loss.requires_grad or loss.item() != 0.0]
    if not active_losses:
        return views["drug_fused"].new_tensor(0.0)
    return torch.stack(active_losses).mean()


def gradient_norm_and_cosine(loss_a, loss_b, parameters):
    params = [p for p in parameters if p.requires_grad]
    grads_a = torch.autograd.grad(loss_a, params, retain_graph=True, allow_unused=True)
    grads_b = torch.autograd.grad(loss_b, params, retain_graph=True, allow_unused=True)

    norm_a = loss_a.new_tensor(0.0)
    norm_b = loss_a.new_tensor(0.0)
    dot = loss_a.new_tensor(0.0)
    for grad_a, grad_b in zip(grads_a, grads_b):
        if grad_a is None or grad_b is None:
            continue
        norm_a = norm_a + grad_a.pow(2).sum()
        norm_b = norm_b + grad_b.pow(2).sum()
        dot = dot + (grad_a * grad_b).sum()
    norm_a = torch.sqrt(norm_a + 1e-12)
    norm_b = torch.sqrt(norm_b + 1e-12)
    cosine = dot / (norm_a * norm_b + 1e-12)
    return norm_a.detach(), norm_b.detach(), cosine.detach()


def balanced_contrastive_weight(main_loss, contrastive_loss, parameters, base_weight=0.1, conflict_scale=0.25):
    main_norm, contrastive_norm, cosine = gradient_norm_and_cosine(main_loss, contrastive_loss, parameters)
    if contrastive_norm.item() == 0:
        return 0.0, main_norm.item(), contrastive_norm.item(), cosine.item()
    weight = base_weight * min(1.0, (main_norm / (contrastive_norm + 1e-12)).item())
    if cosine.item() < 0:
        weight *= conflict_scale
    return weight, main_norm.item(), contrastive_norm.item(), cosine.item()


def row_normalized_sparse(num_rows, num_cols, row_idx, col_idx, device):
    if len(row_idx) == 0:
        indices = torch.zeros((2, 0), dtype=torch.long, device=device)
        values = torch.zeros(0, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, (num_rows, num_cols), device=device).coalesce()
    row_idx = np.asarray(row_idx, dtype=np.int64)
    col_idx = np.asarray(col_idx, dtype=np.int64)
    degrees = np.bincount(row_idx, minlength=num_rows).astype(np.float32)
    values = 1.0 / np.maximum(degrees[row_idx], 1.0)
    indices = torch.tensor(np.vstack([row_idx, col_idx]), dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, (num_rows, num_cols), device=device).coalesce()


def build_knn_graph(features, topk, device):
    n_nodes = features.shape[0]
    if n_nodes == 0:
        return row_normalized_sparse(0, 0, [], [], device)
    x = features.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    k = min(topk, max(1, n_nodes - 1))
    cols = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n_nodes), k)
    cols = cols.reshape(-1)
    return row_normalized_sparse(n_nodes, n_nodes, rows, cols, device)


def build_graphs(train_mask, drug_semantic_features, disease_semantic_features, topk, device):
    disease_idx, drug_idx = np.where(train_mask)
    num_diseases, num_drugs = train_mask.shape
    return {
        "drug_sem": build_knn_graph(drug_semantic_features, topk, device),
        "disease_sem": build_knn_graph(disease_semantic_features, topk, device),
        "drug_to_disease": row_normalized_sparse(num_drugs, num_diseases, drug_idx, disease_idx, device),
        "disease_to_drug": row_normalized_sparse(num_diseases, num_drugs, disease_idx, drug_idx, device),
    }


def pairs_from_masks(train_mask, test_mask, association_matrix):
    train_pairs, test_pairs = [], []
    n_diseases, n_drugs = association_matrix.shape
    for disease_idx in range(n_diseases):
        for drug_idx in range(n_drugs):
            if train_mask[disease_idx, drug_idx]:
                train_pairs.append((drug_idx, disease_idx, int(association_matrix[disease_idx, drug_idx] > 0)))
            elif test_mask[disease_idx, drug_idx]:
                test_pairs.append((drug_idx, disease_idx, int(association_matrix[disease_idx, drug_idx] > 0)))
    return np.asarray(train_pairs, dtype=np.int64), np.asarray(test_pairs, dtype=np.int64)


def standardize_features(x):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] == 0:
        return x
    return StandardScaler().fit_transform(x).astype(np.float32)


def build_disease_text_features(project_root, dataset, n_diseases, output_dim=128):
    idx_to_id_path = os.path.join(project_root, "dataset", dataset, "disease_idx_to_id.npy")
    disease_idx_to_id = np.load(idx_to_id_path, allow_pickle=True).item()
    disease_ids = [disease_idx_to_id[i] for i in range(n_diseases)]

    tsv_path = os.path.join(project_root, "embedding_generation", "4.General Information of Disease.tsv")
    csv_path = os.path.join(project_root, "dataset", dataset, "disease_info.csv")
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t")
        df = df.set_index("DiseaseID")
        fields = ["Disease_Entry", "Disease_Synonymous", "Definitions"]
        texts = []
        for disease_id in disease_ids:
            if disease_id not in df.index:
                texts.append("")
                continue
            row = df.loc[disease_id]
            texts.append(" ".join(str(row[field]) for field in fields if field in df.columns and pd.notna(row[field])))
    else:
        df = pd.read_csv(csv_path).set_index("DiseaseID")
        texts = [str(df.loc[disease_id, "Disease_Name"]) if disease_id in df.index else "" for disease_id in disease_ids]

    vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(texts)
    if tfidf.shape[1] <= output_dim:
        dense = tfidf.toarray().astype(np.float32)
        if dense.shape[1] < output_dim:
            dense = np.pad(dense, ((0, 0), (0, output_dim - dense.shape[1])))
        return dense
    svd = TruncatedSVD(n_components=output_dim, random_state=42)
    return svd.fit_transform(tfidf).astype(np.float32)
def _normalize_column_name(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _pick_column(df, candidates):
    lookup = {_normalize_column_name(column): column for column in df.columns}
    for candidate in candidates:
        column = lookup.get(_normalize_column_name(candidate))
        if column is not None:
            return column
    return None


def _first_existing_file(directory, names):
    for name in names:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


def _safe_text(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _resolve_index(row, idx_col, id_col, id_to_idx, upper_bound):
    if idx_col is not None and pd.notna(row[idx_col]):
        try:
            idx = int(row[idx_col])
            if 0 <= idx < upper_bound:
                return idx
        except Exception:
            pass
    if id_col is not None:
        entity_id = _safe_text(row[id_col])
        if entity_id in id_to_idx:
            return int(id_to_idx[entity_id])
    return None


def _target_id(row, target_col):
    if target_col is None:
        return None
    return _safe_text(row[target_col])


def _load_id_to_idx(project_root, dataset, name):
    path = os.path.join(project_root, "dataset", dataset, f"{name}_idx_to_id.npy")
    if not os.path.exists(path):
        return {}
    idx_to_id = np.load(path, allow_pickle=True).item()
    return {str(entity_id): int(idx) for idx, entity_id in idx_to_id.items()}


def _add_target(target_to_idx, target_sets, entity_idx, target_name):
    if entity_idx is None or target_name is None:
        return False
    target_idx = target_to_idx.setdefault(target_name, len(target_to_idx))
    target_sets[entity_idx].add(target_idx)
    return True


def _load_entity_target_edges(edge_path, entity_kind, id_to_idx, upper_bound, target_to_idx):
    target_sets = [set() for _ in range(upper_bound)]
    if edge_path is None:
        return target_sets, 0

    df = pd.read_csv(edge_path)
    if entity_kind == "drug":
        idx_col = _pick_column(df, ["drug_idx", "drug_index", "DrugIndex"])
        id_col = _pick_column(df, ["drug_id", "DrugID", "drug"])
    else:
        idx_col = _pick_column(df, ["disease_idx", "disease_index", "DiseaseIndex"])
        id_col = _pick_column(df, ["disease_id", "DiseaseID", "disease"])

    target_col = _pick_column(df, [
        "target_id",
        "TargetID",
        "target",
        "protein_id",
        "ProteinID",
        "protein",
        "gene_id",
        "GeneID",
        "gene",
        "target_idx",
        "protein_idx",
        "gene_idx",
    ])
    if target_col is None:
        excluded = {column for column in [idx_col, id_col] if column is not None}
        remaining = [column for column in df.columns if column not in excluded]
        target_col = remaining[0] if remaining else None

    edge_count = 0
    for _, row in df.iterrows():
        entity_idx = _resolve_index(row, idx_col, id_col, id_to_idx, upper_bound)
        if _add_target(target_to_idx, target_sets, entity_idx, _target_id(row, target_col)):
            edge_count += 1
    return target_sets, edge_count


def _load_target_target_edges(edge_path, target_to_idx):
    if edge_path is None:
        return [], 0

    df = pd.read_csv(edge_path)
    src_col = _pick_column(df, [
        "source",
        "from",
        "target1",
        "target1_id",
        "protein1",
        "protein1_id",
        "gene1",
        "gene1_id",
        "node1",
    ])
    dst_col = _pick_column(df, [
        "target",
        "to",
        "target2",
        "target2_id",
        "protein2",
        "protein2_id",
        "gene2",
        "gene2_id",
        "node2",
    ])
    if src_col is None or dst_col is None or src_col == dst_col:
        if len(df.columns) < 2:
            return [], 0
        src_col, dst_col = df.columns[:2]

    edges = []
    for _, row in df.iterrows():
        src_name = _safe_text(row[src_col])
        dst_name = _safe_text(row[dst_col])
        if src_name is None or dst_name is None:
            continue
        src_idx = target_to_idx.setdefault(src_name, len(target_to_idx))
        dst_idx = target_to_idx.setdefault(dst_name, len(target_to_idx))
        if src_idx != dst_idx:
            edges.append((src_idx, dst_idx))
    return edges, len(edges)


def load_target_heterograph(project_root, dataset, n_drugs, n_diseases, edge_dir=None):
    edge_dir = edge_dir or os.path.join(project_root, "dataset", dataset, "heterograph")
    meta = {
        "edge_dir": edge_dir,
        "drug_target_edges": 0,
        "disease_target_edges": 0,
        "target_target_edges": 0,
        "num_targets": 0,
    }
    if not os.path.isdir(edge_dir):
        return None, meta

    drug_target_path = _first_existing_file(edge_dir, [
        "drug_target.csv",
        "drug_targets.csv",
        "drug_protein.csv",
        "drug_proteins.csv",
        "drug_gene.csv",
        "drug_genes.csv",
        "drug_target_edges.csv",
    ])
    disease_target_path = _first_existing_file(edge_dir, [
        "disease_target.csv",
        "disease_targets.csv",
        "disease_protein.csv",
        "disease_proteins.csv",
        "disease_gene.csv",
        "disease_genes.csv",
        "disease_target_edges.csv",
    ])
    target_target_path = _first_existing_file(edge_dir, [
        "target_target.csv",
        "target_targets.csv",
        "protein_protein.csv",
        "ppi.csv",
        "protein_gene.csv",
        "gene_gene.csv",
        "target_edges.csv",
    ])

    target_to_idx = {}
    drug_id_to_idx = _load_id_to_idx(project_root, dataset, "drug")
    disease_id_to_idx = _load_id_to_idx(project_root, dataset, "disease")

    drug_targets, meta["drug_target_edges"] = _load_entity_target_edges(
        drug_target_path, "drug", drug_id_to_idx, n_drugs, target_to_idx
    )
    disease_targets, meta["disease_target_edges"] = _load_entity_target_edges(
        disease_target_path, "disease", disease_id_to_idx, n_diseases, target_to_idx
    )
    target_edges, meta["target_target_edges"] = _load_target_target_edges(target_target_path, target_to_idx)

    target_neighbors = [set() for _ in range(len(target_to_idx))]
    for src_idx, dst_idx in target_edges:
        while max(src_idx, dst_idx) >= len(target_neighbors):
            target_neighbors.append(set())
        target_neighbors[src_idx].add(dst_idx)
        target_neighbors[dst_idx].add(src_idx)

    meta["num_targets"] = len(target_to_idx)
    heterograph = {
        "drug_targets": drug_targets,
        "disease_targets": disease_targets,
        "target_neighbors": target_neighbors,
        "target_to_idx": target_to_idx,
        "meta": meta,
    }
    return heterograph, meta


def build_path_feature_matrix(project_root, dataset, n_drugs, n_diseases, edge_dir=None, standardize=True):
    heterograph, meta = load_target_heterograph(project_root, dataset, n_drugs, n_diseases, edge_dir=edge_dir)
    if heterograph is None:
        return np.zeros((n_drugs, n_diseases, 0), dtype=np.float32), meta
    if meta["drug_target_edges"] == 0 or meta["disease_target_edges"] == 0:
        return np.zeros((n_drugs, n_diseases, 0), dtype=np.float32), meta

    drug_targets = heterograph["drug_targets"]
    disease_targets = heterograph["disease_targets"]
    target_neighbors = heterograph["target_neighbors"]
    n_targets = meta["num_targets"]

    target_degrees = np.zeros(n_targets, dtype=np.float32)
    for targets in drug_targets:
        for target_idx in targets:
            target_degrees[target_idx] += 1.0
    for targets in disease_targets:
        for target_idx in targets:
            target_degrees[target_idx] += 1.0
    for target_idx, neighbors in enumerate(target_neighbors):
        target_degrees[target_idx] += len(neighbors)

    features = np.zeros((n_drugs, n_diseases, 6), dtype=np.float32)
    for drug_idx, d_targets in enumerate(drug_targets):
        if not d_targets:
            continue
        d_count = len(d_targets)
        for disease_idx, s_targets in enumerate(disease_targets):
            if not s_targets:
                continue
            s_count = len(s_targets)
            common = d_targets.intersection(s_targets)

            bridge_count = 0
            for target_idx in d_targets:
                if target_idx < len(target_neighbors):
                    bridge_count += len(target_neighbors[target_idx].intersection(s_targets))

            denom = math.sqrt(d_count * s_count)
            common_count = float(len(common))
            bridge_count = float(bridge_count)
            adamic_adar = sum(1.0 / math.log(max(float(target_degrees[target_idx]), 2.0)) for target_idx in common)
            resource_allocation = sum(1.0 / max(float(target_degrees[target_idx]), 1.0) for target_idx in common)
            features[drug_idx, disease_idx] = np.asarray([
                common_count,
                bridge_count,
                common_count / denom,
                adamic_adar,
                resource_allocation,
                bridge_count / denom,
            ], dtype=np.float32)

    if standardize:
        flat = features.reshape(-1, features.shape[-1])
        features = standardize_features(flat).reshape(features.shape).astype(np.float32)
    return features, meta


def _safe_float_value(value, default=0.0):
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _target_evidence_cache_name(standardize):
    return f"target_evidence_features_{'std' if standardize else 'raw'}.npy"


def _safe_divide_matrix(numerator, denominator):
    return (numerator / np.maximum(denominator, 1e-6)).astype(np.float32)


def build_target_evidence_feature_matrix(
    project_root,
    dataset,
    n_drugs,
    n_diseases,
    edge_dir=None,
    standardize=True,
    use_cache=True,
):
    try:
        from scipy.sparse import csr_matrix
    except Exception as exc:
        print(f"scipy is required for target evidence features: {exc}")
        return np.zeros((n_drugs, n_diseases, 0), dtype=np.float32), {"feature_names": []}

    edge_dir = edge_dir or os.path.join(project_root, "dataset", dataset, "heterograph")
    cache_path = os.path.join(edge_dir, _target_evidence_cache_name(standardize))
    feature_names = [
        "direct_target_overlap_count",
        "direct_target_jaccard",
        "drug_target_coverage",
        "disease_target_coverage",
        "direct_disease_score_sum",
        "direct_disease_score_per_drug_target",
        "direct_disease_score_coverage",
        "direct_moa_weighted_score",
        "ppi_bridge_count",
        "ppi_bridge_count_norm",
        "ppi_bridge_disease_score",
        "ppi_bridge_moa_weighted_score",
    ]
    if use_cache and os.path.exists(cache_path):
        features = np.load(cache_path)
        if features.shape == (n_drugs, n_diseases, len(feature_names)):
            return features.astype(np.float32), {
                "feature_names": feature_names,
                "cache_path": cache_path,
                "cached": True,
            }
        print(f"Ignoring target evidence cache with unexpected shape {features.shape}: {cache_path}")

    if not os.path.isdir(edge_dir):
        return np.zeros((n_drugs, n_diseases, 0), dtype=np.float32), {"feature_names": []}

    drug_target_path = _first_existing_file(edge_dir, ["drug_target.csv", "drug_targets.csv", "drug_protein.csv"])
    disease_target_path = _first_existing_file(edge_dir, ["disease_target.csv", "disease_targets.csv", "disease_protein.csv"])
    target_target_path = _first_existing_file(edge_dir, ["target_target.csv", "target_targets.csv", "ppi.csv"])
    if drug_target_path is None or disease_target_path is None:
        return np.zeros((n_drugs, n_diseases, 0), dtype=np.float32), {"feature_names": []}

    target_to_idx = {}

    def target_index(target_name):
        target_name = _safe_text(target_name)
        if target_name is None:
            return None
        return target_to_idx.setdefault(target_name, len(target_to_idx))

    drug_df = pd.read_csv(drug_target_path)
    drug_idx_col = _pick_column(drug_df, ["DrugIndex", "drug_idx", "drug_index"])
    drug_target_col = _pick_column(drug_df, ["target_id", "gene", "TargetID", "target"])
    moa_col = _pick_column(drug_df, ["moa", "MOA", "is_moa"])
    drug_binary_pairs = set()
    drug_moa_values = {}
    if drug_idx_col is not None and drug_target_col is not None:
        for _, row in drug_df.iterrows():
            try:
                drug_idx = int(row[drug_idx_col])
            except Exception:
                continue
            if not (0 <= drug_idx < n_drugs):
                continue
            t_idx = target_index(row[drug_target_col])
            if t_idx is None:
                continue
            drug_binary_pairs.add((drug_idx, t_idx))
            moa_value = 0.0
            if moa_col is not None:
                moa_value = 1.0 if _safe_float_value(row[moa_col], 0.0) > 0 else 0.0
            if moa_value > drug_moa_values.get((drug_idx, t_idx), 0.0):
                drug_moa_values[(drug_idx, t_idx)] = moa_value

    disease_df = pd.read_csv(disease_target_path)
    disease_idx_col = _pick_column(disease_df, ["DiseaseIndex", "disease_idx", "disease_index"])
    disease_target_col = _pick_column(disease_df, ["target_id", "gene", "TargetID", "target"])
    disease_score_col = _pick_column(disease_df, ["score", "association_score", "confidence"])
    disease_binary_pairs = set()
    disease_score_values = {}
    if disease_idx_col is not None and disease_target_col is not None:
        for _, row in disease_df.iterrows():
            try:
                disease_idx = int(row[disease_idx_col])
            except Exception:
                continue
            if not (0 <= disease_idx < n_diseases):
                continue
            t_idx = target_index(row[disease_target_col])
            if t_idx is None:
                continue
            disease_binary_pairs.add((disease_idx, t_idx))
            score = _safe_float_value(row[disease_score_col], 1.0) if disease_score_col is not None else 1.0
            key = (disease_idx, t_idx)
            if score > disease_score_values.get(key, 0.0):
                disease_score_values[key] = score

    target_weight_values = {}
    target_binary_pairs = set()
    if target_target_path is not None:
        target_df = pd.read_csv(target_target_path)
        src_col = _pick_column(target_df, ["source", "from", "target1", "protein1", "gene1"])
        dst_col = _pick_column(target_df, ["target", "to", "target2", "protein2", "gene2"])
        score_col = _pick_column(target_df, ["score", "combined_score", "confidence"])
        if src_col is not None and dst_col is not None:
            for _, row in target_df.iterrows():
                src_idx = target_index(row[src_col])
                dst_idx = target_index(row[dst_col])
                if src_idx is None or dst_idx is None or src_idx == dst_idx:
                    continue
                score = _safe_float_value(row[score_col], 1.0) if score_col is not None else 1.0
                for edge in [(src_idx, dst_idx), (dst_idx, src_idx)]:
                    target_binary_pairs.add(edge)
                    if score > target_weight_values.get(edge, 0.0):
                        target_weight_values[edge] = score

    n_targets = len(target_to_idx)
    if n_targets == 0 or not drug_binary_pairs or not disease_binary_pairs:
        return np.zeros((n_drugs, n_diseases, 0), dtype=np.float32), {"feature_names": []}

    def sparse_from_pairs(pairs, shape, values=None):
        if values is None:
            rows, cols = zip(*pairs) if pairs else ([], [])
            data = np.ones(len(pairs), dtype=np.float32)
        else:
            rows = [key[0] for key in values]
            cols = [key[1] for key in values]
            data = np.asarray([values[key] for key in values], dtype=np.float32)
        return csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)

    drug_binary = sparse_from_pairs(drug_binary_pairs, (n_drugs, n_targets))
    drug_moa = sparse_from_pairs(None, (n_drugs, n_targets), values=drug_moa_values)
    disease_binary = sparse_from_pairs(disease_binary_pairs, (n_diseases, n_targets))
    disease_score = sparse_from_pairs(None, (n_diseases, n_targets), values=disease_score_values)
    target_binary = sparse_from_pairs(target_binary_pairs, (n_targets, n_targets))
    target_weight = sparse_from_pairs(None, (n_targets, n_targets), values=target_weight_values)

    drug_degree = np.asarray(drug_binary.sum(axis=1)).reshape(-1).astype(np.float32)
    disease_degree = np.asarray(disease_binary.sum(axis=1)).reshape(-1).astype(np.float32)
    disease_score_sum = np.asarray(disease_score.sum(axis=1)).reshape(-1).astype(np.float32)

    direct_count = (drug_binary @ disease_binary.T).toarray().astype(np.float32)
    direct_score = (drug_binary @ disease_score.T).toarray().astype(np.float32)
    direct_moa_score = (drug_moa @ disease_score.T).toarray().astype(np.float32)

    ppi_bridge_count = ((drug_binary @ target_binary) @ disease_binary.T).toarray().astype(np.float32)
    ppi_bridge_score = ((drug_binary @ target_weight) @ disease_score.T).toarray().astype(np.float32)
    ppi_bridge_moa_score = ((drug_moa @ target_weight) @ disease_score.T).toarray().astype(np.float32)

    drug_degree_matrix = drug_degree[:, None]
    disease_degree_matrix = disease_degree[None, :]
    disease_score_matrix = disease_score_sum[None, :]
    denom = np.sqrt(np.maximum(drug_degree_matrix * disease_degree_matrix, 1e-6)).astype(np.float32)
    union = drug_degree_matrix + disease_degree_matrix - direct_count

    features = np.stack(
        [
            direct_count,
            _safe_divide_matrix(direct_count, union),
            _safe_divide_matrix(direct_count, drug_degree_matrix),
            _safe_divide_matrix(direct_count, disease_degree_matrix),
            direct_score,
            _safe_divide_matrix(direct_score, drug_degree_matrix),
            _safe_divide_matrix(direct_score, disease_score_matrix),
            direct_moa_score,
            ppi_bridge_count,
            _safe_divide_matrix(ppi_bridge_count, denom),
            ppi_bridge_score,
            ppi_bridge_moa_score,
        ],
        axis=-1,
    ).astype(np.float32)

    if standardize:
        flat = features.reshape(-1, features.shape[-1])
        features = standardize_features(flat).reshape(features.shape).astype(np.float32)
    if use_cache:
        os.makedirs(edge_dir, exist_ok=True)
        np.save(cache_path, features)
    return features, {
        "feature_names": feature_names,
        "cache_path": cache_path,
        "cached": False,
        "num_targets": n_targets,
        "drug_target_edges": len(drug_binary_pairs),
        "disease_target_edges": len(disease_binary_pairs),
        "target_target_edges": len(target_binary_pairs) // 2,
    }


def _pathway_evidence_cache_name(standardize):
    return f"reactome_pathway_evidence_features_{'std' if standardize else 'raw'}.npy"


def _download_reactome_gmt(gmt_zip_path):
    import urllib.request

    url = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
    os.makedirs(os.path.dirname(gmt_zip_path), exist_ok=True)
    print(f"Downloading Reactome pathway GMT: {url}")
    urllib.request.urlretrieve(url, gmt_zip_path)


def _load_reactome_gene_to_pathways(edge_dir, reactome_gmt_path=None):
    import zipfile

    cache_dir = os.path.join(edge_dir, "cache")
    gmt_zip_path = reactome_gmt_path or os.path.join(cache_dir, "ReactomePathways.gmt.zip")
    if not os.path.exists(gmt_zip_path):
        _download_reactome_gmt(gmt_zip_path)

    gene_to_pathways = {}
    pathway_names = []
    pathway_ids = []
    with zipfile.ZipFile(gmt_zip_path) as zf:
        gmt_names = [name for name in zf.namelist() if name.endswith(".gmt")]
        if not gmt_names:
            raise FileNotFoundError(f"No .gmt file found inside {gmt_zip_path}")
        with zf.open(gmt_names[0]) as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                pathway_name, pathway_id = parts[0], parts[1]
                if not str(pathway_id).startswith("R-HSA-"):
                    continue
                pathway_idx = len(pathway_names)
                pathway_names.append(pathway_name)
                pathway_ids.append(pathway_id)
                for gene in parts[2:]:
                    gene = _safe_text(gene)
                    if gene is None:
                        continue
                    gene_to_pathways.setdefault(gene.upper(), set()).add(pathway_idx)
    return gene_to_pathways, pathway_names, pathway_ids, gmt_zip_path


def build_reactome_pathway_feature_matrix(
    project_root,
    dataset,
    n_drugs,
    n_diseases,
    edge_dir=None,
    reactome_gmt_path=None,
    standardize=True,
    use_cache=True,
):
    try:
        from scipy.sparse import csr_matrix, diags
    except Exception as exc:
        print(f"scipy is required for Reactome pathway features: {exc}")
        return np.zeros((n_drugs, n_diseases, 0), dtype=np.float32), {"feature_names": []}

    edge_dir = edge_dir or os.path.join(project_root, "dataset", dataset, "heterograph")
    cache_path = os.path.join(edge_dir, _pathway_evidence_cache_name(standardize))
    feature_names = [
        "reactome_pathway_overlap_count",
        "reactome_pathway_jaccard",
        "reactome_drug_pathway_coverage",
        "reactome_disease_pathway_coverage",
        "reactome_disease_score_sum",
        "reactome_disease_score_per_drug_pathway",
        "reactome_disease_score_coverage",
        "reactome_moa_pathway_overlap_count",
        "reactome_moa_pathway_coverage",
        "reactome_moa_disease_score_sum",
        "reactome_pathway_adamic_adar",
        "reactome_moa_pathway_adamic_adar",
    ]
    if use_cache and os.path.exists(cache_path):
        features = np.load(cache_path)
        if features.shape == (n_drugs, n_diseases, len(feature_names)):
            return features.astype(np.float32), {
                "feature_names": feature_names,
                "cache_path": cache_path,
                "cached": True,
            }
        print(f"Ignoring Reactome pathway cache with unexpected shape {features.shape}: {cache_path}")

    if not os.path.isdir(edge_dir):
        return np.zeros((n_drugs, n_diseases, 0), dtype=np.float32), {"feature_names": []}
    drug_target_path = _first_existing_file(edge_dir, ["drug_target.csv", "drug_targets.csv", "drug_protein.csv"])
    disease_target_path = _first_existing_file(edge_dir, ["disease_target.csv", "disease_targets.csv", "disease_protein.csv"])
    if drug_target_path is None or disease_target_path is None:
        return np.zeros((n_drugs, n_diseases, 0), dtype=np.float32), {"feature_names": []}

    try:
        gene_to_pathways, pathway_names, pathway_ids, gmt_zip_path = _load_reactome_gene_to_pathways(
            edge_dir,
            reactome_gmt_path=reactome_gmt_path,
        )
    except Exception as exc:
        print(f"Reactome pathway features are unavailable: {exc}")
        return np.zeros((n_drugs, n_diseases, 0), dtype=np.float32), {"feature_names": []}
    n_pathways = len(pathway_names)
    if n_pathways == 0:
        return np.zeros((n_drugs, n_diseases, 0), dtype=np.float32), {"feature_names": []}

    drug_df = pd.read_csv(drug_target_path)
    drug_idx_col = _pick_column(drug_df, ["DrugIndex", "drug_idx", "drug_index"])
    drug_gene_col = _pick_column(drug_df, ["gene", "target_id", "TargetID", "target"])
    moa_col = _pick_column(drug_df, ["moa", "MOA", "is_moa"])
    drug_pairs = set()
    drug_moa_values = {}
    if drug_idx_col is not None and drug_gene_col is not None:
        for _, row in drug_df.iterrows():
            try:
                drug_idx = int(row[drug_idx_col])
            except Exception:
                continue
            if not (0 <= drug_idx < n_drugs):
                continue
            gene = _safe_text(row[drug_gene_col])
            if gene is None:
                continue
            pathways = gene_to_pathways.get(gene.upper())
            if not pathways:
                continue
            moa_value = 0.0
            if moa_col is not None:
                moa_value = 1.0 if _safe_float_value(row[moa_col], 0.0) > 0 else 0.0
            for pathway_idx in pathways:
                drug_pairs.add((drug_idx, pathway_idx))
                if moa_value > drug_moa_values.get((drug_idx, pathway_idx), 0.0):
                    drug_moa_values[(drug_idx, pathway_idx)] = moa_value

    disease_df = pd.read_csv(disease_target_path)
    disease_idx_col = _pick_column(disease_df, ["DiseaseIndex", "disease_idx", "disease_index"])
    disease_gene_col = _pick_column(disease_df, ["gene", "target_id", "TargetID", "target"])
    disease_score_col = _pick_column(disease_df, ["score", "association_score", "confidence"])
    disease_pairs = set()
    disease_score_values = {}
    if disease_idx_col is not None and disease_gene_col is not None:
        for _, row in disease_df.iterrows():
            try:
                disease_idx = int(row[disease_idx_col])
            except Exception:
                continue
            if not (0 <= disease_idx < n_diseases):
                continue
            gene = _safe_text(row[disease_gene_col])
            if gene is None:
                continue
            pathways = gene_to_pathways.get(gene.upper())
            if not pathways:
                continue
            score = _safe_float_value(row[disease_score_col], 1.0) if disease_score_col is not None else 1.0
            for pathway_idx in pathways:
                disease_pairs.add((disease_idx, pathway_idx))
                key = (disease_idx, pathway_idx)
                if score > disease_score_values.get(key, 0.0):
                    disease_score_values[key] = score

    if not drug_pairs or not disease_pairs:
        return np.zeros((n_drugs, n_diseases, 0), dtype=np.float32), {
            "feature_names": [],
            "num_pathways": n_pathways,
            "reactome_gmt_path": gmt_zip_path,
        }

    def sparse_from_pairs(pairs, shape, values=None):
        if values is None:
            rows, cols = zip(*pairs) if pairs else ([], [])
            data = np.ones(len(pairs), dtype=np.float32)
        else:
            rows = [key[0] for key in values]
            cols = [key[1] for key in values]
            data = np.asarray([values[key] for key in values], dtype=np.float32)
        return csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)

    drug_binary = sparse_from_pairs(drug_pairs, (n_drugs, n_pathways))
    drug_moa = sparse_from_pairs(None, (n_drugs, n_pathways), values=drug_moa_values)
    disease_binary = sparse_from_pairs(disease_pairs, (n_diseases, n_pathways))
    disease_score = sparse_from_pairs(None, (n_diseases, n_pathways), values=disease_score_values)

    drug_degree = np.asarray(drug_binary.sum(axis=1)).reshape(-1).astype(np.float32)
    disease_degree = np.asarray(disease_binary.sum(axis=1)).reshape(-1).astype(np.float32)
    disease_score_sum = np.asarray(disease_score.sum(axis=1)).reshape(-1).astype(np.float32)

    overlap_count = (drug_binary @ disease_binary.T).toarray().astype(np.float32)
    disease_score_overlap = (drug_binary @ disease_score.T).toarray().astype(np.float32)
    moa_overlap_count = (drug_moa @ disease_binary.T).toarray().astype(np.float32)
    moa_disease_score = (drug_moa @ disease_score.T).toarray().astype(np.float32)

    pathway_entity_degrees = (
        np.asarray(drug_binary.sum(axis=0)).reshape(-1)
        + np.asarray(disease_binary.sum(axis=0)).reshape(-1)
    ).astype(np.float32)
    adamic_weights = 1.0 / np.log(np.maximum(pathway_entity_degrees, 2.0))
    adamic_diag = diags(adamic_weights.astype(np.float32), offsets=0, format="csr")
    pathway_adamic = ((drug_binary @ adamic_diag) @ disease_binary.T).toarray().astype(np.float32)
    moa_pathway_adamic = ((drug_moa @ adamic_diag) @ disease_binary.T).toarray().astype(np.float32)

    drug_degree_matrix = drug_degree[:, None]
    disease_degree_matrix = disease_degree[None, :]
    disease_score_matrix = disease_score_sum[None, :]
    union = drug_degree_matrix + disease_degree_matrix - overlap_count

    features = np.stack(
        [
            overlap_count,
            _safe_divide_matrix(overlap_count, union),
            _safe_divide_matrix(overlap_count, drug_degree_matrix),
            _safe_divide_matrix(overlap_count, disease_degree_matrix),
            disease_score_overlap,
            _safe_divide_matrix(disease_score_overlap, drug_degree_matrix),
            _safe_divide_matrix(disease_score_overlap, disease_score_matrix),
            moa_overlap_count,
            _safe_divide_matrix(moa_overlap_count, drug_degree_matrix),
            moa_disease_score,
            pathway_adamic,
            moa_pathway_adamic,
        ],
        axis=-1,
    ).astype(np.float32)

    if standardize:
        flat = features.reshape(-1, features.shape[-1])
        features = standardize_features(flat).reshape(features.shape).astype(np.float32)
    if use_cache:
        os.makedirs(edge_dir, exist_ok=True)
        np.save(cache_path, features)
    return features, {
        "feature_names": feature_names,
        "cache_path": cache_path,
        "cached": False,
        "num_pathways": n_pathways,
        "reactome_gmt_path": gmt_zip_path,
        "drug_pathway_edges": len(drug_pairs),
        "disease_pathway_edges": len(disease_pairs),
    }


def tensor_features(drug_features, disease_features, device, path_features=None):
    path_tensor = None
    if path_features is not None and path_features.shape[-1] > 0:
        path_tensor = torch.tensor(path_features, dtype=torch.float32, device=device)
    return {
        "drug": [torch.tensor(feat, dtype=torch.float32, device=device) for feat in drug_features],
        "disease": [torch.tensor(feat, dtype=torch.float32, device=device) for feat in disease_features],
        "num_drugs": drug_features[0].shape[0],
        "path_features": path_tensor,
    }


def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_label = (y_score >= threshold).astype(np.int64)
    return {
        "auroc": roc_auc_score(y_true, y_score),
        "aupr": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_label),
        "accuracy": accuracy_score(y_true, y_label),
        "threshold": threshold,
    }
