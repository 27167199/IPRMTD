import argparse
import csv
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Att import AttentionWeightedPooling
from loader import data_preparation
from hybrid_model import (
    HybridDDA,
    balanced_contrastive_weight,
    build_disease_text_features,
    build_graphs,
    build_path_feature_matrix,
    build_reactome_pathway_feature_matrix,
    build_target_evidence_feature_matrix,
    compute_metrics,
    cross_view_contrastive_loss,
    intra_view_contrastive_loss,
    set_seed,
    standardize_features,
    tensor_features,
)


def parse_args():
    parser = argparse.ArgumentParser(description="IPRMTD final drug-disease association model")

    parser.add_argument("--dataset", default="appoved", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_interval", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--semantic_topk", type=int, default=10)
    parser.add_argument("--fusion_method", default="attention", choices=["attention", "gated"])
    parser.add_argument("--contrastive_weight", type=float, default=0.2)
    parser.add_argument("--intra_contrastive_weight", type=float, default=0.3)
    parser.add_argument("--aux_weight", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--ranking_topks", default="1,3,5,10")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument(
        "--drug_emb_path",
        default=os.path.join(PROJECT_ROOT, "dataset", "appoved", "emb", "drug_embeddings"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disease_emb_path",
        default=os.path.join(PROJECT_ROOT, "dataset", "appoved", "emb", "disease_embeddings.npy"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--heterograph_dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reactome_gmt_path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--disease_TopK", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--drug_TopK", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--pooling_method", default="attention", choices=["mean", "attention"], help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=0.2, help=argparse.SUPPRESS)
    parser.add_argument("--max_contrastive_nodes", type=int, default=512, help=argparse.SUPPRESS)
    parser.add_argument("--max_intra_edges", type=int, default=1024, help=argparse.SUPPRESS)
    parser.add_argument("--no_gradient_balance", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--conflict_scale", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--no_disease_text", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--disease_text_dim", type=int, default=128, help=argparse.SUPPRESS)
    parser.add_argument("--use_path_features", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--no_path_features", dest="use_path_features", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--use_target_evidence_features", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no_target_evidence_features",
        dest="use_target_evidence_features",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no_target_evidence_cache", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--use_reactome_pathway_features", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no_reactome_pathway_features",
        dest="use_reactome_pathway_features",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no_reactome_pathway_cache", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--use_oof_teachers", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--no_oof_teachers", dest="use_oof_teachers", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--teacher_inner_folds", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--teacher_weight", type=float, default=0.05, help=argparse.SUPPRESS)
    parser.add_argument("--teacher_eval_rank_blend", type=float, default=0.02, help=argparse.SUPPRESS)
    parser.add_argument("--teacher_estimators", type=int, default=300, help=argparse.SUPPRESS)
    parser.add_argument("--teacher_max_depth", type=int, default=24, help=argparse.SUPPRESS)
    parser.add_argument("--teacher_min_samples_leaf", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--teacher_gbdt_iter", type=int, default=250, help=argparse.SUPPRESS)
    parser.add_argument("--teacher_gbdt_lr", type=float, default=0.04, help=argparse.SUPPRESS)
    parser.add_argument("--teacher_n_jobs", type=int, default=-1, help=argparse.SUPPRESS)

    return parser.parse_args()


def setup_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def process_drug_embedding(emb_data, pooling_method="attention"):
    m_drug = None
    if isinstance(emb_data, dict) and "embeddings" in emb_data:
        embedding_obj = emb_data["embeddings"]
        if hasattr(embedding_obj, "x"):
            m_seq = embedding_obj.x
        else:
            m_seq = embedding_obj
    elif isinstance(emb_data, torch.Tensor):
        m_seq = emb_data.detach()
    else:
        return None

    if len(m_seq.shape) == 3:
        m_seq = m_seq.squeeze(0)
    if len(m_seq.shape) == 2:
        if pooling_method == "attention" and m_seq.shape[0] > 1:
            pooling_module = AttentionWeightedPooling(embed_dim=m_seq.shape[1])
            m_drug, _ = pooling_module(m_seq.cpu())
            m_drug = m_drug.detach().cpu().numpy()
        else:
            m_drug = m_seq.mean(dim=0).detach().cpu().numpy()
    elif len(m_seq.shape) == 1:
        m_drug = m_seq.detach().cpu().numpy()
    return m_drug


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_embeddings(args):
    disease_emb_path = args.disease_emb_path
    if os.path.isdir(disease_emb_path):
        disease_emb_path = os.path.join(disease_emb_path, "disease_embeddings.npy")
    if not os.path.exists(disease_emb_path):
        raise FileNotFoundError(f"Disease embedding file not found: {disease_emb_path}")
    disease_embeddings = np.load(disease_emb_path).astype(np.float32)

    drug_idx_to_id_path = os.path.join(PROJECT_ROOT, "dataset", args.dataset, "drug_idx_to_id.npy")
    drug_idx_to_id = np.load(drug_idx_to_id_path, allow_pickle=True).item()
    drug_id_to_idx = {str(drug_id): int(idx) for idx, drug_id in drug_idx_to_id.items()}
    n_drugs = len(drug_idx_to_id)

    drug_embeddings = None
    if os.path.isfile(args.drug_emb_path) and args.drug_emb_path.endswith(".npy"):
        drug_embeddings = np.load(args.drug_emb_path).astype(np.float32)
    else:
        first_dim = 384
        collected = {}
        if os.path.isdir(args.drug_emb_path):
            for file_name in os.listdir(args.drug_emb_path):
                if not file_name.endswith(".pt"):
                    continue
                drug_id = file_name.split("_embedded.pt")[0]
                if drug_id not in drug_id_to_idx:
                    continue
                emb_path = os.path.join(args.drug_emb_path, file_name)
                try:
                    emb_data = torch_load_cpu(emb_path)
                    vector = process_drug_embedding(emb_data, args.pooling_method)
                except Exception:
                    vector = None
                if vector is None:
                    continue
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                first_dim = vector.shape[0]
                collected[drug_id_to_idx[drug_id]] = vector

        drug_embeddings = np.zeros((n_drugs, first_dim), dtype=np.float32)
        for drug_idx, vector in collected.items():
            if vector.shape[0] == first_dim:
                drug_embeddings[drug_idx] = vector
        print(f"Loaded drug embeddings for {len(collected)}/{n_drugs} drugs; dim={first_dim}")

    return standardize_features(drug_embeddings), standardize_features(disease_embeddings)


def concat_node_features(features):
    blocks = [np.asarray(feat, dtype=np.float32) for feat in features if feat is not None and feat.shape[1] > 0]
    if len(blocks) == 1:
        return blocks[0]
    return standardize_features(np.concatenate(blocks, axis=1))


def append_pair_feature_blocks(blocks):
    active = [block for block in blocks if block is not None and block.shape[-1] > 0]
    if not active:
        return None
    if len(active) == 1:
        return active[0].astype(np.float32)
    return np.concatenate(active, axis=-1).astype(np.float32)


def prepare_feature_blocks(args, n_drugs, n_diseases, device):
    drug_pretrained, disease_pretrained = load_embeddings(args)
    drug_features = [drug_pretrained]
    disease_features = [disease_pretrained]
    pair_feature_names = []

    if not args.no_disease_text:
        print("Building disease text-description modality...")
        disease_text = build_disease_text_features(
            PROJECT_ROOT,
            args.dataset,
            n_diseases,
            output_dim=args.disease_text_dim,
        )
        disease_features.append(standardize_features(disease_text))

    pair_blocks = []
    if args.use_path_features:
        path_features, meta = build_path_feature_matrix(
            PROJECT_ROOT,
            args.dataset,
            n_drugs,
            n_diseases,
            edge_dir=args.heterograph_dir,
        )
        print(
            "Path features enabled: "
            f"targets={meta.get('num_targets', 'NA')}, "
            f"drug-target={meta.get('drug_target_edges', 0)}, "
            f"disease-target={meta.get('disease_target_edges', 0)}, "
            f"target-target={meta.get('target_target_edges', 0)}, "
            f"dim={path_features.shape[-1]}"
        )
        pair_blocks.append(path_features)
        pair_feature_names.extend(
            [
                "path_direct_overlap_count",
                "path_target_bridge_count",
                "path_normalized_overlap",
                "path_adamic_adar",
                "path_resource_allocation",
                "path_normalized_bridge",
            ][: path_features.shape[-1]]
        )

    if args.use_target_evidence_features:
        target_evidence_features, meta = build_target_evidence_feature_matrix(
            PROJECT_ROOT,
            args.dataset,
            n_drugs,
            n_diseases,
            edge_dir=args.heterograph_dir,
            use_cache=not args.no_target_evidence_cache,
        )
        cache_text = f", loaded from cache: {meta.get('cache_path')}" if meta.get("cached") else ""
        print(
            "Target evidence features enabled: "
            f"targets={meta.get('num_targets', 'NA')}, "
            f"drug-target={meta.get('drug_target_edges', 0)}, "
            f"disease-target={meta.get('disease_target_edges', 0)}, "
            f"target-target={meta.get('target_target_edges', 0)}, "
            f"dim={target_evidence_features.shape[-1]}{cache_text}"
        )
        if meta.get("feature_names"):
            print("Target evidence feature names: " + ", ".join(meta["feature_names"]))
            pair_feature_names.extend(meta["feature_names"])
        else:
            pair_feature_names.extend([f"target_evidence_{i}" for i in range(target_evidence_features.shape[-1])])
        pair_blocks.append(target_evidence_features)

    if args.use_reactome_pathway_features:
        reactome_features, meta = build_reactome_pathway_feature_matrix(
            PROJECT_ROOT,
            args.dataset,
            n_drugs,
            n_diseases,
            edge_dir=args.heterograph_dir,
            reactome_gmt_path=args.reactome_gmt_path,
            use_cache=not args.no_reactome_pathway_cache,
        )
        cache_text = f", loaded from cache: {meta.get('cache_path')}" if meta.get("cached") else ""
        print(
            "Reactome pathway features enabled: "
            f"pathways={meta.get('num_pathways', 'NA')}, "
            f"drug-pathway={meta.get('drug_pathway_edges', 0)}, "
            f"disease-pathway={meta.get('disease_pathway_edges', 0)}, "
            f"dim={reactome_features.shape[-1]}{cache_text}"
        )
        if meta.get("feature_names"):
            print("Reactome pathway feature names: " + ", ".join(meta["feature_names"]))
            pair_feature_names.extend(meta["feature_names"])
        else:
            pair_feature_names.extend([f"reactome_pathway_{i}" for i in range(reactome_features.shape[-1])])
        pair_blocks.append(reactome_features)

    static_pair_features = append_pair_feature_blocks(pair_blocks)
    drug_semantic = concat_node_features(drug_features)
    disease_semantic = concat_node_features(disease_features)
    return (
        drug_features,
        disease_features,
        drug_semantic,
        disease_semantic,
        static_pair_features,
        pair_feature_names,
    )


def pairs_from_mask(mask, association_matrix):
    disease_idx, drug_idx = np.where(mask)
    labels = (association_matrix[disease_idx, drug_idx] > 0).astype(np.int64)
    return np.stack([drug_idx, disease_idx, labels], axis=1).astype(np.int64)


def build_teacher_input(pairs, drug_semantic, disease_semantic, pair_features, args):
    drug_idx = pairs[:, 0]
    disease_idx = pairs[:, 1]
    drug_block = drug_semantic[drug_idx]
    disease_block = disease_semantic[disease_idx]
    blocks = [drug_block, disease_block]

    shared_dim = min(drug_block.shape[1], disease_block.shape[1])
    if shared_dim > 0:
        drug_shared = drug_block[:, :shared_dim]
        disease_shared = disease_block[:, :shared_dim]
        blocks.extend([np.abs(drug_shared - disease_shared), drug_shared * disease_shared])

    if pair_features is not None and pair_features.shape[-1] > 0:
        blocks.append(pair_features[drug_idx, disease_idx])

    return np.concatenate(blocks, axis=1).astype(np.float32)


def make_teacher_models(args, seed):
    return [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=args.teacher_estimators,
                max_depth=args.teacher_max_depth,
                min_samples_leaf=args.teacher_min_samples_leaf,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=args.teacher_n_jobs,
                random_state=seed + 11,
            ),
        ),
        (
            "extra",
            ExtraTreesClassifier(
                n_estimators=args.teacher_estimators,
                max_depth=args.teacher_max_depth,
                min_samples_leaf=args.teacher_min_samples_leaf,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=args.teacher_n_jobs,
                random_state=seed + 23,
            ),
        ),
        (
            "gbdt",
            HistGradientBoostingClassifier(
                max_iter=args.teacher_gbdt_iter,
                learning_rate=args.teacher_gbdt_lr,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=seed + 37,
            ),
        ),
    ]


def predict_teacher_proba(model, x):
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        return np.full(x.shape[0], float(model.classes_[0] == 1), dtype=np.float32)
    class_lookup = {int(label): idx for idx, label in enumerate(model.classes_)}
    return proba[:, class_lookup.get(1, 1)].astype(np.float32)


def build_teacher_feature_columns(probs):
    probs = np.asarray(probs, dtype=np.float32)
    mean = probs.mean(axis=1, keepdims=True)
    disagreement = probs.std(axis=1, keepdims=True)
    return np.concatenate([probs, mean, disagreement], axis=1).astype(np.float32)


def fit_oof_teachers(
    args,
    fold_idx,
    train_pairs,
    test_pairs,
    drug_semantic,
    disease_semantic,
    train_pair_features,
    eval_pair_features,
    pair_feature_names,
):
    x_train = build_teacher_input(train_pairs, drug_semantic, disease_semantic, train_pair_features, args)
    y_train = train_pairs[:, 2].astype(np.int64)
    x_test = build_teacher_input(test_pairs, drug_semantic, disease_semantic, eval_pair_features, args)
    y_test = test_pairs[:, 2].astype(np.int64)

    class_counts = np.bincount(y_train, minlength=2)
    inner_folds = max(2, min(args.teacher_inner_folds, int(class_counts.min())))
    teacher_names = [name for name, _ in make_teacher_models(args, args.seed)]
    oof_probs = np.zeros((len(train_pairs), len(teacher_names)), dtype=np.float32)
    test_probs = np.zeros((len(test_pairs), len(teacher_names)), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=args.seed + fold_idx * 97)

    print(
        f"Fold {fold_idx + 1} OOF teachers: "
        f"models={','.join(teacher_names)}, inner_folds={inner_folds}, feature_dim={x_train.shape[1]}"
    )

    for inner_idx, (fit_idx, valid_idx) in enumerate(splitter.split(x_train, y_train), start=1):
        for teacher_idx, (name, model) in enumerate(make_teacher_models(args, args.seed + fold_idx * 1000 + inner_idx * 101)):
            model.fit(x_train[fit_idx], y_train[fit_idx])
            oof_probs[valid_idx, teacher_idx] = predict_teacher_proba(model, x_train[valid_idx])
        print(f"  teacher inner fold {inner_idx}/{inner_folds}: valid={len(valid_idx)}")

    pair_importance = np.zeros(len(pair_feature_names), dtype=np.float64)
    importance_models = 0
    pair_start = x_train.shape[1] - len(pair_feature_names) if pair_feature_names else None
    for teacher_idx, (name, model) in enumerate(make_teacher_models(args, args.seed + fold_idx * 2000 + 503)):
        model.fit(x_train, y_train)
        test_probs[:, teacher_idx] = predict_teacher_proba(model, x_test)
        if pair_start is not None and hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=np.float64)
            denom = importances.sum()
            if denom > 0 and importances.shape[0] >= pair_start + len(pair_feature_names):
                pair_importance += importances[pair_start : pair_start + len(pair_feature_names)] / denom
                importance_models += 1

    oof_mean = oof_probs.mean(axis=1)
    test_mean = test_probs.mean(axis=1)
    print(
        f"Fold {fold_idx + 1} teacher OOF: "
        f"AUROC={roc_auc_score(y_train, oof_mean):.4f}, "
        f"AUPR={average_precision_score(y_train, oof_mean):.4f}"
    )
    print(
        f"Fold {fold_idx + 1} teacher test-only report: "
        f"AUROC={roc_auc_score(y_test, test_mean):.4f}, "
        f"AUPR={average_precision_score(y_test, test_mean):.4f}"
    )
    if importance_models > 0 and pair_feature_names:
        pair_importance /= float(importance_models)
        top_idx = np.argsort(-pair_importance)[: min(8, len(pair_feature_names))]
        top_text = ", ".join(f"{pair_feature_names[i]}={pair_importance[i]:.4f}" for i in top_idx)
        print(f"Fold {fold_idx + 1} teacher pair-feature importance: {top_text}")

    train_teacher_features = build_teacher_feature_columns(oof_probs)
    test_teacher_features = build_teacher_feature_columns(test_probs)
    return train_teacher_features, test_teacher_features, oof_mean.astype(np.float32), test_mean.astype(np.float32)


def teacher_columns_to_matrices(train_pairs, test_pairs, train_columns, test_columns, n_drugs, n_diseases):
    n_cols = train_columns.shape[1]
    train_matrix = np.zeros((n_drugs, n_diseases, n_cols), dtype=np.float32)
    eval_matrix = np.zeros((n_drugs, n_diseases, n_cols), dtype=np.float32)
    train_matrix[train_pairs[:, 0], train_pairs[:, 1]] = train_columns
    eval_matrix[test_pairs[:, 0], test_pairs[:, 1]] = test_columns
    return train_matrix, eval_matrix


def batch_indices(n_items, batch_size, shuffle=True):
    indices = np.arange(n_items)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n_items, batch_size):
        yield indices[start : start + batch_size]


def aux_bce_loss(outputs, labels):
    losses = []
    for key in ["semantic", "interaction", "fused"]:
        logit = outputs.get(key)
        if logit is not None:
            losses.append(F.binary_cross_entropy_with_logits(logit, labels))
    for logit in outputs.get("single", []):
        losses.append(F.binary_cross_entropy_with_logits(logit, labels))
    if not losses:
        return labels.new_tensor(0.0)
    return torch.stack(losses).mean()


def teacher_soft_targets(teacher_scores, labels):
    return teacher_scores.clamp(0.02, 0.98), torch.ones_like(labels)


def parse_topks(raw_topks):
    topks = []
    for item in str(raw_topks).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("ranking_topks must contain positive integers.")
        topks.append(value)
    return sorted(set(topks))


def _dcg(binary_relevance):
    binary_relevance = np.asarray(binary_relevance, dtype=np.float32)
    if binary_relevance.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(binary_relevance.size, dtype=np.float32) + 2.0)
    return float(np.sum(binary_relevance * discounts))


def ranking_metrics_by_disease(pairs, scores, topks):
    disease_items = {}
    for pair, score in zip(pairs, scores):
        drug_idx, disease_idx, label = int(pair[0]), int(pair[1]), int(pair[2])
        disease_items.setdefault(disease_idx, []).append((drug_idx, int(label), float(score)))

    per_metric = {}
    for k in topks:
        per_metric[f"p_at_{k}"] = []
        per_metric[f"r_at_{k}"] = []
    per_metric["mrr"] = []
    ndcg_k = 10
    per_metric[f"ndcg_at_{ndcg_k}"] = []

    for items in disease_items.values():
        labels = np.asarray([item[1] for item in items], dtype=np.int64)
        if labels.sum() <= 0:
            continue
        scores_for_disease = np.asarray([item[2] for item in items], dtype=np.float32)
        ranked_labels = labels[np.argsort(-scores_for_disease)]
        positive_count = float(labels.sum())

        for k in topks:
            cutoff = min(k, ranked_labels.shape[0])
            hits = float(ranked_labels[:cutoff].sum())
            per_metric[f"p_at_{k}"].append(hits / float(cutoff))
            per_metric[f"r_at_{k}"].append(hits / positive_count)

        positive_ranks = np.where(ranked_labels == 1)[0]
        per_metric["mrr"].append(1.0 / float(positive_ranks[0] + 1))

        cutoff = min(ndcg_k, ranked_labels.shape[0])
        dcg = _dcg(ranked_labels[:cutoff])
        ideal = np.sort(labels)[::-1]
        idcg = _dcg(ideal[:cutoff])
        per_metric[f"ndcg_at_{ndcg_k}"].append(dcg / idcg if idcg > 0 else 0.0)

    metrics = {"ranking_disease_count": len(per_metric["mrr"])}
    for key, values in per_metric.items():
        metrics[key] = float(np.mean(values)) if values else 0.0
    return metrics


def metric_display_name(key):
    if key.startswith("p_at_"):
        return "P@" + key.split("_")[-1]
    if key.startswith("r_at_"):
        return "R@" + key.split("_")[-1]
    if key.startswith("ndcg_at_"):
        return "NDCG@" + key.split("_")[-1]
    if key == "mrr":
        return "MRR"
    return key.upper()


def rank_normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size <= 1:
        return scores
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(scores.size, dtype=np.float32)
    return ranks / float(scores.size - 1)


def train_one_fold(
    args,
    fold_idx,
    association_matrix,
    train_mask,
    test_mask,
    drug_features,
    disease_features,
    drug_semantic,
    disease_semantic,
    static_pair_features,
    static_pair_feature_names,
    device,
    return_artifacts=False,
):
    train_positive_mask = train_mask & (association_matrix > 0)
    train_pairs = pairs_from_mask(train_mask, association_matrix)
    test_pairs = pairs_from_mask(test_mask, association_matrix)

    train_pair_blocks = [static_pair_features]
    eval_pair_blocks = [static_pair_features]
    pair_feature_names = list(static_pair_feature_names)
    train_pair_features = append_pair_feature_blocks(train_pair_blocks)
    eval_pair_features = append_pair_feature_blocks(eval_pair_blocks)
    train_teacher_cols = None
    test_teacher_cols = None
    teacher_train_target = None
    teacher_test_target = None
    teacher_feature_names = []
    if args.use_oof_teachers:
        n_diseases, n_drugs = association_matrix.shape
        train_teacher_cols, test_teacher_cols, teacher_train_target, teacher_test_target = fit_oof_teachers(
            args,
            fold_idx,
            train_pairs,
            test_pairs,
            drug_semantic,
            disease_semantic,
            train_pair_features,
            eval_pair_features,
            pair_feature_names,
        )
        train_teacher_matrix, eval_teacher_matrix = teacher_columns_to_matrices(
            train_pairs,
            test_pairs,
            train_teacher_cols,
            test_teacher_cols,
            n_drugs,
            n_diseases,
        )
        train_pair_features = append_pair_feature_blocks([train_pair_features, train_teacher_matrix])
        eval_pair_features = append_pair_feature_blocks([eval_pair_features, eval_teacher_matrix])
        print(f"Fold {fold_idx + 1}: teacher stacking features appended, dim={train_teacher_cols.shape[1]}")
        teacher_feature_names = [
            "teacher_rf_prob",
            "teacher_extra_prob",
            "teacher_gbdt_prob",
            "teacher_mean",
            "teacher_std",
        ]

    print(f"Fold {fold_idx + 1}: train pairs={len(train_pairs)}, test pairs={len(test_pairs)}")

    graphs = build_graphs(train_positive_mask, drug_semantic, disease_semantic, args.semantic_topk, device)
    features = tensor_features(drug_features, disease_features, device, path_features=train_pair_features)
    eval_features = tensor_features(drug_features, disease_features, device, path_features=eval_pair_features)
    pair_extra_dim = 0 if eval_pair_features is None else int(eval_pair_features.shape[-1])

    drug_input_dims = [feat.shape[1] for feat in drug_features]
    disease_input_dims = [feat.shape[1] for feat in disease_features]

    model = HybridDDA(
        drug_input_dims=drug_input_dims,
        disease_input_dims=disease_input_dims,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        fusion_method=args.fusion_method,
        pair_extra_dim=pair_extra_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_log = {
        "loss": 0.0,
        "main": 0.0,
        "aux": 0.0,
        "teacher": 0.0,
        "cl": 0.0,
        "intra": 0.0,
        "bal_w": 0.0,
        "grad_main": 0.0,
        "grad_cl": 0.0,
        "cos": 0.0,
    }

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_logs = []
        for idx in batch_indices(len(train_pairs), args.batch_size, shuffle=True):
            batch = train_pairs[idx]
            drug_idx = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            disease_idx = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
            labels = torch.tensor(batch[:, 2], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(drug_idx, disease_idx, features, graphs)
            main_loss = F.binary_cross_entropy_with_logits(outputs["main"], labels)
            aux_loss = aux_bce_loss(outputs, labels)
            total_loss = main_loss + args.aux_weight * aux_loss

            teacher_loss = labels.new_tensor(0.0)
            if teacher_train_target is not None and args.teacher_weight > 0:
                soft_targets = torch.tensor(teacher_train_target[idx], dtype=torch.float32, device=device)
                soft_targets, teacher_loss_weights = teacher_soft_targets(soft_targets, labels)
                teacher_loss_values = F.binary_cross_entropy_with_logits(
                    outputs["main"],
                    soft_targets,
                    reduction="none",
                )
                teacher_loss = (teacher_loss_values * teacher_loss_weights).sum() / teacher_loss_weights.sum().clamp_min(1e-8)
                total_loss = total_loss + args.teacher_weight * teacher_loss

            cl_loss = labels.new_tensor(0.0)
            intra_loss = labels.new_tensor(0.0)
            contrastive_objective = labels.new_tensor(0.0)
            if args.contrastive_weight > 0:
                cl_loss = cross_view_contrastive_loss(
                    outputs["views"],
                    max_nodes=args.max_contrastive_nodes,
                    temperature=args.temperature,
                )
                contrastive_objective = contrastive_objective + cl_loss
            if args.intra_contrastive_weight > 0:
                intra_loss = intra_view_contrastive_loss(
                    outputs["views"],
                    graphs,
                    max_edges=args.max_intra_edges,
                    temperature=args.temperature,
                )
                contrastive_objective = contrastive_objective + args.intra_contrastive_weight * intra_loss

            bal_w = 0.0
            grad_main = 0.0
            grad_cl = 0.0
            cos = 0.0
            if contrastive_objective.requires_grad and float(contrastive_objective.detach().cpu()) != 0.0:
                if args.no_gradient_balance:
                    bal_w = args.contrastive_weight
                else:
                    bal_w, grad_main, grad_cl, cos = balanced_contrastive_weight(
                        main_loss,
                        contrastive_objective,
                        model.shared_parameters_for_balance(),
                        base_weight=args.contrastive_weight,
                        conflict_scale=args.conflict_scale,
                    )
                total_loss = total_loss + bal_w * contrastive_objective

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_logs.append(
                {
                    "loss": float(total_loss.detach().cpu()),
                    "main": float(main_loss.detach().cpu()),
                    "aux": float(aux_loss.detach().cpu()),
                    "teacher": float(teacher_loss.detach().cpu()),
                    "cl": float(cl_loss.detach().cpu()),
                    "intra": float(intra_loss.detach().cpu()),
                    "bal_w": float(bal_w),
                    "grad_main": float(grad_main),
                    "grad_cl": float(grad_cl),
                    "cos": float(cos),
                }
            )

        for key in last_log:
            last_log[key] = float(np.mean([log[key] for log in epoch_logs]))

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"Fold {fold_idx + 1} Epoch {epoch:03d} "
                f"loss={last_log['loss']:.4f} "
                f"main={last_log['main']:.4f} "
                f"aux={last_log['aux']:.4f} "
                f"teacher={last_log['teacher']:.4f} "
                f"cl={last_log['cl']:.4f} "
                f"intra={last_log['intra']:.4f} "
                f"bal_w={last_log['bal_w']:.4f} "
                f"grad_main={last_log['grad_main']:.4f} "
                f"grad_cl={last_log['grad_cl']:.4f} "
                f"cos={last_log['cos']:.4f}"
            )

    y_true, y_score = predict(model, test_pairs, eval_features, graphs, args.batch_size, device)
    if teacher_test_target is not None and args.teacher_eval_rank_blend > 0:
        blend = min(max(args.teacher_eval_rank_blend, 0.0), 1.0)
        y_score = (1.0 - blend) * rank_normalize_scores(y_score) + blend * rank_normalize_scores(teacher_test_target)
        print(f"Fold {fold_idx + 1}: rank-blended neural score with teacher mean, teacher_eval_rank_blend={blend:.3f}")
    metrics = compute_metrics(y_true, y_score, threshold=0.5)
    ranking_topks = parse_topks(args.ranking_topks)
    ranking_metrics = ranking_metrics_by_disease(test_pairs, y_score, ranking_topks)
    metrics.update(ranking_metrics)
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    print(
        f"Fold {fold_idx + 1} done. "
        f"AUROC: {metrics['auroc']:.4f}, "
        f"AUPR: {metrics['aupr']:.4f}, "
        f"F1: {metrics['f1']:.4f}, "
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"Threshold: {metrics['threshold']:.3f}"
    )
    print(
        f"Fold {fold_idx + 1} score stats: "
        f"all=[{y_score.min():.4f}, {y_score.mean():.4f}, {y_score.max():.4f}], "
        f"pos_mean={pos_scores.mean():.4f}, "
        f"neg_mean={neg_scores.mean():.4f}"
    )
    ranking_text = ", ".join(
        f"{metric_display_name(key)}={metrics[key]:.4f}"
        for key in [*(f"p_at_{k}" for k in ranking_topks), *(f"r_at_{k}" for k in ranking_topks), "mrr", "ndcg_at_10"]
    )
    print(
        f"Fold {fold_idx + 1} ranking "
        f"(diseases={metrics['ranking_disease_count']}): {ranking_text}"
    )
    if return_artifacts:
        artifacts = {
            "model": model,
            "train_pairs": train_pairs,
            "test_pairs": test_pairs,
            "train_positive_mask": train_positive_mask,
            "drug_features": drug_features,
            "disease_features": disease_features,
            "drug_semantic": drug_semantic,
            "disease_semantic": disease_semantic,
            "static_pair_features": static_pair_features,
            "static_pair_feature_names": static_pair_feature_names,
            "pair_feature_names": pair_feature_names,
            "train_pair_features": train_pair_features,
            "eval_pair_features": eval_pair_features,
            "graphs": graphs,
            "features": features,
            "eval_features": eval_features,
            "pair_extra_dim": pair_extra_dim,
            "teacher_train_target": teacher_train_target,
            "teacher_test_target": teacher_test_target,
            "train_teacher_cols": train_teacher_cols if args.use_oof_teachers else None,
            "test_teacher_cols": test_teacher_cols if args.use_oof_teachers else None,
            "teacher_feature_names": teacher_feature_names,
            "y_true": y_true,
            "y_score": y_score,
            "metrics": metrics,
        }
        return metrics, artifacts
    return metrics


@torch.no_grad()
def predict(model, pairs, features, graphs, batch_size, device):
    model.eval()
    scores = []
    labels = []
    for idx in batch_indices(len(pairs), batch_size, shuffle=False):
        batch = pairs[idx]
        drug_idx = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
        disease_idx = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
        outputs = model(drug_idx, disease_idx, features, graphs)
        scores.append(torch.sigmoid(outputs["main"]).detach().cpu().numpy())
        labels.append(batch[:, 2].astype(np.int64))
    return np.concatenate(labels), np.concatenate(scores)


def main():
    args = parse_args()
    setup_seed(args.seed)
    if args.output_csv is None:
        results_dir = os.path.join(PROJECT_ROOT, "results", "hybrid")
        os.makedirs(results_dir, exist_ok=True)
        args.output_csv = os.path.join(results_dir, f"{args.dataset}_{args.n_splits}fold.csv")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    print("Hybrid training started.")
    _, _, association_matrix, all_train_mask, all_test_mask, _ = data_preparation(args)
    n_diseases, n_drugs = association_matrix.shape
    print(f"Dataset: {args.dataset}, drugs={n_drugs}, diseases={n_diseases}, positives={int(association_matrix.sum())}")

    (
        drug_features,
        disease_features,
        drug_semantic,
        disease_semantic,
        static_pair_features,
        static_pair_feature_names,
    ) = prepare_feature_blocks(args, n_drugs, n_diseases, device)

    fold_metrics = []
    for fold_idx in range(args.n_splits):
        metrics = train_one_fold(
            args,
            fold_idx,
            association_matrix,
            all_train_mask[fold_idx],
            all_test_mask[fold_idx],
            drug_features,
            disease_features,
            drug_semantic,
            disease_semantic,
            static_pair_features,
            static_pair_feature_names,
            device,
        )
        fold_metrics.append(metrics)

    if fold_metrics:
        print("\nAverage Results:")
        ranking_topks = parse_topks(args.ranking_topks)
        summary_keys = [
            "auroc",
            "aupr",
            "f1",
            "accuracy",
            *(f"p_at_{k}" for k in ranking_topks),
            *(f"r_at_{k}" for k in ranking_topks),
            "mrr",
            "ndcg_at_10",
        ]
        for key in summary_keys:
            if key not in fold_metrics[0]:
                continue
            values = np.asarray([metrics[key] for metrics in fold_metrics], dtype=np.float32)
            print(f"{metric_display_name(key)}: {values.mean():.4f} +/- {values.std():.4f}")
        rows = [{"fold": idx + 1, "split_mode": "random", "dataset": args.dataset, **metric} for idx, metric in enumerate(fold_metrics)]
        with open(args.output_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved fold metrics to: {args.output_csv}")

    print("Hybrid training completed.")


if __name__ == "__main__":
    main()
