
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_disease_sim_Matrix(disease_similarity, disease_disease_topk):
    n = disease_similarity.shape[0]
    m = np.zeros((n, n), np.float32)
    s = disease_similarity
    for i in range(n):
        sim = {j: s[i][j] for j in range(n) if i != j}
        for k, (j, _) in enumerate(sorted(sim.items(), key=lambda d: d[1], reverse=True)):
            if k >= disease_disease_topk:
                break
            m[i][j] = s[i][j]
    return m

def get_drug_sim_Matrix(drug_similarity, drug_drug_topk):
    n = drug_similarity.shape[0]
    m = np.zeros((n, n), np.float32)
    s = drug_similarity
    for i in range(n):
        sim = {j: s[i][j] for j in range(n) if i != j}
        for k, (j, _) in enumerate(sorted(sim.items(), key=lambda d: d[1], reverse=True)):
            if k >= drug_drug_topk:
                break
            m[i][j] = s[i][j]
    return m

def build_interaction_matrix(filepath):
    associations_path = os.path.join(filepath, "drug_disease_associations.csv")
    disease_idx_to_id_path = os.path.join(filepath, "disease_idx_to_id.npy")
    drug_idx_to_id_path = os.path.join(filepath, "drug_idx_to_id.npy")

    if not os.path.exists(associations_path):
        raise FileNotFoundError(f"File not found: {associations_path}")

    if os.path.exists(disease_idx_to_id_path) and os.path.exists(drug_idx_to_id_path):
        disease_count = len(np.load(disease_idx_to_id_path, allow_pickle=True).item())
        drug_count = len(np.load(drug_idx_to_id_path, allow_pickle=True).item())
    else:
        associations = pd.read_csv(associations_path)
        disease_count = int(associations["DiseaseIndex"].max()) + 1
        drug_count = int(associations["DrugIndex"].max()) + 1

    interactions = np.zeros((disease_count, drug_count), dtype=np.int8)
    associations = pd.read_csv(associations_path)
    interactions[
        associations["DiseaseIndex"].astype(int).to_numpy(),
        associations["DrugIndex"].astype(int).to_numpy()
    ] = 1

    return interactions


def load_appoved_dataset(filepath):
    interactions_path = os.path.join(filepath, "drug_disease_association_matrix.npy")
    if not os.path.exists(interactions_path):
        interactions = build_interaction_matrix(filepath)
        np.save(interactions_path, interactions)
    else:
        interactions = np.load(interactions_path)
    n_diseases, n_drugs = interactions.shape

    disease_idx_to_id_path = os.path.join(filepath, "disease_idx_to_id.npy")
    drug_idx_to_id_path = os.path.join(filepath, "drug_idx_to_id.npy")
    if os.path.exists(disease_idx_to_id_path) and os.path.exists(drug_idx_to_id_path):
        disease_idx_to_id = np.load(disease_idx_to_id_path, allow_pickle=True).item()
        drug_idx_to_id = np.load(drug_idx_to_id_path, allow_pickle=True).item()
        disease_name = list(disease_idx_to_id.values())
        drug_name = list(drug_idx_to_id.values())
    else:
        disease_name = np.arange(n_diseases)
        drug_name = np.arange(n_drugs)

    disease_sim = np.eye(n_diseases, dtype=np.float32)
    drug_sim = np.eye(n_drugs, dtype=np.float32)

    return drug_sim, disease_sim, drug_name, n_drugs, disease_name, n_diseases, interactions

def data_preparation(args):
    if args.dataset != "appoved":
        raise AssertionError("Only 'appoved' dataset is supported in this trimmed loader.")
    path = os.path.join(PROJECT_ROOT, "dataset", args.dataset)
    drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_appoved_dataset(path)

    args.n_diseases, args.n_drugs = interactions.shape
    kfold = KFold(n_splits=args.n_splits, shuffle=True)

    pos_row, pos_col = np.nonzero(interactions)
    neg_row, neg_col = np.where(interactions == 0)
    pos_count = len(pos_row)

    train_data, test_data = [], []
    for train_pos_idx, test_pos_idx in kfold.split(np.arange(pos_count)):
        train_mask = np.zeros_like(interactions, dtype=bool)
        test_mask = np.zeros_like(interactions, dtype=bool)

        train_pos_row = pos_row[train_pos_idx]
        train_pos_col = pos_col[train_pos_idx]
        test_pos_row = pos_row[test_pos_idx]
        test_pos_col = pos_col[test_pos_idx]

        n_train_pos = len(train_pos_row)
        neg_indices_shuffled = np.random.permutation(len(neg_row))

        train_neg_idx = neg_indices_shuffled[:n_train_pos]
        train_neg_row = neg_row[train_neg_idx]
        train_neg_col = neg_col[train_neg_idx]

        n_test_pos = len(test_pos_row)
        test_neg_idx = neg_indices_shuffled[n_train_pos:n_train_pos + n_test_pos]
        test_neg_row = neg_row[test_neg_idx]
        test_neg_col = neg_col[test_neg_idx]

        train_mask[train_pos_row, train_pos_col] = True
        train_mask[train_neg_row, train_neg_col] = True
        test_mask[test_pos_row, test_pos_col] = True
        test_mask[test_neg_row, test_neg_col] = True

        train_data.append(train_mask)
        test_data.append(test_mask)

    disease_disease_sim_Matrix = get_disease_sim_Matrix(disease_sim, args.disease_TopK)
    drug_drug_sim_Matrix = get_drug_sim_Matrix(drug_sim, args.drug_TopK)
    truth_label = interactions
    pos_weight = 1.0

    return disease_disease_sim_Matrix, drug_drug_sim_Matrix, truth_label, train_data, test_data, pos_weight
