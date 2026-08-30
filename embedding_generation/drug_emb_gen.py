import argparse
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import Data
from rdkit import Chem


class DrugEmbeddingGenerator:
    def __init__(self,
                 drug_id_path,
                 drug_info_path,
                 output_dir,
                 device=None,
                 drug_encoder_path='DeepChem/ChemBERTa-77M-MTR',
                 max_length=512,
                 local_files_only=False):

        self.drug_id_path = drug_id_path
        self.drug_info_path = drug_info_path
        self.output_dir = output_dir
        self.max_length = max_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        load_kwargs = {"local_files_only": local_files_only}
        try:
            self.drug_encoder = AutoModel.from_pretrained(drug_encoder_path, **load_kwargs)
            try:
                self.drug_tokenizer = AutoTokenizer.from_pretrained(drug_encoder_path, **load_kwargs)
            except ValueError:
                self.drug_tokenizer = AutoTokenizer.from_pretrained(
                    drug_encoder_path, use_fast=False, **load_kwargs
                )
        except OSError as exc:
            raise OSError(
                "Cannot load ChemBERTa model. If this machine cannot access Hugging Face, "
                "download DeepChem/ChemBERTa-77M-MTR on another machine, copy the model "
                "directory here, then run with --model_path /path/to/model "
                "--local_files_only."
            ) from exc

        self.drug_encoder = self.drug_encoder.to(self.device)
        os.makedirs(output_dir, exist_ok=True)

    def is_valid_smiles(self, smiles):
        if pd.isna(smiles) or not smiles:
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def generate_token_edges(self, S_trim):
        n = len(S_trim)
        edges = []
        for i in range(n - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        return np.array(edges).T

    def generate_embeddings(self, drug_ids, drug_info_df):
        missing_smiles_drugs = []
        filtered_df = drug_info_df[drug_info_df["DrugID"].isin(drug_ids)].copy()

        for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
            drug_id = row['DrugID']
            smiles = row['Canonical_SMILES']

            if not self.is_valid_smiles(smiles):
                missing_smiles_drugs.append(drug_id)
                continue

            output_path = os.path.join(self.output_dir, f"{drug_id}_embedded.pt")
            if os.path.exists(output_path):
                continue

            try:
                S_trim = self.drug_tokenizer.encode(
                    smiles,
                    truncation=True,
                    max_length=self.max_length
                )
                tokens = torch.tensor(S_trim).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    M_seq = self.drug_encoder(tokens).last_hidden_state

                edges = self.generate_token_edges(S_trim)
                node_ids = [True] * len(S_trim)

                data = {
                    'embeddings': Data(
                        x=M_seq.squeeze(0).cpu(),
                        edge_index=torch.tensor(edges, dtype=torch.long)
                    ),
                    'Drug_ID': drug_id,
                    'SMILES': smiles,
                    'node_ids': np.array(node_ids, dtype=bool),
                }

                torch.save(data, output_path)

            except Exception:
                missing_smiles_drugs.append(drug_id)

        return len(filtered_df), missing_smiles_drugs

    def run(self):
        drug_ids = load_ids_from_mapping(self.drug_id_path)

        drug_info_df = pd.read_csv(self.drug_info_path, sep='\t')
        processed_count, missing_smiles_drugs = self.generate_embeddings(drug_ids, drug_info_df)

        return processed_count, missing_smiles_drugs


def load_ids_from_mapping(file_path):
    ids_data = np.load(file_path, allow_pickle=True)
    if isinstance(ids_data, np.ndarray) and ids_data.shape == ():
        ids_data = ids_data.item()

    if isinstance(ids_data, dict):
        if all(isinstance(k, (int, np.integer)) for k in ids_data.keys()):
            return [ids_data[k] for k in sorted(ids_data.keys())]
        if all(isinstance(v, (int, np.integer)) for v in ids_data.values()):
            return [k for k, _ in sorted(ids_data.items(), key=lambda item: item[1])]
        return list(ids_data.values())

    return list(ids_data)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate drug embeddings with ChemBERTa.")
    parser.add_argument(
        "--model_path",
        default=os.environ.get("CHEMBERTA_MODEL_PATH", "DeepChem/ChemBERTa-77M-MTR"),
        help="Hugging Face model id or local ChemBERTa model directory."
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        help="Load only local model files and do not contact Hugging Face."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    drug_id_path = os.path.join(project_root, "dataset", "appoved", "drug_idx_to_id.npy")
    drug_info_path = os.path.join(project_root, "embedding_generation", "1.General Information of Drug.tsv")
    output_dir = os.path.join(project_root, "dataset", "appoved", "emb", "drug_embeddings")

    generator = DrugEmbeddingGenerator(
        drug_id_path=drug_id_path,
        drug_info_path=drug_info_path,
        output_dir=output_dir,
        drug_encoder_path=args.model_path,
        local_files_only=args.local_files_only
    )

    processed_count, missing_smiles_drugs = generator.run()


if __name__ == "__main__":
    main()
