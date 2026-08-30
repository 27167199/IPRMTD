import argparse
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import os
from tqdm import tqdm


class DiseaseTextEncoder:
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", output_dim=128,
                 device=None, local_files_only=False):
        self.model_name = model_name
        self.output_dim = output_dim

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        load_kwargs = {"local_files_only": local_files_only}
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name, **load_kwargs)
            self.model = BertModel.from_pretrained(model_name, **load_kwargs)
        except OSError as exc:
            raise OSError(
                "Cannot load BioBERT model. If this machine cannot access Hugging Face, "
                "download dmis-lab/biobert-base-cased-v1.1 on another machine, copy the "
                "model directory here, then run with --model_path /path/to/model "
                "--local_files_only."
            ) from exc
        self.model = self.model.to(self.device)

        if output_dim is not None and output_dim != 768:
            self.projection = torch.nn.Linear(768, output_dim).to(self.device)
        else:
            self.projection = None

        self.model.eval()

    def encode_batch(self, texts, batch_size=32):
        D_emb_list = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                D_cls = outputs.last_hidden_state[:, 0, :]

                if self.projection is not None:
                    D_cls = self.projection(D_cls)

                D_emb_list.append(D_cls.cpu().numpy())

        return np.vstack(D_emb_list)

    def process_disease_data(self, disease_ids, disease_info_df, text_fields=None,
                             output_path="disease_embeddings.npy", save_mapping=True):
        if text_fields is None:
            text_fields = [
                ("Disease_Entry", 1.0),
                ("Disease_Synonymous", 0.8),
                ("Definitions", 1.2),
            ]

        filtered_df = disease_info_df.set_index("DiseaseID").reindex(disease_ids).reset_index()
        ids = filtered_df["DiseaseID"].tolist()

        field_embeddings = {}
        for field_name, weight in text_fields:
            if field_name in filtered_df.columns:
                texts = []
                for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
                    if pd.notna(row[field_name]) and row[field_name]:
                        texts.append(str(row[field_name]))
                    else:
                        texts.append("")

                D_field = self.encode_batch(texts)
                field_embeddings[field_name] = (D_field, weight)

        D_fused = np.zeros((len(filtered_df), self.output_dim if self.output_dim else 768))
        total_weight = 0

        for field_name, (D_field, weight) in field_embeddings.items():
            D_fused += D_field * weight
            total_weight += weight

        if total_weight > 0:
            D_fused /= total_weight

        id_to_index = {disease_id: idx for idx, disease_id in enumerate(ids)}

        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            np.save(output_path, D_fused)

            if save_mapping:
                mapping_path = output_path.replace('.npy', '_id_mapping.pkl')
                pd.to_pickle(id_to_index, mapping_path)

                mapping_npy_path = output_path.replace('.npy', '_id_mapping.npy')
                np.save(mapping_npy_path, id_to_index)

        return D_fused, id_to_index


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
    parser = argparse.ArgumentParser(description="Generate disease embeddings with BioBERT.")
    parser.add_argument(
        "--model_path",
        default=os.environ.get("BIOBERT_MODEL_PATH", "dmis-lab/biobert-base-cased-v1.1"),
        help="Hugging Face model id or local BioBERT model directory."
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
    disease_id_path = os.path.join(project_root, "dataset", "appoved", "disease_idx_to_id.npy")
    disease_info_path = os.path.join(project_root, "embedding_generation", "4.General Information of Disease.tsv")
    output_emb_path = os.path.join(project_root, "dataset", "appoved", "emb", "disease_embeddings.npy")

    os.makedirs(os.path.dirname(output_emb_path), exist_ok=True)

    disease_ids = load_ids_from_mapping(disease_id_path)
    if not disease_ids:
        return

    disease_info_df = pd.read_csv(disease_info_path, sep='\t')

    encoder = DiseaseTextEncoder(
        model_name=args.model_path,
        output_dim=128,
        local_files_only=args.local_files_only
    )

    text_fields = [
        ("Disease_Entry", 1.0),
        ("Disease_Synonymous", 0.8),
        ("Definitions", 1.5),
    ]

    embeddings, id_mapping = encoder.process_disease_data(
        disease_ids=disease_ids,
        disease_info_df=disease_info_df,
        text_fields=text_fields,
        output_path=output_emb_path
    )

    index_to_id = {idx: did for did, idx in id_mapping.items()}
    np.save(output_emb_path.replace('.npy', '_index_to_id.npy'), index_to_id)


if __name__ == "__main__":
    main()
