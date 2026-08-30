# IPRMTD

## Integrating Multi-source Biological Evidence and Multi-view Graph Learning for Drug Repurposing

## Overview

IPRMTD is a multi-source biological evidence integration framework for drug repurposing based on multi-view graph learning. The framework aims to identify potential drug–disease associations by integrating heterogeneous biological information, including drug–target interactions, disease–target relationships, protein–protein interactions, pathway knowledge, and biomedical evidence.

By constructing a heterogeneous biological network and learning comprehensive representations of drugs and diseases, IPRMTD captures complex relationships among drugs, targets, diseases, and biological pathways, providing a computational approach for discovering potential therapeutic candidates.

------

## Framework

The main workflow of IPRMTD includes:

1. **Multi-source biological data integration**
   - Drug information
   - Disease information
   - Drug–target interactions
   - Disease–target associations
   - Target–target interaction networks
   - Biological pathway knowledge
2. **Multi-view graph representation learning**
   - Construction of heterogeneous biological graphs
   - Learning drug and disease embeddings
   - Integration of different biological evidence views
3. **Drug repurposing prediction**
   - Prediction of potential drug–disease associations
   - Ranking candidate drugs for specific diseases
   - Case study validation using known therapeutic evidence

------

## Directory Structure

```
IPRMTD/
│
├── code/
│   ├── Att.py                 # Attention-related modules
│   ├── model.py               # Main hybrid graph learning model
│   ├── loader.py              # Dataset loading and preprocessing
│   ├── main.py                # Model training and prediction
│
├── dataset/
│   └── appoved/
│       ├── drug_info.csv                  # Drug information
│       ├── disease_info.csv               # Disease information
│       ├── drug_disease_associations.csv  # Known drug-disease associations
│       ├── drug_idx_to_id.npy             # Drug index mapping
│       ├── disease_idx_to_id.npy          # Disease index mapping
│       │
│       └── heterograph/
│           ├── drug_target.csv            # Drug-target interaction data
│           ├── disease_target.csv         # Disease-target association data
│           ├── target_target.csv          # Target interaction network
│           └── evidence features          # Biological evidence features
│
├── embedding_generation/
│   ├── drug_emb_gen.py        # Drug embedding generation
│   └── disease_emb_gen.py     # Disease embedding generation
│
└── README.md
```

Due to large file sizes, the  `reactome_pathway_evidence_features_std.npy` and `target_evidence_features_std.npy`  are **not** included in this repository.  If you need these files, please send a request email to: yancheng01@hnucm.edu.cn

------

## Requirements

The framework is implemented using Python.

Recommended environment:

```
Python >= 3.8
PyTorch
NumPy
Pandas
Scikit-learn
DGL / PyTorch Geometric (depending on implementation)
RDKit (for molecular feature processing)
```

------

## Data Preparation

The dataset folder contains processed biological networks required for model training.

Main input files:

Drug information

```
drug_info.csv
```

Contains drug identifiers and basic drug information.

Disease information

```
disease_info.csv
```

Contains disease identifiers and disease-related information.

Drug–disease associations

```
drug_disease_associations.csv
```

Contains known drug–disease relationships used for model training and evaluation.

Heterogeneous biological network

Located in:

```
dataset/appoved/heterograph/
```

Including:

- Drug–target relationships
- Disease–target relationships
- Target interaction networks
- Pathway evidence features

------

## Usage

### 1. Generate drug and disease embeddings

Generate initial feature representations:

```bash
python embedding_generation/drug_emb_gen.py

python embedding_generation/disease_emb_gen.py
```

------

### 2. Train the IPRMTD model

Run:

```bash
python code/main_hybrid.py
```

The model will:

- Load heterogeneous biological graphs
- Learn multi-view representations
- Optimize drug–disease association prediction
- Generate ranking scores for candidate drugs

------

## Evaluation

IPRMTD can be evaluated using:

- Cross-validation
- AUC
- AUPR
- Precision@K
- Recall@K
- Case study validation

Known drug–disease associations can be used as references for assessing prediction reliability.

------

## Biological Evidence Integration

IPRMTD integrates multiple biological evidence sources:

| Evidence type               | Function                        |
| --------------------------- | ------------------------------- |
| Drug–target interactions    | Describe drug mechanisms        |
| Disease–target associations | Represent disease mechanisms    |
| Target interactions         | Capture biological connectivity |
| Pathway information         | Provide functional context      |
| Literature evidence         | Support validation              |

------

## Citation

If you use IPRMTD in your research, please cite:

```
IPRMTD: Integrating Multi-source Biological Evidence and Multi-view Graph Learning for Drug Repurposing.
```

------

## Contact

If you have any questions or comments, please feel free to email Cheng Yan([yancheng01@hnucm.edu.cn](mailto:yancheng01@hnucm.edu.cn)).
