# Protein-Ligand Binding Affinity (pK) Prediction

A three-branch graph neural network that predicts protein-ligand binding affinity from 3D
structure, combining a ligand molecular graph, a protein-pocket graph, and global molecular
features.

`pK = -log10(Ki in M) = 9 - log10(Ki in nM)`

## Quick start

```bash
pip install -r requirements.txt
```

Predict affinity for ligand/pocket pairs:

```bash
python predict.py --csv examples/pairs.csv \
                  --model checkpoints/random_run2.pt \
                  --output predictions.csv
```

The input CSV needs `ligand_path` and `pocket_path` columns (`pdb_id` optional). Ligands may
be SDF or MOL2, pockets PDB or MOL2. Output reports predicted pK and Ki (nM) per complex.

Train a model:

```bash
python train.py --csv data/pl_table_with_paths.csv \
                --cache_dir data/cache \
                --assay_filter ki \
                --epochs 50 --hidden_dim 256 --batch_size 32 \
                --split random --seed 0 \
                --save checkpoints/my_model.pt
```

Training reads the preprocessed sample cache (one built ligand graph + pocket graph + label
per complex, ~350 MB). It is not in the repository; available on request, or rebuilt from the
raw PDBbind/scPDB structures.

## Data

| Stage | Samples |
|---|---|
| PDBbind v2020, total | 19,037 |
| After standard filters | 8,914 |
| + scPDB + BindingDB extension (2,359) | 11,272 |
| Ki only (training set) | 5,596 |

**Filters:** exact (`=`) values, non-covalent, resolution <= 2.5 A, pK 2-12, validated
structures.

Mean pK = 6.09 (806 nM affinity).

## Ligand features

**32 atom + 8 bond + 51 graph = 91 total**

| Level | Purpose | Features | Dims |
|---|---|---|---|
| Atom | Chemical identity | atomic number, mass, VdW radius | 3 |
| Atom | Electrostatics | formal charge | 1 |
| Atom | H-bonding | is donor, is acceptor | 2 |
| Atom | Geometry | hybridization (SP/SP2/SP3), aromaticity | 4 |
| Atom | Connectivity | degree, valence, num hydrogens | 3 |
| Atom | Ring info | in ring, ring size | 2 |
| Atom | Crippen | logP_i, MR_i | 2 |
| Atom | Cross-context | 5 residue types x 3 distance bins (0-6, 6-10, 10-15 A) | 15 |
| Bond | Bond type | single / double / triple / aromatic | 4 |
| Bond | Bond properties | conjugated, in ring, distance, stereo | 4 |
| Graph | Molecular descriptors | counts, TPSA, MolWt, HBD/HBA, rings | 9 |
| Graph | Pharmacophore 3D | 6 types x (count, centroid, spread, max dist) + pairwise | 42 |

Gasteiger charges were replaced by per-atom Crippen contributions. Cross-context counts nearby
pocket residues by chemistry type at three distance shells — a per-node version of what a
global PLIF descriptor did previously.

## Pocket features

**47 residue + 8 AA embedding + 9 edge + 79 graph = 143 total**

| Level | Purpose | Features | Dims |
|---|---|---|---|
| Residue | Chemical properties | charge, hydrophobicity, polarity, aromaticity, volume | 7 |
| Residue | H-bonding | HBD, HBA flags | 2 |
| Residue | Size & composition | heavy atom count, fraction C/N/O/S, backbone fraction | 6 |
| Residue | Residue shape | spatial spread of atoms | 1 |
| Residue | Refined properties | hydrophobicity score, charge value | 2 |
| Residue | Spatial context | dist to center, min/mean neighbour dist, close/medium counts | 5 |
| Residue | 3D position | centered coordinates (X, Y, Z) | 3 |
| Residue | Cross-context | 7 atom types x 3 distance bins (0-6, 6-10, 10-15 A) | 21 |
| Residue | AA identity | learned embedding (nn.Embedding, 21 types -> 8 dims) | 8 |
| Edge | Distance | normalized distance, exponential decay | 2 |
| Edge | Interaction potential | charge product, aromatic-aromatic, hydrophobic-hydrophobic, H-bond | 4 |
| Edge | Distance bins | close / medium / far one-hot (0-6, 6-10, 10-12 A) | 3 |
| Graph | Composition | residue type fractions, HBD/HBA, hydrophobicity, volume | 10 |
| Graph | Binding composition | Cys, His, Gly, Pro fractions | 4 |
| Graph | Shape (PCA) | eigenvalues, radius of gyration, pairwise stats, bounding box | 11 |
| Graph | Interaction hotspots | 6 types x (count, centroid, spread, density, fraction) | 54 |

Cross-context features are intentionally symmetric with the ligand side, encoding the local
interaction environment from both the ligand and the residue perspective. Cross-context bins
are residue-level (ligand atoms <-> pocket residue); edge bins are edge-level (pocket residue
<-> pocket residue).

## Model architecture

```
Ligand graph (32 atom + 8 bond)  -->  GINEConv x4        -->  h_lig (256)
                                      residual, BatchNorm,          |
                                      max pool                      |
                                                                    |
Pocket graph (47 residue + 9 edge -->  GINEConv x4        -->  h_poc (256)
              + AA embed 8)           residual, BatchNorm,          |
                                      max pool                      |
                                                                    |
Global features                  -->  Linear 180 -> 256  -->  h_global (256)
(lig 51 + poc 79 + shape 50 = 180)                                  |
                                                                    |
        L2-normalize each branch, concat [h_lig, h_poc, h_global] = 768
          -->  Linear(768, 256) -> ReLU -> Dropout(0.30)
          -->  Linear(256, 128) -> ReLU -> Dropout(0.30)
          -->  Linear(128, 1)   -> predicted (standardized) affinity
```

Node, edge and graph features are standardized inside the model using buffers stored in the
checkpoint, so inference needs no external statistics.

## Training

| Component | Setting |
|---|---|
| Optimizer | AdamW (lr=1e-4, weight_decay=5e-2) |
| Scheduler | CosineAnnealingLR (1e-4 -> 1e-6) |
| Loss | MSE (standardized targets) |
| Gradient clipping | max_norm=1.0 |
| Augmentation | Coordinate noise (sigma=0.05 A) |
| Dropout | 0.30 |
| Epochs | 50 |
| Batch size | 32 |
| Hidden dim | 256 |

| Strategy | Train / Val / Test | Description |
|---|---|---|
| Random | 80 / 10 / 10 | Shuffled — comparable to literature |
| Similarity | 80 / 10 / 10 | Similarity-grouped — chemistry held out of test |

Per-epoch metrics are written to `logs/`. The best-validation checkpoint is saved with
`model_state`, `y_mean`, `y_std` and `model_config`, which is the format inference expects.

## Results

The model is trained on standardized pKi targets; fold-error is computed after converting
predictions back to Ki. MAE and RMSE are in standardized units.

| Split | Checkpoint | R2 | Rp | MAE | RMSE |
|---|---|---|---|---|---|
| Random — run 1 | `random_run1.pt` | 0.581 | 0.766 | 0.482 | 0.670 |
| Random — run 2 | `random_run2.pt` | 0.658 | 0.817 | 0.432 | 0.574 |
| Random — run 3 | `random_run3.pt` | 0.652 | 0.809 | 0.431 | 0.599 |
| **Random — mean ± std** | | **0.630 ± 0.035** | **0.797 ± 0.022** | **0.448 ± 0.024** | **0.614 ± 0.041** |
| Similarity split | `similarity_run1.pt` | 0.377 | 0.622 | 0.542 | 0.686 |

Each row's per-epoch training log is in `logs/`, named to match the checkpoint.

Best random-split run (n=561): median fold-error 6.15x, within 2x 21.6%, within 5x 46.2%,
within 10x 61.7%.

**Similarity split:** a Morgan fingerprint is computed per ligand, pairwise Tanimoto similarity
is taken across the set, and compounds at similarity >= 0.75 are grouped into clusters. Whole
clusters go entirely to train, val or test, so no near-duplicate chemistry leaks across splits.

## Data flow

```
TRAINING                                 INFERENCE

Dataset                                  Input
(ligand + pocket files, affinity)        (ligand + pocket files)
        |                                        |
Split into train / val / test            Load trained model
        |                                        |
Feature extraction                       Feature extraction
        |                                        |
Model                                    Predict affinity
(ligand GNN + pocket GNN + global)               |
        |                                Output CSV
Loss + optimizer (train weights)
        |
Save best model
```

## Code structure

```
train.py                        training entry point (CLI)
predict.py                      inference entry point (CLI)
src/
  data/
    build_pl_table.py           parse the PDBbind index, apply the affinity filters
    add_paths_to_pl_table.py    validate structure files, attach their paths
    dataset.py                  PDBbindDataset — lazy load + cache
    fingerprints.py             Morgan fingerprint index used by the similarity split
  features/
    ligand_graph.py             ligand atom / bond / graph features
    pocket_graph.py             pocket residue / edge / graph features
    interactions.py             per-node cross-context features
  model/
    model.py                    GNNEncoder + AffinityModel
  training/
    trainer.py                  train loop, splits, evaluate
tools/
  consolidate_cache.py          deduplicate and slim the sample cache
tests/
  test_smoke.py                 feature, checkpoint and split regression tests
examples/                       ligand/pocket pair for the smoke test
checkpoints/
  random_run1.pt                random split, seed 42
  random_run2.pt                random split, seed 0
  random_run3.pt                random split, seed 1
  similarity_run1.pt            similarity split, seed 42
logs/
  random_run*.csv               per-epoch metrics for each random run
  similarity_run1.csv           per-epoch metrics for the similarity run
data/                           affinity tables + fingerprint index
```

## Test case

Selected test-set complexes, best random-split model:

| PDB ID | True Ki | Predicted Ki |
|---|---|---|
| 3i5x | 4.00 uM | 4.05 uM |
| 1h4w | 22.00 uM | 23.26 uM |
| 3e2s | 86.00 uM | 97.07 uM |
| 4q1d | 16.60 nM | 19.78 nM |
| 6evr | 8.80 nM | 11.13 nM |
| 3hiv | 7.50 nM | 5.46 nM |
| 4oc5 | 85.00 nM | 122.8 nM |
| 1fdt | 650.0 pM | 1,727.0 pM |
| 5g57 | 3.98 nM | 14.27 nM |
| 3loo | 860.0 nM | 21,078.9 nM |

## Requirements

Python 3.10+, PyTorch 2.3+, PyTorch Geometric 2.6+, RDKit, BioPython, SciPy, NumPy.

```bash
pip install -r requirements.txt
pytest tests/ -q
```
