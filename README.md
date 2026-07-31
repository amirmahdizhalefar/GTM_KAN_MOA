# GTM-KAN-MoA

A drug–target interaction (DTI) prediction framework that combines **G**enerative
**T**opographic **M**apping (manifold learning on biological networks), **M**echanism-**o**f-**A**ction
(MoA) feature extraction, and a **K**olmogorov–**A**rnold **N**etwork (KAN) classifier.

## Overview

- **GTM** is applied directly to seven raw biological network matrices (drug–drug,
  protein–protein, drug–protein, drug–disease, protein–disease, drug–side-effect)
  to learn a low-dimensional manifold embedding — no PCA pre-reduction.
- **MoA extractors** compute mechanism-of-action-aware features for drugs (from
  SMILES, via RDKit Morgan fingerprints/descriptors) and proteins (from FASTA
  sequences, via ESM2 embeddings).
- **Fusion + classifier**: a gated multi-head cross-attention fusion module
  (residual connections, LayerNorm, SE channel attention) feeds a KAN classifier
  — with Focal Loss and label smoothing for class imbalance — falling back to an
  enhanced MLP if `pykan` isn't installed.
- Evaluated on the **Luo benchmark** (708 drugs, 1512 proteins) and a **second
  benchmark** (151 drugs, 285 proteins) from the BMC Biology dataset.

## Repository structure

```
.
├── data_loader.py     # Loads the 7 network matrices; fetches/caches SMILES (PubChem) & FASTA (UniProt)
├── gtm.py              # Generative Topographic Mapping (numerically stable EM implementation)
├── gtm_kan_moa.py       # MoA feature extractors + fusion + KAN/MLP classifier
├── gtm_net.py           # Integrates GTM fitting with the PyTorch pipeline; training/evaluation
├── run_gtm_net.py       # CLI entry point
├── requirements.txt
└── README.md
```

## Data

This repo ships **code only** — the data and trained-model folders (~262 MB total)
are intentionally not tracked in git:

| Folder                | Contents                                                              | Size    |
|------------------------|------------------------------------------------------------------------|---------|
| `data/`                | Seven biological network matrices + raw interaction files             | ~71 MB  |
| `dti/`                 | Preprocessed DTI labels/folds for the Luo benchmark                   | ~73 MB  |
| `new_datasets/`        | Second benchmark (151 drugs / 285 proteins)                           | ~7 MB   |
| `gtmnet/seq_cache/`    | Cached ESM2 protein embeddings + PubChem/UniProt sequence cache       | ~9 MB   |
| `model/`               | Trained fold checkpoints (`fold_0_best.pt` … `fold_4_best.pt`) + config | ~104 MB |

**To reproduce:** download the data bundle from `<ADD LINK — e.g. Zenodo/OSF>` and
place the folders at the repo root so the layout matches the table above.
`data_loader.py` auto-detects the repo root by walking up from its own location
until it finds `data/sevenNets/` or `new_datasets/`, so no path configuration is
needed once the folders are in place.

## Installation

Developed with **Python 3.10**.

```bash
git clone <YOUR-REPO-URL>
cd GTM-KAN-MoA
pip install -r requirements.txt
```

If you use ESM2/HuggingFace models that require authentication, set your token
first: `export HF_TOKEN=your_token_here` (never commit tokens to the repo).

## Usage

```bash
python run_gtm_net.py --dataset luo    # Luo benchmark (708 drugs, 1512 proteins)
python run_gtm_net.py --dataset new    # Second benchmark (151 drugs, 285 proteins)
python run_gtm_net.py --dataset both   # Both, sequentially
python run_gtm_net.py --dataset luo --n_iter 50   # override GTM EM iterations (default 30)
```

Results are written as JSON to the output directory reported at the top of the run log.

## Results

<!-- TODO: replace with your final headline table before publishing. -->
GTM-KAN-MoA is compared against recent DTI baselines (CE-DTI, DHGT-DTI, HGMAIB,
DTI-RME, FBRWPC, GHCDTI, among others) under 5-fold cross-validation on the Luo
benchmark. Statistical significance across methods was assessed with
Friedman/Nemenyi tests, where GTM-KAN-MoA obtained the best (lowest) mean rank.

## Citation

<!-- TODO: fill in once published -->
```bibtex
@article{gtmkanmoa2026,
  title   = {GTM-KAN-MoA: TITLE},
  author  = {AUTHORS},
  journal = {Bioinformatics},
  year    = {2026},
  note    = {Under review}
}
```

## License

<!-- TODO: pick a license — MIT is a common permissive default for academic
     code if you don't have a specific requirement. Add it via GitHub's
     "Add file" -> "Create new file" -> LICENSE, which offers templates. -->
