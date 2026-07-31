"""
run_gtm_net.py ── GTM-KAN-MoA Entry Point
=============================================================
Usage:
    python run_gtm_net.py --dataset luo
    python run_gtm_net.py --dataset new
    python run_gtm_net.py --dataset both

SMILES and FASTA sequences are fetched automatically from PubChem and UniProt
the first run, then cached in gtmnet/seq_cache/ for all subsequent runs.
"""

import os
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import (
    load_luo_dataset, load_new_dataset, load_luo_folds,
    _ROOT, OUT_DIR
)
from gtm_net import GTMNet


def _parse_args():
    parser = argparse.ArgumentParser(description="GTM-KAN-MoA: Advanced DTI framework")
    parser.add_argument('--dataset', choices=['luo', 'new', 'both'], default='luo')
    parser.add_argument('--n_iter', type=int, default=30,
                        help="GTM EM iterations (default: 30)")
    args, _ = parser.parse_known_args()
    return args


def print_table(summary: dict, ds: str) -> None:
    print(f"\n{'=' * 56}\n  RESULTS — {ds.upper()}\n{'=' * 56}")
    print(f"  {'Metric':<10} {'Mean':>8} {'±Std':>8}")
    print("  " + "-" * 30)
    for metric in ['auroc', 'aupr', 'acc', 'prec', 'rec', 'f1']:
        if metric in summary and 'mean' in summary[metric]:
            m = summary[metric]['mean']
            s = summary[metric]['std']
            print(f"  {metric.upper():<10} {m:>8.4f} {s:>8.4f}")
    print('=' * 56)


def main():
    args = _parse_args()
    datasets = ['luo', 'new'] if args.dataset == 'both' else [args.dataset]

    print(f"\n[Environment]")
    print(f"  ROOT DIR : {_ROOT}")
    print(f"  OUT DIR  : {OUT_DIR}")

    for ds in datasets:
        print(f"\n{'#'*60}\n# GTM-KAN-MoA | Dataset: {ds.upper()}\n{'#'*60}")
        t0 = time.time()

        # Load raw network matrices + inject drug_ids / prot_ids
        data = load_luo_dataset() if ds == 'luo' else load_new_dataset()

        # Build model — sequences are resolved automatically inside fit()
        model = GTMNet(n_iter=args.n_iter, verbose=True)
        model.fit(data)

        if ds == 'luo':
            folds   = load_luo_folds()
            summary = model.evaluate_cv(folds, is_luo=True)
        else:
            from sklearn.model_selection import train_test_split
            summary = model.evaluate_random_split(n_splits=5) \
                if hasattr(model, 'evaluate_random_split') \
                else model.evaluate_cv(load_luo_folds(1), is_luo=False)

        elapsed = time.time() - t0
        print(f"\n  Total runtime: {elapsed:.1f}s")
        print_table(summary, ds)

        out_j = os.path.join(OUT_DIR, f'gtm_kan_moa_{ds}_results.json')
        with open(out_j, 'w') as fh:
            json.dump({k: v for k, v in summary.items()
                       if isinstance(v, dict) and 'mean' in v}, fh, indent=2)
        print(f"  Results saved → {out_j}")


if __name__ == '__main__':
    main()
