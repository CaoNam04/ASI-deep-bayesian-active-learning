"""Entry point for the ISIC 2016 melanoma experiment (section 5.5).

Example:
    python -m isic.main --data-dir isic_data/images --csv isic_data/labels.csv

Runs BALD and uniform acquisition over n_splits random test splits, with
repetitions per split, and saves AUC / positive-count curves per split.
"""
import argparse
import os

import numpy as np
import torch

from isic.active_learning import run_isic_active_learning
from isic.config import ISICConfig


def parse_args():
    p = argparse.ArgumentParser(description="ISIC 2016 Bayesian active learning")
    p.add_argument("--data-dir", default=None, help="folder of lesion images")
    p.add_argument("--csv", default=None, help="labels CSV (image_id,label)")
    p.add_argument("--splits", type=int, default=2)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--freeze-features", action="store_true",
                   help="train only the classifier head (faster)")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ISICConfig()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.csv:
        cfg.csv_path = args.csv
    cfg.n_splits = args.splits
    cfg.repetitions = args.reps
    cfg.max_steps = args.steps
    cfg.epochs = args.epochs
    cfg.freeze_features = args.freeze_features

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    print(f"Using device: {device}")
    os.makedirs(cfg.results_dir, exist_ok=True)

    for split in range(cfg.n_splits):
        for acq in ["BALD", "uniform"]:
            aucs, poss = [], []
            for rep in range(cfg.repetitions):
                auc, pos = run_isic_active_learning(cfg, acq, split, rep, device)
                aucs.append(auc)
                poss.append(pos)
            aucs, poss = np.stack(aucs), np.stack(poss)
            np.save(os.path.join(cfg.results_dir,
                    f"auc_split{split}_{acq}.npy"), aucs)
            np.save(os.path.join(cfg.results_dir,
                    f"pos_split{split}_{acq}.npy"), poss)
            print(f"  split {split} {acq}: final AUC "
                  f"{aucs[:, -1].mean():.3f} +/- {aucs[:, -1].std():.3f}")

    print(f"\nSaved results to {cfg.results_dir}/")


if __name__ == "__main__":
    main()
