"""Reproduce Table 2 (section 5.4): test error at 1000 labelled images.

Run the MNIST experiments with a large validation set first, e.g.:
    python main.py --acquisition VAR_RATIOS --val-size 5000 --steps 98
    python main.py --acquisition BALD       --val-size 5000 --steps 98
    ...
then print the comparison table:
    python report_table.py --labelled 1000
"""
import argparse
import os

import numpy as np

from config import ACQUISITIONS, Config

# Test errors of semi-supervised methods reported in the paper (Table 2),
# shown for context. These are NOT recomputed here.
SEMI_SUPERVISED = {
    "Semi-sup. Embedding (Weston 2012)": 5.73,
    "Transductive SVM (Weston 2012)": 5.38,
    "MTC (Rifai 2011)": 3.64,
    "Pseudo-label (Lee 2013)": 3.46,
    "AtlasRBF (Pitelis 2014)": 3.68,
    "DGN (Kingma 2014)": 2.40,
    "Virtual Adversarial (Miyato 2015)": 1.32,
    "Ladder Network Gamma (Rasmus 2015)": 1.53,
    "Ladder Network full (Rasmus 2015)": 0.84,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", type=int, default=1000,
                    help="number of labelled images to report error at")
    args = ap.parse_args()

    cfg = Config()
    # Map "number of labelled images" to the acquisition-step index.
    step = (args.labelled - cfg.initial_train_size) // cfg.queries_per_step

    print("Semi-supervised methods (from the paper, for reference):")
    for name, err in SEMI_SUPERVISED.items():
        print(f"  {name:38s} {err:5.2f}%")

    print(f"\nActive learning (this repo) at {args.labelled} labelled images:")
    for acq in ACQUISITIONS:
        path = os.path.join(cfg.results_dir, f"test_acc_{acq}.npy")
        if not os.path.exists(path):
            print(f"  {acq:38s}   (no results: {path})")
            continue
        runs = np.load(path)                  # (experiments, steps+1)
        if step >= runs.shape[1]:
            print(f"  {acq:38s}   (only {runs.shape[1]-1} steps run)")
            continue
        acc = runs[:, step].mean()
        err = 100.0 - acc
        print(f"  {acq:38s} {err:5.2f}%   (acc {acc:5.2f}%)")


if __name__ == "__main__":
    main()
