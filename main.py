"""Command-line entry point.

Examples:
    python main.py --acquisition BALD
    python main.py --acquisition VAR_RATIOS --experiments 3 --steps 100
    python main.py --acquisition RANDOM --quick   # tiny run for a smoke test
"""
import argparse
import os

import numpy as np
import torch

from active_learning import run_active_learning
from config import ACQUISITIONS, Config


def parse_args():
    p = argparse.ArgumentParser(description="Deep Bayesian Active Learning (MNIST)")
    p.add_argument("--acquisition", choices=ACQUISITIONS, default="BALD",
                   help="acquisition function to use")
    p.add_argument("--experiments", type=int, default=3,
                   help="number of repetitions to average over")
    p.add_argument("--steps", type=int, default=100,
                   help="number of acquisition steps")
    p.add_argument("--epochs", type=int, default=50,
                   help="training epochs per acquisition round")
    p.add_argument("--mc-samples", type=int, default=50,
                   help="MC-dropout forward passes when scoring the pool")
    p.add_argument("--val-size", type=int, default=None,
                   help="validation set size (use 5000 to reproduce section 5.4)")
    p.add_argument("--seed", type=int, default=1, help="base random seed")
    p.add_argument("--mc-eval", action="store_true",
                   help="use MC-dropout for test evaluation (paper-faithful, slower)")
    p.add_argument("--deterministic", action="store_true",
                   help="deterministic CNN acquisition (no MC dropout) for Figure 2")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA exists")
    p.add_argument("--quick", action="store_true",
                   help="tiny config for a fast smoke test")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.experiments = args.experiments
    cfg.acquisition_steps = args.steps
    cfg.epochs = args.epochs
    cfg.mc_samples = args.mc_samples
    cfg.seed = args.seed
    cfg.mc_eval = args.mc_eval
    cfg.deterministic = args.deterministic
    if args.val_size is not None:
        cfg.val_size = args.val_size

    if args.quick:  # fast settings just to verify the pipeline runs
        cfg.acquisition_steps = 3
        cfg.epochs = 2
        cfg.experiments = 1
        cfg.mc_samples = 5
        cfg.pool_subset = 200
        cfg.test_size = 2000

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.results_dir, exist_ok=True)

    # Run repetitions and stack the accuracy curves.
    all_runs = []
    for exp in range(cfg.experiments):
        seed = cfg.seed + exp
        print(f"\n=== {args.acquisition} | experiment {exp + 1}/"
              f"{cfg.experiments} (seed {seed}) ===")
        acc = run_active_learning(cfg, args.acquisition, seed, device)
        all_runs.append(acc)

    all_runs = np.stack(all_runs)  # (experiments, steps+1)
    suffix = "_deterministic" if cfg.deterministic else ""
    out = os.path.join(cfg.results_dir,
                       f"test_acc_{args.acquisition}{suffix}.npy")
    np.save(out, all_runs)
    print(f"\nSaved results to {out}")
    print(f"Final mean test accuracy: {all_runs[:, -1].mean():.2f}%")


if __name__ == "__main__":
    main()
