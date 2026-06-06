"""Plot saved accuracy curves to reproduce Figure 1 of the paper.

Reads results/test_acc_<ACQ>.npy files and draws test accuracy vs. number of
acquired images, averaged over experiment repetitions (with std shading).

Usage:
    python plot_results.py
"""
import os

import numpy as np

from config import ACQUISITIONS, Config


def main():
    import matplotlib.pyplot as plt

    cfg = Config()
    plt.figure(figsize=(8, 5))

    for acq in ACQUISITIONS:
        path = os.path.join(cfg.results_dir, f"test_acc_{acq}.npy")
        if not os.path.exists(path):
            print(f"skip {acq}: {path} not found")
            continue

        runs = np.load(path)              # (experiments, steps+1)
        mean = runs.mean(axis=0)
        std = runs.std(axis=0)

        # x-axis: number of labelled images at each step.
        n_acquired = (cfg.initial_train_size
                      + np.arange(len(mean)) * cfg.queries_per_step)

        plt.plot(n_acquired, mean, label=acq)
        plt.fill_between(n_acquired, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Number of acquired images")
    plt.ylabel("Test accuracy (%)")
    plt.title("Active learning on MNIST (Bayesian CNN)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(cfg.results_dir, "accuracy_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
