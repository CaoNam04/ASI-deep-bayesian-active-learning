"""Reproduce Figure 2: Bayesian (red) vs deterministic (blue) CNN.

Compares the two model types for BALD, Variation Ratios and Max Entropy. Run
each acquisition function in both modes first, e.g.:

    python main.py --acquisition BALD
    python main.py --acquisition BALD --deterministic
    ... (same for VAR_RATIOS and MAX_ENTROPY)

then:
    python plot_figure2.py
"""
import os

import numpy as np

from config import Config

FUNCS = ["BALD", "VAR_RATIOS", "MAX_ENTROPY"]


def main():
    import matplotlib.pyplot as plt

    cfg = Config()
    fig, axes = plt.subplots(1, len(FUNCS), figsize=(5 * len(FUNCS), 4),
                             squeeze=False)

    for ax, acq in zip(axes[0], FUNCS):
        for suffix, color, label in [("", "C3", "Bayesian"),
                                     ("_deterministic", "C0", "Deterministic")]:
            path = os.path.join(cfg.results_dir, f"test_acc_{acq}{suffix}.npy")
            if not os.path.exists(path):
                print(f"skip {acq}{suffix}: {path} not found")
                continue
            runs = np.load(path)
            mean, std = runs.mean(axis=0), runs.std(axis=0)
            n = cfg.initial_train_size + np.arange(len(mean)) * cfg.queries_per_step
            ax.plot(n, mean, color=color, label=label)
            ax.fill_between(n, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(acq)
        ax.set_xlabel("Number of acquired images")
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(cfg.results_dir, "figure2_bayesian_vs_deterministic.png")
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
