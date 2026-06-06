"""Reproduce Figure 5: AUC and number of acquired positives per split.

Usage:
    python -m isic.plot_results
"""
import os

import numpy as np

from isic.config import ISICConfig


def main():
    import matplotlib.pyplot as plt

    cfg = ISICConfig()
    fig, axes = plt.subplots(cfg.n_splits, 2, figsize=(11, 4 * cfg.n_splits),
                             squeeze=False)

    for split in range(cfg.n_splits):
        for acq, color in [("BALD", "C0"), ("uniform", "C2")]:
            auc_path = os.path.join(cfg.results_dir, f"auc_split{split}_{acq}.npy")
            pos_path = os.path.join(cfg.results_dir, f"pos_split{split}_{acq}.npy")
            if not (os.path.exists(auc_path) and os.path.exists(pos_path)):
                continue
            auc = np.load(auc_path)            # (reps, steps+1)
            pos = np.load(pos_path)
            steps = np.arange(auc.shape[1])
            se_auc = auc.std(0) / np.sqrt(auc.shape[0])

            ax_a, ax_p = axes[split][0], axes[split][1]
            ax_a.plot(steps, auc.mean(0), color=color, label=acq)
            ax_a.fill_between(steps, auc.mean(0) - se_auc, auc.mean(0) + se_auc,
                              color=color, alpha=0.2)
            ax_p.plot(steps, pos.mean(0), color=color, label=acq)

            ax_a.set_title(f"AUC (split {split})")
            ax_a.set_xlabel("Acquisition step"); ax_a.set_ylabel("AUC")
            ax_a.legend(); ax_a.grid(alpha=0.3)
            ax_p.set_title(f"# positives acquired (split {split})")
            ax_p.set_xlabel("Acquisition step")
            ax_p.set_ylabel("# positive examples")
            ax_p.legend(); ax_p.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(cfg.results_dir, "isic_figure5.png")
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
