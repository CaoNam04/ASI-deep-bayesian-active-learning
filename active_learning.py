"""The active-learning loop.

For each acquisition step we:
  1. (re)train a fresh model on the current labelled set,
  2. record test accuracy,
  3. score a random subset of the pool with the acquisition function,
  4. move the top-k points from the pool into the training set.
The model is reset every step to isolate the effect of the acquisition function
(as done in the paper), at the cost of longer runtime.
"""
import numpy as np
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from acquisition import select_queries
from config import Config
from data import make_active_learning_split
from engine import build_model, evaluate, train


def run_active_learning(cfg: Config, acquisition: str, seed: int, device, verbose=True):
    """Run one full active-learning experiment; return per-step test accuracies."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    split = make_active_learning_split(cfg, rng)
    labelled = split["init"]                  # grows over time
    pool_x, pool_y = split["pool_x"], split["pool_y"]
    val_ds, test_ds = split["val"], split["test"]

    test_acc_history = []

    for step in range(cfg.acquisition_steps + 1):
        # 1. Train a fresh model on the current labelled data (best-val checkpoint).
        n_points = len(labelled)
        model, optimizer = build_model(cfg, n_points, device)
        train(model, optimizer, labelled, val_ds, cfg, device)

        # 2. Evaluate on the fixed test set.
        test_acc, _ = evaluate(model, test_ds, cfg, device, mc=cfg.mc_eval)
        val_acc, _ = evaluate(model, val_ds, cfg, device)
        test_acc_history.append(test_acc)
        if verbose:
            print(f"[{acquisition}] step {step:3d} | "
                  f"train={n_points:4d} | val={val_acc:5.2f}% | test={test_acc:5.2f}%")

        if step == cfg.acquisition_steps:
            break  # no acquisition after the last evaluation

        # 3. Subsample the pool for efficiency, then score it.
        sub = rng.choice(len(pool_x), size=min(cfg.pool_subset, len(pool_x)),
                         replace=False)
        sub_x, sub_y = pool_x[sub], pool_y[sub]
        chosen = select_queries(acquisition, model, sub_x,
                                cfg.queries_per_step, cfg.mc_samples, device, rng,
                                deterministic=cfg.deterministic)

        # 4. Add chosen points to the labelled set, remove them from the pool.
        new_x, new_y = sub_x[chosen], sub_y[chosen]
        labelled = ConcatDataset([labelled, TensorDataset(new_x, new_y)])

        global_idx = sub[chosen]
        keep = np.ones(len(pool_x), dtype=bool)
        keep[global_idx] = False
        pool_x, pool_y = pool_x[keep], pool_y[keep]

    return np.array(test_acc_history)
