"""Active-learning loop for ISIC melanoma diagnosis (section 5.5).

Each acquisition step resets the model to its pre-trained ImageNet weights and
fine-tunes from scratch (as in the paper) to isolate the acquisition effect.
We track test AUC and the number of positive (malignant) examples acquired.
"""
import copy

import numpy as np
import torch

from isic.acquisition import select_queries
from isic.config import ISICConfig
from isic.data import make_split, make_train_dataset, ISICDataset
from isic.engine import evaluate_auc, train
from isic.model import build_vgg16


def run_isic_active_learning(cfg: ISICConfig, acquisition: str, split_seed: int,
                             rep_seed: int, device, pretrained=True, verbose=True):
    """Run one ISIC AL experiment; return (auc_per_step, positives_per_step)."""
    rng = np.random.default_rng(10_000 * split_seed + rep_seed)
    torch.manual_seed(rep_seed)

    paths, labels, init_idx, pool_idx, test_ds = make_split(
        cfg, split_seed, rep_seed)

    labelled = list(init_idx)
    pool = list(pool_idx)

    # Build once to capture the pristine pre-trained weights, then reset to them.
    model = build_vgg16(cfg, pretrained=pretrained).to(device)
    pristine = copy.deepcopy(model.state_dict())

    auc_hist, pos_hist = [], []

    for step in range(cfg.max_steps + 1):
        # Reset to pre-trained weights and fine-tune on the current labelled set.
        model.load_state_dict(pristine)
        train_ds = make_train_dataset(cfg, paths, labels, labelled)
        train(model, train_ds, cfg, device, n_train_points=len(labelled))

        auc = evaluate_auc(model, test_ds, cfg, device)
        n_pos = int(np.sum(labels[np.array(labelled)] == 1))
        auc_hist.append(auc)
        pos_hist.append(n_pos)
        if verbose:
            print(f"[{acquisition} | split {split_seed} rep {rep_seed}] "
                  f"step {step} | labelled={len(labelled)} | pos={n_pos} | AUC={auc:.3f}")

        if step == cfg.max_steps or not pool:
            break

        # Score the pool and move the chosen points into the labelled set.
        pool_ds = ISICDataset(paths[np.array(pool)], labels[np.array(pool)],
                              cfg.img_size)
        chosen = select_queries(acquisition, model, pool_ds,
                                cfg.queries_per_step, cfg.mc_samples, device, rng)
        chosen_global = [pool[i] for i in chosen]
        labelled.extend(chosen_global)
        pool = [p for i, p in enumerate(pool) if i not in set(chosen)]

    return np.array(auc_hist), np.array(pos_hist)
