"""Acquisition functions for the ISIC experiment (section 5.5).

Only BALD and a uniform baseline are used here: Variation Ratios fails on this
imbalanced data because nearly all images get the same modal-class probability.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from isic.model import enable_dropout

_EPS = 1e-12


@torch.no_grad()
def mc_dropout_probs(model, dataset, T, device, batch_size=16):
    """Return MC-dropout softmax probabilities of shape (N, T, 2)."""
    model.eval()
    enable_dropout(model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    samples = []
    for _ in range(T):
        probs = []
        for x, _ in loader:
            logits = model(x.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu())
        samples.append(torch.cat(probs, dim=0))
    return torch.stack(samples, dim=1).numpy()


def bald(probs):
    """Mutual information = entropy(mean) - mean(entropy)."""
    mean_p = probs.mean(axis=1)
    entropy_of_mean = -(mean_p * np.log(mean_p + _EPS)).sum(axis=1)
    entropy_per_pass = -(probs * np.log(probs + _EPS)).sum(axis=2)
    return entropy_of_mean - entropy_per_pass.mean(axis=1)


def select_queries(name, model, pool_dataset, n_queries, T, device, rng):
    """Return indices (into the pool) of the points to label next."""
    n = len(pool_dataset)
    n_queries = min(n_queries, n)
    if name == "uniform":
        return rng.choice(n, size=n_queries, replace=False)
    probs = mc_dropout_probs(model, pool_dataset, T, device)
    scores = bald(probs)
    return scores.argsort()[-n_queries:][::-1].copy()
