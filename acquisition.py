"""Acquisition functions for Bayesian active learning.

All functions consume MC-dropout predictions of shape (N, T, C):
    N = number of pool points, T = MC samples, C = classes.
Each returns a score per point; higher score = more worth labelling.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import enable_dropout

_EPS = 1e-12  # numerical stability for logs


@torch.no_grad()
def mc_dropout_predictions(model, x, T, device, batch_size=256):
    """Run T stochastic forward passes; return softmax probs (N, T, C)."""
    model.eval()
    enable_dropout(model)  # keep dropout active for sampling

    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    samples = []  # one (N, C) array per MC pass
    for _ in range(T):
        probs = []
        for (batch,) in loader:
            logits = model(batch.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu())
        samples.append(torch.cat(probs, dim=0))
    return torch.stack(samples, dim=1).numpy()  # (N, T, C)


# ---- Individual acquisition scores ----

def max_entropy(probs):
    """Predictive entropy of the mean prediction."""
    mean_p = probs.mean(axis=1)                       # (N, C)
    return -(mean_p * np.log(mean_p + _EPS)).sum(axis=1)


def bald(probs):
    """Mutual information = entropy(mean) - mean(entropy). Captures disagreement."""
    mean_p = probs.mean(axis=1)
    entropy_of_mean = -(mean_p * np.log(mean_p + _EPS)).sum(axis=1)
    entropy_per_pass = -(probs * np.log(probs + _EPS)).sum(axis=2)  # (N, T)
    mean_entropy = entropy_per_pass.mean(axis=1)
    return entropy_of_mean - mean_entropy


def variation_ratios(probs):
    """1 - fraction of MC passes that agree with the modal class."""
    preds = probs.argmax(axis=2)                      # (N, T) hard labels
    N, T = preds.shape
    scores = np.empty(N)
    for i in range(N):
        counts = np.bincount(preds[i], minlength=probs.shape[2])
        scores[i] = 1.0 - counts.max() / T
    return scores


def mean_std(probs):
    """Mean over classes of the per-class standard deviation across MC passes."""
    std_per_class = probs.std(axis=1)                 # (N, C)
    return std_per_class.mean(axis=1)


# ---- Dispatcher ----

_FUNCS = {
    "BALD": bald,
    "VAR_RATIOS": variation_ratios,
    "MAX_ENTROPY": max_entropy,
    "MEAN_STD": mean_std,
}


def score_pool(name, model, pool_x, T, device):
    """Return acquisition scores for every point in `pool_x`."""
    probs = mc_dropout_predictions(model, pool_x, T, device)
    return _FUNCS[name](probs)


def select_queries(name, model, pool_x, n_queries, T, device, rng):
    """Pick indices of the top-n points to label (random for the baseline)."""
    if name == "RANDOM":
        return rng.choice(len(pool_x), size=n_queries, replace=False)
    scores = score_pool(name, model, pool_x, T, device)
    # Top-n indices, highest score first. .copy() avoids a negative stride
    # (PyTorch can't index with negative-stride numpy arrays).
    return scores.argsort()[-n_queries:][::-1].copy()
