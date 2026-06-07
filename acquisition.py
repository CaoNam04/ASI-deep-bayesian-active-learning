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


@torch.no_grad()
def deterministic_predictions(model, x, device, batch_size=256):
    """Single deterministic forward pass (dropout OFF); returns (N, 1, C).

    Models a point-mass posterior q*(w) = delta(w - theta): captures only
    aleatoric uncertainty, not epistemic. Used for the deterministic CNN.
    """
    model.eval()  # dropout off
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    probs = []
    for (batch,) in loader:
        logits = model(batch.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(probs, dim=0).unsqueeze(1).numpy()  # (N, 1, C)


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
    """1 - max over classes of the MEAN predicted probability.

    This is the paper's definition (VR = 1 - max_y p(y|x)). Using the mean
    predictive probability (rather than hard-label vote counts) makes it well
    defined for a single deterministic pass too, which we need for Figure 2.
    """
    mean_p = probs.mean(axis=1)                       # (N, C)
    return 1.0 - mean_p.max(axis=1)


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


def score_pool(name, model, pool_x, T, device, deterministic=False):
    """Return acquisition scores for every point in `pool_x`."""
    if deterministic:
        probs = deterministic_predictions(model, pool_x, device)  # (N, 1, C)
    else:
        probs = mc_dropout_predictions(model, pool_x, T, device)  # (N, T, C)
    return _FUNCS[name](probs)


def select_queries(name, model, pool_x, n_queries, T, device, rng,
                   deterministic=False):
    """Pick indices of the top-n points to label (random for the baseline)."""
    if name == "RANDOM":
        return rng.choice(len(pool_x), size=n_queries, replace=False)
    scores = score_pool(name, model, pool_x, T, device, deterministic)
    # Tiny jitter breaks ties randomly. For acquisition functions that go
    # degenerate on a deterministic model (BALD and Mean STD become all-zero),
    # this makes selection effectively random -- the expected behaviour, since a
    # deterministic CNN has no epistemic uncertainty to exploit.
    scores = scores + rng.normal(0, 1e-9, size=scores.shape)
    # Top-n indices, highest score first. .copy() avoids a negative stride.
    return scores.argsort()[-n_queries:][::-1].copy()
