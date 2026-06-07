"""Central configuration for the Deep Bayesian Active Learning experiments.

All defaults follow the MNIST setup described in Gal, Islam & Ghahramani (2017),
"Deep Bayesian Active Learning with Image Data" (https://arxiv.org/abs/1703.02910).
"""
from dataclasses import dataclass, field
from typing import List


# Acquisition functions implemented in this repo.
ACQUISITIONS: List[str] = [
    "BALD",          # Bayesian Active Learning by Disagreement
    "VAR_RATIOS",    # Variation Ratios
    "MAX_ENTROPY",   # Predictive entropy
    "MEAN_STD",      # Mean standard deviation
    "RANDOM",        # Uniform random baseline
]


@dataclass
class Config:
    # ---- Data / active-learning split (paper, section 5.1) ----
    initial_train_size: int = 20      # 2 balanced points per class
    val_size: int = 100               # small, realistic validation set
    pool_size: int = 40000            # unlabelled pool drawn from the train set
    test_size: int = 10000            # standard MNIST test set

    # ---- Active-learning loop ----
    acquisition_steps: int = 100      # number of acquisition iterations
    queries_per_step: int = 10        # points labelled at each step
    pool_subset: int = 2000           # random pool subsample scored per step (speed)
    mc_samples: int = 50              # T: MC-dropout forward passes for scoring

    # ---- Model / training ----
    num_classes: int = 10
    epochs: int = 50                  # epochs per (re)training round
    batch_size: int = 128
    lr: float = 1e-3                  # Adam learning rate
    weight_decay_base: float = 3.5    # decay = base / num_train_points ...
    weight_decay_max: float = 5e-3    # ... capped so small-N decay isn't extreme
    dropout_conv: float = 0.25
    dropout_dense: float = 0.5
    num_filters: int = 32
    kernel_size: int = 4
    pool_size_kernel: int = 2
    dense_units: int = 128

    # ---- Experiment management ----
    experiments: int = 3              # repetitions to average over
    seed: int = 1
    mc_eval: bool = False             # use MC-dropout for test evaluation (paper-faithful)
    deterministic: bool = False       # deterministic CNN for acquisition (Figure 2)
    results_dir: str = "results"
    data_dir: str = "data"
