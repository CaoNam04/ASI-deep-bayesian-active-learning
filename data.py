"""MNIST loading and the initial active-learning data split.

We split the standard MNIST training set into:
    - a tiny balanced initial labelled set (2 per class -> 20 points),
    - a small validation set,
    - a large unlabelled pool,
and keep the standard 10k test set untouched.
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

from config import Config


def _load_mnist(cfg: Config):
    """Download MNIST and return tensors scaled to [0, 1]."""
    tf = transforms.ToTensor()
    train = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=tf)
    test = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=tf)

    # .data is uint8 [N,28,28]; add channel dim and scale to [0,1].
    x_train = train.data.unsqueeze(1).float() / 255.0
    y_train = train.targets.clone()
    x_test = test.data.unsqueeze(1).float() / 255.0
    y_test = test.targets.clone()
    return x_train, y_train, x_test, y_test


def make_active_learning_split(cfg: Config, rng: np.random.Generator):
    """Build (initial_train, val, pool, test) datasets.

    `rng` controls the random partition so each experiment repetition differs.
    Returns a dict of TensorDatasets plus the raw pool tensors (needed because
    the pool shrinks as points are acquired).
    """
    x_train, y_train, x_test, y_test = _load_mnist(cfg)

    # Shuffle the full training set.
    perm = rng.permutation(len(y_train))
    x_train, y_train = x_train[perm], y_train[perm]

    # Balanced initial training set: pick the first `per_class` of each digit.
    per_class = cfg.initial_train_size // cfg.num_classes
    init_idx = []
    for c in range(cfg.num_classes):
        cls_idx = np.where(y_train.numpy() == c)[0][:per_class]
        init_idx.extend(cls_idx.tolist())
    init_idx = np.array(init_idx)

    # Remaining indices (exclude the initial set) for val and pool.
    mask = np.ones(len(y_train), dtype=bool)
    mask[init_idx] = False
    rest = np.where(mask)[0]
    val_idx = rest[: cfg.val_size]
    pool_idx = rest[cfg.val_size : cfg.val_size + cfg.pool_size]

    init_ds = TensorDataset(x_train[init_idx], y_train[init_idx])
    val_ds = TensorDataset(x_train[val_idx], y_train[val_idx])
    test_ds = TensorDataset(x_test[:cfg.test_size], y_test[:cfg.test_size])

    # Pool kept as raw tensors so we can remove acquired points easily.
    pool_x = x_train[pool_idx].clone()
    pool_y = y_train[pool_idx].clone()

    return {
        "init": init_ds,
        "val": val_ds,
        "test": test_ds,
        "pool_x": pool_x,
        "pool_y": pool_y,
    }
