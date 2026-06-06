"""ISIC 2016 data loading and the active-learning splits (section 5.5).

Expected layout (override with --data-dir / --csv):
    isic_data/images/ISIC_0000000.jpg, ...
    isic_data/labels.csv   with columns: image_id,label
        label is 0/1 or "benign"/"malignant".

The 900-image set is unbalanced (727 benign, 173 malignant), so:
  - the test set is balanced (test_neg benign + test_pos malignant),
  - the initial training set mirrors the paper (80 benign + 20 malignant),
  - the rest forms the pool.
"""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from isic.config import ISICConfig

# ImageNet normalisation (VGG16 was trained with these statistics).
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _base_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _aug_transform(size):
    # Flip-based augmentation used by the paper for positive examples.
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class ISICDataset(Dataset):
    """Lesion images addressed by index, with optional augmentation."""

    def __init__(self, paths, labels, size, augment=False):
        self.paths = list(paths)
        self.labels = list(labels)
        self.tf = _aug_transform(size) if augment else _base_transform(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


def load_index(cfg: ISICConfig):
    """Read the CSV and return arrays of image paths and binary labels."""
    df = pd.read_csv(cfg.csv_path)
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("image_id") or cols.get("image") or df.columns[0]
    label_col = cols.get("label") or df.columns[1]

    paths, labels = [], []
    for _, row in df.iterrows():
        name = str(row[id_col])
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            name += ".jpg"
        raw = str(row[label_col]).strip().lower()
        y = 1 if raw in ("1", "1.0", "malignant") else 0
        paths.append(os.path.join(cfg.data_dir, name))
        labels.append(y)
    return np.array(paths), np.array(labels)


def make_split(cfg: ISICConfig, split_seed: int, rep_seed: int):
    """Build (initial labelled indices, pool indices, test dataset) for one run.

    `split_seed` fixes the balanced test set; `rep_seed` shuffles initial/pool.
    Returns paths, labels, and the index sets so the pool can shrink over time.
    """
    paths, labels = load_index(cfg)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]

    # Fixed balanced test set, chosen by split_seed.
    rng_split = np.random.default_rng(split_seed)
    test_pos = rng_split.choice(pos, cfg.test_pos, replace=False)
    test_neg = rng_split.choice(neg, cfg.test_neg, replace=False)
    test_idx = np.concatenate([test_pos, test_neg])

    # Remaining pos/neg available for training + pool.
    train_pos = np.setdiff1d(pos, test_pos)
    train_neg = np.setdiff1d(neg, test_neg)

    # Initial labelled set, shuffled by rep_seed.
    rng_rep = np.random.default_rng(1000 + rep_seed)
    rng_rep.shuffle(train_pos)
    rng_rep.shuffle(train_neg)
    init_idx = np.concatenate([train_neg[:cfg.init_neg], train_pos[:cfg.init_pos]])
    pool_idx = np.concatenate([train_neg[cfg.init_neg:], train_pos[cfg.init_pos:]])
    rng_rep.shuffle(pool_idx)

    test_ds = ISICDataset(paths[test_idx], labels[test_idx], cfg.img_size)
    return paths, labels, init_idx, pool_idx, test_ds


def make_train_dataset(cfg: ISICConfig, paths, labels, labelled_idx):
    """Build a training dataset, oversampling positives with flip augmentation."""
    idx = list(labelled_idx)
    # Add augmented copies of each positive example.
    aug_idx = [i for i in idx if labels[i] == 1] * cfg.pos_augment
    all_idx = np.array(idx + aug_idx)

    # Non-augmented for originals + augmented transform for the extra positives.
    base = ISICDataset(paths[np.array(idx)], labels[np.array(idx)], cfg.img_size,
                       augment=False)
    if aug_idx:
        aug = ISICDataset(paths[np.array(aug_idx)], labels[np.array(aug_idx)],
                          cfg.img_size, augment=True)
        from torch.utils.data import ConcatDataset
        return ConcatDataset([base, aug])
    return base
