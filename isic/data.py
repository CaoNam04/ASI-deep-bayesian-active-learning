"""ISIC 2016 data loading and the active-learning splits (section 5.5).

Expected layout (override with --data-dir / --csv):
    isic_data/images/ISIC_0000000.jpg, ...     # 900 dermoscopic JPEG images
    isic_data/labels.csv                        # classification ground truth

IMPORTANT: the PNG files bundled with the ISIC 2016 Task-3B download are
*segmentation masks* (single-channel 8-bit: 0 = background, 255 = lesion), NOT
benign/malignant labels. They tell you the lesion's shape, not its diagnosis,
so they cannot drive this classification task. The class labels live in a
separate ground-truth CSV (image_id, benign/malignant) -- point --csv at it.

CSV format (header optional):
    ISIC_0000000,benign
    ISIC_0000001,malignant
labels may be benign/malignant or 0/1 (or 0.0/1.0).

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


def _to_binary(raw) -> int:
    """Map a label cell to 1 (malignant) or 0 (benign)."""
    s = str(raw).strip().lower()
    if s in ("malignant", "1", "1.0", "true", "yes"):
        return 1
    try:
        return 1 if float(s) >= 0.5 else 0   # handles 0.0 / 1.0 and confidences
    except ValueError:
        return 0                              # benign / anything else


def load_index(cfg: ISICConfig):
    """Read the classification CSV and return arrays of image paths + binary labels.

    Note: PNG segmentation masks are NOT labels (see module docstring). This
    reader expects the benign/malignant ground-truth CSV. It tolerates a missing
    header and either string or numeric labels.
    """
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(
            f"Classification labels CSV not found at '{cfg.csv_path}'.\n"
            "The PNG files in the ISIC download are segmentation masks, not "
            "labels. Download the ISIC 2016 Task-3 'Training Ground Truth' CSV "
            "(image_id,benign/malignant) and point --csv at it.")

    # ISIC's official CSV has no header; processed copies sometimes add one.
    df = pd.read_csv(cfg.csv_path, header=None, dtype=str)
    first_cell = str(df.iloc[0, 0]).strip().lower()
    if first_cell in ("image", "image_id", "image_name", "id", "name"):
        df = df.iloc[1:].reset_index(drop=True)   # drop the header row

    paths, labels = [], []
    for _, row in df.iterrows():
        name = str(row[0]).strip()
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            name += ".jpg"
        paths.append(os.path.join(cfg.data_dir, name))
        labels.append(_to_binary(row[1]))
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
