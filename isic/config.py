"""Configuration for the ISIC 2016 melanoma active-learning experiment (section 5.5)."""
from dataclasses import dataclass


@dataclass
class ISICConfig:
    # ---- Data layout (set --data-dir / --csv on the command line) ----
    img_size: int = 224               # VGG16 input size
    data_dir: str = "isic_data/images"        # folder of .jpg lesion images
    csv_path: str = "isic_data/labels.csv"    # CSV: image_id,label (label 0/1 or benign/malignant)

    # ---- Balanced test set (per split) ----
    test_neg: int = 100               # benign test images
    test_pos: int = 100               # malignant test images

    # ---- Initial labelled set ----
    init_neg: int = 80                # benign
    init_pos: int = 20                # malignant

    # ---- Active-learning loop ----
    queries_per_step: int = 100       # images labelled per acquisition
    max_steps: int = 4                # acquisition steps (paper shows 0..4)
    mc_samples: int = 20              # MC-dropout passes (paper uses 20 at test time)

    # ---- Training (fine-tuning VGG16) ----
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4                  # small LR for fine-tuning
    length_scale_sq: float = 0.5      # l^2 in the weight-decay formula
    dropout_p: float = 0.5
    pos_augment: int = 3              # extra flipped copies of each positive

    # ---- Experiment management ----
    n_splits: int = 2                 # number of random test splits
    repetitions: int = 3              # repeats per split (averaged)
    seed: int = 0
    freeze_features: bool = False     # if True, only train the classifier head
    results_dir: str = "results_isic"
