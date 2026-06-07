"""Configuration for the ISIC 2016 melanoma active-learning experiment (section 5.5).

Defaults follow the procedure described in Gal, Islam & Ghahramani (2017), sec. 5.5.
Each value is annotated with its source in the paper.
"""
from dataclasses import dataclass


@dataclass
class ISICConfig:
    # ---- Data layout (override with --data-dir / --csv) ----
    img_size: int = 224               # VGG16 input resolution
    data_dir: str = "isic_data/images"        # folder of .jpg lesion images
    csv_path: str = "isic_data/labels.csv"    # classification CSV (image_id,benign/malignant)

    # ---- Balanced test set, per split (paper: "set aside 100 negative and
    #      100 positive examples") ----
    test_neg: int = 100               # benign test images
    test_pos: int = 100               # malignant test images

    # ---- Initial labelled set (paper: "80 negative examples and 20 positive") ----
    init_neg: int = 80
    init_pos: int = 20

    # ---- Active-learning loop ----
    queries_per_step: int = 100       # "select the 100 most informative images"
    max_steps: int = 4                # paper's Fig. 5 shows acquisition steps 0..4
    mc_samples: int = 20              # "MC dropout with 20 samples" at test time

    # ---- Training: fine-tuning VGG16 (paper: model of Agarwal et al. 2016) ----
    epochs: int = 100                 # "trained ... for 100 epochs until convergence"
    batch_size: int = 8               # "batch size 8"
    lr: float = 1e-4                  # "small learning rate" (exact value unspecified)
    dropout_p: float = 0.5            # p in the weight-decay formula and FC dropout
    length_scale_sq: float = 0.5      # l^2 in the weight-decay formula
    # Weight decay applied in engine.py: (1 - p) * l^2 / N  ->  0.25 / N.
    pos_augment: int = 2              # flip positives vertically AND horizontally (2 copies)

    # ---- Experiment management ----
    n_splits: int = 2                 # "two different random splits"
    repetitions: int = 3              # "repeat our experiments three times and average"
    seed: int = 0
    freeze_features: bool = False     # True => train only the classifier head (faster)
    results_dir: str = "results_isic"
