# Deep Bayesian Active Learning with Image Data

A clean, from-scratch PyTorch implementation of

> Yarin Gal, Riashat Islam, Zoubin Ghahramani.
> **Deep Bayesian Active Learning with Image Data.** ICML 2017.
> [arXiv:1703.02910](https://arxiv.org/abs/1703.02910)

The project trains a **Bayesian CNN** (a CNN with dropout used as approximate
Bayesian inference via **MC-dropout**) and uses its predictive uncertainty to
decide which unlabelled images are most worth labelling. It reproduces the MNIST
active-learning experiment comparing five acquisition functions.

## Acquisition functions

| Name          | Idea                                                            |
|---------------|----------------------------------------------------------------|
| `BALD`        | Mutual information between prediction and weights (disagreement)|
| `VAR_RATIOS`  | 1 − fraction of MC passes agreeing with the modal class        |
| `MAX_ENTROPY` | Predictive entropy of the mean prediction                      |
| `MEAN_STD`    | Mean per-class standard deviation across MC passes             |
| `RANDOM`      | Uniform random baseline                                        |

## Project structure

```
deep-bayesian-active-learning/
├── config.py            # all hyperparameters (paper defaults)
├── model.py             # Bayesian CNN + MC-dropout helper
├── data.py              # MNIST loading and active-learning split
├── acquisition.py       # MC-dropout predictions + acquisition functions
├── engine.py            # train / evaluate routines
├── active_learning.py   # the active-learning loop
├── main.py              # command-line entry point (MNIST)
├── plot_results.py      # reproduce Figure 1
├── report_table.py      # section 5.4: test-error table at 1000 labels
├── run_all.bat          # run every acquisition (Windows)
├── isic/                # section 5.5: ISIC 2016 melanoma experiment
│   ├── config.py        #   ISIC hyperparameters
│   ├── model.py         #   Bayesian VGG16 (fine-tuned, dropout head)
│   ├── data.py          #   ISIC loading, balanced splits, augmentation
│   ├── acquisition.py   #   BALD + uniform (binary)
│   ├── engine.py        #   train + AUC evaluation (MC dropout)
│   ├── active_learning.py #  ISIC AL loop (reset to pre-trained each step)
│   ├── main.py          #   ISIC entry point
│   └── plot_results.py  #   reproduce Figure 5
├── requirements.txt
└── README.md
```

## Setup (Windows + NVIDIA RTX GPU)

1. **Install Python 3.10+** from [python.org](https://www.python.org/downloads/)
   (tick "Add Python to PATH" during install).

2. **Create and activate a virtual environment** (in PowerShell or CMD):

   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install PyTorch with CUDA.** Pick the command for your CUDA version from
   <https://pytorch.org/get-started/locally/>. For a recent RTX card:

   ```bat
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install the remaining dependencies:**

   ```bat
   pip install -r requirements.txt
   ```

5. **Verify the GPU is visible:**

   ```bat
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   You should see `True` and your RTX card's name.

## Usage

Quick smoke test (a few minutes, just checks the pipeline works):

```bat
python main.py --acquisition BALD --quick
```

Single full experiment:

```bat
python main.py --acquisition BALD --experiments 3 --steps 100
```

Run all acquisition functions and plot the comparison:

```bat
run_all.bat
```

Plot from already-saved results:

```bat
python plot_results.py
```

### Useful flags

| Flag             | Meaning                                        |
|------------------|------------------------------------------------|
| `--acquisition`  | `BALD`, `VAR_RATIOS`, `MAX_ENTROPY`, `MEAN_STD`, `RANDOM` |
| `--experiments`  | repetitions to average over (default 3)        |
| `--steps`        | acquisition steps (default 100)                |
| `--epochs`       | training epochs per round (default 50)         |
| `--mc-samples`   | MC-dropout passes when scoring (default 50)    |
| `--mc-eval`      | use MC-dropout for test evaluation (paper-faithful, slower) |
| `--cpu`          | force CPU even if CUDA is available            |
| `--quick`        | tiny config for a fast smoke test              |

## Reproducing the paper's experiments

### Sections 5.1–5.3 (MNIST, acquisition comparison)

```bat
run_all.bat
python plot_results.py
```

### Section 5.4 (comparison to semi-supervised learning)

Same MNIST pipeline but with a large validation set (5000) and accuracy read off
at 1000 labelled images, then printed as a Table-2-style error table:

```bat
python main.py --acquisition VAR_RATIOS --val-size 5000 --steps 98
python main.py --acquisition BALD       --val-size 5000 --steps 98
python main.py --acquisition MAX_ENTROPY --val-size 5000 --steps 98
python main.py --acquisition RANDOM      --val-size 5000 --steps 98
python report_table.py --labelled 1000
```

### Section 5.5 (ISIC 2016 melanoma diagnosis)

Fine-tunes a Bayesian VGG16 and compares BALD vs uniform acquisition with the
AUC metric on the imbalanced ISIC 2016 lesion dataset. Code lives in `isic/`.

**Get the data.** Download the ISIC 2016 *Task 3 (classification)* training set
from <https://challenge.isic-archive.com/data/> (900 dermoscopic images +
ground-truth CSV with benign/malignant labels). Arrange it as:

```
isic_data/
├── images/        # ISIC_0000000.jpg, ...
└── labels.csv     # columns: image_id,label   (label = 0/1 or benign/malignant)
```

**Run:**

```bat
python -m isic.main --data-dir isic_data/images --csv isic_data/labels.csv
python -m isic.plot_results
```

Useful flags: `--splits` (random test splits, default 2), `--reps` (repeats per
split, default 3), `--steps` (acquisition steps, default 4), `--epochs`
(fine-tuning epochs, default 100), `--freeze-features` (train only the head, much
faster), `--cpu`.

## Training stability

The model is **reset and retrained from scratch at every acquisition step** (as
in the paper) to isolate the effect of the acquisition function. To avoid a
single bad epoch collapsing the reported accuracy, training keeps the weights
with the **best validation accuracy** rather than the final-epoch weights, and
weight decay is applied only to the dense layers (capped at `weight_decay_max`).
If you still see noisy curves, increase `--experiments` to average over more runs.

## Notes on runtime

The model is **reset and retrained from scratch at every acquisition step** (as
in the paper) to isolate the effect of the acquisition function. With 100 steps
× 3 repetitions this is the slow part; an RTX GPU handles it comfortably, but you
can reduce `--steps` or `--experiments` for quicker results. Expected outcome:
`BALD` and `VAR_RATIOS` reach low test error with far fewer labels than `RANDOM`.

## License

MIT
