# Building the report on Overleaf

1. Create a new Overleaf project (Blank Project) or upload this `report/` folder.
2. Upload `main.tex` and keep the `figures/` folder alongside it.
3. Set the compiler to **pdfLaTeX** (Menu -> Compiler).
4. Press **Recompile**. It builds even with no figures (grey placeholder boxes
   appear where plots will go).

## Adding your plots

Run the experiments locally, then copy the generated PNGs into `figures/`
using these exact names (referenced by `main.tex`):

| Plot                                   | Produced by                  | Place in              |
|----------------------------------------|------------------------------|-----------------------|
| `accuracy_comparison.png`              | `python plot_results.py`     | `figures/`            |
| `figure2_bayesian_vs_deterministic.png`| `python plot_figure2.py`     | `figures/`            |
| `isic_figure5.png`                     | `python -m isic.plot_results`| `figures/`            |

(The scripts save them under `results/` or `results_isic/`; just copy them into
`report/figures/`.) Re-compile and the placeholders are replaced by your figures.

## Filling in the report

Search for `TODO` in `main.tex` — each marks a spot to add your own text,
numbers, or commentary (abstract, results discussion, the two result tables).
