@echo off
REM Reproduce Figure 2: run BALD, Var Ratios, Max Entropy with BOTH a Bayesian
REM CNN (MC dropout) and a deterministic CNN, then plot the comparison.

setlocal
set EXPERIMENTS=3
set STEPS=100

for %%A in (BALD VAR_RATIOS MAX_ENTROPY) do (
    echo === %%A : Bayesian CNN ===
    python main.py --acquisition %%A --experiments %EXPERIMENTS% --steps %STEPS%
    echo === %%A : Deterministic CNN ===
    python main.py --acquisition %%A --experiments %EXPERIMENTS% --steps %STEPS% --deterministic
)

echo Plotting Figure 2...
python plot_figure2.py

echo Done.
endlocal
