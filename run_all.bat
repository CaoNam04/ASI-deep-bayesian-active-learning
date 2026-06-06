@echo off
REM Run every acquisition function in sequence, then plot the comparison.
REM Usage:  run_all.bat
REM Tip:    edit --experiments / --steps below for shorter runs.

setlocal

set EXPERIMENTS=3
set STEPS=100

for %%A in (RANDOM MEAN_STD MAX_ENTROPY BALD VAR_RATIOS) do (
    echo ============================================
    echo Running acquisition: %%A
    echo ============================================
    python main.py --acquisition %%A --experiments %EXPERIMENTS% --steps %STEPS%
)

echo Generating comparison plot...
python plot_results.py

echo Done.
endlocal
