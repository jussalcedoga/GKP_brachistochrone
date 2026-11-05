#!/usr/bin/env bash
# Single entry point: regenerate every figure of the Letter and of the Supplementary Information from data/, then
# check them against the published PNG files and the stored figure facts.  No long simulation is rerun.
#   bash make_figures.sh [-j N]          (default N = 4 parallel figures; about 4 minutes, about 2 minutes with -j 8)
# The two renders of the transfer at m = 1.17 and 1.40 (Supplementary Table 12) run too.
# Outputs: output/paper/figures/fig1.png ... fig4.png, output/paper/si_figures/S01.png ... (the paper's file names).
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
JOBS=4
if [ "${1:-}" = "-j" ]; then JOBS="$2"; fi
"$PYTHON" figures/make_figures.py all -j "$JOBS"
"$PYTHON" tools/compare_figures.py
