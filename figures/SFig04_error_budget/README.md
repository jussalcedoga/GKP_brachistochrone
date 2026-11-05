# Supplementary Fig. 4: Error budget of the trade-off family

first-order error along the time-photon trade-off family.

- **File in the paper:** `si_figures/S05.png`
- **Make target:** `make SFig4` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig04_error_budget/plot.py`
- **Run time:** about 50 s on one core
- **Input data:**
  - `data/manuscript/fig3_time_vs_photon/results/fig3.json`
  - `data/manuscript_figures/analytics_audit/cache/{family,photon}_{gkp,flip}.pkl`
- **Outputs:** `output/manuscript_figures/supplementary_information/S05_error_budget/S05_error_budget.png` (and `.pdf`)
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`).
