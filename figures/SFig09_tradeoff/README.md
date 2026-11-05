# Supplementary Fig. 9: Following against arriving

minimum closeness against target infidelity over the tracking weights.

- **File in the paper:** `si_figures/S23.png`
- **Make target:** `make SFig9` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig09_tradeoff/plot.py`
- **Run time:** about 25 s on one core
- **Input data:**
  - `data/manuscript_figures/analytics_audit/round3/tradeoff.json`
- **Outputs:** `output/manuscript_figures/supplementary_information/S23_tradeoff/S23_tradeoff.png` (and `.pdf`)
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`).
