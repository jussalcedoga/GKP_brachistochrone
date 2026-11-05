# Supplementary Fig. 3: The price of saving photons

extra time against the photon exposure saved.

- **File in the paper:** `si_figures/S03.png`
- **Make target:** `make SFig3` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig03_time_photon_comparison/plot.py`
- **Run time:** about 35 s on one core
- **Input data:**
  - `data/manuscript/fig3_time_vs_photon/results/fig3.json`
- **Outputs:** `output/manuscript_figures/supplementary_information/S03_time_photon_comparison/S03_time_photon_comparison.png` (and `.pdf`)
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`).
