# Supplementary Fig. 1: Photon-optimal path against the brachistochrone

analytic and numerical photon-optimal paths.

- **File in the paper:** `si_figures/S02.png`
- **Make target:** `make SFig1` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig01_photon_path_analytic/plot.py`
- **Run time:** about 35 s on one core
- **Input data:**
  - `data/manuscript/fig1_brachistochrone/results/fig1.json`
  - `data/manuscript/fig2_photon_optimal/results/fig2.json`
  - `data/manuscript/fig3_time_vs_photon/results/fig3.json`
  - `data/manuscript/si/state_gallery/results/`, `.../squeezer_capacity/results/`
  - `data/manuscript_figures/analytics_audit/cache/photon_{gkp,cat}.pkl`
- **Outputs:** `output/manuscript_figures/supplementary_information/S02_photon_path_analytic/S02_photon_path_analytic.png` (and `.pdf`)
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`).
