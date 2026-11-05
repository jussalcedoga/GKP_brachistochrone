# Supplementary Fig. 7: Tangential capacity of squeezing

how conditional squeezing pushes the state along a path.

- **File in the paper:** `si_figures/S07.png`
- **Make target:** `make SFig7` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig07_squeezer_capacity/plot.py`
- **Run time:** about 40 s on one core
- **Input data:**
  - `data/manuscript/si/squeezer_capacity/results/squeezer_capacity.json`
  - `data/manuscript/fig2_photon_optimal/results/fig2.json`
- **Outputs:** `output/manuscript_figures/supplementary_information/S07_squeezer_capacity/S07_squeezer_capacity.png` (and `.pdf`)
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`).
