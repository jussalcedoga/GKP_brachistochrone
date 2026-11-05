# Supplementary Fig. 2: State gallery

photon number along both paths for eleven targets.

- **File in the paper:** `si_figures/S01.png`
- **Make target:** `make SFig2` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig02_state_gallery/plot.py`
- **Run time:** about 55 s on one core
- **Input data:**
  - `data/manuscript/si/state_gallery/results/state_gallery.json`, `state_gallery_wigner.npz`
- **Outputs:** `output/manuscript_figures/supplementary_information/S01_state_gallery/S01_state_gallery.png` (and `.pdf`); the arrays of its Wigner panels in `output/manuscript_figures/supplementary_information/S01_state_gallery/S01_state_gallery_wigner.npz`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`).
