# Supplementary Fig. 6: Aligned generator of the flip

the periodic potential h cos(sqrt(pi) p) as a generator of the flip.

- **File in the paper:** `si_figures/S06.png`
- **Make target:** `make SFig6` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig06_aligned_generator/plot.py`
- **Run time:** about 30 s on one core
- **Input data:**
  - `data/manuscript/si/aligned_generator/results/aligned_generator.json`
- **Outputs:** `output/manuscript_figures/supplementary_information/S06_aligned_generator/S06_aligned_generator.png` (and `.pdf`)
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`).
