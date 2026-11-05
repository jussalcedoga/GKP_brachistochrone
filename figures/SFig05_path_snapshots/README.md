# Supplementary Fig. 5: States along the three paths

Wigner functions along the three paths between the logical states.

- **File in the paper:** `si_figures/S04.png`
- **Make target:** `make SFig5` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig05_path_snapshots/plot.py`
- **Run time:** about 40 s on one core
- **Input data:**
  - `data/manuscript_figures/supplementary_information/_cache/path_*.npz` (dense reference paths)
- **Outputs:** `output/manuscript_figures/supplementary_information/S04_path_snapshots/S04_path_snapshots.png` (and `.pdf`); the arrays of its Wigner panels in `output/manuscript_figures/supplementary_information/S04_path_snapshots/S04_path_snapshots_wigner.npz`
- **Numbers shown in the figure:** `output/manuscript_figures/supplementary_information/S04_path_snapshots/S04_path_snapshots_facts.json`; stored copy `data/manuscript_figures/supplementary_information/S04_path_snapshots/S04_path_snapshots_facts.json`, also as CSV in `data/csv/figure_facts/S04_path_snapshots_facts.csv`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`) and the `*_facts.json` with the stored copy in `data/`.
