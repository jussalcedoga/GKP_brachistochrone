# Supplementary Fig. 14: The same at m = 7.27

as Supplementary Fig. 13, at 7.27 times the minimal time of each path.

- **File in the paper:** `si_figures/S22.png`
- **Make target:** `make SFig14` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/three_paths.py` with `--target flip --k 3 --landscape` (shared; run this figure alone with `python figures/SFig14_logical_flip_m727/plot.py`)
- **Run time:** about 90 s on one core
- **Input data:**
  - `data/manuscript/fig4_cs_path_following/results/traces80/` (Lindblad traces of every tracking run)
  - `data/manuscript/fig4_cs_path_following/results/rho80/` (oscillator states of the representative starts)
  - `data/manuscript_figures/supplementary_information/_cache/path_*.npz` (dense reference paths)
- **Outputs:** `output/manuscript_figures/supplementary_information/S20_S22_flip_durations/S22_flip_m727.png` (and `.pdf`); the arrays of its Wigner panels in `output/manuscript_figures/supplementary_information/S20_S22_flip_durations/S22_flip_m727_wigner.npz`
- **Numbers shown in the figure:** `output/manuscript_figures/supplementary_information/S20_S22_flip_durations/S22_flip_m727_facts.json`; stored copy `data/manuscript_figures/supplementary_information/S20_S22_flip_durations/S22_flip_m727_facts.json`, also as CSV in `data/csv/figure_facts/S22_flip_m727_facts.csv`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`) and the `*_facts.json` with the stored copy in `data/`.
