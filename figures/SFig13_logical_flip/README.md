# Supplementary Fig. 13: The transfer |0_L> -> |1_L> along three paths (m = 2.81)

time-optimal, trade-off and photon-optimal paths, CS, CD and SNAP, 20 starts each.

- **File in the paper:** `si_figures/S16.png`
- **Make target:** `make SFig13` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/three_paths.py` with `--target flip --k 2 --landscape` (shared; run this figure alone with `python figures/SFig13_logical_flip/plot.py`)
- **Run time:** about 85 s on one core
- **Input data:**
  - `data/manuscript/fig4_cs_path_following/results/traces80/` (Lindblad traces of every tracking run)
  - `data/manuscript/fig4_cs_path_following/results/rho80/` (oscillator states of the representative starts)
  - `.../results/wigner/flip_*_cs_k2.npz`
  - `data/manuscript_figures/supplementary_information/_cache/path_*.npz` (dense reference paths)
- **Outputs:** `output/manuscript_figures/supplementary_information/S16_logical_flip/S16_logical_flip.png` (and `.pdf`); the arrays of its Wigner panels in `output/manuscript_figures/supplementary_information/S16_logical_flip/S16_logical_flip_wigner.npz`
- **Numbers shown in the figure:** `output/manuscript_figures/supplementary_information/S16_logical_flip/S16_logical_flip_facts.json`; stored copy `data/manuscript_figures/supplementary_information/S16_logical_flip/S16_logical_flip_facts.json`, also as CSV in `data/csv/figure_facts/S16_logical_flip_facts.csv`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`) and the `*_facts.json` with the stored copy in `data/`.
