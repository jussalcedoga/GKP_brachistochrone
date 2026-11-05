# Supplementary Fig. 11: As Fig. 4, for the cat state

the layout of Fig. 4 for the even cat state of amplitude 2.

- **File in the paper:** `si_figures/S10.png`
- **Make target:** `make SFig11` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/Fig4/plot_fig4.py` with `--target cat --landscape` (shared; run this figure alone with `python figures/SFig11_tracking_cat/plot.py`)
- **Run time:** about 65 s on one core
- **Input data:**
  - `data/manuscript/fig4_cs_path_following/results/traces80/` (Lindblad traces of every tracking run)
  - `data/manuscript/fig4_cs_path_following/results/rho80/` (oscillator states of the representative starts)
  - `.../results/wigner/cat_geodesic_cs_k2.npz`
- **Outputs:** `output/manuscript_figures/supplementary_information/S10_tracking_cat/S10_tracking_cat.png` (and `.pdf`); the arrays of its Wigner panels in `output/manuscript_figures/supplementary_information/S10_tracking_cat/S10_tracking_cat_wigner.npz`
- **Numbers shown in the figure:** `output/manuscript_figures/supplementary_information/S10_tracking_cat/S10_tracking_cat_facts.json`; stored copy `data/manuscript_figures/supplementary_information/S10_tracking_cat/S10_tracking_cat_facts.json`, also as CSV in `data/csv/figure_facts/S10_tracking_cat_facts.csv`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`) and the `*_facts.json` with the stored copy in `data/`; as Fig. 4.
