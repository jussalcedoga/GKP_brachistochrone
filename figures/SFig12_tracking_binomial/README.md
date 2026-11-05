# Supplementary Fig. 12: As Fig. 4, for the binomial-code state

the layout of Fig. 4 for (|0>+|4>)/sqrt(2); the SNAP rows use the warm-started runs.

- **File in the paper:** `si_figures/S11.png`
- **Make target:** `make SFig12` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/Fig4/plot_fig4.py` with `--target binomial --snap-warm --landscape` (shared; run this figure alone with `python figures/SFig12_tracking_binomial/plot.py`)
- **Run time:** about 65 s on one core
- **Input data:**
  - `data/manuscript/fig4_cs_path_following/results/traces80/` (Lindblad traces of every tracking run)
  - `data/manuscript/fig4_cs_path_following/results/rho80/` (oscillator states of the representative starts)
  - `.../results/wigner/binomial_geodesic_cs_k2.npz`
  - `data/manuscript_figures/sims/results/snap_warm/`
- **Outputs:** `output/manuscript_figures/supplementary_information/S11_tracking_binomial/S11_tracking_binomial.png` (and `.pdf`); the arrays of its Wigner panels in `output/manuscript_figures/supplementary_information/S11_tracking_binomial/S11_tracking_binomial_wigner.npz`
- **Numbers shown in the figure:** `output/manuscript_figures/supplementary_information/S11_tracking_binomial/S11_tracking_binomial_facts.json`; stored copy `data/manuscript_figures/supplementary_information/S11_tracking_binomial/S11_tracking_binomial_facts.json`, also as CSV in `data/csv/figure_facts/S11_tracking_binomial_facts.csv`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`) and the `*_facts.json` with the stored copy in `data/`; as Fig. 4.
