# Supplementary Fig. 10: As Fig. 4, for Fock |2>

the layout of Fig. 4 for the Fock state |2> (a half arc), drawn by the Fig. 4 script.

- **File in the paper:** `si_figures/S09.png`
- **Make target:** `make SFig10` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/Fig4/plot_fig4.py` with `--target fock2 --landscape` (shared; run this figure alone with `python figures/SFig10_tracking_fock2/plot.py`)
- **Run time:** about 65 s on one core
- **Input data:**
  - `data/manuscript/fig4_cs_path_following/results/traces80/` (Lindblad traces of every tracking run)
  - `data/manuscript/fig4_cs_path_following/results/rho80/` (oscillator states of the representative starts)
  - `.../results/wigner/fock2_geodesic_cs_k2.npz`
- **Outputs:** `output/manuscript_figures/supplementary_information/S09_tracking_fock2/S09_tracking_fock2.png` (and `.pdf`); the arrays of its Wigner panels in `output/manuscript_figures/supplementary_information/S09_tracking_fock2/S09_tracking_fock2_wigner.npz`
- **Numbers shown in the figure:** `output/manuscript_figures/supplementary_information/S09_tracking_fock2/S09_tracking_fock2_facts.json`; stored copy `data/manuscript_figures/supplementary_information/S09_tracking_fock2/S09_tracking_fock2_facts.json`, also as CSV in `data/csv/figure_facts/S09_tracking_fock2_facts.csv`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`) and the `*_facts.json` with the stored copy in `data/`; as Fig. 4.
