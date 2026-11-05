# Supplementary Fig. 15: Pulses of the Fig. 4 runs

controls of one representative start per gate set at 824 ns.

- **File in the paper:** `si_figures/S25.png`
- **Make target:** `make SFig15` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig15_pulses_tracking/plot.py`
- **Run time:** about 25 s on one core
- **Input data:**
  - `data/manuscript_figures/sims/results/pulses/` (`tracking_gkp_*_T824.npz`, `closed_tracking_gkp_*_T824.npz`, `pulses_stats_A_rows.json`)
  - `data/manuscript_figures/fig4/fig4_facts.json`
- **Outputs:** `output/manuscript_figures/supplementary_information/S25_pulses_tracking/S25.png` (and `.pdf`)
- **Numbers shown in the figure:** `output/manuscript_figures/supplementary_information/S25_pulses_tracking/S25_facts.json`; stored copy `data/manuscript_figures/supplementary_information/S25_pulses_tracking/S25_facts.json`, also as CSV in `data/csv/figure_facts/S25_facts.csv`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`) and the `*_facts.json` with the stored copy in `data/`; compares the printed rows of Supplementary Table 9 with the data before drawing.
