# Supplementary Fig. 16: Target-only pulses

pulses of the target-only GKP preparation of Supplementary Table 15 at 412 ns.

- **File in the paper:** `si_figures/S26.png`
- **Make target:** `make SFig16` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig16_pulses_target/plot.py`
- **Run time:** about 20 s on one core
- **Input data:**
  - `data/manuscript_figures/sims/results/pulses/` (`closed_target_*_T412.npz`)
- **Outputs:** `output/manuscript_figures/supplementary_information/S26_pulses_target/S26.png` (and `.pdf`)
- **Numbers shown in the figure:** `output/manuscript_figures/supplementary_information/S26_pulses_target/S26_facts.json`; stored copy `data/manuscript_figures/supplementary_information/S26_pulses_target/S26_facts.json`, also as CSV in `data/csv/figure_facts/S26_facts.csv`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`) and the `*_facts.json` with the stored copy in `data/`; compares the printed rows of Supplementary Table 15 with the data before drawing.
