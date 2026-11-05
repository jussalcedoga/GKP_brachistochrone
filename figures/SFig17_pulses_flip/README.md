# Supplementary Fig. 17: Pulses of the flip gate

flip-gate pulses at 824 ns, representative starts and the best CD start.

- **File in the paper:** `si_figures/S27.png`
- **Make target:** `make SFig17` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig17_pulses_flip/plot.py`
- **Run time:** about 20 s on one core
- **Input data:**
  - `data/manuscript_figures/sims/results/pulses/` (`flip_target_*_T824.npz`, `closed_flip_target_*_T824.npz`)
- **Outputs:** `output/manuscript_figures/supplementary_information/S27_pulses_flip/S27.png` (and `.pdf`)
- **Numbers shown in the figure:** `output/manuscript_figures/supplementary_information/S27_pulses_flip/S27_facts.json`; stored copy `data/manuscript_figures/supplementary_information/S27_pulses_flip/S27_facts.json`, also as CSV in `data/csv/figure_facts/S27_facts.csv`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`) and the `*_facts.json` with the stored copy in `data/`; compares the printed rows of Supplementary Table 13 with the data before drawing.
