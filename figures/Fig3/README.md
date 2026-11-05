# Fig. 3: Path states halfway and the path-following schematic

(a) Wigner functions halfway along the time- and photon-optimal paths, for preparation (left pair) and for |0_L> -> |1_L> (right pair, photon-optimal path at the vacuum); (b) schematic of the path-following objective.

- **File in the paper:** `figures/fig3.png`
- **Make target:** `make Fig3` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/Fig3/plot_fig3.py`
- **Run time:** about 30 s on one core
- **Input data:**
  - `data/manuscript/fig2_photon_optimal/results/fig2.json`
  - `data/manuscript_figures/supplementary_information/_cache/path_*.npz` (dense reference paths)
- **Outputs:** `output/manuscript_figures/fig3/fig3.png` (and `.pdf`); the arrays of its Wigner panels in `output/manuscript_figures/fig3/fig3_wigner.npz`
- **Numbers shown in the figure:** `output/manuscript_figures/fig3/fig3_facts.json`; stored copy `data/manuscript_figures/fig3/fig3_facts.json`, also as CSV in `data/csv/figure_facts/fig3_facts.csv`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`) and the `*_facts.json` with the stored copy in `data/`.
