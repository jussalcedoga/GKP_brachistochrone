# Fig. 4: Following the GKP brachistochrone with CS, CD and SNAP

two-column figure: (a) state in the plane of |0> and |chi>, (b) closeness F, (c) target fidelity F_T at 412, 824 and 2133 ns, 20 starts per gate set (thin: every start, thick: the representative start, bands: mean +- s.d.); (d) Wigner functions and closeness of the path and of each gate set at the numbered times. The same runs give Table I (`tables/letter_table1_tracking.py`).

- **File in the paper:** `figures/fig4.png`
- **Make target:** `make Fig4` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/Fig4/plot_fig4.py` with `--wide`
- **Run time:** about 70 s on one core
- **Input data:**
  - `data/manuscript/fig4_cs_path_following/results/traces80/` (Lindblad traces of every tracking run)
  - `data/manuscript/fig4_cs_path_following/results/rho80/` (oscillator states of the representative starts)
  - `.../results/wigner/gkp_geodesic_cs_k2.npz` (grid)
- **Outputs:** `output/manuscript_figures/fig4/fig4.png` (and `.pdf`); the arrays of its Wigner panels in `output/manuscript_figures/fig4/fig4_wigner.npz`
- **Numbers shown in the figure:** `output/manuscript_figures/fig4/fig4_facts.json`; stored copy `data/manuscript_figures/fig4/fig4_facts.json`, also as CSV in `data/csv/figure_facts/fig4_facts.csv`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`) and the `*_facts.json` with the stored copy in `data/`; the script asserts that every Wigner panel shows the plotted run (closeness recomputed from the state within 2e-3 of the trace).
