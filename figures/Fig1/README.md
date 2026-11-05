# Fig. 1: Optimal paths for GKP preparation and the logical flip

(a, b) brachistochrone, photon-optimal and trade-off paths projected onto a sphere of real states coloured by the mean photon number; (c, d) Wigner functions and target fidelities at the numbered points of the brachistochrones (computed exactly from the path states).

- **File in the paper:** `figures/fig1.png`
- **Make target:** `make Fig1` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/Fig1/plot_fig1.py`
- **Run time:** about 1 min on one core
- **Input data:**
  - `data/manuscript/fig1_brachistochrone/results/fig1.json`, `fig1_wigner.npz` (grid)
  - `data/manuscript/fig2_photon_optimal/results/fig2.json`
  - `data/manuscript/fig3_time_vs_photon/results/fig3.json`
- **Outputs:** `output/manuscript_figures/fig1/fig1.png` (and `.pdf`); the arrays of its Wigner panels in `output/manuscript_figures/fig1/fig1_wigner.npz`
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`).
