# Fig. 2: Photon number, photon exposure and target fidelity along the paths

brachistochrone against photon-optimal path for preparation and for the transfer between the logical states: analytic closed form (bands), numerical optimum (dashed), lower bound m(F) (dotted).

- **File in the paper:** `figures/fig2.png`
- **Make target:** `make Fig2` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/Fig2/compute_fig2.py, then figures/Fig2/plot_fig2.py`
- **Run time:** about 40 s on one core
- **Input data:**
  - `data/manuscript/fig2_photon_optimal/results/fig2.json` (numerical optima; the closed form, m(F) and the bound are computed by `compute_fig2.py`)
- **Outputs:** `output/manuscript_figures/fig2/fig2.png` (and `.pdf`)
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`).
