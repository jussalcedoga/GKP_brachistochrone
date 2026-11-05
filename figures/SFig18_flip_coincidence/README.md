# Supplementary Fig. 18: Photon-optimal flips that keep the qubit encoded

photon exposure of code-preserving flips around the brachistochrone.

- **File in the paper:** `si_figures/S28.png`
- **Make target:** `make SFig18` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig18_flip_coincidence/plot.py`
- **Run time:** about 35 s on one core
- **Input data:**
  - `data/manuscript_review/flip_coincidence/num_curves.npz`, `num_facts.json`
- **Outputs:** `output/manuscript_figures/supplementary_information/S28_flip_coincidence/S28_note.png` (and `.pdf`)
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`).
