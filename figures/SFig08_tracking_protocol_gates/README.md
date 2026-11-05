# Supplementary Fig. 8: The tracking protocol

the tracking protocol for CS, CD and SNAP (GKP preparation, 824 ns).

- **File in the paper:** `si_figures/S08.png`
- **Make target:** `make SFig8` (from the repository root, after `pip install -r requirements.txt`)
- **Script:** `figures/SFig08_tracking_protocol_gates/plot.py`
- **Run time:** about 25 s on one core
- **Input data:**
  - `data/manuscript/fig4_cs_path_following/results/traces80/` (Lindblad traces of every tracking run)
- **Outputs:** `output/manuscript_figures/supplementary_information/S08_tracking_protocol_gates/S08_tracking_protocol_gates.png` (and `.pdf`)
- **Checks:** `make figures` ends with `tools/compare_figures.py`, which compares the PNG with the published file (md5 and pixel size, `figures/reference_png.json`).
