# CSV copies of the result tables

Written by [`tools/export_csv.py`](../../tools/export_csv.py) from the stored JSON results in `data/`, for auditing
with any spreadsheet or language. Numbers carry 17 significant digits, so every stored double is reproduced exactly.
Nothing here is recomputed; the JSON, JSON-lines and NumPy files remain the primary record.

| file | rows | content |
|---|---|---|
| `tracking_runs.csv` | one per tracking run | `target` (gkp, fock2, cat, binomial, flip), `path` (geodesic = brachistochrone, photon, tradeoff), `gate_set` (cs, cd, snap), duration `T_ns` and its index `k`, random start `seed`, slices `N`, Fock cutoff `D` of the Lindblad evaluation, selected tracking weight `w_star`; `F_T_final` and `C` under closed evolution, `F_T_final_noisy` and `C_noisy` under the Lindblad equation (the values in the figures and tables), photon exposure `expo_ns` (closed) and `expo_L_ns` (Lindblad). The traces along $t$ are in the JSON file named in `file` (`data/manuscript/fig4_cs_path_following/results/traces80/`) |
| `target_only_runs.csv` | one per start | target-only GKP preparation: `F60` (closed, cutoff 60), `F90_closed`, `F90_noisy` (cutoff 90, Lindblad) |
| `flip_gate_runs.csv` | one per optimization | the logical flip as a gate: settings, scores at cutoff 60 (`F_T_D60[i]`, `C_D60[i]`), and the evaluation at cutoff 90 (`eval.*`) |
| `figure_facts/<figure>_facts.csv` | (key, value) | every number a figure shows, from its `*_facts.json` (nested keys joined by dots, list items by `[i]`) |

Which runs enter which figure: Fig. 4 and Table I use `target=gkp`, `path=geodesic`, `k` = 1, 2, 3 (412, 824,
2133 ns), 20 starts per gate set and duration (batches `main` and `seed`, see `src/mfcommon.py` and
`simulations/tracking/load.py`).
