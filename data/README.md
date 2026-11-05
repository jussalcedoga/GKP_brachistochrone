# Stored results

This folder holds the results of the long simulations and analyses that the figures, tables and numbers of the paper
are built from, about 340 MB, no file above 10 MB. [`SHA256SUMS`](SHA256SUMS) gives the SHA-256 of every file except
this README and the checksum file itself; check with `make verify` (or `python tools/verify_data.py`) from the
repository root, or with `cd data && sha256sum -c SHA256SUMS`.

The files keep their relative paths from the working tree in which they were produced, so that every script finds them
where it always did (see `src/layout.py`):

- `manuscript/` holds the results of the path computations and of the tracking runs;
- `manuscript_figures/` holds the results of the figure-specific simulations and analyses, and the numbers each figure
  shows (`*_facts.json`);
- `manuscript_review/flip_coincidence/` holds the numerics of the photon-optimal flips that keep the qubit encoded;
- `paper_quotes/` holds the printed tables of the paper (the tabular blocks only, not the manuscript) that the table and
  figure scripts compare with the data;
- `csv/` holds CSV copies of the result tables (one row per optimization run) and of the numbers each figure shows,
  written by `tools/export_csv.py` (see [`csv/README.md`](csv/README.md)).

The folder names of the working tree do not follow the figure numbers of the paper (for example
`manuscript/fig1_brachistochrone/` feeds Fig. 1, Fig. 2 and several SI figures). The top-level
[`README.md`](../README.md) maps every figure, table and number to its files.

Formats: JSON and JSON lines (`.json`, `.jsonl`), NumPy archives (`.npz`, read with `numpy.load`), and a few Python
pickles (`manuscript_figures/analytics_audit/cache/*.pkl`, read by `simulations/analysis/audit.py`; load pickles only
from a source you trust).

## Contents

| folder | files | size | content | written by | read by |
|---|---|---|---|---|---|
| `manuscript/fig1_brachistochrone/results/` | 2 | 3.3 MB | brachistochrones of the logical states and the flip, Wigner snapshots | `simulations/paths/fig1_brachistochrone.py` | Fig. 1, SFigs. 1–2, Table 3, numbers |
| `manuscript/fig2_photon_optimal/results/` | 1 | 1.6 MB | numerical photon-optimal paths | `simulations/paths/fig2_photon_optimal.py` | Figs. 1–3, SFigs. 1, 2, 7, numbers |
| `manuscript/fig3_time_vs_photon/results/` | 2 | 0.6 MB | the time–photon trade-off family, $w^*$ | `simulations/paths/fig3_time_vs_photon.py` | Fig. 1, SFigs. 1–4, numbers |
| `manuscript/si/*/results/` | 10 | 3.1 MB | state gallery, tangential capacity, aligned generator, flip route test | `simulations/paths/si_*.py` | SFigs. 1, 2, 6, 7, Table 3 |
| `manuscript/fig4_cs_path_following/results/jobs80/` | 837 | 26 MB | the optimized tracking pulses (piecewise-constant controls) of every run, with their settings and selection data | `simulations/tracking/compute.py` | `traces.py`, analyses |
| `manuscript/fig4_cs_path_following/results/traces80/` | 833 | 123 MB | closed and Lindblad traces of every tracking pulse at cutoff 90: $x$, $z$, closeness, $F_T$, $\bar n$ | `simulations/tracking/traces.py` | Fig. 4, SFigs. 8, 10–14, tables, analyses |
| `manuscript/fig4_cs_path_following/results/rho80/` | 47 | 41 MB | oscillator states at the snapshot times, for the representative starts shown in the Wigner panels (and in the $m=1.17$, $1.40$ transfer renders) only | `simulations/tracking/traces.py` | Fig. 4, SFigs. 10–14, Table 12 |
| `manuscript/fig4_cs_path_following/results/wigner/` | 16 | 69 MB | Wigner functions used by the figures and the $m=1.17$, $1.40$ renders | `simulations/tracking/wigner_snapshots.py` | Fig. 4, SFigs. 10–14, Table 12 |
| `manuscript_figures/sims/results/` (`*.jsonl`) | 3 | 2.8 MB | target-only preparation (Supplementary Table 15), the flip gate (Table 11), one-branch CS | `simulations/gate_sets/capability_noisy.py`, `flip_gate.py`, `cs_onesided.py` | numbers, Supplementary Table 15, analyses |
| `manuscript_figures/sims/results/snap_warm/` | 82 | 7.4 MB | warm-started SNAP runs | `simulations/gate_sets/snap_warm.py` | SFig. 12 |
| `manuscript_figures/sims/results/{jobs,jobs80,traces80}/` | 45 | 2.5 MB | restart test of the SNAP runs that stay at the vacuum | `simulations/gate_sets/restart_trivial.py` | – (cited in the text) |
| `manuscript_figures/sims/results/pulses/` | 75 | 8.9 MB | the 324 stored pulses as arrays, statistics, closed traces, smoothing test, checks; `README_A.md`, `README_B.md` describe them | `simulations/gate_sets/pulses_*.py`, `capability_pulses.py` | SFigs. 15–17, Table 14, numbers |
| `manuscript_figures/analytics_audit/cache/` | 13 | 0.7 MB | cached photon-optimal paths and trade-off families | `simulations/analysis/audit.py` | SFigs. 1, 2, 4 |
| `manuscript_figures/analytics_audit/photon_flip/results/` | 5 | 9.0 MB | transfers through the vacuum | `simulations/analysis/photon_flip/audit_photon*.py` | numbers, Table 2 |
| `manuscript_figures/analytics_audit/round4/results/` | 1 | 1 kB | the closed forms applied to each half of a transfer through the vacuum | `simulations/analysis/round4/transfer_closed.py` | Table 2 |
| `manuscript_figures/analytics_audit/round3/` | 10 | 0.2 MB | analyses of the stored pulses (trade-off, parity, route test, Pauli frame, flip-gate summary, …) | `simulations/analysis/round3/*.py` | SFig. 9, Tables 5, 9, 11, numbers |
| `manuscript_figures/analytics_audit/review_oct28/` | 3 | 0.6 MB | robustness and independent Lindblad cross-check | `simulations/analysis/review_oct28/*.py` | Tables 7–9, 13, numbers |
| `manuscript_figures/analytics_audit/round6/cs_proposal/` | 2 | 0.2 MB | CS with the Hamiltonian of the cited proposal | `simulations/analysis/cs_proposal/cs_proposal_hamiltonian.py` | numbers (0.996) |
| `manuscript_figures/supplementary_information/_cache/` | 9 | 5.3 MB | dense reference paths (state vectors along each path) | `figures/three_paths.py` (computed if missing) | Fig. 3, SFigs. 5, 13, 14 |
| `manuscript_figures/fig*/`, `manuscript_figures/supplementary_information/S*/` | 16 | 0.5 MB | the numbers shown in each figure (`*_facts.json`) and `fig2_extra.json`, as produced for the paper | the figure scripts | tables, numbers, `tools/compare_figures.py` |
| `manuscript_review/flip_coincidence/` | 2 | 0.6 MB | photon-optimal flips that keep the qubit encoded | `simulations/flip_note/num_core.py` | SFig. 18 |
| `paper_quotes/` | 2 | 20 kB | the printed Table I and Supplementary Tables 2, 7, 8, 9, 11, 13 and 15 (tabular blocks only) | – | Table I and SI table checks, SFigs. 15–17, pulse checks |
| `csv/` | ≈20 | ≈1 MB | CSV copies: every tracking run, every target-only and flip-gate optimization, the numbers of every figure | `tools/export_csv.py` | anyone auditing by hand |

## Large result sets (Zenodo)

The following result sets are not needed for any figure, table or number of the paper and are too large for the
repository. They will be deposited on Zenodo (`https://doi.org/10.5281/zenodo.XXXXXXX`) with the same relative paths.
[`zenodo/`](zenodo/) lists every file with its md5, path and size in bytes (three columns); after downloading, unpack
into `data/` and check with

```bash
cd data && awk '{print $1"  "$2}' zenodo/<list>.md5 | md5sum -c
```

| set | files | size | content | md5 list | regenerate with |
|---|---|---|---|---|---|
| `manuscript/fig4_cs_path_following/results/rho80/` | 833 | 681 MB | oscillator states (closed and Lindblad) at the snapshot times of every tracking run (`data/` holds the 47 the figures and the $m=1.17$, $1.40$ renders show) | [`zenodo/manuscript__fig4_cs_path_following__results__rho80.md5`](zenodo/manuscript__fig4_cs_path_following__results__rho80.md5) | `python simulations/tracking/traces.py --src jobs80 --workers 64` (all 833 runs, minutes on 64 workers) |
| `manuscript/fig4_cs_path_following/results/wigner/` | 108 | 477 MB | Wigner functions of the snapshot states of every group of runs (`data/` holds the 16 the figures and those renders use) | [`zenodo/manuscript__fig4_cs_path_following__results__wigner.md5`](zenodo/manuscript__fig4_cs_path_following__results__wigner.md5) | `python simulations/tracking/wigner_snapshots.py` (needs all of `rho80/`) |
| `manuscript/fig4_cs_path_following/results/jobs/` | 837 | 25 MB | the tracking pulses at the optimization cutoff 60, before the level-3 re-polish at cutoff 80 that produced `jobs80/` | [`zenodo/manuscript__fig4_cs_path_following__results__jobs.md5`](zenodo/manuscript__fig4_cs_path_following__results__jobs.md5) | `python simulations/tracking/compute.py <batch> --workers 64` |
| `manuscript_figures/sims/results/snap_warm/rho/` | 40 | 32 MB | snapshot states of the warm-started SNAP runs (`data/` holds the one shown) | [`zenodo/manuscript_figures__sims__results__snap_warm__rho.md5`](zenodo/manuscript_figures__sims__results__snap_warm__rho.md5) | `python simulations/gate_sets/snap_warm.py --workers 16` |
| `manuscript_figures/sims/results/rho80/` | 15 | 7.6 MB | snapshot states of the restart test | [`zenodo/manuscript_figures__sims__results__rho80.md5`](zenodo/manuscript_figures__sims__results__rho80.md5) | `python simulations/gate_sets/restart_trivial.py` |

The Zenodo record will also hold a snapshot of this repository, so that the code, the data in `data/` and the large
sets above are archived together under one DOI.
