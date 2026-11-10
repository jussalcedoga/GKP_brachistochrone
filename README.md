# Time- and photon-optimal paths for GKP states: code and data

Code and stored results for the paper

> A. Curtis, J. S. Salcedo-Gallo, K. Limanta, S. Leonov, P. Chin, S. Bedkihal, and M. Fitzpatrick,
> *Time- and photon-optimal paths for the preparation and logical flip of Gottesman–Kitaev–Preskill states*.

- **Paper (Letter):** journal link to be added (`https://doi.org/XX.XXXX/XXXXX`), preprint `https://arxiv.org/abs/XXXX.XXXXX`
- **Supplementary Information:** link to be added (published with the Letter; also in the arXiv record)
- **Archived copy of this repository and the large result files:** Zenodo, `https://doi.org/10.5281/zenodo.XXXXXXX` (see [`data/README.md`](data/README.md))

Every figure of the Letter (Figs. 1–4) and of the Supplementary Information (Supplementary Figs. 1–18), every table, and
the numbers quoted in the text can be regenerated from this repository. The stored results of the long simulations are
included in [`data/`](data/), so nothing has to be re-simulated to rebuild the figures and check the numbers. The
simulation scripts are included too, with the command lines, hardware and run times of the production runs.

## Overview

| you want to … | run | what it checks |
|---|---|---|
| confirm that the stored data are intact | `make verify` | SHA-256 of every file in `data/` against [`data/SHA256SUMS`](data/SHA256SUMS) |
| rebuild every figure | `make figures` | each PNG against the published file (md5, pixel size) and the numbers each figure shows (`*_facts.json`) against their stored copies |
| rebuild the tables and the quoted numbers | `make numbers` | every cell of Table I and of Supplementary Tables 2, 7–9, 13 and 15 against the printed cell; every key number of the Letter against the data (`PASS`/`FAIL`, exit status 1 on any failure) |
| rerun the analyses of the stored results | `make analyses` | the regenerated analysis files against the stored ones (relative 1e-9) |
| test the shared library | `make test` | the GKP states, the brachistochrone Hamiltonians, the gate sets and the GRAPE gradient against independent constructions |
| audit a number by hand | open [`data/csv/`](data/csv/) | one row per optimization run (every start, nothing excluded), in plain CSV |
| rerun the long simulations | see [Long simulations](#long-simulations-and-analyses) | the production command lines, run times and hardware |

Each figure has its own folder in [`figures/`](figures/) with its script and a `README.md` (what it shows, the
command, the input files, the checks); [`figures/README.md`](figures/README.md) is the index.

## Contents

1. [Quick start](#quick-start)
2. [Repository layout](#repository-layout)
3. [Figures](#figures)
4. [Tables](#tables)
5. [Numbers quoted in the text](#numbers-quoted-in-the-text)
6. [Long simulations and analyses](#long-simulations-and-analyses)
7. [Data](#data)
8. [Auditing a number](#auditing-a-number)
9. [Reproducibility notes](#reproducibility-notes)
10. [License and citation](#license-and-citation)

## Quick start

Requirements:
- Linux or macOS with `bash` and `make`.
- Python 3.12 (the results were produced with CPython 3.12.14) and the pinned packages of [`requirements.txt`](requirements.txt) (numpy, scipy, matplotlib, mpmath).
- For the figure text (matplotlib with `text.usetex`): LaTeX with `amsmath`, `amssymb`, `bm`, `helvet` and `sfmath`, plus `dvipng` and Ghostscript. On Debian or Ubuntu, `apt install texlive-latex-extra texlive-fonts-recommended dvipng ghostscript cm-super` provides them.

```bash
git clone https://github.com/jussalcedoga/GKP_brachistochrone.git && cd GKP_brachistochrone
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make verify       # integrity of data/ (SHA-256 of every file, a few seconds)
make figures      # every figure of the paper from data/, then the check against the paper (about 4 min with the
                  # default 4 parallel jobs; `make figures JOBS=8` about 2 min)
make numbers      # the tables and the quoted numbers, with their checks (under 1 min)
make analyses     # optional: rerun the analyses of the stored results in a copy of data/ and compare (about 5 min)
make test         # unit tests of the shared library
```

The regenerated figures land in `output/paper/figures/fig1.png` … `fig4.png` and `output/paper/si_figures/S01.png` …,
under the file names the paper uses (see the [figure table](#figures)). `make figures` ends with
`tools/compare_figures.py`, which compares each PNG with the md5 and pixel size of the published file
([`figures/reference_png.json`](figures/reference_png.json)) and each figure's numbers (`*_facts.json`) with the stored
copy in `data/`. `make figures` also renders the transfer at $m=1.17$ and $1.40$ (not paper figures), whose facts
files hold those two columns of Supplementary Table 12, so the check covers every column of that table. For a pixel
comparison with the published files, unpack the arXiv source of the paper (it holds `figures/` and `si_figures/`) and
run `python tools/compare_figures.py --ref <that folder>` (needs `pillow`).

One figure at a time: `make Fig4` or `make SFig13` (Letter Fig. N is `FigN`, Supplementary Fig. N is `SFigN`), or
`python figures/make_figures.py --list` for the commands. To run a script by hand, first `source env.sh` (it puts
`src/` and the simulation modules on the Python path), then for example `python figures/Fig1/plot_fig1.py`.

Without `make`, run `bash make_figures.sh`, `bash make_numbers.sh` and `bash make_analyses.sh`.

## Repository layout

| folder | content |
|---|---|
| [`src/common/`](src/common/) | the shared model: parameters with sources (`params.py`), GKP and other states (`states.py`), brachistochrone Hamiltonians (`brachistochrone.py`), photon-optimal paths (`paths.py`, `photon_core.py`), the gate sets and GRAPE with exact gradients (`grape.py`), the path-tracking protocol and its stage schedule (`tracking.py`), the Lindblad master equation (`lindblad.py`), Wigner functions (`wigner.py`), the figure style (`style.py`) |
| [`src/mfcommon.py`](src/mfcommon.py) | figure helpers: colours, the representative start, run grouping, exact Wigner functions on the shown window |
| [`src/layout.py`](src/layout.py) | where the stored results (`data/`) and the outputs (`output/`) live (see [Reproducibility notes](#reproducibility-notes)) |
| [`figures/`](figures/) | one folder per paper figure, named by its number in the paper (`Fig1/` … `Fig4/`, `SFig01_…/` … `SFig18_…/`), each with its script and a `README.md`; Supplementary Figs. 10–12 are drawn by `Fig4/plot_fig4.py` and Supplementary Figs. 13–14 by `three_paths.py`, and their folders hold a `plot.py` that renders exactly that figure; `make_figures.py` runs them all |
| [`tables/`](tables/) | scripts that rebuild the tables from `data/` and compare them with the printed rows |
| [`numbers/`](numbers/) | scripts that regenerate the numbers quoted in the Letter and the SI and check them |
| [`simulations/`](simulations/) | the scripts that produced the stored results: `paths/` (time- and photon-optimal paths), `tracking/` (GRAPE path tracking and its Lindblad evaluation), `gate_sets/` (target-only and flip-gate optimizations, pulse extraction, smoothing), `analysis/` (analyses of the stored pulses), `flip_note/` (photon-optimal flips that keep the qubit encoded). See [`simulations/README.md`](simulations/README.md) |
| [`tests/`](tests/) | tests of the shared library against independent constructions and of the GRAPE gradient |
| [`data/`](data/) | stored results (≈340 MB), under their original relative paths; [`data/README.md`](data/README.md) lists them and the Zenodo deposit; [`data/csv/`](data/csv/) holds CSV copies of the result tables; [`data/SHA256SUMS`](data/SHA256SUMS) the checksums |
| [`tools/`](tools/) | `verify_data.py` (data integrity), `compare_figures.py` (figures against the paper), `compare_data.py` (two data trees), `export_csv.py` (the CSV copies) |
| `output/` | everything regenerated (not tracked) |

## Figures

Run time: wall time of one figure on one core of an Intel Xeon Gold 6416H (the figures run in parallel with `-j`).
"traces80" stands for `data/manuscript/fig4_cs_path_following/results/traces80/` (the Lindblad traces of all 833
tracking runs), "rho80" for the oscillator states of the representative runs in `…/results/rho80/`, "pulses" for
`data/manuscript_figures/sims/results/pulses/`.

### Letter

| figure | file in the paper | script | input data (in `data/`) | time | command |
|---|---|---|---|---|---|
| Fig. 1 | `figures/fig1.png` | [`figures/Fig1/compute_sphere.py`](figures/Fig1/compute_sphere.py) (the common sphere of (a, b), span{\|0⟩, \|0_L⟩, \|1_L⟩}), then [`plot_fig1.py`](figures/Fig1/plot_fig1.py) | `manuscript/fig1_brachistochrone/results/fig1.json`, `fig1_wigner.npz`; `manuscript/fig2_photon_optimal/results/fig2.json`; `manuscript/fig3_time_vs_photon/results/fig3.json` | 1 min | `make Fig1` |
| Fig. 2 | `figures/fig2.png` | [`figures/Fig2/compute_fig2.py`](figures/Fig2/compute_fig2.py), then [`plot_fig2.py`](figures/Fig2/plot_fig2.py) | `manuscript/fig2_photon_optimal/results/fig2.json` (closed form, $m(F)$ and bound computed by `compute_fig2.py`) | 40 s | `make Fig2` |
| Fig. 3 | `figures/fig3.png` | [`figures/Fig3/plot_fig3.py`](figures/Fig3/plot_fig3.py) | `fig2.json`; reference paths `manuscript_figures/supplementary_information/_cache/path_*.npz` | 30 s | `make Fig3` |
| Fig. 4 (two columns) | `figures/fig4.png` | [`figures/Fig4/plot_fig4.py`](figures/Fig4/plot_fig4.py) `--wide` | traces80 (20 starts per gate set and duration); rho80 of the 3 representative starts; `…/results/wigner/gkp_geodesic_cs_k2.npz` | 70 s | `make Fig4` |

### Supplementary Information

The SI includes its figures as `si_figures/SNN.png`; the file numbers are those of the working tree and differ from the
figure numbers in the SI. The table gives both.

| figure | file in the paper | content | script | input data (in `data/`) | time | command |
|---|---|---|---|---|---|---|
| Supplementary Fig. 1 | `si_figures/S02.png` | photon-optimal path against the brachistochrone, analytic and numerical | [`figures/SFig01_photon_path_analytic/plot.py`](figures/SFig01_photon_path_analytic/plot.py) | `fig1.json`, `fig2.json`, `fig3.json`, `manuscript/si/state_gallery/results/`, `manuscript/si/squeezer_capacity/results/`, `manuscript_figures/analytics_audit/cache/photon_{gkp,cat}.pkl` | 35 s | `make SFig1` |
| Supplementary Fig. 2 | `si_figures/S01.png` | photon number along both paths for eleven targets (state gallery) | [`figures/SFig02_state_gallery/plot.py`](figures/SFig02_state_gallery/plot.py) | `manuscript/si/state_gallery/results/state_gallery.json`, `state_gallery_wigner.npz`, as above | 55 s | `make SFig2` |
| Supplementary Fig. 3 | `si_figures/S03.png` | price of saving photons (extra time against exposure saved) | [`figures/SFig03_time_photon_comparison/plot.py`](figures/SFig03_time_photon_comparison/plot.py) | `fig3.json` | 35 s | `make SFig3` |
| Supplementary Fig. 4 | `si_figures/S05.png` | first-order error along the trade-off family | [`figures/SFig04_error_budget/plot.py`](figures/SFig04_error_budget/plot.py) | `fig3.json`, `manuscript_figures/analytics_audit/cache/{family,photon}_{gkp,flip}.pkl` | 50 s | `make SFig4` |
| Supplementary Fig. 5 | `si_figures/S04.png` | Wigner functions along the three paths between the logical states | [`figures/SFig05_path_snapshots/plot.py`](figures/SFig05_path_snapshots/plot.py) | `_cache/path_flip_{geodesic,photon,tradeoff}.npz` | 40 s | `make SFig5` |
| Supplementary Fig. 6 | `si_figures/S06.png` | the periodic potential $h\cos(\sqrt\pi\hat p)$ as a generator of the flip | [`figures/SFig06_aligned_generator/plot.py`](figures/SFig06_aligned_generator/plot.py) | `manuscript/si/aligned_generator/results/aligned_generator.json` | 30 s | `make SFig6` |
| Supplementary Fig. 7 | `si_figures/S07.png` | how conditional squeezing pushes the state along a path (tangential capacity) | [`figures/SFig07_squeezer_capacity/plot.py`](figures/SFig07_squeezer_capacity/plot.py) | `manuscript/si/squeezer_capacity/results/squeezer_capacity.json`, `fig2.json` | 40 s | `make SFig7` |
| Supplementary Fig. 8 | `si_figures/S08.png` | the tracking protocol for CS, CD and SNAP (GKP, 824 ns) | [`figures/SFig08_tracking_protocol_gates/plot.py`](figures/SFig08_tracking_protocol_gates/plot.py) | traces80 | 25 s | `make SFig8` |
| Supplementary Fig. 9 | `si_figures/S23.png` | following against arriving (minimum closeness against target infidelity) | [`figures/SFig09_tradeoff/plot.py`](figures/SFig09_tradeoff/plot.py) | `manuscript_figures/analytics_audit/round3/tradeoff.json` | 25 s | `make SFig9` |
| Supplementary Fig. 10 | `si_figures/S09.png` | as Fig. 4, for Fock $\ket2$ | [`figures/Fig4/plot_fig4.py`](figures/Fig4/plot_fig4.py) `--target fock2 --landscape` | traces80, rho80, `wigner/fock2_geodesic_cs_k2.npz` | 65 s | `make SFig10` |
| Supplementary Fig. 11 | `si_figures/S10.png` | as Fig. 4, for the cat | `plot_fig4.py --target cat --landscape` | traces80, rho80, `wigner/cat_geodesic_cs_k2.npz` | 65 s | `make SFig11` |
| Supplementary Fig. 12 | `si_figures/S11.png` | as Fig. 4, for the binomial-code state (SNAP warm-started) | `plot_fig4.py --target binomial --snap-warm --landscape` | traces80, rho80, `wigner/binomial_geodesic_cs_k2.npz`, `manuscript_figures/sims/results/snap_warm/` | 65 s | `make SFig12` |
| Supplementary Fig. 13 | `si_figures/S16.png` | the transfer $\ket{0_L}\to\ket{1_L}$ along three paths, $m=2.81$ | [`figures/three_paths.py`](figures/three_paths.py) `--target flip --k 2 --landscape` | traces80, rho80 of 9 representative starts, `wigner/flip_*_cs_k2.npz`, `_cache/path_flip_*.npz` | 85 s | `make SFig13` |
| Supplementary Fig. 14 | `si_figures/S22.png` | the same at $m=7.27$ | `three_paths.py --target flip --k 3 --landscape` | as above, $k=3$ | 90 s | `make SFig14` |
| Supplementary Fig. 15 | `si_figures/S25.png` | pulses of the Fig. 4 tracking runs (824 ns) | [`figures/SFig15_pulses_tracking/plot.py`](figures/SFig15_pulses_tracking/plot.py) | pulses: `tracking_gkp_*_T824.npz`, `closed_tracking_gkp_*_T824.npz`, `pulses_stats_A_rows.json`; `manuscript_figures/fig4/fig4_facts.json` | 25 s | `make SFig15` |
| Supplementary Fig. 16 | `si_figures/S26.png` | target-only pulses of Supplementary Table 15 (412 ns) | [`figures/SFig16_pulses_target/plot.py`](figures/SFig16_pulses_target/plot.py) | pulses: `closed_target_*_T412.npz` | 20 s | `make SFig16` |
| Supplementary Fig. 17 | `si_figures/S27.png` | pulses of the flip gate (824 ns) | [`figures/SFig17_pulses_flip/plot.py`](figures/SFig17_pulses_flip/plot.py) | pulses: `flip_target_*_T824.npz`, `closed_flip_target_*_T824.npz` | 20 s | `make SFig17` |
| Supplementary Fig. 18 | `si_figures/S28.png` | photon-optimal logical flips that keep the qubit encoded | [`figures/SFig18_flip_coincidence/plot.py`](figures/SFig18_flip_coincidence/plot.py) | `manuscript_review/flip_coincidence/num_curves.npz`, `num_facts.json` | 35 s | `make SFig18` |

Every tracking figure follows the same rules (in `src/mfcommon.py`): thick lines and Wigner panels show one
representative start (the start whose Lindblad $(x,z,F)$ traces lie closest to the averages over the starts), never an
average over starts; bands show the mean ± sample s.d. over all starts; every start is included.

`plot_fig4.py` without `--wide` draws the one-column variant of Fig. 4 (same data and numbers), with `--landscape`
the layout of Supplementary Figs. 10–12. Three scripts compare printed table rows with the data before drawing
(Supplementary Figs. 15–17). They read those rows
from `data/paper_quotes/` (the printed tables only, not the manuscript), or from the paper sources if `GKP_PAPER_TEX`
names their folder.

## Tables

| table | content | script (command) | data (in `data/`) | time |
|---|---|---|---|---|
| Letter Table I | errors and closeness of the Fig. 4 runs: $\epsilon_1$ of the path, $1-F_T$ and minimum closeness $C$ of CS, CD and SNAP at 412, 824 and 2133 ns, mean (s.d.) over 20 starts | [`tables/letter_table1_tracking.py`](tables/letter_table1_tracking.py) (checks every run's stored $C$ and $F_T$ against its own traces, then every printed cell; also writes the unrounded statistics as CSV) | traces80, `manuscript_figures/fig4/fig4_facts.json` | 5 s |
| Supplementary Table 1 | target fidelity with the normal-ordered $H_{\rm opt}$ truncated at order $K$ (60-digit arithmetic) | [`tables/si_table1_truncation_order.py`](tables/si_table1_truncation_order.py) | none (computed) | 10 s on 32 cores |
| Supplementary Table 2 | photon-optimal preparations and transfers through the vacuum | [`tables/si_tables_2_7_8_9_13.py`](tables/si_tables_2_7_8_9_13.py) (`python tables/si_tables_2_7_8_9_13.py`); inputs from [`simulations/analysis/photon_flip/audit_photon2.py`](simulations/analysis/photon_flip/audit_photon2.py) (100 s), [`simulations/analysis/round4/transfer_closed.py`](simulations/analysis/round4/transfer_closed.py) (closed-form columns) and `figures/Fig2/compute_fig2.py` | `manuscript_figures/analytics_audit/photon_flip/results/audit_photon2.json`, `…/round4/results/transfer_closed.json`, `manuscript_figures/fig2/results/fig2_extra.json`, `fig2.json` | 2 s |
| Supplementary Table 3 | state gallery: Bures angle, exposure and extra time per target | [`tables/si_table3_state_gallery.py`](tables/si_table3_state_gallery.py) | `manuscript/si/state_gallery/results/state_gallery.json` | 5 s |
| Supplementary Table 4 | physical parameters | the values are defined, with their sources, in [`src/common/params.py`](src/common/params.py) | – | – |
| Supplementary Table 5 | Hamiltonians that carry $\ket{0_L}$ along a path, applied to $\ket{1_L}$ | [`simulations/analysis/round3/route_test.py`](simulations/analysis/round3/route_test.py) | writes `manuscript_figures/analytics_audit/round3/route_test.json` and `.md` | 10 s |
| Supplementary Table 6 | stage schedule of the tracking optimization | `SCHEDULE` and `LADDER` in [`src/common/tracking.py`](src/common/tracking.py) | – | – |
| Supplementary Tables 7, 8, 13 | final states of GKP preparation (stabilizers), energy uncertainty and photon number of the pulses, transfer pulses applied to the code space | [`tables/si_tables_2_7_8_9_13.py`](tables/si_tables_2_7_8_9_13.py); inputs from [`simulations/analysis/review_oct28/robustness.py`](simulations/analysis/review_oct28/robustness.py) (items 2, 4 and 1) | `manuscript_figures/analytics_audit/review_oct28/robustness.json` | 2 s (the inputs: long, see below) |
| Supplementary Table 9 | stored pulses under closed evolution and three noise models | [`tables/si_tables_2_7_8_9_13.py`](tables/si_tables_2_7_8_9_13.py); inputs from `robustness.py` (item 6) and [`simulations/analysis/round3/rates_parity.py`](simulations/analysis/round3/rates_parity.py) (Ramsey-derived dephasing, 2 min) | `robustness.json`, `round3/rates_parity.json` | 2 s |
| Supplementary Tables 10 and 12 | preparation by the three gate sets; the transfer along three paths at four durations | [`tables/si_tables_10_12.py`](tables/si_tables_10_12.py) (writes `output/manuscript_figures/review_round/tab_prep.tex`, `tab_flip.tex`) | the `*_facts.json` of Fig. 4, Supplementary Figs. 10–14 and the $m=1.17$, $1.40$ renders (all regenerated by `make figures` and compared with these stored copies) | 1 s |
| Supplementary Table 11 | the logical flip as a gate | [`simulations/analysis/round3/flip_gate_summary.py`](simulations/analysis/round3/flip_gate_summary.py) | `manuscript_figures/sims/results/flip_gate.jsonl` → `round3/flip_gate_summary.json` | 3 s |
| Supplementary Table 14 | statistics of the optimized pulses | [`numbers/pulses_text_facts.py`](numbers/pulses_text_facts.py) (key `table` of `output/numbers/pulses_text_facts.json`) | pulses | 1 s |
| Supplementary Table 15 | target-only GKP preparation, mean (s.d.) [best] of 8 starts | [`tables/target_only_preparation.py`](tables/target_only_preparation.py) (every printed cell checked); also checked in `numbers/letter_numbers.py` | `manuscript_figures/sims/results/capability_noisy.jsonl` | 1 s |

`make numbers` runs the table scripts that take seconds. `tables/si_tables_2_7_8_9_13.py` writes every cell of
Supplementary Tables 2, 7, 8, 9 and 13 from the data in the notation and at the precision of the printed cell (mean,
s.d. in parentheses in units of the last digit, best start in brackets) and compares it with the printed row
(`PASS`/`FAIL` per row, rows in `output/tables/si_tables_2_7_8_9_13.tex`). The printed rows are read from
[`data/paper_quotes/`](data/paper_quotes/) (the tabular blocks only, not the manuscript), or from the paper sources if
`GKP_PAPER_TEX` names their folder. The analyses that write into `data/` (Tables 5, 11, the closed-form columns of
Table 2, the inputs of Supplementary Figs. 9 and 18, and others) are rerun by `make analyses` in a copy of `data/`, and
`tools/compare_data.py` compares the regenerated files with the stored ones (numbers to a relative 1e-9; run times and
recorded script names are ignored).

## Numbers quoted in the text

| numbers | script | data | check |
|---|---|---|---|
| every key number of the Letter: the speed limit, $T^*$ = 140, 237 and 294 ns, $\langle0_L\|1_L\rangle$, $\bar n$ of the logical states, the target-only fidelities, the Fig. 1 weight $\geq0.987$, the photon savings (7% in time for 8% in exposure, at most 26%, the transfer at 294, 350 and 397 ns), the capacity $0.51\omega$, the flip-gate fidelities, the parity-detectable share of the error, the displacement-rate bound of the experiments, the 15–35% exposure saving of CS, … | [`numbers/letter_numbers.py`](numbers/letter_numbers.py) | the stored results listed in the script, and an independent GKP construction written in the script (analytic benchmarks: the overlaps, photon numbers and Bures angles of the finite-energy GKP states, the brachistochrone durations) | each number is recomputed and tested against the printed value (`PASS`/`FAIL`, exit status 1 on any `FAIL`). With `GKP_PAPER_TEX=<folder with main.tex and supplement.tex>` it also reports (`MISS`, not a failure) any quoted phrase that the final text words differently |
| the reductions of $\epsilon_1$ of the transfer (22–26% without, about 40% with cavity dephasing), with the photon-optimal transfer rebuilt from its stored nodes; the photon-optimal time of $\ket{1_L}$ at $\Delta=0.3$ (printed for information) | [`numbers/letter_numbers2.py`](numbers/letter_numbers2.py) | `paths.npz`, `audit_photon2.json`, `fig3.json`, `state_gallery.json` | `PASS`/`FAIL` against the printed values |
| the numbers of SI Sec. X (optimized pulses) and Supplementary Table 14 | [`numbers/pulses_text_facts.py`](numbers/pulses_text_facts.py) | pulses, `S2{5,6,7}_facts.json` | 143 assertions against the raw arrays |
| the numbers each figure shows (fidelities, closeness, representative starts, counts of starts) | each figure script writes `*_facts.json` | as for the figure | `tools/compare_figures.py` compares them with the stored copies |

## Long simulations and analyses

The stored results come from the scripts in [`simulations/`](simulations/). They ran on one compute server with two
Intel Xeon Gold 6416H processors (72 hardware threads, 1 TB RAM, Ubuntu), one thread per worker process
(`OMP_NUM_THREADS=1`). Rerunning them is not needed for the figures or the numbers. Each script writes its results under
`data/` at the path its readers expect, so a rerun overwrites the stored file; to keep `data/` intact, copy it and point
`GKP_DATA` at the copy, for example

```bash
cp -r data /scratch/gkp_data && export GKP_DATA=/scratch/gkp_data
python simulations/tracking/compute.py seeds4 --workers 64      # rerun one batch of tracking runs
python tools/compare_data.py data "$GKP_DATA" --only manuscript/fig4_cs_path_following/results/jobs80
```

| stored result (in `data/`) | what | script and production command | run time of the production run |
|---|---|---|---|
| `manuscript/fig4_cs_path_following/results/jobs80/*.json` | the optimized tracking pulses (GRAPE with exact gradients, L-BFGS-B), 837 runs | [`simulations/tracking/compute.py`](simulations/tracking/compute.py): `all --workers 64` (grid jobs, sets `main`/`si`/`val`) then `level3 --workers 64`; the random-start batches `seeds`, `seeds2`, `seeds3`, `seeds4`, `snap` (Fig. 4, Supplementary Figs. 8–14), each followed by `level3 --only seeds --guard 0.55 --workers 64` (re-polish at cutoff 80); see [`simulations/README.md`](simulations/README.md) | grid jobs 94 runs about 2.5 h; `seeds` 272 runs 3.5 h, `seeds2` 68 runs 1.0 h, `seeds3` 90 runs 1.3 h, `seeds4` 96 runs 1.0 h, `snap` 211 runs 2.6 h, each on 64 workers; `level3` 3–10 min per batch |
| `…/results/traces80/*.json`, `…/results/rho80/*.npz` | Lindblad evaluation of every pulse at cutoff 90 (device rates), traces and snapshot states | [`simulations/tracking/traces.py`](simulations/tracking/traces.py) `--src jobs80 --workers 64` | 1–2 min per batch on 64 workers |
| `…/results/wigner/*.npz` | Wigner functions of the snapshot states | [`simulations/tracking/wigner_snapshots.py`](simulations/tracking/wigner_snapshots.py) (needs all of `rho80/`, on Zenodo) | minutes |
| `manuscript_figures/sims/results/capability_noisy.jsonl` | target-only GKP preparation, Table I | [`simulations/gate_sets/capability_noisy.py`](simulations/gate_sets/capability_noisy.py) | 32 starts, 13 ± 2 min per start on one core |
| `manuscript_figures/sims/results/flip_gate.jsonl` | the logical flip as a gate (Supplementary Table 11) | [`simulations/gate_sets/flip_gate.py`](simulations/gate_sets/flip_gate.py) `--workers 60` | 96 cases, 20–100 min per case |
| `manuscript_figures/sims/results/snap_warm/` | warm-started SNAP runs (Supplementary Fig. 12) | [`simulations/gate_sets/snap_warm.py`](simulations/gate_sets/snap_warm.py) `--workers 16` | 40 cases, 15–30 min each |
| `manuscript_figures/sims/results/cs_onesided.jsonl` | CS with one branch squeezed | [`simulations/gate_sets/cs_onesided.py`](simulations/gate_sets/cs_onesided.py) `--workers 16` | 40 cases, 15–40 min each |
| `manuscript_figures/analytics_audit/round6/cs_proposal/` | CS with the Hamiltonian of the cited proposal (Kerr, idle-branch rotation; the 0.996 of SI Sec. VIII) | [`simulations/analysis/cs_proposal/cs_proposal_hamiltonian.py`](simulations/analysis/cs_proposal/cs_proposal_hamiltonian.py) `--workers 16` | not recorded |
| `manuscript_figures/sims/results/pulses/` | the 324 stored pulses as arrays, their statistics, closed traces, smoothing test, independent checks (SI Sec. X) | [`simulations/gate_sets/`](simulations/gate_sets/): `capability_pulses.py --workers 32`, `pulses_extract.py`, `pulses_reproduce.py --workers 28`, `pulses_stats.py --workers 28`, `pulses_closed_target.py`, `pulses_smooth.py --workers 32`, `pulses_facts.py`, `pulses_check_B.py --workers 16`, `pulses_facts_B.py`, `pulses_fig_check.py` | `pulses_smooth.py` 3.3 h on 32 workers; `pulses_reproduce.py` 4 min on 24 workers; the others minutes |
| `manuscript_figures/analytics_audit/review_oct28/robustness.json`, `lindblad_crosscheck.json` | gates vs transfers, stabilizers, resources, noise budget, CS-proposal rates (Supplementary Tables 7–9, 13), with an independent Lindblad integrator | [`simulations/analysis/review_oct28/robustness.py`](simulations/analysis/review_oct28/robustness.py) `--workers 30` (then `--item6`), [`lindblad_crosscheck.py`](simulations/analysis/review_oct28/lindblad_crosscheck.py) `--workers 12` | not recorded (long: Lindblad runs of every stored flip and GKP pulse) |
| `manuscript/fig1_brachistochrone/results/`, `fig2_photon_optimal/results/`, `fig3_time_vs_photon/results/`, `manuscript/si/*/results/` | brachistochrones, photon-optimal paths, the trade-off family, the state gallery, the tangential capacity, the aligned generator | [`simulations/paths/`](simulations/paths/) (`fig1_brachistochrone.py`, `fig2_photon_optimal.py`, `fig3_time_vs_photon.py`, `si_state_gallery.py`, `si_squeezer_capacity.py`, `si_aligned_generator.py`) | minutes each |
| `manuscript_figures/analytics_audit/cache/*.pkl`, `photon_flip/results/` | cached photon-optimal paths and the analytic checks; transfers through the vacuum | [`simulations/analysis/audit.py`](simulations/analysis/audit.py) (about 10 min, one core), [`photon_flip/audit_photon.py`, `audit_photon2.py`](simulations/analysis/photon_flip/) | minutes (`audit_photon2.py` 100 s) |
| `manuscript_figures/analytics_audit/round4/results/transfer_closed.json` | the closed form and its one-mode form applied to each half of a transfer through the vacuum (Supplementary Table 2) | [`simulations/analysis/round4/transfer_closed.py`](simulations/analysis/round4/transfer_closed.py) | 6 s (`make analyses`) |
| `manuscript_figures/analytics_audit/round3/*.json` | analyses of the stored pulses (trade-off, parity, route test, Pauli frame, flip-gate summary, stabilizers along the paths, code-space averages, half-rate CS) | [`simulations/analysis/round3/`](simulations/analysis/round3/) | seconds to 2 min each (`make analyses`) |
| `manuscript_review/flip_coincidence/` | photon-optimal flips that keep the qubit encoded (Supplementary Fig. 18) | [`simulations/flip_note/num_core.py`](simulations/flip_note/num_core.py) | about 1 min |

Quick checks of the simulation pipeline: `python simulations/tracking/compute.py <batch> --smoke` (3 iterations per
stage, into `results/smoke/`), `python simulations/gate_sets/flip_gate.py --smoke`,
`python simulations/gate_sets/capability_pulses.py --smoke`. Point `GKP_DATA` at a scratch folder for these.

## Data

[`data/README.md`](data/README.md) lists the stored results, their formats, producers and readers, and describes the
Zenodo deposit of the large result sets that no figure or number needs (the oscillator states and Wigner functions of
all tracking runs and the tracking pulses at the optimization cutoff 60, 1.2 GB).

- **Formats.** Open formats only: JSON and JSON lines for results and traces, NumPy `.npz` archives (zip files of `.npy`
  arrays) for states, Wigner functions and pulses, CSV for the tables, and a few Python pickles of cached paths (see
  `data/README.md`).
- **CSV copies.** [`data/csv/`](data/csv/) holds every optimization run as one CSV row (`tracking_runs.csv`, 833 tracking
  runs; `target_only_runs.csv`; `flip_gate_runs.csv`) and every number each figure shows (`figure_facts/`), with 17
  significant digits, for auditing without Python (`make csv` rewrites them from the JSON files).
- **Integrity.** [`data/SHA256SUMS`](data/SHA256SUMS) gives the SHA-256 of every file in `data/` except `README.md` and
  itself; check with `make verify` (`python tools/verify_data.py`, no third-party packages) or
  `cd data && sha256sum -c SHA256SUMS`.

## Auditing a number

Every printed number leads to raw results in a few steps. For example, the minimum closeness $C=0.90(1)$ of CS at
824 ns in Table I:

1. `python tables/letter_table1_tracking.py` recomputes the cell (mean and sample s.d. over 20 starts) and compares it
   with the printed table; the unrounded statistics are written to `output/manuscript_figures/review_round/table1_tracking.csv`.
2. The 20 starts are the rows of [`data/csv/tracking_runs.csv`](data/csv/tracking_runs.csv) with `target=gkp`,
   `path=geodesic`, `gate_set=cs`, `k=2` (the column `C_noisy`).
3. Each row names its JSON file in `data/manuscript/fig4_cs_path_following/results/traces80/`, which holds the closeness
   trace `FL` along $t$ (the stored `C_noisy` is its minimum, which the table script checks for every run), and the
   optimized pulse is in `jobs80/` under the same name.
4. `python simulations/tracking/traces.py` re-evaluates the stored pulses with the Lindblad equation (see
   [Long simulations](#long-simulations-and-analyses)), and `python simulations/tracking/compute.py` reruns the
   optimization itself.

## Reproducibility notes

- **Paths.** The scripts were written in the working tree where the results were produced, and each one found its
  inputs relative to its own location. Here the code is grouped by role and the stored results keep their original
  relative paths under `data/` (`data/manuscript/…`, `data/manuscript_figures/…`, `data/manuscript_review/…`). One call,
  `layout.vfile(__file__)`, gives every script its original location, mapped into `data/` for the simulations and
  analyses and into `output/` for the figure, number and table scripts. Only paths, imports and docstrings were changed
  (docstrings: repository paths, and no notes on which review round or work session wrote a script). The numerics of
  every script are those that produced the paper. The exceptions are disclosed: `figures/Fig2/plot_fig2.py` creates
  its output folder, and the check scripts in `numbers/` follow the wording of the final text, read the stored Fig. 1
  weight, check the 15–35% exposure saving at all four durations and the $\epsilon_1$ reductions, and exit with status 1
  on a failed check.
- **"deck".** Some docstrings, file names and keys (`validate_deck_*.json`, `si/deck_crosscheck/`, the noise set
  `"deck"`, the `val` runs) refer to the unpublished protocol notes (a slide deck) in which the tracking protocol was
  first written down. The protocol as used is specified in SI Sec. VII and Supplementary Table 6, and in
  `src/common/tracking.py`. The `val` runs and `si/deck_crosscheck/` reproduce the settings of those notes as a
  cross-check only.
- **Determinism.** On the production machine, `make figures` reproduces 20 of the 22 PNG files of the paper byte for
  byte (`tools/compare_figures.py` reports `IDENTICAL`). Supplementary Fig. 10 differs from the published file by at
  most 2/255 in any pixel value. Supplementary Fig. 18 differs in about 2600 of its 12 million pixels (339 by more than
  16/255), all in the anti-aliasing of the $\bar n_{\mathcal C}$ and $\bar n_{\min}$ labels; the script and data are
  identical and reruns are deterministic, so the difference comes from the TeX glyph rendering. On other machines, the
  fonts, the LaTeX version and the floating-point library can change the PNG bytes, but not the numbers in the
  `*_facts.json` files, which are compared separately.
- **Software of the pulse extraction.** The scripts that extracted the 324 stored pulses into arrays (`simulations/gate_sets/pulses_extract.py`, `pulses_reproduce.py`, `pulses_stats.py`, `pulses_facts.py`) ran with Python 3.14.4, numpy 2.3.5 and scipy 1.16.3; `pulses_reproduce.py` reproduces every stored fidelity and closeness from the extracted arrays to 6e-14.
- **Statistics.** Mean ± sample s.d. (ddof = 1) over all random starts, every start included (including SNAP starts that
  stay at the vacuum, which are counted in the captions).
- **Environment variables.** `GKP_DATA` (stored results, default `data/`), `GKP_OUT` (outputs, default `output/`),
  `GKP_PAPER_TEX` (optional folder with the paper sources for the wording checks).
- **Units.** Times in ns, angular rates in µs⁻¹ (rad/µs), exposures in photon ns.

## License and citation

The code is released under the MIT License ([`LICENSE`](LICENSE)). The stored results in `data/` are released under the
same terms. If you use the code or the data, please cite the paper (see [`CITATION.cff`](CITATION.cff)); the citation
will be completed with the journal reference when it is available. Each release of this repository that accompanies a
version of the paper is tagged (for example `v1.0`) and archived on Zenodo with a DOI ([`.zenodo.json`](.zenodo.json)
holds the deposit metadata).

Questions and bug reports: please open an issue in this repository.
