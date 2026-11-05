# Simulations and analyses

These scripts produced the stored results in [`../data/`](../data/). Rerunning them is not needed to rebuild the figures
or check the numbers. Run them from the repository root after `source env.sh`. Each one writes its results under
`$GKP_DATA` (default `data/`) at the path its readers expect, so point `GKP_DATA` at a copy of `data/` to keep the
stored files, and compare afterwards with `python tools/compare_data.py data "$GKP_DATA"`.

Production hardware: one server with two Intel Xeon Gold 6416H processors (72 hardware threads, 1 TB RAM), one thread
per worker process (`OMP_NUM_THREADS=1`), CPython 3.12.14 with the packages of `requirements.txt` (the pulse extraction
of `gate_sets/pulses_extract.py`, `pulses_reproduce.py`, `pulses_stats.py` and `pulses_facts.py` ran with Python 3.14.4,
numpy 2.3.5 and scipy 1.16.3, and `pulses_reproduce.py` shows that the stored numbers do not depend on this, to 6e-14).

## Model and protocol

All simulations share `src/common/`:
- the gate sets of the Letter, Eq. (8): conditional squeezing (CS), echoed conditional displacement (ECD) and
  SNAP-displacement, with the bounds $|\Omega|\le10g$ and branch amplitudes $\le g$, $g=7.5~\mu\mathrm{s}^{-1}$
  (`params.py`, `grape.py`);
- piecewise-constant controls on $N=50$ slices (100 at 2133 ns), exact gradients and L-BFGS-B (`grape.py`);
- the tracking objective and its stage schedule (`tracking.py`, Supplementary Table 6);
- the Lindblad master equation with the device rates (`lindblad.py`, `params.NOISE`);
- Fock cutoffs: 60 for the optimization, 80 for the level-3 re-polish, 90 for every reported evaluation.

## Tracking runs (Fig. 4, Supplementary Figs. 8, 10–14, Tables 7–10, 12, 13)

`tracking/compute.py` optimizes the tracking pulses, `tracking/traces.py` evaluates them (closed and Lindblad, cutoff
90), `tracking/wigner_snapshots.py` computes Wigner functions of the snapshot states, `tracking/load.py` groups the runs.
The production sequence was

```bash
python simulations/tracking/compute.py all --workers 64          # grid jobs (sets main, si, val), 94 jobs, about 2.5 h
python simulations/tracking/compute.py level3 --workers 64       # re-polish every chosen pulse at cutoff 80
for b in seeds seeds2 seeds3 seeds4 snap; do                      # random starts (20 per gate set for Fig. 4)
  python simulations/tracking/compute.py $b --workers 64
  python simulations/tracking/compute.py level3 --only seeds --guard 0.55 --workers 64
  python simulations/tracking/traces.py --src jobs80 --workers 64
done
python simulations/tracking/wigner_snapshots.py
```

| batch | runs | wall time on 64 workers |
|---|---|---|
| `seeds` | 272 | 3.5 h (`level3` 10 min) |
| `seeds2` | 68 | 1.0 h (`level3` 3 min) |
| `seeds3` | 90 | 1.3 h (`level3` 4 min) |
| `seeds4` | 96 | 1.0 h (`level3` 4 min) |
| `snap` | 211 | 2.6 h (`level3` 10 min) |
| `traces.py` | 90–362 per batch | 1–2 min per batch |

`compute.py <batch> --smoke` runs the whole pipeline with 3 iterations per stage into `results/smoke/`.

## Gate-set optimizations and pulses (Table I, Supplementary Tables 11, 14, Figs. 15–17)

| script | result | production command and time |
|---|---|---|
| `gate_sets/capability_noisy.py` | target-only GKP preparation (Letter Table I) | 32 starts, 13 ± 2 min per start on one core |
| `gate_sets/capability_pulses.py` | the same with the pulses stored | `--workers 32` |
| `gate_sets/flip_gate.py` | the logical flip as a gate | `--workers 60`, 96 cases of 20–100 min |
| `gate_sets/snap_warm.py` | warm-started SNAP runs | `--workers 16`, 40 cases of 15–30 min |
| `gate_sets/cs_onesided.py` | CS with one branch squeezed | `--workers 16`, 40 cases of 15–40 min |
| `gate_sets/restart_trivial.py` | restart test of the SNAP runs that stay at the vacuum | minutes |
| `gate_sets/pulses_extract.py`, `pulses_reproduce.py --workers 28`, `pulses_stats.py --workers 28`, `pulses_closed_target.py`, `pulses_facts.py` | the 324 pulses as arrays, reproduced to $\le2\times10^{-13}$, their statistics and closed traces | minutes each (`pulses_reproduce.py` 4 min on 24 workers) |
| `gate_sets/pulses_smooth.py --workers 32`, `pulses_check_B.py --workers 16`, `pulses_facts_B.py`, `pulses_fig_check.py` | Gaussian smoothing test and independent checks | `pulses_smooth.py` 3.3 h on 32 workers, the others minutes |

## Paths (Figs. 1–3, Supplementary Figs. 1–7)

`paths/fig1_brachistochrone.py`, `paths/fig2_photon_optimal.py`, `paths/fig3_time_vs_photon.py`,
`paths/si_state_gallery.py`, `paths/si_squeezer_capacity.py`, `paths/si_aligned_generator.py` compute the brachistochrones,
the photon-optimal paths, the trade-off family, the state gallery, the tangential capacity and the aligned generator
(single core each).

## Analyses of the stored results

| script | result (under `data/manuscript_figures/analytics_audit/`) | time |
|---|---|---|
| `analysis/audit.py` | `cache/*.pkl` (photon-optimal paths, trade-off families) and the analytic checks | about 10 min, one core |
| `analysis/photon_flip/audit_photon.py`, `audit_photon2.py` | transfers through the vacuum (`photon_flip/results/`) | about 2 min each |
| `analysis/round4/transfer_closed.py` | the closed form and its one-mode form on each half of a transfer through the vacuum (closed-form columns of Table 2, `round4/results/transfer_closed.json`) | 6 s |
| `analysis/round3/*.py` | trade-off (SFig. 9), parity and Ramsey dephasing (Table 9), route test (Table 5), Pauli frame, flip-gate summary (Table 11), stabilizers along the paths, code-space averages, half-rate CS, one-branch summary | seconds to 2 min each |
| `analysis/review_oct28/robustness.py --workers 30` (then `--item6`) | gates vs transfers, stabilizers, resources, noise budget, CS-proposal rates (Tables 7–9, 13) | long (Lindblad runs of every stored flip and GKP pulse) |
| `analysis/review_oct28/lindblad_crosscheck.py --workers 12` | an independent Lindblad integrator against the stored traces | long |
| `analysis/cs_proposal/cs_proposal_hamiltonian.py --workers 16` | CS with the Hamiltonian of the cited proposal (Kerr, idle-branch rotation) | long |
| `flip_note/num_core.py` | photon-optimal flips that keep the qubit encoded (SFig. 18) | about 1 min |

`bash make_analyses.sh` (or `make analyses`) reruns the quick ones in a copy of `data/` and compares the results with the
stored files.
