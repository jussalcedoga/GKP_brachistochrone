# Stored pulses of the paper (result set A, Nov 1 2026)

Pulses of the Fig. 4 tracking benchmark and of the logical flip as a gate, converted from the stored slice variables to
Cartesian controls, re-simulated against the stored numbers, and characterized (bounds, conditional part, qubit
rotation, entanglement). No new optimization. Target-only GKP pulses (Table I) and the smoothing test are result set B
(files without the suffix `_A`, folder `B_scripts/`).

## How the files were made

Scripts in `simulations/gate_sets/`, run in this order:

| script | does | writes |
|---|---|---|
| `pulses_extract.py` | slice variables X -> controls through `model.controls()`, representative flags, stored scores | `tracking_gkp_*.npz`, `flip_*.npz`, `pulses_index_A.json` |
| `pulses_reproduce.py --workers 24` | re-simulates every extracted pulse with the routine that wrote the stored numbers, compares | `reproduce_A.jsonl`, `reproduce_A.json`, `pulses_reproduce_A.log` |
| `pulses_stats.py --workers 16` | per-start control statistics and closed-evolution purity, P_e, nbar (cutoff 90) | `pulses_stats_A.json`, `pulses_stats_A_rows.json`, `closed_*.npz`, `pulses_stats_A.log` |
| `pulses_facts.py` | assembles every number and evaluates the physics checks by explicit rules | `pulses_facts_A.json` |

Compute environment. All runs on the compute server described in the top-level README, from a copy of the read-only inputs in the same
relative layout, with Python 3.14.4, numpy 2.3.5 and scipy 1.16.3 (not the pinned runtime of requirements.txt). The
reproduction below (max deviation 6e-14) shows that the environment does not change any stored number.

## Data selection

* Tracking, `tracking_gkp_<gate>_T<412|824|2133>.npz`: exactly the 20 starts per gate set and duration of main-text
  Fig. 4, that is `mfcommon.groups()[("gkp", "geodesic", gate, k)]`, k = 1, 2, 3 (traces80 of
  manuscript/fig4_cs_path_following, X from its jobs80). The SNAP rows of Fig. 4 are these traces80 runs (seeds 1-20).
  The warm-started runs of `sims/snap_warm.py` enter only S11 (binomial word, `--snap-warm`) and are not used here.
  Seed 2 of CS and ECD is the grid job `main_gkp_geodesic_<gate>_Tg<3.09|6.18|16.0>`. Checks passed at extraction:
  n = 20, the representative equals `rep_seed` of `fig4/fig4_facts.json`, and the number of trivial runs equals
  `n_trivial` there (SNAP 13 at 412 ns, 2 at 824 ns, 0 otherwise).
* Flip, `flip_target_<gate>_T<412|824>.npz` (gate objective alone) and `flip_track_<gate>_T<412|824>_W<10|70>.npz`
  (gate tracking, the two rungs of Supplementary Table 11; the other seven rungs stay in flip_gate.jsonl only),
  8 starts each (seeds 3000-3007). The scores reproduce Supplementary Table 11 (for example CS gate only 824 ns
  0.9860 +- 0.0031, ECD tracking 824 ns W = 70 0.8916 +- 0.1154).

Representative start.
* Tracking: `mfcommon.representative` (Lindblad x, z, F traces closest to the mean). For SNAP at 412 ns that start
  (seed 9) never leaves the vacuum. `rep_nontrivial` marks the start chosen by the same rule among the non-trivial
  starts (seed 1).
* Flip: no (x, z, F) traces exist for gate pulses, so the representative is the start whose score (six-state average
  with X_L, best frame update, Lindblad, cutoff 90) is closest to the mean score of its case (ties, lowest seed). For ECD
  gate-only at 824 ns this is a stalled pulse (seed 3006, 0.774), not the one start that implements the flip
  (seed 3002, 0.991).

## Conventions and units

Controls follow Letter Eq. (7) and SI Sec. VII. Omega = Omega_x + i Omega_y = 10 g r e^{i f_Omega}, branch amplitude
u_j = g a_j e^{i f_j} (xi_j for CS, varepsilon_j for ECD), SNAP cavity drive varepsilon = g a_g e^{i f_g} at
chi_d = g, (a_e, f_e) unused for SNAP. g = 7.5 us^-1. Slices are piecewise constant on [t_k, t_{k+1}),
t_k = k T/N. Stored complex amplitudes are normalized: `Omega_over_bound` = Omega/(10 g), `Omega_over_g`,
`u_g`, `u_e`, `eps` in units of g. For CS and ECD, u_+ = (u_g + u_e)/2 (unconditional) and u_- = (u_g - u_e)/2
(sigma_z-conditional), since sum_j |j><j| (x) G(u_j) = 1 (x) G(u_+) + sigma_z (x) G(u_-) for G linear in u.
Qubit basis index 0 = |g>, sigma_z|g> = +|g>. Statistics are mean +- sample s.d. (ddof 1) over all starts, every start
included (trivial SNAP runs too), n stated.

## Files and keys

`pulses_index_A.json`: `description`, `keys_common`, `keys_tracking`, `keys_flip`, and `cases` (one entry per npz:
file, family, gate, T_ns, N, n_starts, seeds, rep_seed, rep_rule, score, score_mean, score_sd; tracking also k,
rep_nontrivial_seed, n_trivial, trivial_seeds; flip also W, tabulated, score_best, rep_score).

`tracking_gkp_<gate>_T<ns>.npz` (arrays over starts, sorted by seed):
X (n, N, 6) slice variables (r, f_Omega, a_g, f_g, a_e, f_e); N; T_ns; dt_ns; t_edges_ns (N+1); g_per_us;
Omega_max_per_us; chi_d_over_g (1 for SNAP, nan otherwise); Omega_over_g, Omega_over_bound (n, N) complex;
u_g, u_e (CS, ECD) or eps (SNAP) (n, N) complex in units of g; seed; job_id; w_star (selected tracking weight);
F_T_closed, C_closed, F_T_noisy, C_noisy (stored traces80 values, cutoff 90); trivial; representative; rep_nontrivial;
tL_ns, FTL, FL, nbarL (n, 8N+1) Lindblad traces of traces80 (target fidelity, closeness, photon number).

`flip_<mode>_<gate>_T<ns>[_W<w>].npz`: the common keys above plus W (nan for target); avg6_opt_lind (the score),
avg6_lind, phi_opt_lind, F_0to1_lind, F_1to0_lind, avg6_opt_closed, avg6_closed, nbar_max_closed (stored eval, cutoff
90, independent integrator); F_T_D60, C_D60 (n, 3) per input 0_L, 1_L, +_L at cutoff 60; representative.

`closed_<case>.npz`: closed evolution at cutoff 90, 8 points per slice. t_ns (8N+1), sub = 8, inputs ("0" for
tracking from |0>|g>; "0", "1", "+" for the flip, code words of `flip_gate.code`, qubit in |g>), seed,
representative, and (n, inputs, 8N+1) arrays purity = Tr rho_osc^2 (equal to the qubit purity, the joint state is pure),
Pe = excited-qubit population, nbar = <a^dag a>. Slice ends are every 8th point.

`reproduce_A.jsonl`: one line per re-simulated pulse (324): kind, id or line, gate, k or mode, seed, T_ns, W, `dev`
(absolute deviation per stored quantity), stored and new values, wall_s, `_key`. `reproduce_A.json`: maxima per case
(`max_dev_FT_C`, `max_dev_traces` for tracking, `max_dev_metric`, `max_dev_grape_closed` for the flip), global maxima,
number of errors.

`pulses_stats_A.json`: per case, `controls` and `closed` statistics (mean, sd, n, min, max) of every per-start number
below, `controls_rep` and `closed_rep` (the representative start), tracking also max_F_T_closed_check, n_trivial,
trivial_osc_max, trivial_drive_mean, nontrivial_controls, nontrivial_closed, flip also F_0to1_closed, F_1to0_closed and
their maximal deviation from the stored closed eval. `pulses_stats_A_rows.json`: the per-start numbers.

Per-start control numbers (fractions of slices unless stated): osc_at_bound (a >= 0.999, both branches pooled for CS and
ECD), osc_at_zero (a <= 0.001), osc_mean (time-averaged a), osc_g_at_bound, osc_e_at_bound, both_branches_at_bound,
drive_at_bound (r >= 0.999), drive_at_zero, drive_mean (time-averaged r), rotation_angle_over_pi (int|Omega|dt / pi; the
bound 10 g T / pi is 9.84, 19.67, 50.93 at 412, 824, 2133 ns), drive_le_chi (|Omega| <= chi_d = g, r <= 0.1),
cond_frac (int|u_-|^2 / int(|u_+|^2 + |u_-|^2)), um_mean, up_mean (time-averaged |u_-|, |u_+| over g), um_at_bound,
up_at_bound (|u_-|, |u_+| >= 0.99 g), overlap_both / drive_only / cond_only / neither (r >= 0.5 and |u_-| >= 0.5 g),
corr_r_um (Pearson correlation of r and |u_-| over the slices), jumps_Omega, jumps_um, jumps_up, jumps_eps (phase jumps
> pi/2 between consecutive slices with both amplitudes > 0.1 of the bound), osc_max (SNAP, largest |eps|/g).
Per-start closed numbers (per input): purity_min (slice ends), purity_min_fine (8 per slice), t_purity_min_over_T,
purity_final, purity_mean, Pe_max, Pe_mean, Pe_final, nbar_max, nbar_final, nbar_mean.

`pulses_facts_A.json`: `cases` (index entries plus rotation_angle_bound_over_pi), `reproduction`, `stats` (compact per
case, plus osc_bound_or_zero, drive_bound_or_zero and, for the flip, closed_min_over_inputs), and `checks`, each with
its rule and numbers: bang_bang, conditional_part, entanglement, entanglement_ranking_tracking,
rotations_vs_conditional_part, pure_conditional_at_bound, ecd_flip_stall, snap_zero_drive, qubit_drive_strength,
gate_vs_tracking_824ns, snap_flip_tracking_412ns.

## Reproduction

All 324 extracted pulses re-simulated, 0 errors.
* Tracking (180 pulses), `traces.run` at cutoff 90 (closed and Lindblad, 8 points per slice): max |dev| of F_T and C
  (closed and Lindblad) 6.4e-14, of every stored time trace 2.1e-13, per case <= 6.4e-14.
* Flip (144 pulses, gate only and W = 10, 70), `flip_gate.evaluate`: max |dev| of every gate metric 3.3e-16. A third
  propagator (the GRAPE models of common/) gives the closed six-state averages to <= 1.2e-14.
* The closed propagation of `pulses_stats.py` reproduces the stored closed F_T to 5.3e-11 and the stored closed
  F(0_L -> 1_L), F(1_L -> 0_L) to 1.1e-13.

## Findings (numbers from pulses_facts_A.json, mean +- s.d., n = 20 tracking, n = 8 flip)

* Saturation fades with duration. In the tracking pulses the oscillator amplitudes sit at a bound (upper or lower) in
  0.89 +- 0.04, 0.60 +- 0.05, 0.21 +- 0.06 of the slice-branches for CS at 412, 824, 2133 ns, and 1.00, 0.98, 0.59 for
  ECD. The CS gate-only flip at 412 ns is bang-bang in the oscillator amplitude (0.998 +- 0.005). The qubit drive is
  never bang-bang by the 90% rule (at most 0.81, CS flip at 412 ns).
* The conditional part dominates. Conditional fraction of CS tracking pulses 0.85 +- 0.03, 0.79 +- 0.02, 0.66 +- 0.03
  (minimum over starts 0.60), ECD 0.97, 0.97, 0.76. ECD tracking pulses use u_e = -u_g at the bound, the ECD generator
  itself, in 0.89 +- 0.06 and 0.73 +- 0.11 of the slices at 412 and 824 ns. CS rarely does (0.20, 0.10).
* Entanglement is made and undone. Minimum purity of CS tracking pulses 0.77 +- 0.01, 0.86 +- 0.02, 0.87 +- 0.01,
  ECD 0.57, 0.56, 0.81, SNAP non-trivial 0.80 (n = 7), 0.57 (n = 18), 0.59, mean final purity 0.966-0.9995 in every
  tracking case (the final purity is bounded below by F_T^2, so it is not independent of the fidelity). CS stays closest to product
  states at 824 and 2133 ns but not at 412 ns, where the non-trivial SNAP starts barely move (F_T 0.636 +- 0.010).
* The flip needs far more entanglement than preparation. Every CS gate-only pulse at 824 ns passes through a nearly
  maximally entangled state (minimum purity over the inputs 0.514 +- 0.012) and holds up to 5.5 +- 0.7 photons,
  against a minimum purity of 0.86 +- 0.02 and at most 2.13 +- 0.03 photons for CS preparation tracking at the same
  duration.
* ECD flip stall. The one ECD start that implements the flip (seed 3002, 0.991) entangles maximally (minimum purity
  0.500). Four of the five starts stalled at 0.773-0.774 stay close to product states (minimum purity 0.91-0.99) with a
  mostly unconditional drive (conditional fraction 0.09-0.35), whereas the fifth and the two at 0.727 entangle
  maximally. Gate tracking makes every ECD start entangle maximally at 824 ns (0.502 +- 0.006 at W = 10,
  0.500 +- 0.000 at W = 70) and raises the conditional fraction (0.53 +- 0.32 to 0.69-0.72).
* Rotations and conditional squeezing act together, not in alternation. In CS pulses at 412 and 824 ns the qubit drive
  (r >= 0.5) and the conditional part (|u_-| >= 0.5 g) are on in the same slice in 0.76 +- 0.05 and 0.70 +- 0.06 of the
  slices, and their correlation is weak (-0.17 +- 0.10, -0.03 +- 0.11). Only at 2133 ns does the drive act mostly
  alone (0.41). The sequential gate picture of SI Sec. V is a decomposition, not the shape of these pulses.
* Qubit drive strength. Time-averaged |Omega| is 8.2 g, 7.8 g, 6.5 g for CS tracking, 4.1-5.3 g for ECD and about
  2 g for non-trivial SNAP, where |Omega| <= chi_d in 0.41-0.44 of the slices (0.07-0.13 for CS).
* SNAP zero-drive runs. The 13 (412 ns) and 2 (824 ns) trivial runs have a cavity drive of at most 1.3e-4 g while the
  qubit drive is on (0.35 +- 0.07 and 0.46 +- 0.11 of its bound), and end at F_T = |<0|0_L>|^2 = 0.5441.
* Observation (not a claim). SNAP gate-tracking pulses at 412 ns barely drive the cavity (max 0.05 +- 0.04 g at W = 10)
  and accumulate chi_d int P_e dt = (1.01 +- 0.02) pi/2, consistent with a quarter turn of phase space through the
  dispersive term, F(0_L -> 1_L) = 0.4995 and a score of 0.663, close to the 2/3 of a quarter turn against X_L.

## What did not hold or needs care

* Bang-bang structure is not generic. It appears only at the shortest durations and for the oscillator amplitudes.
* "CS entangles least" fails at 412 ns (see above).
* A literal interleaving of rotations and conditional gates is not what the pulses do (CS at 412 and 824 ns).
* Entanglement is not fully undone in the low-score flip cases (final purity, minimum over inputs, 0.958 +- 0.031 for
  CS gate only at 412 ns, 0.83 +- 0.07 for SNAP gate only at 824 ns).
* The ECD stall is not uniformly unentangled (one of five stalled starts at 824 ns entangles maximally).
* The representative ECD gate-only pulse at 824 ns is a stalled pulse. A figure of the ECD gate pulse must say which
  start it shows.
