# data-B: target-only pulses of Table I and the smoothing test of the pulses

Written by `manuscript_figures/sims/pulses_facts_B.py` from the data files (every number below is in `pulses_facts_B.json`). Statistics are mean and sample s.d. over all starts (none excluded).

## 1. Target-only GKP preparation at 412 ns (Letter Table I), with the pulses

`sims/capability_pulses.py` repeats `sims/capability_noisy.py` (imported, numerics unchanged) for CS, ECD, SNAP (chi_d = g, 4g), seeds 1000-1007, and stores the slice variables X and the Cartesian controls in `capability_pulses.jsonl`.

Per-start reproduction against `capability_noisy.jsonl` (32 starts): max |dF60| = 0.0e+00, max |dF90_closed| = 0.0e+00, max |dF90_noisy| = 0.0e+00 (reproduced to 1e-6: True).

| gate set | reproduced mean (s.d.) [best] | quoted in Table I | mean, s.d., best match |
|---|---|---|---|
| CS | 0.9976(2) [0.998] | 0.9976(2) [0.998] | True, True, True |
| ECD | 0.797(15) [0.813] | 0.797(15) [0.813] | True, True, True |
| SNAP | 0.629(9) [0.644] | 0.629(9) [0.644] | True, True, True |
| SNAP (chi_d = 4g) | 0.839(23) [0.876] | 0.839(23) [0.876] | True, True, True |

Wall time per start 788 +- 130 s (one core).

## 2. Smoothing test

Each Cartesian control is convolved with a normalized Gaussian of s.d. sigma, with zero drive before t = 0 and after T (duration kept, so the integrated drive near both ends is lower). Closed form (error functions), averaged over sub-slices of h = 0.2575 ns (M = 32 per slice at 412 ns, 64 at 824 ns), evaluated at cutoff 90 with `common/lindblad.evolve` and `CSModel.propagate` (preparation) or `flip_gate.evaluate` (flip gate). Delta = smoothed minus the same start at sigma = 0 on the same grid. Diagnostic `hold`: first and last slice values continued outside [0, T] (bandwidth only, no end loss).

Bounds over all 1232 rows: max |Omega|/10g - 1 = 1.3e-13 and max |branch amplitude|/g - 1 = 1.3e-13 for the smoothed pulses (sigma = 0: 4.4e-16, 4.4e-16), so every smoothed control stays in its bound disk up to rounding; kernel weights >= -5.9e-309, row sums - 1 <= 1.3e-13. For SNAP the branch column is the cavity drive.

Half-power bandwidth of the filter: sigma = 1 ns -> 133 MHz (sigma/Delta t = 0.12 at 412 ns), sigma = 2 ns -> 66 MHz (sigma/Delta t = 0.24 at 412 ns), sigma = 4 ns -> 33 MHz (sigma/Delta t = 0.49 at 412 ns).

Convolution check (closed form against a brute-force quadrature, CS 412 ns start 11): max deviation 1.4e-05 g (sigma = 1.0 ns), 3.6e-06 g (sigma = 2.0 ns), 1.5e-06 g (sigma = 4.0 ns).

Grid convergence (sigma = 1 ns, |change| of F_T or gate metric when h is halved):

- track_cs_412_s11_sigma1: M 16->32: F_T_noisy 1.2e-05, F_T_closed 1.2e-05; M 32->64: F_T_noisy 3.3e-06, F_T_closed 2.9e-06
- track_cd_412_s9_sigma1: M 16->32: F_T_noisy 2.7e-06, F_T_closed 2.4e-06; M 32->64: F_T_noisy 7.3e-07, F_T_closed 6.0e-07
- track_snap_824_s5_sigma1: M 32->64: F_T_noisy 1.6e-07, F_T_closed 4.1e-07; M 64->128: F_T_noisy 1.7e-07, F_T_closed 1.0e-07
- track_cs_824_s16_sigma1: M 32->64: F_T_noisy 2.1e-06, F_T_closed 9.0e-07; M 64->128: F_T_noisy 7.6e-07, F_T_closed 2.2e-07
- flip_cs_824_s3000_sigma1: M 16->32: avg6_opt_noisy 5.9e-06, avg6_opt_closed 7.1e-06

Baseline (sigma = 0 on the sub-slice grid) against the stored values, max |difference|:

- flip_cd_824: avg6_opt_noisy 3.3e-16, avg6_opt_closed 1.3e-15, avg6_noisy 2.2e-16, F_0to1_noisy 3.3e-16
- flip_cs_824: avg6_opt_noisy 3.3e-16, avg6_opt_closed 6.7e-16, avg6_noisy 1.7e-16, F_0to1_noisy 3.3e-16
- flip_snap_824: avg6_opt_noisy 3.3e-16, avg6_opt_closed 3.2e-15, avg6_noisy 1.7e-16, F_0to1_noisy 3.3e-16
- target_cd_412: F_T_noisy 1.5e-06, F_T_closed 3.8e-13
- target_cs_412: F_T_noisy 6.2e-06, F_T_closed 3.2e-13
- target_snap_412: F_T_noisy 3.8e-07, F_T_closed 2.7e-13
- target_snap4_412: F_T_noisy 2.6e-06, F_T_closed 2.0e-13
- track_cd_412: F_T_noisy 1.0e-06, C_noisy 1.9e-07, F_T_closed 2.7e-13, C_closed 2.5e-13
- track_cd_824: F_T_noisy 2.7e-06, C_noisy 5.6e-07, F_T_closed 1.0e-12, C_closed 5.7e-13
- track_cs_412: F_T_noisy 3.1e-06, C_noisy 1.0e-06, F_T_closed 5.4e-13, C_closed 2.4e-13
- track_cs_824: F_T_noisy 3.2e-06, C_noisy 1.6e-06, F_T_closed 9.3e-13, C_closed 5.4e-13
- track_snap_412: F_T_noisy 2.2e-07, C_noisy 2.2e-07, F_T_closed 2.3e-13, C_closed 2.1e-13
- track_snap_824: F_T_noisy 1.7e-06, C_noisy 2.8e-07, F_T_closed 6.0e-13, C_closed 3.0e-13

Change of the figures of merit (units of 1e-3, mean +- s.d. over starts [min, max]):

| case | n | value at sigma = 0 | quantity | sigma = 1 ns | sigma = 2 ns | sigma = 4 ns | hold, 2 ns | hold, 4 ns |
|---|---|---|---|---|---|---|---|---|
| flip_cd_824 | 8 | 0.7889 +- 0.0841 | avg6_opt_noisy | +0.11 +- 0.40 [-0.26, +0.80] | -0.19 +- 0.68 [-1.30, +1.12] | -2.97 +- 3.20 [-8.07, +0.85] | -0.36 +- 0.76 | -2.39 +- 4.42 |
| flip_cd_824 | 8 | 0.7950 +- 0.0843 | avg6_opt_closed | +0.11 +- 0.40 [-0.29, +0.79] | -0.21 +- 0.70 [-1.37, +1.12] | -3.04 +- 3.28 [-8.26, +0.86] | -0.36 +- 0.77 | -2.43 +- 4.48 |
| flip_cs_824 | 8 | 0.9860 +- 0.0031 | avg6_opt_noisy | -0.37 +- 0.10 [-0.51, -0.20] | -2.00 +- 0.57 [-2.85, -1.25] | -16.11 +- 5.88 [-24.99, -7.74] | -0.79 +- 0.43 | -10.84 +- 4.87 |
| flip_cs_824 | 8 | 0.9975 +- 0.0024 | avg6_opt_closed | -0.37 +- 0.09 [-0.49, -0.20] | -2.03 +- 0.56 [-2.82, -1.26] | -16.42 +- 5.92 [-25.22, -7.94] | -0.82 +- 0.42 | -11.11 +- 4.88 |
| flip_snap_824 | 8 | 0.7443 +- 0.0587 | avg6_opt_noisy | -0.73 +- 1.09 [-2.84, +0.36] | -2.10 +- 2.19 [-6.34, +0.32] | -9.48 +- 5.36 [-18.84, -2.43] | -0.19 +- 0.26 | -3.63 +- 2.25 |
| flip_snap_824 | 8 | 0.7509 +- 0.0593 | avg6_opt_closed | -0.74 +- 1.10 [-2.88, +0.36] | -2.14 +- 2.21 [-6.44, +0.30] | -9.66 +- 5.40 [-19.15, -2.52] | -0.21 +- 0.26 | -3.74 +- 2.28 |
| target_cd_412 | 8 | 0.7970 +- 0.0154 | F_T_noisy | -0.74 +- 0.10 [-0.85, -0.59] | -2.14 +- 0.28 [-2.50, -1.72] | -7.32 +- 1.18 [-8.74, -5.28] | -0.60 +- 0.19 | -2.77 +- 0.87 |
| target_cd_412 | 8 | 0.7978 +- 0.0155 | F_T_closed | -0.75 +- 0.10 [-0.85, -0.59] | -2.15 +- 0.28 [-2.51, -1.73] | -7.35 +- 1.19 [-8.78, -5.30] | -0.60 +- 0.19 | -2.79 +- 0.88 |
| target_cs_412 | 8 | 0.9976 +- 0.0002 | F_T_noisy | -0.31 +- 0.09 [-0.46, -0.20] | -1.73 +- 0.59 [-2.83, -1.04] | -12.38 +- 5.49 [-24.36, -7.40] | -0.45 +- 0.24 | -6.15 +- 3.59 |
| target_cs_412 | 8 | 0.9999 +- 0.0001 | F_T_closed | -0.31 +- 0.09 [-0.46, -0.21] | -1.75 +- 0.59 [-2.87, -1.06] | -12.49 +- 5.52 [-24.52, -7.47] | -0.46 +- 0.25 | -6.22 +- 3.61 |
| target_snap_412 | 8 | 0.6288 +- 0.0093 | F_T_noisy | -0.41 +- 0.08 [-0.53, -0.33] | -0.92 +- 0.18 [-1.18, -0.75] | -2.25 +- 0.58 [-2.90, -1.52] | -0.01 +- 0.01 | -0.07 +- 0.05 |
| target_snap_412 | 8 | 0.6291 +- 0.0093 | F_T_closed | -0.42 +- 0.08 [-0.53, -0.33] | -0.93 +- 0.18 [-1.19, -0.75] | -2.26 +- 0.58 [-2.92, -1.52] | -0.01 +- 0.01 | -0.07 +- 0.05 |
| target_snap4_412 | 8 | 0.8389 +- 0.0231 | F_T_noisy | -0.88 +- 0.21 [-1.16, -0.55] | -4.12 +- 1.68 [-6.75, -2.53] | -29.77 +- 13.72 [-52.17, -14.04] | -2.80 +- 1.56 | -27.10 +- 13.52 |
| target_snap4_412 | 8 | 0.8401 +- 0.0232 | F_T_closed | -0.88 +- 0.21 [-1.16, -0.55] | -4.13 +- 1.68 [-6.76, -2.53] | -29.86 +- 13.77 [-52.31, -14.10] | -2.80 +- 1.56 | -27.18 +- 13.57 |
| track_cd_412 | 20 | 0.8074 +- 0.0094 | F_T_noisy | -0.80 +- 0.11 [-1.02, -0.56] | -2.28 +- 0.32 [-2.89, -1.63] | -7.51 +- 1.17 [-9.54, -5.47] | -0.69 +- 0.27 | -3.06 +- 1.10 |
| track_cd_412 | 20 | 0.8082 +- 0.0095 | F_T_closed | -0.81 +- 0.11 [-1.03, -0.56] | -2.29 +- 0.32 [-2.91, -1.63] | -7.54 +- 1.18 [-9.58, -5.49] | -0.69 +- 0.27 | -3.07 +- 1.10 |
| track_cd_412 | 20 | 0.6218 +- 0.0339 | C_noisy | +1.30 +- 0.67 [+0.04, +2.30] | +2.75 +- 1.42 [-0.43, +4.96] | +6.63 +- 3.50 [-1.25, +12.43] | +1.32 +- 1.21 | +3.48 +- 3.28 |
| track_cd_824 | 20 | 0.9795 +- 0.0033 | F_T_noisy | -0.34 +- 0.11 [-0.55, -0.12] | -1.32 +- 0.30 [-2.09, -0.83] | -6.50 +- 1.41 [-10.22, -4.93] | -0.32 +- 0.17 | -2.63 +- 1.21 |
| track_cd_824 | 20 | 0.9836 +- 0.0033 | F_T_closed | -0.34 +- 0.11 [-0.56, -0.11] | -1.33 +- 0.30 [-2.12, -0.81] | -6.56 +- 1.44 [-10.35, -4.97] | -0.32 +- 0.18 | -2.66 +- 1.23 |
| track_cd_824 | 20 | 0.5124 +- 0.0620 | C_noisy | -0.03 +- 2.15 [-4.53, +4.12] | -0.07 +- 4.15 [-8.91, +7.17] | +0.69 +- 8.64 [-17.14, +12.34] | +0.94 +- 1.40 | +2.60 +- 4.31 |
| track_cs_412 | 20 | 0.9903 +- 0.0002 | F_T_noisy | -0.32 +- 0.26 [-0.87, +0.08] | -2.32 +- 1.16 [-5.50, -0.58] | -19.15 +- 8.80 [-43.28, -7.06] | -1.13 +- 0.59 | -12.66 +- 5.72 |
| track_cs_412 | 20 | 0.9926 +- 0.0003 | F_T_closed | -0.33 +- 0.26 [-0.87, +0.08] | -2.33 +- 1.17 [-5.53, -0.58] | -19.24 +- 8.84 [-43.49, -7.09] | -1.13 +- 0.59 | -12.72 +- 5.75 |
| track_cs_412 | 20 | 0.8254 +- 0.0201 | C_noisy | -0.60 +- 2.00 [-3.82, +3.97] | -1.77 +- 4.00 [-8.13, +4.73] | -5.31 +- 9.50 [-19.95, +13.45] | +0.37 +- 1.95 | -0.38 +- 6.17 |
| track_cs_824 | 20 | 0.9901 +- 0.0002 | F_T_noisy | -0.24 +- 0.41 [-1.00, +0.48] | -1.28 +- 0.95 [-2.70, +0.35] | -9.87 +- 4.34 [-22.25, -4.24] | -0.49 +- 0.37 | -6.66 +- 2.92 |
| track_cs_824 | 20 | 0.9952 +- 0.0003 | F_T_closed | -0.24 +- 0.43 [-1.05, +0.51] | -1.29 +- 0.98 [-2.73, +0.40] | -9.96 +- 4.40 [-22.51, -4.25] | -0.48 +- 0.38 | -6.72 +- 2.96 |
| track_cs_824 | 20 | 0.8962 +- 0.0101 | C_noisy | +0.47 +- 2.14 [-2.63, +4.12] | +0.65 +- 4.28 [-5.89, +7.55] | -2.56 +- 8.93 [-16.83, +12.03] | +0.51 +- 1.70 | -1.29 +- 4.52 |
| track_snap_412 | 20 | 0.5762 +- 0.0452 | F_T_noisy | -0.17 +- 0.23 [-0.54, +0.00] | -0.39 +- 0.54 [-1.20, +0.00] | -0.99 +- 1.39 [-2.92, +0.00] | -0.00 +- 0.01 | -0.02 +- 0.04 |
| track_snap_412 | 20 | 0.5763 +- 0.0454 | F_T_closed | -0.17 +- 0.23 [-0.54, +0.00] | -0.39 +- 0.54 [-1.21, +0.00] | -1.00 +- 1.40 [-2.94, +0.00] | -0.00 +- 0.01 | -0.02 +- 0.04 |
| track_snap_412 | 20 | 0.4849 +- 0.0892 | C_noisy | +1.11 +- 1.59 [-0.00, +4.14] | +2.20 +- 3.15 [-0.00, +8.11] | +4.42 +- 6.32 [-0.00, +16.13] | +0.29 +- 0.59 | +0.62 +- 1.17 |
| track_snap_824 | 20 | 0.8344 +- 0.1008 | F_T_noisy | -0.42 +- 0.18 [-0.66, +0.00] | -1.21 +- 0.49 [-1.81, +0.00] | -4.39 +- 1.74 [-6.33, +0.00] | -0.02 +- 0.04 | -0.62 +- 0.42 |
| track_snap_824 | 20 | 0.8376 +- 0.1019 | F_T_closed | -0.43 +- 0.19 [-0.67, +0.00] | -1.23 +- 0.51 [-1.83, +0.00] | -4.42 +- 1.76 [-6.42, +0.00] | -0.01 +- 0.04 | -0.61 +- 0.41 |
| track_snap_824 | 20 | 0.1928 +- 0.1232 | C_noisy | +0.38 +- 2.33 [-2.56, +4.59] | +0.77 +- 4.78 [-5.15, +9.75] | +1.61 +- 9.97 [-10.51, +21.59] | +0.41 +- 0.41 | +1.02 +- 1.33 |

## 3. Control statistics of the target-only pulses (412 ns, mean +- s.d. over 8 starts)

Definitions of data-A (`pulses_stats.control_stats`, copied): fractions of slices with the oscillator amplitude at its bound (a >= 0.999, both branches pooled for CS and ECD) or at zero, the same for the qubit drive, conditional fraction int|u_-|^2 / int(|u_+|^2 + |u_-|^2) with u_+- = (u_g +- u_e)/2, qubit rotation int|Omega|dt / pi (bound 10 g T / pi = 9.84 at 412 ns). For SNAP the oscillator amplitude is the cavity drive.
Copy checked against data-A's rows on 48 flip gate pulses: max |difference| 5.3e-15.

| pulses | osc at bound | osc at zero | drive at bound | mean drive / 10g | conditional fraction | qubit rotation / pi |
|---|---|---|---|---|---|---|
| target_cd_412 | 0.90 +- 0.07 | 0.10 +- 0.07 | 0.28 +- 0.06 | 0.40 +- 0.03 | 0.94 +- 0.03 | 3.96 +- 0.30 |
| target_cs_412 | 0.69 +- 0.15 | 0.09 +- 0.06 | 0.46 +- 0.12 | 0.74 +- 0.07 | 0.86 +- 0.04 | 7.31 +- 0.71 |
| target_snap4_412 | 0.92 +- 0.05 | 0.08 +- 0.05 | 0.34 +- 0.11 | 0.55 +- 0.09 |  | 5.42 +- 0.92 |
| target_snap_412 | 0.85 +- 0.06 | 0.15 +- 0.06 | 0.05 +- 0.01 | 0.19 +- 0.02 |  | 1.90 +- 0.16 |

Relative L2 change of the controls (mean over starts): flip_cd_824 sigma 1: drive 0.102, osc 0.079; flip_cd_824 sigma 4: drive 0.204, osc 0.159; flip_cs_824 sigma 1: drive 0.135, osc 0.125; flip_cs_824 sigma 4: drive 0.272, osc 0.250; flip_snap_824 sigma 1: drive 0.170, osc 0.056; flip_snap_824 sigma 4: drive 0.341, osc 0.113; target_cd_412 sigma 1: drive 0.127, osc 0.077; target_cd_412 sigma 4: drive 0.258, osc 0.152; target_cs_412 sigma 1: drive 0.163, osc 0.127; target_cs_412 sigma 4: drive 0.339, osc 0.260; target_snap_412 sigma 1: drive 0.149, osc 0.067; target_snap_412 sigma 4: drive 0.310, osc 0.133; target_snap4_412 sigma 1: drive 0.211, osc 0.063; target_snap4_412 sigma 4: drive 0.446, osc 0.125; track_cd_412 sigma 1: drive 0.135, osc 0.073; track_cd_412 sigma 4: drive 0.275, osc 0.145; track_cd_824 sigma 1: drive 0.136, osc 0.074; track_cd_824 sigma 4: drive 0.273, osc 0.149; track_cs_412 sigma 1: drive 0.170, osc 0.115; track_cs_412 sigma 4: drive 0.349, osc 0.232; track_cs_824 sigma 1: drive 0.147, osc 0.108; track_cs_824 sigma 4: drive 0.295, osc 0.218; track_snap_412 sigma 1: drive 0.164, osc 0.187; track_snap_412 sigma 4: drive 0.348, osc 0.398; track_snap_824 sigma 1: drive 0.147, osc 0.059; track_snap_824 sigma 4: drive 0.296, osc 0.119.

## 4. Derived summaries (from the table above)

Mean figure of merit of each gate set at sigma = 0, 1, 2, 4 ns (zero drive outside [0, T]) and the order of the gate sets:

- track_412: sigma 0: CS 0.9903, ECD 0.8074, SNAP 0.5762; sigma 1: CS 0.9899, ECD 0.8065, SNAP 0.5760; sigma 2: CS 0.9879, ECD 0.8051, SNAP 0.5758; sigma 4: CS 0.9711, ECD 0.7998, SNAP 0.5752 (same order at every sigma: True)
- track_824: sigma 0: CS 0.9901, ECD 0.9795, SNAP 0.8344; sigma 1: CS 0.9898, ECD 0.9792, SNAP 0.8340; sigma 2: CS 0.9888, ECD 0.9782, SNAP 0.8332; sigma 4: CS 0.9802, ECD 0.9730, SNAP 0.8301 (same order at every sigma: True)
- target_412: sigma 0: CS 0.9976, SNAP (chi_d = 4g) 0.8389, ECD 0.7970, SNAP 0.6288; sigma 1: CS 0.9973, SNAP (chi_d = 4g) 0.8380, ECD 0.7962, SNAP 0.6284; sigma 2: CS 0.9958, SNAP (chi_d = 4g) 0.8348, ECD 0.7948, SNAP 0.6279; sigma 4: CS 0.9852, SNAP (chi_d = 4g) 0.8091, ECD 0.7897, SNAP 0.6265 (same order at every sigma: True)
- flip_824: sigma 0: CS 0.9860, ECD 0.7889, SNAP 0.7443; sigma 1: CS 0.9857, ECD 0.7890, SNAP 0.7435; sigma 2: CS 0.9840, ECD 0.7887, SNAP 0.7422; sigma 4: CS 0.9699, ECD 0.7859, SNAP 0.7348 (same order at every sigma: True)

Largest loss of a single start and largest mean loss (units of 1e-3): sigma 1 ns: start -2.84 (flip_snap_824), mean -0.88 (target_snap4_412); sigma 2 ns: start -6.75 (target_snap4_412), mean -4.12 (target_snap4_412); sigma 4 ns: start -52.17 (target_snap4_412), mean -29.77 (target_snap4_412).

Growth of the mean loss with sigma, exponent p of |Delta| ~ sigma^p between successive sigma (log2 of the ratio; only where both losses exceed 1e-5):

- flip_cd_824: 2->4 p = 4.0
- flip_cd_824_hold: 1->2 p = 2.2, 2->4 p = 2.7
- flip_cs_824: 1->2 p = 2.4, 2->4 p = 3.0
- flip_cs_824_hold: 1->2 p = 3.5, 2->4 p = 3.8
- flip_snap_824: 1->2 p = 1.5, 2->4 p = 2.2
- flip_snap_824_hold: 2->4 p = 4.2
- target_cd_412: 1->2 p = 1.5, 2->4 p = 1.8
- target_cd_412_hold: 1->2 p = 2.1, 2->4 p = 2.2
- target_cs_412: 1->2 p = 2.5, 2->4 p = 2.8
- target_cs_412_hold: 1->2 p = 3.8, 2->4 p = 3.8
- target_snap_412: 1->2 p = 1.2, 2->4 p = 1.3
- target_snap_412_hold: 2->4 p = 2.7
- target_snap4_412: 1->2 p = 2.2, 2->4 p = 2.9
- target_snap4_412_hold: 1->2 p = 3.3, 2->4 p = 3.3
- track_cd_412: 1->2 p = 1.5, 2->4 p = 1.7
- track_cd_412_hold: 1->2 p = 2.1, 2->4 p = 2.2
- track_cd_824: 1->2 p = 2.0, 2->4 p = 2.3
- track_cd_824_hold: 1->2 p = 2.5, 2->4 p = 3.1
- track_cs_412: 1->2 p = 2.8, 2->4 p = 3.0
- track_cs_412_hold: 1->2 p = 3.1, 2->4 p = 3.5
- track_cs_824: 1->2 p = 2.4, 2->4 p = 2.9
- track_cs_824_hold: 1->2 p = 3.5, 2->4 p = 3.8
- track_snap_412: 1->2 p = 1.2, 2->4 p = 1.4
- track_snap_824: 1->2 p = 1.5, 2->4 p = 1.9
- track_snap_824_hold: 2->4 p = 5.3

Share of the mean loss due to the lower drive near the ends, 1 - Delta(hold)/Delta(zero):

- flip_cd_824: sigma 2: -0.86, sigma 4: 0.20
- flip_cs_824: sigma 1: 0.82, sigma 2: 0.60, sigma 4: 0.33
- flip_snap_824: sigma 1: 1.01, sigma 2: 0.91, sigma 4: 0.62
- target_cd_412: sigma 1: 0.81, sigma 2: 0.72, sigma 4: 0.62
- target_cs_412: sigma 1: 0.90, sigma 2: 0.74, sigma 4: 0.50
- target_snap_412: sigma 1: 1.00, sigma 2: 0.99, sigma 4: 0.97
- target_snap4_412: sigma 1: 0.67, sigma 2: 0.32, sigma 4: 0.09
- track_cd_412: sigma 1: 0.80, sigma 2: 0.70, sigma 4: 0.59
- track_cd_824: sigma 1: 0.83, sigma 2: 0.76, sigma 4: 0.60
- track_cs_412: sigma 1: 0.61, sigma 2: 0.52, sigma 4: 0.34
- track_cs_824: sigma 1: 0.81, sigma 2: 0.62, sigma 4: 0.32
- track_snap_412: sigma 1: 1.00, sigma 2: 0.99, sigma 4: 0.98
- track_snap_824: sigma 1: 1.02, sigma 2: 0.99, sigma 4: 0.86

## 5. Independent checks (`sims/pulses_check_B.py` -> `pulses_check_B.json`)

- Completeness: 1232 rows, 1232 distinct (pulse, sigma, edge); starts per case flip_cd_824 8, flip_cs_824 8, flip_snap_824 8, target_cd_412 8, target_cs_412 8, target_snap4_412 8, target_snap_412 8, track_cd_412 20, track_cd_824 20, track_cs_412 20, track_cs_824 20, track_snap_412 20, track_snap_824 20; every sigma and edge has every start: True. Starts equal data-A's independent extraction (mfcommon.groups, flip_gate.jsonl): True, trivial SNAP starts equal: True, max |X - X_A| = 0.0e+00; target seeds 1000-1007 of every gate set: True.
- Bounds over every row: max |Omega|/10g - 1 = 1.3e-13, max |branch|/g - 1 = 1.3e-13, polar round trip 2.8e-14.
- Statistics recomputed from the rows: 228 means and s.d. of Delta, max |difference| 0.0e+00.
- Table I: 32 of 32 starts bitwise equal to capability_noisy.jsonl (F60, F90_closed, F90_noisy); stored C equals the formulas of Eq. (7) to 0.0e+00; the entries re-formatted from the 8 starts equal the current main.tex: CS True, ECD True, SNAP True, SNAP (chi_d = 4g) True.
- Spot checks (17 pulses: track|cs|412|11|0.0|zero, track|cs|412|11|4.0|zero, track|cs|412|11|1.0|hold, track|cd|412|9|2.0|zero, track|snap|412|9|4.0|zero, track|snap|412|6|4.0|zero, track|cs|824|16|4.0|zero, track|cd|824|18|2.0|hold, track|snap|824|5|4.0|zero, target|cs|412|1000|4.0|zero, target|cd|412|1004|1.0|hold, target|snap|412|1000|2.0|zero, target|snap4|412|1001|4.0|zero, target|snap4|412|1001|0.0|zero, flip|cs|824|3000|4.0|zero, flip|snap|824|3007|2.0|zero, flip|cd|824|3003|4.0|hold): smoothed pulses rebuilt independently (pointwise exact convolution with the normal CDF, 8-point Gauss-Legendre average per sub-slice, controls from the formulas of Eq. (7)) and evaluated with the independent integrator of analytics_audit/review_oct28/lindblad_crosscheck.py (exact expm_multiply per sub-slice; flip: flip_gate.evaluate). Max |difference| from the rows: F_T noisy 1.0e-06, F_T closed 1.2e-13, C noisy 2.1e-07, C closed 1.2e-12, flip gate metric noisy 8.0e-14, closed 8.4e-14; bounds of the rebuilt pulses - 1 <= 4.4e-16. The noisy F_T difference is the same at sigma = 0 (Lindblad splitting of common/lindblad.py against exact exponentials), so it cancels in Delta.

