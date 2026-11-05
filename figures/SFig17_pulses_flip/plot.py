"""Supplementary Fig. S27: the logical-flip gate pulses at 824 ns (gate objective alone, Supplementary Table 11).

Pulses of sims/flip_gate.py (mode target), T = 824 ns, N = 50 slices, 8 starts per gate set (seeds 3000-3007), scored by
the six-state average fidelity with X_L after the best frame update (Lindblad, cutoff 90).  Shown: for each gate set the
start whose score is closest to the mean over its 8 starts (ties: lowest seed), and in addition the one CD start that
implements the flip (the best start), since the representative CD start stalls at the plain-displacement value.
Purity row: the smallest oscillator purity over the inputs |0_L>, |1_L>, |+_L> (closed evolution, cutoff 90), band
mean +- s.d. over the 8 starts for the representative columns.  Representation as S25 (sims/pulses_figlib.py).
Data: results/pulses/flip_target_<gate>_T824.npz and closed_flip_target_<gate>_T824.npz (pulses_extract.py,
pulses_stats.py); the shown runs are checked against data-A's per-start numbers (pulses_stats_A_rows.json).
Writes S27.png, S27_facts.json and caption.tex into this folder.
  python figures/SFig17_pulses_flip/plot.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "sims"))
import numpy as np  # noqa: E402
import pulses_figlib as PF  # noqa: E402

T = 824
z = np.load(PF.R / f"flip_target_cd_T{T}.npz")
sc = np.asarray(z["avg6_opt_lind"], float)
best = int(z["seed"][int(np.argmax(sc))])
n_flip = int(np.sum(sc > 0.78))                  # starts above the best plain displacement (0.780, Sec. IX)
KEYS = ("cs", "cd", "cd_best", "snap")
cols = [PF.flip_column("cs", T), PF.flip_column("cd", T), PF.flip_column("cd", T, seed=best, title="CD, best start"),
        PF.flip_column("snap", T)]
for k, c in zip(KEYS, cols):
    PF.check_against_A(c, f"flip_target_{c['gate']}_T{T}.npz")
print("wrote", PF.draw(cols, HERE / "S27.png", layout="cols", pur_label=r"$\min\,\mathrm{Tr}\,\rho^2$"))
f = {k: c["facts"] for k, c in zip(KEYS, cols)}
f.update(T_ns=cols[0]["T_ns"], N=cols[0]["N"], dt_ns=cols[0]["T_ns"] / cols[0]["N"], cd_starts_above_0780=n_flip,
         cd_best_seed=best)
(HERE / "S27_facts.json").write_text(json.dumps(f, indent=1))
n = f["cs"]["n"]
assert n_flip == 1 and f["cd"]["seed"] != best
# Supplementary Table 11 (tab:gate2), row 'gate only, 824 ns': the quoted mean +- s.d. [best] must match the data
import re  # noqa: E402
si = _L.paper_text("supplement.tex")
row = next(l for l in si.splitlines() if l.startswith(r"gate only, 824~ns &"))
tab = re.findall(r"\$([0-9.]+)\\pm([0-9.]+)\$ \[([0-9.]+)\]", row)
assert len(tab) == 3, row
PM = {}
for g, (mq, sq, bq) in zip(("cs", "cd", "snap"), tab):
    m, s, _ = f[g]["F_stat"]
    assert (f"{m:.3f}", f"{s:.3f}", f"{f[g]['F_best']:.3f}") == (mq, sq, bq), (g, m, s, mq, sq, bq)
    PM[g] = f"${mq}\\pm{sq}$"
# claims of the caption, from the facts of the drawn runs
assert f["cd"]["shape"]["cond_frac"] < 0.2 and min(f["cd"]["purity_min_rep"]) > 0.95       # stalled CD start
assert max(f["cs"]["purity_min_rep"]) < 0.6 and max(f["cd_best"]["purity_min_rep"]) < 0.6   # nearly maximal
assert f["cs"]["purity_final_min_over_inputs_rep"] > 0.99 and f["cd_best"]["purity_final_min_over_inputs_rep"] > 0.99
assert f["snap"]["purity_final_min_over_inputs_rep"] < 0.8                                   # left entangled
cap = (r"\caption{Pulses of the logical flip optimized for the gate objective alone at $T="
       + f"{f['T_ns']:.0f}" + r"$~ns, scored by the six-state average fidelity with $X_L$ under noise (Supplementary "
       r"Table~\ref{tab:gate2}). For each gate set a column shows the start whose score is closest to the mean over "
       rf"its {n} starts, and for CD a further column shows the best start, the only one that implements the flip. "
       rf"Panels as in Supplementary Fig.~\ref{{fig:S25}}, with $N={f['N']}$ slices of {f['dt_ns']:.2f}~ns, except "
       r"that (d) shows the smallest oscillator purity over the inputs $\ket{0_L}$, $\ket{1_L}$, and $\ket{+_L}$, with "
       rf"the band over the {n} starts for the representative columns. The shown pulses score "
       rf"{f['cs']['F_rep']:.3f} (CS), {f['cd']['F_rep']:.3f} and {f['cd_best']['F_rep']:.3f} (CD), and "
       rf"{f['snap']['F_rep']:.3f} (SNAP), against {PM['cs']}, {PM['cd']}, and {PM['snap']} over the {n} starts. The "
       r"representative CD pulse, one of the stalled starts, stays close to a product state and acts mainly through "
       r"its common part, as a plain displacement would. The best CD pulse and the CS pulse instead pass through "
       r"nearly maximally entangled states and disentangle by the end, whereas the SNAP pulse leaves qubit and "
       r"oscillator partly entangled." + "}\n" + r"\label{fig:S27}" + "\n")
(HERE / "caption.tex").write_text(cap)
print(cap)
for k in KEYS:
    print(k, f[k])
