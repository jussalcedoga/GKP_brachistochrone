"""Supplementary Fig. S26: the target-only GKP pulses at 412 ns of Letter Table I.

Vacuum -> |0_L> (Delta = 0.4) optimized for the final fidelity alone (tracking weight zero, the stages of
sims/capability_noisy.py), T = 412 ns, N = 50 slices, CS, CD, SNAP (chi_d = g) and SNAP (chi_d = 4 g), 8 starts each
(seeds 1000-1007).  Shown: for each gate set the start whose Lindblad fidelity is closest to the mean over its 8 starts
(ties: lowest seed).  Representation as S25 (sims/pulses_figlib.py).  Data: results/pulses/closed_target_<gk>_T412.npz
(sims/pulses_closed_target.py, from capability_pulses.jsonl, closed F_T reproduced to 4e-14).  Before drawing, the means
over the starts are checked against the rows of the target-only table (tab:capability, Supplementary Table 15) in
supplement.tex of the paper (read only).
Writes S26.png, S26_facts.json and caption.tex into this folder.
  python figures/SFig16_pulses_target/plot.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import re
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "sims"))
import pulses_figlib as PF  # noqa: E402

GK = ("cs", "cd", "snap", "snap4")
cols = [PF.target_column(gk) for gk in GK]
for c in cols:
    PF.shape_facts(c)
# Table I of the Letter: mean (s.d.) [best] at 412 ns; the means of the 8 starts must match the quoted digits
TAB = {"cs": "CS", "cd": "CD", "snap": r"SNAP, $\chi_d=g$", "snap4": r"SNAP, $\chi_d=4g$"}
tex = _L.paper_text("supplement.tex")   # the target-only table: SI (Nov 5)
tex = tex[tex.index(r"\label{tab:capability}"):]                               # its rows only, not other tables
tex = tex[:tex.index(r"\end{tabular}")]
for gk, c in zip(GK, cols):
    row = next(l for l in tex.splitlines() if l.startswith(TAB[gk] + " &"))
    entry = re.search(r"&\s*([0-9.]+\([0-9]+\))", row).group(1)            # 412-ns column, mean (s.d.)
    c["facts"]["table_I_412"] = entry
    c["facts"]["table_I_pm"] = PF.table_pm(entry, *c["facts"]["F_stat"][:2])   # asserts mean and s.d. digits
print("wrote", PF.draw(cols, HERE / "S26.png", layout="cols"))
f = {gk: c["facts"] for gk, c in zip(GK, cols)}
f.update(T_ns=cols[0]["T_ns"], N=cols[0]["N"], dt_ns=cols[0]["T_ns"] / cols[0]["N"])
(HERE / "S26_facts.json").write_text(json.dumps(f, indent=1))
n = f["cs"]["n"]
assert all(f[g]["n"] == n for g in GK)
pm = [f"${f[g]['table_I_pm']}$" for g in GK]
cap = (r"\caption{Pulses of the GKP preparation of Table~I at $T=" + f"{f['T_ns']:.0f}" + r"$~ns, optimized for the "
       r"final fidelity alone without a path constraint, for CS, CD, and SNAP with $\chi_d=g$ and $4g$. Each column "
       rf"shows the start whose fidelity under noise is closest to the mean over its {n} starts. Panels as in "
       rf"Supplementary Fig.~\ref{{fig:S25}}, with $N={f['N']}$ slices of {f['dt_ns']:.2f}~ns, the fixed conditional "
       r"term $\tfrac12\chi_d\sigma_z\otimes\hat n$ of SNAP at the $\chi_d$ of each column, and the band in (d) over "
       rf"the {n} starts. The shown runs reach $F_T={f['cs']['F_rep']:.3f}$ (CS), {f['cd']['F_rep']:.3f} (CD), "
       rf"{f['snap']['F_rep']:.3f} ($\chi_d=g$), and {f['snap4']['F_rep']:.3f} ($\chi_d=4g$), against {pm[0]}, "
       rf"{pm[1]}, {pm[2]}, and {pm[3]} over the {n} starts (Table~I)." + "}\n" + r"\label{fig:S26}" + "\n")
(HERE / "caption.tex").write_text(cap)
print(cap)
for g in GK:
    print(g, f[g])
