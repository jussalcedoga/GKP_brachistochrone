"""Supplementary Fig. S25: the pulses of the Fig. 4 tracking runs whose Wigner functions Fig. 4(d) shows.

GKP preparation |0> -> |0_L> (Delta = 0.4) along the brachistochrone at T = 824 ns, N = 50 slices, the representative
start of CS, CD and SNAP (mfcommon.representative, the start whose Lindblad (x, z, F) traces are closest to the averages
over the 20 starts, the runs of Fig. 4).  Numbered markers at t_k = kT/5, the times of the Wigner functions of Fig. 4(d).
Representation and data: simulations/gate_sets/pulses_figlib.py (results/pulses/ of pulses_extract.py and
pulses_stats.py).  Before drawing, the shown runs are checked against fig4/fig4_facts.json (seed, F_T, mean F_T) and
against data-A's per-start numbers (pulses_stats_A_rows.json).
Writes S25.png, S25_facts.json and caption.tex into this folder (--layout rows: the rejected candidate with one row per
gate set, into _candidates/S25_rows.png).
  python figures/SFig15_pulses_tracking/plot.py [--layout rows]
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "sims"))
import pulses_figlib as PF  # noqa: E402

LAYOUT = sys.argv[sys.argv.index("--layout") + 1] if "--layout" in sys.argv else "cols"
T = 824
GATES = ("cs", "cd", "snap")
cols = [PF.tracking_column(g, T) for g in GATES]
fig4 = json.loads((PF.MF / "fig4" / "fig4_facts.json").read_text())["columns"]["2"]
for c in cols:                                   # the shown runs are those of Fig. 4(d)
    assert c["facts"]["seed"] == fig4[c["gate"]]["rep_seed"], (c["gate"], c["facts"]["seed"])
    assert abs(c["facts"]["F_rep"] - fig4[c["gate"]]["rep_F_T"]) < 1e-12
    assert abs(c["facts"]["F_stat"][0] - fig4[c["gate"]]["F_T_mean"]) < 1e-12
    assert c["facts"]["n_trivial"] == fig4[c["gate"]]["n_trivial"]
    PF.check_against_A(c, f"tracking_gkp_{c['gate']}_T{T}.npz")
if LAYOUT != "cols":
    print("wrote", PF.draw(cols, HERE / "_candidates" / "S25_rows.png", layout="rows", snapshots=True, row_h=2.2))
    sys.exit()
print("wrote", PF.draw(cols, HERE / "S25.png", layout="cols", snapshots=True))
f = {c["gate"]: c["facts"] for c in cols}
f.update(T_ns=cols[0]["T_ns"], N=cols[0]["N"], dt_ns=cols[0]["T_ns"] / cols[0]["N"])
(HERE / "S25_facts.json").write_text(json.dumps(f, indent=1))
cs, cd, sn = (f[g] for g in GATES)
n = cs["n"]
assert cd["n"] == n and sn["n"] == n
assert cs["n_trivial"] == 0 and cd["n_trivial"] == 0 and 1 < sn["n_trivial"] < 4
# mean +- s.d. over the 20 starts in the rounding of Supplementary Table tab:proposal (column 'echo', the rates of
# Fig. 4), asserted against the data by PF.table_pm (audit, Nov 2: 0.9901 +- 0.0002 instead of 0.99006 +- 0.00019)
import re  # noqa: E402
si = _L.paper_text("supplement.tex")
LAB = {"cs": "CS", "cd": "CD", "snap": "SNAP"}
PM = {}
for g in GATES:
    trow = next(l for l in si.splitlines() if l.startswith(f"GKP, {LAB[g]}, {T}~ns &"))
    entry = [c.strip() for c in trow.rstrip("\\").split("&")][2]           # pulses & closed & echo & ...
    assert re.fullmatch(r"[0-9.]+\([0-9]+\)", entry), trow
    PM[g] = "$" + PF.table_pm(entry, *f[g]["F_stat"][:2]) + "$"
cap = (r"\caption{Pulses of the GKP preparation runs of Fig.~4 at $T=" + f"{f['T_ns']:.0f}" + r"$~ns, the runs whose "
       r"Wigner functions Fig.~4(d) shows. Each column shows one run per gate set, the start whose $(x,z,F)$ traces "
       rf"lie closest to the averages over its {n} starts. The controls are constant on $N={f['N']}$ slices of "
       rf"{f['dt_ns']:.2f}~ns and are drawn relative to their bounds (dotted lines at $\pm1$), with the real part "
       r"solid, the imaginary part dashed, and the range from minus to plus the modulus shaded. (a)~Qubit drive "
       r"$\Omega=\Omega_x+i\Omega_y$, bounded by $|\Omega|\leq10g$. (b,~c)~Conditional part $u_-=(u_g-u_e)/2$ and "
       r"common part $u_+=(u_g+u_e)/2$ of the branch amplitudes $u_j=\xi_j$ (CS) or $\varepsilon_j$ (CD) of Eq.~(7), "
       r"which act on the oscillator as $1\otimes G(u_+)+\sigma_z\otimes G(u_-)$ [Eq.~\eqref{eq:upm}]. The "
       r"conditional-squeezing gate of Sec.~\ref{app:cs} has $u_+=0$. For SNAP the cavity drive acts on both qubit "
       r"states alike, so $u_+=\varepsilon$ and $u_-=0$, and the only conditional term is the fixed "
       r"$\tfrac12\chi_d\sigma_z\otimes\hat n$ of the dispersive coupling, $\chi_d=g$. (d)~Purity $\mathrm{Tr}\,\rho^2$ "
       r"of the oscillator state $\rho$ in closed evolution for the shown run (line) and as mean $\pm$ s.d. over the "
       rf"{n} starts (band, limited to the physical range from 1/2 to 1). A purity below 1 signals qubit--oscillator "
       r"entanglement, and maximal entanglement gives $1/2$ (dotted line). Numbered markers and vertical dotted lines "
       r"give the times $t_k=kT/5$ of the Wigner functions in Fig.~4(d). Under the noise of "
       rf"Eq.~\eqref{{eq:lindblad}} the shown runs reach $F_T={cs['F_rep']:.3f}$ (CS), {cd['F_rep']:.3f} (CD), and "
       rf"{sn['F_rep']:.3f} (SNAP), against {PM['cs']}, {PM['cd']}, and {PM['snap']} over all {n} starts, including "
       rf"{['no', 'one', 'two', 'three'][sn['n_trivial']]} SNAP starts that stay at the vacuum." + "}\n"
       r"\label{fig:S25}" + "\n")
(HERE / "caption.tex").write_text(cap)
print(cap)
for g in GATES:
    print(g, f[g])
