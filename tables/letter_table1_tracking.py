"""Table I of the Letter: errors and closeness of the GKP preparation runs of Fig. 4 (CS, CD and SNAP along the
brachistochrone at 412, 824 and 2133 ns, 20 random starts each), recomputed from the stored Lindblad traces.

For every run, the stored scalars are first checked against the run's own traces: the minimum closeness C equals the
minimum of the closeness trace FL, and the final fidelity F_T equals the last point of the target-fidelity trace FTL.
Each cell is then the mean (sample s.d.) over the 20 starts, with no start excluded:
  * eps_1, the first-order error of the path itself [Eq. (7) of the Letter], from the Fig. 4 facts (linear in T);
  * 1 - F_T in percent, two significant digits, s.d. in units of the last digit;
  * C with two decimals, s.d. in units of the last digit.
Rounding is half up on the decimal value of the binary double.  The rows are written as LaTeX and CSV next to this
script, and compared cell by cell with the printed table (tab:tracking) of the paper sources.

Usage:  python tables/letter_table1_tracking.py      (exit status 1 on any mismatch; outputs in output/)
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import math
import pathlib
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

R = _L.DATA                                                      # stored results (src/layout.py)
OUTD = _L.vfile(__file__).parent
TR = R / "manuscript" / "fig4_cs_path_following" / "results" / "traces80"
FACTS = R / "manuscript_figures" / "fig4" / "fig4_facts.json"
TEX = _L.paper_text("main.tex")
TGS = (2.58, 3.09, 6.18, 16.0)                                    # grid jobs: the duration index k of their T_g
GATES = ("cs", "cd", "snap")
NAME = {"cs": "CS", "cd": "CD", "snap": "SNAP"}
fails = []


def check(ok, what):
    print(("PASS  " if ok else "FAIL  ") + what)
    if not ok:
        fails.append(what)


# runs of GKP preparation along the brachistochrone, grouped by gate set and duration index k = 1, 2, 3 (as load.py)
groups = {}
for p in sorted(TR.glob("*.json")):
    r = json.loads(p.read_text())
    jb = r["job"]
    if jb.get("name") != "gkp" or jb.get("ref") != "geodesic":
        continue
    if jb["set"] in ("main", "si"):
        if jb["Tg"] not in TGS:
            continue
        k, seed = TGS.index(jb["Tg"]), 2
    elif jb["set"] == "seed":
        k, seed = jb["k"], jb["seed"]
    else:
        continue
    if k in (1, 2, 3):
        groups.setdefault((jb["gate"], k), []).append((seed, r))

stats = {}
for (g, k), lst in sorted(groups.items()):
    seeds = [s for s, _ in lst]
    check(len(lst) == 20 and len(set(seeds)) == 20, f"{NAME[g]}, k={k}: 20 runs with distinct seeds")
    dC = max(abs(r["C_noisy"] - min(r["FL"])) for _, r in lst)
    dF = max(abs(r["F_T_final_noisy"] - r["FTL"][-1]) for _, r in lst)
    check(dC == 0 and dF == 0, f"{NAME[g]}, k={k}: stored C and F_T equal the traces (max diff {max(dC, dF):.1e})")
    C = np.array([r["C_noisy"] for _, r in lst])
    F = np.array([r["F_T_final_noisy"] for _, r in lst])
    stats[(g, k)] = dict(C=(C.mean(), C.std(ddof=1)), F=(F.mean(), F.std(ddof=1)), T=lst[0][1]["T_ns"], n=len(lst))

facts = json.loads(FACTS.read_text())["columns"]
for (g, k), s in stats.items():                                   # the same statistics as the figure
    c = facts[str(k)][g]
    d = max(abs(s["C"][0] - c["C_mean"]), abs(s["C"][1] - c["C_sd"]), abs(s["F"][0] - c["F_T_mean"]),
            abs(s["F"][1] - c["F_T_sd"]))
    check(d < 1e-12, f"{NAME[g]}, k={k}: equal to the Fig. 4 facts (max diff {d:.1e})")
eps = {k: facts[str(k)]["eps1_path"] for k in (1, 2, 3)}
rat = [eps[k] / facts[str(k)]["T_ns"] for k in (1, 2, 3)]
check(max(rat) - min(rat) <= 1e-12 * max(rat), "eps_1 of the path is linear in T")


def q(x, d):
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)


def two_sig(x):
    return 1 - math.floor(math.log10(abs(x)))


def ms(m, s, d):
    """mean with d decimals and the s.d. in units of the last digit"""
    ss = int(q(float(s) * 10 ** d, 0))
    return f"{q(m, d)}({ss})" if ss > 0 else f"{q(m, d)}"


cols = []
for k in (1, 2, 3):
    e = 100 * eps[k]
    col = {"T_ns": f"{facts[str(k)]['T_ns']:.0f}", "eps1_pct": f"{q(e, two_sig(e))}"}
    for g in GATES:
        mF, sF = stats[(g, k)]["F"]
        col[g + "_infid_pct"] = ms(100 * (1 - mF), 100 * sF, two_sig(100 * (1 - mF)))
        col[g + "_C"] = ms(*stats[(g, k)]["C"], 2)
    cols.append(col)

rows = [r" & \multicolumn{3}{c}{$1-F_T$ (\%)} & \multicolumn{3}{c}{$C$}\\\cline{2-4}\cline{5-7}",
        r"$T$ (ns) & " + " & ".join(c["T_ns"] for c in cols * 2) + r"\\\hline",
        r"$\epsilon_1$ (\%) & " + " & ".join(c["eps1_pct"] for c in cols) + r" & & & \\"]
rows += [rf"{NAME[g]} & " + " & ".join(c[g + "_infid_pct"] for c in cols) + " & " + " & ".join(c[g + "_C"] for c in cols)
         + r"\\" for g in GATES]
tex_out = "\n".join(rows) + "\n"
(OUTD / "table1_tracking.tex").write_text(tex_out)
with open(OUTD / "table1_tracking.csv", "w") as f:              # unrounded statistics, for any further audit
    f.write("gate_set,T_ns,n_starts,eps1_path,F_T_mean,F_T_sd,C_mean,C_sd\n")
    for (g, k), s in sorted(stats.items(), key=lambda kv: (GATES.index(kv[0][0]), kv[0][1])):
        f.write(f"{NAME[g]},{s['T']:.6f},{s['n']},{eps[k]:.12g},{s['F'][0]:.12g},{s['F'][1]:.12g},"
                f"{s['C'][0]:.12g},{s['C'][1]:.12g}\n")
print(tex_out, end="")

# the printed table, cell by cell
if r"\label{tab:tracking}" in TEX:
    blk = TEX[TEX.index(r"\label{tab:tracking}"):]
    blk = blk[:blk.index(r"\end{tabular}")]
    def cells(line):
        return [x.strip().rstrip("\\").strip() for x in line.split("&")]

    printed = {cells(l)[0]: cells(l) for l in blk.splitlines() if re.match(r"^(\$\\epsilon_1\$|CS |CD |SNAP )", l.strip())}
    for line in rows[2:]:                                         # the eps_1 row and one row per gate set
        c_ = cells(line.strip())
        check(printed.get(c_[0]) == c_, f"row '{c_[0]}': recomputed {c_[1:]}, printed "
                                         f"{printed[c_[0]][1:] if c_[0] in printed else 'missing'}")
    check(len(printed) == len(rows) - 2, f"{len(printed)} printed rows, {len(rows) - 2} recomputed")
else:
    check(False, "Table tab:tracking not found in the paper source")
print(f"\n{len(fails)} mismatch(es): {fails}" if fails else "\nTABLE I MATCHES THE DATA (every cell)")
sys.exit(1 if fails else 0)
