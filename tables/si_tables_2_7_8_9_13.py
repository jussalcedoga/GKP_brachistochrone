"""Supplementary Tables 2, 7, 8, 9 and 13, rebuilt from the stored results and checked against the printed rows.

  Table 2   photon-optimal preparations and transfers through the vacuum     fig2.json, fig2_extra.json,
                                                                             audit_photon2.json, transfer_closed.json
  Table 7   final states of GKP preparation (stabilizers, photon number)     robustness.json (checks, item 2)
  Table 8   energy uncertainty and photon number of the representative pulses robustness.json (item 4)
  Table 9   stored pulses under closed evolution and three noise models      robustness.json (item 6), rates_parity.json
  Table 13  the transfer pulses at m = 2.81 applied to the code space        robustness.json (item 1)

Every cell is written from the data in the notation and at the precision of the printed cell (a mean with its s.d. in
parentheses in units of the last digit, the best start in brackets, 1x10^-5 style powers, "<10^-8", "--") and compared
with it. The printed rows are read from data/paper_quotes/supplement_quoted_rows.tex (the tabular blocks of the SI), or
from supplement.tex of the paper if GKP_PAPER_TEX names its folder. Two cells of Table 2 are computed here from stored
curves: the brachistochrone exposure of the preparation of |1_L>, J_geo = n (theta/2 - sin(2 theta)/4)/sin^2(theta)/omega
(SI Sec. II.1), and its bound B = int_0^theta m(cos^2 r) dr (Eq. (5) of the Letter, as in src/common/photon_core.py),
from the minimal-photon curve m(F) of |1_L> stored in fig2_extra.json. The same two formulas applied to |0_L> are
compared with the stored values as a check.

Usage (repository root, after `source env.sh`):  python tables/si_tables_2_7_8_9_13.py
Writes output/tables/si_tables_2_7_8_9_13.tex (the regenerated rows); exit status 1 if any cell differs.
"""
import json
import math
import re
import sys

import numpy as np

import layout as _L

D = _L.DATA
AA = D / "manuscript_figures" / "analytics_audit"
SUPP = _L.paper_text("supplement.tex")


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


class V:
    """a value: mean, optional sample s.d. and best start"""

    def __init__(self, mean, sd=None, best=None):
        self.mean, self.sd, self.best = float(mean), sd, best

    @classmethod
    def stat(cls, s, best=None):
        return cls(s["mean"], s["sd"], s["max"] if best is None else best)


NUM = re.compile(r"^(\$?)(-?\d+(?:\.(\d+))?)(?:\((\d+)\))?(\$?)(?:\s*\[(\d+(?:\.(\d+))?)\])?$")
POW = re.compile(r"^\$(?:(\d)\\times)?10\^\{(-?\d+)\}\$$")
BELOW = re.compile(r"^\$<10\^\{(-?\d+)\}\$$")


def power(x, times):
    e = math.floor(math.log10(x))
    m = round(x / 10 ** e)
    if m == 10:
        m, e = 1, e + 1
    return f"${m}\\times10^{{{e}}}$" if (times or m != 1) else f"$10^{{{e}}}$"


def render(printed, v):
    """the value v (V, a label string, or None for an empty cell) in the notation and precision of the printed cell"""
    p = printed.strip()
    if isinstance(v, str):
        return v
    if v is None:
        return p if p in ("--", "") else "(no value)"
    m = BELOW.match(p)
    if m:
        return p if v.mean < 10.0 ** int(m.group(1)) else power(v.mean, True)
    m = POW.match(p)
    if m:
        return power(v.mean, m.group(1) is not None)
    m = NUM.match(p)
    if not m:
        return f"(cannot parse '{p}')"
    d = len(m.group(3) or "")
    s = f"{v.mean:.{d}f}"
    if m.group(4) is not None:
        s += f"({int(round(v.sd * 10 ** d))})"
    s = m.group(1) + s + m.group(5)
    if m.group(6) is not None:
        s += f" [{v.best:.{len(m.group(7) or '')}f}]"
    return s


def printed_rows(label):
    """data rows of the printed table \\label{tab:<label>}, as lists of stripped cells"""
    blk = SUPP[SUPP.index(r"\label{tab:%s}" % label):]
    blk = blk[:blk.index(r"\end{tabular}")].splitlines()
    head = next(i for i, l in enumerate(blk) if r"\hline" in l)
    out = []
    for l in blk[head + 1:]:
        if "&" not in l:
            continue
        l = l.strip()
        l = re.sub(r"\\\\(\\hline)?$", "", l).strip()
        out.append([c.strip() for c in l.split("&")])
    return out


TEX = []
nfail = 0


def check(num, label, rows):
    global nfail
    pr = printed_rows(label)
    TEX.append(f"% Supplementary Table {num} (tab:{label})")
    if len(pr) != len(rows):
        print(f"FAIL  Supplementary Table {num}: {len(rows)} rows regenerated, {len(pr)} printed")
        nfail += 1
    for mine, theirs in zip(rows, pr):
        cells = [render(t, v) for t, v in zip(theirs, mine)]
        bad = [(c, t) for c, t in zip(cells, theirs) if c != t] + ([("(cells)", f"{len(mine)} vs {len(theirs)}")]
                                                                  if len(mine) != len(theirs) else [])
        TEX.append(" & ".join(cells) + r"\\")
        name = ", ".join(c for c in cells[:2] if c and not NUM.match(c))
        if bad:
            nfail += 1
            print(f"FAIL  Supplementary Table {num:2d}  {name}: " + "; ".join(f"data {c} vs printed {t}" for c, t in bad))
        else:
            print(f"PASS  Supplementary Table {num:2d}  {name}: " + " & ".join(cells[1:]))


# ------------------------------------------------------------------ Table 2 (tab:vacuum)
F2 = load(D / "manuscript/fig2_photon_optimal/results/fig2.json")
X = load(D / "manuscript_figures/fig2/results/fig2_extra.json")
A2 = load(AA / "photon_flip/results/audit_photon2.json")
TC = load(AA / "round4/results/transfer_closed.json")
OM = F2["params"]["omega"] / 1e3                       # omega in 1/ns


def j_geo(m):
    th = math.acos(math.sqrt(m["F_vacuum"]))
    return m["n_target"] * (th / 2 - math.sin(2 * th) / 4) / math.sin(th) ** 2 / OM


def bound(m):
    r = np.linspace(0, math.acos(math.sqrt(m["F_vacuum"])), 4001)
    return float(np.trapezoid(np.interp(np.cos(r) ** 2, m["F"], m["m"], left=0.0), r)) / OM


G = F2["prep"]["gkp"]
print(f"check of the two formulas on |0_L>: J_geo {j_geo(X['m_0L']):.2f} (stored {G['geodesic']['J_ns']:.2f}), "
      f"B {bound(X['m_0L']):.2f} (stored {X['gkp_bound_ns']:.2f}) photon ns")
rows = [["preparation of $\\ket{0_L}$", V(G["geodesic"]["J_ns"]), V(G["photon"]["J_ns"]), None,
         V(X["modal_gkp"]["J_ns"]), V(X["closed_gkp"]["J_ns"]), V(X["gkp_bound_ns"])],
        ["preparation of $\\ket{1_L}$", V(j_geo(X["m_1L"])), V(A2["flip_D0.4"]["J_prep_f_ns"]), None,
         V(X["flip_modal"]["J_ns"] - X["modal_gkp"]["J_ns"]), V(X["flip_closed"]["J_ns"] - X["closed_gkp"]["J_ns"]),
         V(bound(X["m_1L"]))]]
for lab, ka, kt, b in (("$\\ket{0_L}\\to\\ket{1_L}$, $\\Delta=0.4$", "flip_D0.4", "flip, Delta=0.4", X["flip_bound"]["B2_ns"]),
                       ("$\\ket{0_L}\\to\\ket{1_L}$, $\\Delta=0.3$", "flip_D0.3", "flip, Delta=0.3", None),
                       ("$\\ket{0_L}\\to\\ket{+_L}$", "0L_to_plusL", "|0_L> -> |+_L>", None),
                       ("cat $\\gamma=2\\to\\ket{0_L}$", "cat_to_0L", "cat(2) -> |0_L>", None)):
    a = A2[ka]
    rows.append([lab, V(a["J_geo_ns"]), V(a["J_free_ns"]), V(a["J_sum_ns"]), V(TC[kt]["modal"]["J_ns"]),
                 V(TC[kt]["one-mode"]["J_ns"]), None if b is None else V(b)])
check(2, "vacuum", rows)

# ------------------------------------------------------------------ Tables 7, 8, 9, 13 (robustness.json)
RB = load(AA / "review_oct28/robustness.json")
RP = load(AA / "round3/rates_parity.json")["groups"]
GL = {"cs": "CS", "cd": "CD", "snap": "SNAP"}
KT = (("k1", 412), ("k2", 824), ("k3", 2133))
Q = ("F_T", "S_X", "S_Z", "Z_L", "nbar")                # Z_L = <D(beta)>, the logical Z displacement

rows = [[lab] + [V(RB["checks"][key][q]) for q in Q] for lab, key in (("target $\\ket{0_L}$", "target_0L"), ("vacuum", "vacuum"))]
for g in GL:
    for k, T in KT:
        s = RB["item2_gkp_quality"][f"{g}/{k}"]["stats"]
        rows.append([f"{GL[g]}, {T}~ns"] + [V.stat(s[q]) for q in Q])
check(7, "stab", rows)

P4 = RB["item4_resources"]["pulses"]
rows = []
for name, nl in (("gkp", "GKP"), ("flip", "transfer")):
    for T in ((412, 824) if name == "gkp" else (824,)):
        for g in GL:
            c = [p for p in P4 if p["gate"] == g and p["name"] == name and round(p["T_ns"]) == T]
            if len(c) > 1:                             # SNAP at 412 ns: the representative of the non-trivial starts
                c = [p for p in c if p["nbar_max"] > 1e-6]
            p = c[0]
            rows.append([f"{GL[g]}, {nl}, {T}~ns" + ("$^a$" if (g, name, T) == ("snap", "gkp", 412) else "")]
                        + [V(p[q]) for q in ("dH_osc_avg", "dH_osc_max", "nbar_max", "nbar_avg", "P70_at_peak")])
# printed order: CS, CD, SNAP at 412 ns, then at 824 ns, then the transfer
check(8, "resources", rows)

G6 = RB["item6_csproposal_noise"]["groups"]
rows = []
for name, nl, ks in (("gkp", "GKP", KT), ("flip", "transfer", (("k2", 824),))):
    for g in GL:
        for k, T in ks:
            key = f"{name}/{g}/{k}"
            rows.append([f"{nl}, {GL[g]}, {T}~ns", V.stat(G6[key]["F_closed"]), V.stat(G6[key]["F_yale"]),
                         V.stat(RP[key]["ramsey_F"]), V.stat(G6[key]["F_csp"])])
check(9, "proposal", rows)

I1 = RB["item1_flip_gate"]
rows = []
for ref, rl in (("geodesic", "brachistochrone"), ("tradeoff", "trade-off"), ("photon", "photon optimal")):
    for i, g in enumerate(GL):
        L = I1["groups"][f"{ref}/{g}"]["lind"]
        rows.append([rl if i == 0 else "", GL[g], V.stat(L["F_0to1"]), V.stat(L["F_1to0"]),
                     V.stat(L["avg6_opt"], best=L["best"]["avg6_opt"])])
h = next(x for x in I1["H_opt"] if round(x["T_ns"]) == 824)["lind"]
rows.append(["$H_{\\rm opt}$, 824~ns", "", V(h["F_0to1"]), V(h["F_1to0"]), V(h["avg6_opt"])])
check(13, "gate", rows)

out = _L.out_dir("tables") / "si_tables_2_7_8_9_13.tex"
out.write_text("\n".join(TEX) + "\n", encoding="utf-8")
print(f"\nwrote {out}")
print(f"{nfail} row(s) differ from the printed tables" if nfail else
      "ALL ROWS OF SUPPLEMENTARY TABLES 2, 7, 8, 9 AND 13 MATCH THE DATA")
sys.exit(1 if nfail else 0)
