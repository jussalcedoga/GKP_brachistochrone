"""Supplementary Table 15: target fidelity of GKP preparation optimized for the target alone (no path), mean (s.d.)
[best] over 8 starts, under the Lindblad equation with the device rates at Fock cutoff 90.

Reads data/manuscript_figures/sims/results/capability_noisy.jsonl (written by simulations/gate_sets/capability_noisy.py),
writes the LaTeX rows to output/tables/target_only_preparation.tex and compares every row with the printed table
(tab:capability), read from data/paper_quotes/ or from the paper sources if GKP_PAPER_TEX is set. The s.d. is the
sample s.d. (ddof = 1) in units of the last printed digit of the mean, with the decimals of the printed table (4 for CS,
3 otherwise). Exit status 1 on any mismatch.

Usage:  python tables/target_only_preparation.py
"""
import json
import re
import statistics as st
import sys

import layout as L

ROWS = [("cs", "CS", 4), ("cd", "CD", 3), ("snap", r"SNAP, $\chi_d=g$", 3), ("snap4", r"SNAP, $\chi_d=4g$", 3)]
rows = [json.loads(l) for l in open(L.DATA / "manuscript_figures/sims/results/capability_noisy.jsonl")]


def cell(gate, T, nd):
    v = [r["F90_noisy"] for r in rows if r["gate"] == gate and round(r["T_ns"]) == T]
    m, s, b = st.mean(v), st.stdev(v), max(v)
    return f"{m:.{nd}f}({round(s * 10 ** nd):d}) [{b:.3f}]", len(v)


out = []
for gate, label, nd in ROWS:
    (c412, n1), (c824, n2) = cell(gate, 412, nd), cell(gate, 824, nd)
    assert n1 == n2 == 8, (gate, n1, n2)
    out.append(f"{label} & {c412} & {c824}\\\\")
txt = "\n".join(out) + "\n"
(L.out_dir("tables") / "target_only_preparation.tex").write_text(txt)
print(txt, end="")

tex = L.paper_text("supplement.tex")
blk = tex[tex.index(r"\label{tab:capability}"):]
blk = blk[:blk.index(r"\end{tabular}")]
printed = {l.split("&")[0].strip(): l.strip() for l in blk.splitlines() if re.match(r"^(CS|CD|SNAP)", l.strip())}
fails = []
for line in out:
    lab = line.split("&")[0].strip()
    ok = printed.get(lab) == line
    print(f"{'PASS' if ok else 'FAIL'}  {lab}: printed {printed.get(lab)!r}")
    if not ok:
        fails.append(lab)
print(f"{len(fails)} mismatch(es)" if fails else "SUPPLEMENTARY TABLE 15 MATCHES THE DATA (every cell)")
sys.exit(1 if fails else 0)
