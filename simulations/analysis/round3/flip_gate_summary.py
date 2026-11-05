"""Round 3: summary of sims/results/flip_gate.jsonl, the logical flip as a gate.
target  six-state average fidelity with X_L after the best frame update (Lindblad, Yale rates, cutoff 90): best, mean
        +- s.d. over the starts; closed-evolution value; F(|0_L> -> |1_L>) and F(|1_L> -> |0_L>)
track   the same at every weight of the ladder, and the minimum closeness over the three inputs (|0_L>, |1_L>, |+_L>)
        to their brachistochrone images (closed evolution, cutoff 60)
Writes flip_gate_summary.json.
  python simulations/analysis/round3/flip_gate_summary.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib

import numpy as np

import mfcommon as mf

HERE = _L.vfile(__file__).parent
rows = [json.loads(l) for l in open(mf.MF / "sims" / "results" / "flip_gate.jsonl")]
out = {}


def ms(v):
    v = np.asarray(v, float)
    return dict(best=float(v.max()), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)))


for mode in ("target", "track"):
    for gate in ("cs", "cd", "snap"):
        for T in (412, 824):
            rs = [r for r in rows if r["mode"] == mode and r["gate"] == gate and round(r["T_ns"]) == T]
            if not rs:
                continue
            key = f"{mode}/{gate}/{T}"
            if mode == "target":
                ev = [r["eval"] for r in rs]
                out[key] = dict(avg6=ms([e["lind"]["avg6_opt"] for e in ev]), avg6_closed=ms([e["closed"]["avg6_opt"] for e in ev]),
                                F01=ms([e["lind"]["F_0to1"] for e in ev]), F10=ms([e["lind"]["F_1to0"] for e in ev]),
                                C=ms([min(r["C_D60"]) for r in rs]), nbar_max=ms([e["nbar_max"] for e in ev]))
                o = out[key]
                print(f"{key}: avg6 best {o['avg6']['best']:.4f} mean {o['avg6']['mean']:.4f}+-{o['avg6']['sd']:.4f} (n={o['avg6']['n']}) "
                      f"closed {o['avg6_closed']['mean']:.4f}; F01 {o['F01']['mean']:.3f} F10 {o['F10']['mean']:.3f}")
            else:
                W = [g["w"] for g in rs[0]["rungs"]]
                A = np.array([[g["eval"]["lind"]["avg6_opt"] for g in r["rungs"]] for r in rs])
                C = np.array([[min(g["C_D60"]) for g in r["rungs"]] for r in rs])
                F01 = np.array([[g["eval"]["lind"]["F_0to1"] for g in r["rungs"]] for r in rs])
                out[key] = dict(W=W, avg6_mean=A.mean(0).tolist(), avg6_sd=A.std(0, ddof=1).tolist() if len(rs) > 1 else [0] * len(W),
                                avg6_best=A.max(0).tolist(), C_mean=C.mean(0).tolist(), C_sd=C.std(0, ddof=1).tolist() if len(rs) > 1 else [0] * len(W),
                                F01_mean=F01.mean(0).tolist(), n=len(rs))
                o = out[key]
                print(f"{key} (n={len(rs)}): " + "  ".join(f"W={w:g}: avg6 {a:.4f} [{b:.4f}] C {c:.3f}" for w, a, b, c in
                                                          zip(W, o["avg6_mean"], o["avg6_best"], o["C_mean"]) if w in (10, 150, 2000)))
(HERE / "flip_gate_summary.json").write_text(json.dumps(out, indent=1))
