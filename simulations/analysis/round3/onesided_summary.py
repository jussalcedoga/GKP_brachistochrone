"""Round 3: summary of sims/results/cs_onesided.jsonl (single-branch conditional squeezing, as proposed in Del Grosso et al.)
against the two-branch CS of the benchmark: target-only fidelity (best, mean +- s.d.) and, along the weight ladder, the
closeness and noisy fidelity (cutoff 60, as stored by the benchmark) at W = 10 and W = 2000, with the two-branch values
from tradeoff.json and the ECD values for reference.  Writes onesided_summary.json.
  python simulations/analysis/round3/onesided_summary.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib

import numpy as np

import mfcommon as mf

HERE = _L.vfile(__file__).parent
rows = [json.loads(l) for l in open(mf.MF / "sims" / "results" / "cs_onesided.jsonl")]
two = json.loads((HERE / "tradeoff.json").read_text())
cap = [json.loads(l) for l in open(mf.MF / "sims" / "results" / "capability_noisy.jsonl")]
out = {"target": {}, "track": {}}
for T in sorted({round(r["T_ns"]) for r in rows if r["mode"] == "target"}):
    v = np.array([r["F90_noisy"] for r in rows if r["mode"] == "target" and round(r["T_ns"]) == T])
    w = np.array([r["F90_noisy"] for r in cap if r["gate"] == "cs" and round(r["T_ns"]) == T])
    out["target"][T] = dict(one_best=float(v.max()), one_mean=float(v.mean()), one_sd=float(v.std(ddof=1)), n=len(v),
                            two_best=float(w.max()), two_mean=float(w.mean()))
    print(f"target {T} ns: single-branch best {v.max():.4f} mean {v.mean():.4f}+-{v.std(ddof=1):.4f} (n={len(v)}); "
          f"two-branch best {w.max():.4f} mean {w.mean():.4f}")
for k, T in ((1, 412), (2, 824), (3, 2133)):
    rs = [r for r in rows if r["mode"] == "track" and round(r["T_ns"]) == T]
    if not rs:
        continue
    C = np.array([[g["C_ends"] for g in r["rungs"]] for r in rs])
    F = np.array([[g["F_loss"] for g in r["rungs"]] for r in rs])
    C9 = np.array([[g["C90"] for g in r["rungs"]] for r in rs])
    F9 = np.array([[g["F_loss90"] for g in r["rungs"]] for r in rs])
    t2, cd = two[f"cs/k{k}"], two[f"cd/k{k}"]
    dom = bool(np.all((C.mean(0) > np.array(cd["C_mean"])) & (F.mean(0) > np.array(cd["F_mean"]))))
    out["track"][T] = dict(n=len(rs), C=C.mean(0).tolist(), F=F.mean(0).tolist(), C_sd=C.std(0, ddof=1).tolist(),
                           F_sd=F.std(0, ddof=1).tolist(), C90=C9.mean(0).tolist(), F90=F9.mean(0).tolist(),
                           two_C=t2["C_mean"], two_F=t2["F_mean"], ecd_C=cd["C_mean"], ecd_F=cd["F_mean"],
                           dominates_ecd_every_W=dom)
    print(f"track {T} ns (n={len(rs)}): single-branch W=10 C={C.mean(0)[0]:.3f} F={F.mean(0)[0]:.4f}, W=2000 "
          f"C={C.mean(0)[-1]:.3f} F={F.mean(0)[-1]:.4f} | two-branch W=10 C={t2['C_mean'][0]:.3f} F={t2['F_mean'][0]:.4f}, "
          f"W=2000 C={t2['C_mean'][-1]:.3f} F={t2['F_mean'][-1]:.4f} | ECD W=10 C={cd['C_mean'][0]:.3f} F={cd['F_mean'][0]:.4f}"
          f" | single-branch dominates ECD at every W: {dom}")
(HERE / "onesided_summary.json").write_text(json.dumps(out, indent=1))
