"""Round 3 (Oct 28): the benchmark without a fidelity threshold.

Every run of Fig. 4 (GKP preparation along the brachistochrone; CS, ECD, SNAP; 412, 824, 2133 ns; 20 starts) polished
its pulse at each tracking weight of the ladder W = 10, 20, ..., 2000 and stored, for every rung, the closed-evolution
minimum closeness C and the Lindblad target fidelity F_T (cutoff 60, Yale rates).  The threshold F_T >= 0.99 only selected
which rung is shown in Fig. 4.  Here every rung of every start is used: for each gate set, duration and weight, mean and
s.d. over the starts of C and F_T (bisection rungs excluded; trivial SNAP runs included), and the pairwise comparison at
equal weight.  Writes tradeoff.json next to this file.
  python simulations/analysis/round3/tradeoff.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib

import numpy as np

import mfcommon as mf

HERE = _L.vfile(__file__).parent
JOBS = mf.MS / "fig4_cs_path_following" / "results" / "jobs80"
LADDER = (10.0, 20.0, 40.0, 70.0, 150.0, 300.0, 600.0, 1000.0, 2000.0)


def main():
    G = mf.groups()
    out = {}
    for k in (1, 2, 3):
        for gate in ("cs", "cd", "snap"):
            runs = [json.loads((JOBS / f"{r['job']['id']}.json").read_text()) for r in G[("gkp", "geodesic", gate, k)]]
            C = np.full((len(runs), len(LADDER)), np.nan)
            F = np.full_like(C, np.nan)
            for i, r in enumerate(runs):
                for rung in r["ladder"]:
                    if rung.get("bisection"):
                        continue
                    j = LADDER.index(float(rung["w"]))
                    C[i, j], F[i, j] = rung["C_ends"], rung["F_loss"]
            T_ns = 1e3 * runs[0]["T_us"]
            out[f"{gate}/k{k}"] = dict(T_ns=T_ns, n=len(runs), W=list(LADDER),
                                       C_mean=np.nanmean(C, 0).tolist(), C_sd=np.nanstd(C, 0, ddof=1).tolist(),
                                       F_mean=np.nanmean(F, 0).tolist(), F_sd=np.nanstd(F, 0, ddof=1).tolist(),
                                       C_best=np.nanmax(C, 0).tolist(), F_best=np.nanmax(F, 0).tolist(),
                                       C=C.tolist(), F=F.tolist())
    for k in (1, 2, 3):
        cs = out[f"cs/k{k}"]
        print(f"T = {cs['T_ns']:.0f} ns")
        for gate in ("cs", "cd", "snap"):
            o = out[f"{gate}/k{k}"]
            print(f"  {gate:4s} " + "  ".join(f"W={w:g}: C={c:.3f} F={f:.4f}" for w, c, f in
                                             zip(LADDER, o["C_mean"], o["F_mean"]) if w in (10, 150, 2000)))
        for gate in ("cd", "snap"):
            o = out[f"{gate}/k{k}"]
            # does CS at its lowest-closeness rung (W = 10) already follow more closely, and arrive better, than the
            # other gate set at its highest-closeness rung (W = 2000) and at its best-fidelity rung?
            jF = int(np.nanargmax(o["F_mean"]))
            print(f"  CS(W=10) vs {gate}: C {cs['C_mean'][0]:.3f} vs max {np.nanmax(o['C_mean']):.3f}; "
                  f"F_T {cs['F_mean'][0]:.4f} vs best rung {o['F_mean'][jF]:.4f} (W={LADDER[jF]:g}); "
                  f"CS dominates at every W: {all(cs['C_mean'][j] > o['C_mean'][j] and cs['F_mean'][j] > o['F_mean'][j] for j in range(len(LADDER)))}")
    (HERE / "tradeoff.json").write_text(json.dumps(out))
    print("wrote", HERE / "tradeoff.json")


if __name__ == "__main__":
    main()
