"""Round 3 (Oct 28): conditional squeezing exactly as proposed by Del Grosso et al., PRA 111, 042606 (2025), Eq. (2): the
squeeze acts on one qubit branch only, (i/2)(xi^* a^2 - xi a^dag^2) (x) |e><e|, |xi| <= g = 7.5 us^-1 (CSModel model
"onesided"), with the same qubit drive |Omega| <= 10 g.  The benchmark of the manuscript lets both branches squeeze
independently (model "two"); this run tests whether the conclusions hold for the proposed single-branch gate.
GKP preparation (vacuum -> |0_L>, Delta = 0.4):
  target  target-only optimization (tracking weight 0; the stages of capability_noisy.py), T = 412, 824 ns, seeds
          1000-1007 (the starts of capability_noisy.py); evaluated at cutoff 90, closed and Lindblad (Yale rates)
  track   the tracking policy along the brachistochrone (stages A-E of tracking.SCHEDULE and the polish ladder), T = 412,
          824, 2133 ns, seeds 1-8; every rung kept: closed F_T and C at cutoff 60 and the Lindblad F_T at cutoff 60, as in
          the ladders of the main benchmark (results/jobs80), plus closed C, F_T and Lindblad F_T at cutoff 90
Writes sims/results/cs_onesided.jsonl.
  OMP_NUM_THREADS=1 python simulations/gate_sets/cs_onesided.py --workers 16
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import multiprocessing as mp
import pathlib
import sys
import time

MF = _L.vfile(__file__).parents[1]
MS = MF.parent / "manuscript"
for p in (MS, MS / "fig4_cs_path_following"):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402

from common import grape, lindblad, tracking  # noqa: E402
from common.params import G_SQ  # noqa: E402
import compute as CP  # noqa: E402

OUT = MF / "sims" / "results" / "cs_onesided.jsonl"
ZERO = dict(kappa=0.0, kphi=0.0, g1=0.0, gphi=0.0, kerr=0.0)


def model(D):
    return grape.CSModel(D, G_SQ, 10 * G_SQ, "onesided")


def run(case):
    mode, Tg, seed = case
    t0 = time.time()
    T, N = Tg / G_SQ, CP.NSL[Tg]
    noise = lindblad.rates("Yale")
    M = model(60)
    S, Tt = CP.transfer("gkp", 60)
    psi0 = grape.joint(S)
    M9 = model(90)
    S9, T9 = CP.transfer("gkp", 90)
    X = grape.random_start(N, np.random.default_rng(seed))
    out = dict(mode=mode, T_ns=1e3 * T, N=N, seed=seed)
    if mode == "target":
        path = tracking.Path(np.array([S, Tt]))
        for wf, it in ((200.0, 800), (2e4, 800), (1e5, 1200)):
            X, _, inf = tracking.stage(M, X, N, T, psi0, path, 0.0, 0.0, wf, 0.99999, it)
        out.update(F60=float(inf["F_T"]),
                   F90_closed=float(tracking.lindblad_F(M9, X, T, N, grape.joint(S9), T9, ZERO)),
                   F90_noisy=float(tracking.lindblad_F(M9, X, T, N, grape.joint(S9), T9, noise)))
    else:
        dense = CP.reference_dense("gkp", "geodesic", 60)
        path = tracking.Path(dense)
        path9 = tracking.Path(CP.pad(dense, 90))
        for (nm, frac, thr, w, wf, thf, it, free) in tracking.SCHEDULE:
            X, _, inf = tracking.stage(M, X, N, frac * T, psi0, path, thr, w, wf, thf, it, free)
        pol = tracking.POLISH
        rungs = []
        for w in tracking.LADDER:
            X, _, inf = tracking.stage(M, X, N, T, psi0, path, pol["thr"], w, pol["w_fin"], pol["thr_fin"], pol["iters"])
            d9 = tracking.diagnostics(M9, X, T, N, grape.joint(S9), path9, T9)
            rungs.append(dict(w=w, F_T=float(inf["F_T"]), C_ends=float(inf["C"]),
                              F_loss=float(tracking.lindblad_F(M, X, T, N, psi0, Tt, noise)),
                              C90=float(d9["C"]), F_T90=float(d9["F_T"]),
                              F_loss90=float(tracking.lindblad_F(M9, X, T, N, grape.joint(S9), T9, noise)),
                              X=X.tolist()))
        out.update(rungs=rungs)
    out["wall_s"] = time.time() - t0
    return out


if __name__ == "__main__":
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16
    cases = ([("track", Tg, s) for Tg in (16.0, 6.18, 3.09) for s in range(1, 9)]
             + [("target", Tg, s) for Tg in (3.09, 6.18) for s in range(1000, 1008)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["mode"], round(r["T_ns"]), r["seed"]))
    cases = [c for c in cases if (c[0], round(1e3 * c[1] / G_SQ), c[2]) not in done]
    print(f"{len(cases)} cases", flush=True)
    with mp.Pool(W) as pool, open(OUT, "a") as f:
        for r in pool.imap_unordered(run, cases):
            f.write(json.dumps(r) + "\n")
            f.flush()
            last = r.get("F90_noisy", r.get("rungs", [{}])[-1].get("F_loss90"))
            print(r["mode"], round(r["T_ns"]), r["seed"], f"{r['wall_s']:.0f}s", last, flush=True)
    open(OUT.with_suffix(".done"), "w").write("done\n")
