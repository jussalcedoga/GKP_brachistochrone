"""Round 3 (Oct 28): SNAP--displacement control without the zero-drive trap.

In the benchmark, every SNAP start for the binomial word at 412 and 824 ns, and 13 and 2 of the 20 GKP starts at 412 and
824 ns, stayed at the zero-drive point, which is stationary for the tracking objective when the oscillator starts in the
vacuum (a displacement creates odd-parity components orthogonal to every parity-even path state).  Those runs measure the
optimizer, not the gate set.  Here every SNAP run (chi_d = g, same bounds, slices and stage machinery) first optimizes the
target alone (tracking weight 0; the stages of capability_noisy.py), which leaves the zero-drive point, and then follows
the brachistochrone along the weight ladder W = 10 ... 2000 warm-started from that pulse (the polish stage of the
benchmark at every rung).  The rung shown is chosen by the rule of the benchmark: the largest W whose pulse keeps
F_T >= 0.99 under the Lindblad equation (cutoff 60), else the best F_T.  Cases: binomial word at 412, 824, 2133 ns and GKP
|0_L> at 412 and 824 ns, 8 starts each (seeds 4000-4007).  Every chosen pulse is then traced at cutoff 90 by
simulations/tracking/traces.py (imported, with its folders redirected), in the format of Fig. 4.
Writes sims/results/snap_warm/{jobs,traces,rho}/ and sims/results/snap_warm/summary.jsonl.
  OMP_NUM_THREADS=1 python simulations/gate_sets/snap_warm.py --workers 16
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
import traces as TR  # noqa: E402

OUT = MF / "sims" / "results" / "snap_warm"
JOBS, TRACES, RHO = OUT / "jobs", OUT / "traces", OUT / "rho"
ZERO = dict(kappa=0.0, kphi=0.0, g1=0.0, gphi=0.0, kerr=0.0)
CASES = ([("binomial", k, s) for k in (3, 2, 1) for s in range(4000, 4008)]
         + [("gkp", k, s) for k in (2, 1) for s in range(4000, 4008)])


def run(case):
    name, k, seed = case
    t0 = time.time()
    Tg = CP.TG[k]
    T, N = Tg / G_SQ, CP.NSL[Tg]
    jid = f"warm_{name}_geodesic_snap_k{k}_s{seed}"
    M = tracking.make_model("snap", 60, G_SQ, 10 * G_SQ)
    S, Tt = CP.transfer(name, 60)
    psi0 = grape.joint(S)
    path = tracking.Path(CP.reference_dense(name, "geodesic", 60))
    noise = lindblad.rates("Yale")
    X = grape.random_start(N, np.random.default_rng(seed))
    for wf, it in ((200.0, 800), (2e4, 800), (1e5, 1200)):          # target only
        X, _, inf = tracking.stage(M, X, N, T, psi0, path, 0.0, 0.0, wf, 0.99999, it)
    M9 = tracking.make_model("snap", 90, G_SQ, 10 * G_SQ)
    S9, T9 = CP.transfer(name, 90)
    target_only = dict(F60=float(inf["F_T"]), F90_closed=float(tracking.lindblad_F(M9, X, T, N, grape.joint(S9), T9, ZERO)),
                       F90_noisy=float(tracking.lindblad_F(M9, X, T, N, grape.joint(S9), T9, noise)))
    pol = tracking.POLISH
    rungs, Xs = [], {}
    for w in tracking.LADDER:                                        # warm-started tracking ladder
        X, _, inf = tracking.stage(M, X, N, T, psi0, path, pol["thr"], w, pol["w_fin"], pol["thr_fin"], pol["iters"])
        Fl = float(tracking.lindblad_F(M, X, T, N, psi0, Tt, noise))
        rungs.append(dict(w=w, F_T=float(inf["F_T"]), C_ends=float(inf["C"]), F_loss=Fl))
        Xs[w] = X.copy()
    ok = [r for r in rungs if r["F_loss"] >= tracking.F_LOSS_MIN]
    w_star = max(r["w"] for r in ok) if ok else max(rungs, key=lambda r: r["F_loss"])["w"]
    job = dict(set="seed", name=name, ref="geodesic", gate="snap", k=k, seed=seed, g=G_SQ, noise="Yale", Tg=Tg,
               pace="path", id=jid, protocol="warm start from the target-only optimum")
    res = dict(job=job, T_us=T, N=N, w_star=w_star, ladder=rungs, target_only=target_only, X=Xs[w_star].tolist(),
               wall_s=time.time() - t0)
    (JOBS / f"{jid}.json").write_text(json.dumps(res))
    TR.JOBS, TR.OUT, TR.RHO = JOBS, TRACES, RHO
    _, msg = TR.run(jid)
    tr = json.loads((TRACES / f"{jid}.json").read_text())
    return dict(id=jid, name=name, k=k, T_ns=1e3 * T, seed=seed, target_only=target_only, w_star=w_star,
                C=tr["C"], C_noisy=tr["C_noisy"], F_T=tr["F_T_final"], F_T_noisy=tr["F_T_final_noisy"],
                wall_s=time.time() - t0)


if __name__ == "__main__":
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16
    for d in (JOBS, TRACES, RHO):
        d.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in TRACES.glob("*.json")}
    cases = [c for c in CASES if f"warm_{c[0]}_geodesic_snap_k{c[1]}_s{c[2]}" not in done]
    print(f"{len(cases)} cases", flush=True)
    with mp.Pool(W) as pool, open(OUT / "summary.jsonl", "a") as f:
        for r in pool.imap_unordered(run, cases):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["id"], f"{r['wall_s']:.0f}s target-only {r['target_only']['F90_noisy']:.4f} | W*={r['w_star']:g} "
                  f"C={r['C']:.3f} F_T(noisy)={r['F_T_noisy']:.4f}", flush=True)
    open(OUT / "done", "w").write("done\n")
