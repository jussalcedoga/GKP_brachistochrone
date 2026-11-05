"""Oct 28: target-only capability of the gate sets under the budget of Fig. 4, at the rigor of the main
benchmark.  Optimization with the tracking weight W = 0 (final hinge only; the stages of manuscript/tools/capability.py)
at Fock cutoff 60, closed evolution; evaluation of the final pulse at cutoff 90, closed and under the Lindblad equation
with the Yale rates; 8 random starts per case (seeds 1000-1007, independent of the benchmark starts).
Cases: GKP preparation (vacuum -> |0_L>, Delta = 0.4) at 412 and 824 ns for CS, ECD, SNAP (chi_d = g) and SNAP with
chi_d = 4 g.  Separates 'cannot follow the path' from 'cannot reach the target in time T', and tests how SNAP depends on
its dispersive shift.
OMP_NUM_THREADS=1 python simulations/gate_sets/capability_noisy.py --workers 32"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import multiprocessing as mp
import pathlib
import sys

MF = _L.vfile(__file__).parents[1]
MS = MF.parent / "manuscript"
sys.path.insert(0, str(MS))
sys.path.insert(0, str(MS / "fig4_cs_path_following"))
import numpy as np  # noqa: E402

from common import grape, lindblad, tracking  # noqa: E402
from common.params import G_SQ  # noqa: E402
import compute as CP  # noqa: E402

OUT = MF / "sims" / "results" / "capability_noisy.jsonl"
GATES = {"cs": ("cs", None), "cd": ("cd", None), "snap": ("snap", 1.0), "snap4": ("snap", 4.0)}
CASES = [(gk, Tg, seed) for Tg in (3.09, 6.18) for gk in GATES for seed in range(1000, 1008)]
ZERO = dict(kappa=0.0, kphi=0.0, g1=0.0, gphi=0.0, kerr=0.0)


def model(gk, D):
    gate, chi = GATES[gk]
    if gate == "snap":
        return tracking.SNAPModel(D, G_SQ, 10 * G_SQ, chi=chi * G_SQ)
    return tracking.make_model(gate, D, G_SQ, 10 * G_SQ)


def run(case):
    gk, Tg, seed = case
    T, N = Tg / G_SQ, CP.NSL[Tg]
    M = model(gk, 60)
    S, Tt = CP.transfer("gkp", 60)
    path = tracking.Path(np.array([S, Tt]))                    # unused when the tracking weight is zero
    X = grape.random_start(N, np.random.default_rng(seed))
    for wf, it in ((200.0, 800), (2e4, 800), (1e5, 1200)):
        X, _, inf = tracking.stage(M, X, N, T, grape.joint(S), path, 0.0, 0.0, wf, 0.99999, it)
    M9 = model(gk, 90)
    S9, T9 = CP.transfer("gkp", 90)
    F_closed = tracking.lindblad_F(M9, X, T, N, grape.joint(S9), T9, ZERO)
    F_noisy = tracking.lindblad_F(M9, X, T, N, grape.joint(S9), T9, lindblad.rates("Yale"))
    return dict(gate=gk, T_ns=1e3 * T, seed=seed, F60=float(inf["F_T"]), F90_closed=float(F_closed),
                F90_noisy=float(F_noisy))


if __name__ == "__main__":
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 32
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with mp.Pool(W) as pool, open(OUT, "w") as f:
        for r in pool.imap_unordered(run, CASES):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r, flush=True)
    rows = [json.loads(l) for l in open(OUT)]
    print("\nGKP preparation, target only (W = 0), Lindblad (Yale) at cutoff 90: best / mean of 8 starts")
    for T in sorted({r["T_ns"] for r in rows}):
        for gk in GATES:
            v = np.array([r["F90_noisy"] for r in rows if r["gate"] == gk and r["T_ns"] == T])
            print(f"  T={T:5.0f} ns {gk:6s}: best {v.max():.4f}  mean {v.mean():.4f} +- {v.std():.4f}  (n={len(v)})")
    open(MF / "sims" / "results" / "capability_noisy.done", "w").write("done\n")
