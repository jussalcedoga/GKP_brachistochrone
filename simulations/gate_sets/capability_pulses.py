"""Pulses of the target-only GKP preparation at 412 ns (Letter Table I, label tab:capability), with the pulses kept.

sims/capability_noisy.py wrote the fidelities of Table I but not the pulses.  This script repeats its 412 ns cases
exactly (its model(), GATES, ZERO and transfers are imported from capability_noisy.py, and its run() is copied below
without any change of the numerics) for CS, ECD, SNAP (chi_d = g) and SNAP (chi_d = 4 g), seeds 1000-1007, and also
stores, for every start,
  X      the final slice variables (N rows of r, f_Omega, a_g, f_g, a_e, f_e; Omega = 10 g r e^{i f_Omega},
         branch amplitude g a_j e^{i f_j}; for SNAP (a_g, f_g) is the cavity drive varepsilon and (a_e, f_e) is unused),
  C      the Cartesian controls of the model (Omega_x, Omega_y, Re, Im of the g-branch amplitude, Re, Im of the e-branch
         amplitude) in 1/us, from model.controls(X).
The fidelities F60, F90_closed, F90_noisy are recomputed exactly as in capability_noisy.py, so that each start can be
compared with capability_noisy.jsonl (never overwritten).
Writes sims/results/pulses/capability_pulses.jsonl (--smoke: capability_pulses_smoke.jsonl, one case).
  OMP_NUM_THREADS=1 python simulations/gate_sets/capability_pulses.py --workers 32 [--smoke]
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import multiprocessing as mp
import pathlib
import sys
import time

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE))
import capability_noisy as CN  # noqa: E402  (sets the paths of manuscript/ and imports compute as CN.CP)
import numpy as np  # noqa: E402

from common import grape, lindblad, tracking  # noqa: E402
from common.params import G_SQ  # noqa: E402

OUTDIR = CN.MF / "sims" / "results" / "pulses"
SMOKE = "--smoke" in sys.argv
OUT = OUTDIR / ("capability_pulses_smoke.jsonl" if SMOKE else "capability_pulses.jsonl")
TG = 3.09                                                   # 412 ns
CASES = [(gk, TG, seed) for gk in CN.GATES for seed in range(1000, 1008)]
COLS = ["Omega_x", "Omega_y", "Re_g", "Im_g", "Re_e", "Im_e"]


def run(case):
    """capability_noisy.run, unchanged numerics, plus the pulse"""
    gk, Tg, seed = case
    t0 = time.time()
    T, N = Tg / G_SQ, CN.CP.NSL[Tg]
    M = CN.model(gk, 60)
    S, Tt = CN.CP.transfer("gkp", 60)
    path = tracking.Path(np.array([S, Tt]))                    # unused when the tracking weight is zero
    X = grape.random_start(N, np.random.default_rng(seed))
    for wf, it in ((200.0, 800), (2e4, 800), (1e5, 1200)):
        X, _, inf = tracking.stage(M, X, N, T, grape.joint(S), path, 0.0, 0.0, wf, 0.99999, it)
    M9 = CN.model(gk, 90)
    S9, T9 = CN.CP.transfer("gkp", 90)
    F_closed = tracking.lindblad_F(M9, X, T, N, grape.joint(S9), T9, CN.ZERO)
    F_noisy = tracking.lindblad_F(M9, X, T, N, grape.joint(S9), T9, lindblad.rates("Yale"))
    C, _ = M9.controls(X)
    gate, chi = CN.GATES[gk]
    return dict(gate=gk, T_ns=1e3 * T, seed=seed, F60=float(inf["F_T"]), F90_closed=float(F_closed),
                F90_noisy=float(F_noisy), model=gate, chi_d_over_g=chi, N=int(N), dt_ns=1e3 * T / N, g=G_SQ,
                Omega_max=10 * G_SQ, X_cols=["r", "f_Omega", "a_g", "f_g", "a_e", "f_e"], X=X.tolist(), C_cols=COLS,
                C=C.tolist(), wall_s=time.time() - t0)


if __name__ == "__main__":
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 32
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases = CASES[:1] if SMOKE else CASES
    if OUT.exists() and not SMOKE:                           # resume: keep finished starts
        done = {(r["gate"], r["seed"]) for r in map(json.loads, open(OUT))}
        cases = [c for c in cases if (c[0], c[2]) not in done]
    print(f"{len(cases)} cases, {min(W, max(len(cases), 1))} workers", flush=True)
    with mp.Pool(min(W, max(len(cases), 1))) as pool, open(OUT, "a" if not SMOKE else "w") as f:
        for r in pool.imap_unordered(run, cases):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["gate"], r["seed"], f"{r['wall_s']:.0f}s F60={r['F60']:.10f} F90_closed={r['F90_closed']:.10f} "
                  f"F90_noisy={r['F90_noisy']:.10f}", flush=True)
    if not SMOKE:
        open(OUT.with_suffix(".done"), "w").write("done\n")
