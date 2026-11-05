"""Conditional squeezing with the Hamiltonian of the cited proposal.

Del Grosso et al., PRA 111, 042606 (2025), Eq. (2), in the frame rotating at the cavity frequency of the squeezed
branch:  H_I = |e><e| (x) (i/2)(xi^* a^2 - xi a^dag^2) + Delta n (x) |g><g|,
with chi/2pi = 8 MHz (chi a^dag a sigma_z), hence Delta = 2 chi = 2 pi x 16 MHz = 100.5 us^-1 (their Delta t = 20.1 in
200 ns), and g_d = 7.5 us^-1 (r = 1.5 in 200 ns).  The manuscript's single-branch model ('onesided') omits the Delta term.
A transmon with chi_full = 2 pi x 16 MHz and anharmonicity 150 MHz also imprints a cavity self-Kerr
K ~ chi_full^2 / (4 alpha) = 2 pi x 0.43 MHz = 2.68 us^-1 (the proposal's own Eq. (30): K/omega ~ 1e-4).

Same protocol as simulations/gate_sets/cs_onesided.py, mode 'target' (target-only, W = 0, stages
(200, 800), (2e4, 800), (1e5, 1200) at cutoff 60, seeds 1000-1007, T = 412 and 824 ns, N = 50, |Omega| <= 10 g), with a
drift H0 added to the model:
  disp       H0 = Delta |g><g| (x) n
  disp_kerr  H0 = Delta |g><g| (x) n - (K/2) 1 (x) a^dag^2 a^2
Evaluated at cutoff 90, closed and Lindblad (Yale rates).  Compare with cs_onesided.jsonl (same seeds, H0 = 0).
  OMP_NUM_THREADS=1 python simulations/analysis/cs_proposal/cs_proposal_hamiltonian.py --workers 16
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

HERE = _L.vfile(__file__).parent
MF = HERE.parents[2]
MS = MF.parent / "manuscript"
for p in (MS, MS / "fig4_cs_path_following"):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402

from common import grape, lindblad, tracking  # noqa: E402
from common.params import G_SQ  # noqa: E402
import compute as CP  # noqa: E402

OUT = HERE / "cs_proposal_hamiltonian.jsonl"
DELTA = 2 * 2 * math.pi * 8.0          # rad/us: 2 chi, chi/2pi = 8 MHz
KERR = 2 * math.pi * (16.0 ** 2 / (4 * 150.0))   # rad/us: chi_full^2/(4 alpha), MHz -> rad/us
ZERO = dict(kappa=0.0, kphi=0.0, g1=0.0, gphi=0.0, kerr=0.0)


def model(D, case):
    M = grape.CSModel(D, G_SQ, 10 * G_SQ, "onesided")
    n = np.diag(np.arange(D)).astype(complex)
    a = np.diag(np.sqrt(np.arange(1, D)), 1).astype(complex)
    Pg = np.diag([1.0, 0.0]).astype(complex)
    H0 = DELTA * np.kron(Pg, n)
    if case == "disp_kerr":
        H0 = H0 - 0.5 * KERR * np.kron(np.eye(2), a.conj().T @ a.conj().T @ a @ a)
    M.H0 = H0
    return M


def run(job):
    case, Tg, seed = job
    t0 = time.time()
    T, N = Tg / G_SQ, CP.NSL[Tg]
    M = model(60, case)
    S, Tt = CP.transfer("gkp", 60)
    psi0 = grape.joint(S)
    path = tracking.Path(np.array([S, Tt]))
    X = grape.random_start(N, np.random.default_rng(seed))
    for wf, it in ((200.0, 800), (2e4, 800), (1e5, 1200)):
        X, _, inf = tracking.stage(M, X, N, T, psi0, path, 0.0, 0.0, wf, 0.99999, it)
    M9 = model(90, case)
    S9, T9 = CP.transfer("gkp", 90)
    out = dict(case=case, T_ns=1e3 * T, seed=seed, F60=float(inf["F_T"]),
               F90_closed=float(tracking.lindblad_F(M9, X, T, N, grape.joint(S9), T9, ZERO)),
               F90_noisy=float(tracking.lindblad_F(M9, X, T, N, grape.joint(S9), T9, lindblad.rates("Yale"))),
               wall_s=time.time() - t0, X=X.tolist())
    return out


if __name__ == "__main__":
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16
    jobs = [(c, Tg, s) for c in ("disp", "disp_kerr") for Tg in (3.09, 6.18) for s in range(1000, 1008)]
    print(f"Delta = {DELTA:.2f} us^-1 = {DELTA / G_SQ:.2f} g;  K = {KERR:.3f} us^-1 = {KERR / G_SQ:.3f} g", flush=True)
    with mp.Pool(W) as pool, open(OUT, "w") as f:
        for r in pool.imap_unordered(run, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["case"], round(r["T_ns"]), r["seed"], f"{r['wall_s']:.0f}s", f"F60={r['F60']:.4f}",
                  f"closed90={r['F90_closed']:.4f} noisy90={r['F90_noisy']:.4f}", flush=True)
    rows = [json.loads(l) for l in open(OUT)]
    base = [json.loads(l) for l in open(MF / "sims" / "results" / "cs_onesided.jsonl")]
    base = [r for r in base if r["mode"] == "target"]
    print("\nGKP preparation, target only, single-branch CS; Lindblad (Yale) at cutoff 90: mean +- sd [best] of 8 starts")
    for T in (412.0, 824.0):
        v = np.array([r["F90_noisy"] for r in base if round(r["T_ns"]) == T])
        print(f"  T={T:.0f} ns  manuscript model (H0=0): {v.mean():.4f} +- {v.std(ddof=1):.4f} [{v.max():.4f}]")
        for c in ("disp", "disp_kerr"):
            v = np.array([r["F90_noisy"] for r in rows if r["case"] == c and round(r["T_ns"]) == T])
            vc = np.array([r["F90_closed"] for r in rows if r["case"] == c and round(r["T_ns"]) == T])
            if len(v):
                print(f"  T={T:.0f} ns  {c:10s}: noisy {v.mean():.4f} +- {v.std(ddof=1):.4f} [{v.max():.4f}]   "
                      f"closed {vc.mean():.4f} [{vc.max():.4f}]  (n={len(v)})")
    open(OUT.with_suffix(".done"), "w").write("done\n")
