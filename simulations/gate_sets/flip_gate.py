"""Round 3 (Oct 28): the logical flip as a gate.

A logical flip must act on the whole code space C = span{|0_L>, |1_L>}, not only on |0_L>.  The flip brachistochrone
H_opt = -i omega (|0_L><1_L^perp| - h.c.) leaves C invariant and rotates every code state along a great circle, so the
gate version of the tracking policy asks the pulse to carry the three inputs |0_L>, |1_L> and |+_L> ~ |0_L> + |1_L>
(ancilla in |g>) along their images U_opt(s theta)|mu>, s = t/T.  The superposition input fixes the relative phase and
penalizes an ancilla left entangled with the logical state.  Objective: the mean over the three inputs of the tracking
loss of src/src/common/tracking.py (same hinge, thresholds, stage schedule A-E, polish and weight ladder, same
bounds |Omega| <= 10 g, |xi_q|, |eps_q| <= g, chi_d = g, N slices), optimized at Fock cutoff 60.

Two modes, 8 random starts each (seeds 3000-3007), CS, ECD generator and SNAP, T = 412 and 824 ns (stage D without the
free progress profile):
  track    the gate-tracking policy; every rung of the weight ladder is kept (no fidelity threshold is applied)
  target   the target-only gate optimization (tracking weight 0; final hinge only, stages of capability_noisy.py)
Every pulse (every rung in track mode) is evaluated at cutoff 90 by the independent integrator of
analytics_audit/review_oct28 (Lindblad, Yale rates): the channel on the code span from |0><0|, |1><1|, |0><1| (x) |g><g|,
the six-state average fidelity with X_L after the best virtual frame update e^{i phi Z_L/2} (robustness.gate_metrics),
and the closed-evolution closeness of each input.
Writes sims/results/flip_gate.jsonl.
  OMP_NUM_THREADS=1 python simulations/gate_sets/flip_gate.py --workers 60 [--smoke]
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

MF = _L.vfile(__file__).parents[1]
MS = MF.parent / "manuscript"
for p in (MS, MS / "fig4_cs_path_following", MF / "analytics_audit" / "review_oct28"):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from common import grape, states, tracking  # noqa: E402
from common.params import G_SQ, GKP_DELTA, OMEGA_REF  # noqa: E402
import compute as CP  # noqa: E402
import robustness as RB  # noqa: E402

OUT = MF / "sims" / "results" / "flip_gate.jsonl"
D_OPT = 60
SMOKE = "--smoke" in sys.argv
GATES = ("cs", "cd", "snap")
TGS = (3.09, 6.18)                      # 412 and 824 ns (m = 1.40 and 2.81 T*); 2133 ns omitted (three inputs, N = 100)
SEEDS = range(3000, 3008)


def code(D):
    """real code words with <0_L|1_L> = c > 0, |1_L^perp>, theta, and the three inputs"""
    k0 = np.real(states.gkp(GKP_DELTA, 0, D))
    k1 = np.real(states.gkp(GKP_DELTA, 1, D))
    k0 /= np.linalg.norm(k0)
    k1 /= np.linalg.norm(k1)
    if k0 @ k1 < 0:
        k1 = -k1
    c = float(k0 @ k1)
    e1 = (k1 - c * k0) / math.sqrt(1 - c * c)
    plus = (k0 + k1) / np.linalg.norm(k0 + k1)
    return k0, k1, e1, c, math.acos(c), {"0": k0, "1": k1, "+": plus}


def image_path(v, k0, e1, theta, m=801):
    """U_opt(s theta)|v> for s in [0, 1]: rotation exp(-i theta s sigma_y) on the plane (k0, e1)"""
    a, b = v @ k0, v @ e1
    s = np.linspace(0, 1, m)[:, None]
    ca, sa = np.cos(s * theta), np.sin(s * theta)
    return ((a * ca - b * sa) * k0[None, :] + (a * sa + b * ca) * e1[None, :]).astype(complex)


def gate_cost(model, x, N, dt, psis, paths, w, thr, w_fin, thr_fin, lam_top=1e3, free=False):
    tot, grad, Fs = 0.0, 0.0, []
    for psi0, path in zip(psis, paths):
        c, g, info = tracking.cost_grad(model, x, N, dt, psi0, path, w, thr, w_fin, thr_fin, lam_top, free)
        tot += c / len(psis)
        grad = grad + g / len(psis)
        Fs.append(info["F"])
    return tot, grad, dict(F=np.array(Fs))


def gate_stage(model, X, N, T, psis, paths, thr, w, w_fin, thr_fin, iters, free=False, d0=None):
    dt = T / N
    bounds = [(0.0, 1.0) if j % 2 == 0 else (None, None) for _ in range(N) for j in range(6)]
    x0 = X.ravel()
    if free:
        d0 = np.ones(N) if d0 is None else d0
        x0 = np.r_[x0, d0]
        bounds = bounds + [(1e-3, None)] * N
    fun = lambda x: gate_cost(model, x, N, dt, psis, paths, w, thr, w_fin, thr_fin, 1e3, free)[:2]  # noqa: E731
    r = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                 options=dict(maxiter=iters, maxfun=2 * iters, ftol=1e-14, gtol=1e-10, maxcor=30))
    c, _, info = gate_cost(model, r.x, N, dt, psis, paths, w, thr, w_fin, thr_fin, 1e3, free)
    F = info["F"]
    return r.x[:6 * N].reshape(N, 6), dict(cost=c, F_T=F[:, -1].tolist(), C=F.min(1).tolist(), nit=int(r.nit))


def evaluate(X, gate, T):
    """cutoff 90, independent integrator: closed and Lindblad gate metrics"""
    B, _ = RB.code()
    dt = T / len(X)
    from scipy.sparse.linalg import expm_multiply
    Psi = np.stack([RB.joint(B[0]), RB.joint(B[1])], axis=1)
    nmax = 0.0
    for x in X:
        Psi = expm_multiply((-1j * RB.LC.hamiltonian(gate, x)).tocsr() * dt, Psi)
        nmax = max(nmax, max(RB.pops(Psi[:, c])[0] for c in range(2)))
    D = RB.D
    A = np.array([[[np.vdot(B[k], Psi[q * D:(q + 1) * D, i]) for i in range(2)] for k in range(2)] for q in range(2)])
    closed = RB.gate_metrics(np.einsum("qki,qlj->ijkl", A, A.conj()))
    u0, u1 = RB.joint(B[0]), RB.joint(B[1])
    V0 = np.stack([np.outer(u0, u0.conj()).ravel(), np.outer(u1, u1.conj()).ravel(), np.outer(u0, u1.conj()).ravel()],
                  axis=1)
    V = RB.lindblad_final(X, gate, T, V0)
    lind = RB.gate_metrics(RB.K_from_osc(RB.osc_of(V[:, 0]), RB.osc_of(V[:, 1]), RB.osc_of(V[:, 2])))
    keep = ("avg6", "avg6_opt", "phi_opt", "F_0to1", "F_1to0")
    return dict(closed={k: closed[k] for k in keep}, lind={k: lind[k] for k in keep}, nbar_max=float(nmax))


def run(case):
    mode, gate, Tg, seed = case
    t0 = time.time()
    T, N = Tg / G_SQ, CP.NSL[Tg]
    M = tracking.make_model(gate, D_OPT, G_SQ, 10 * G_SQ)
    k0, k1, e1, c, theta, inp = code(D_OPT)
    names = ("0", "1", "+")
    psis = [grape.joint(inp[nm].astype(complex)) for nm in names]
    paths = [tracking.Path(image_path(inp[nm], k0, e1, theta)) for nm in names]
    X = grape.random_start(N, np.random.default_rng(seed))
    it = (lambda n: 3) if SMOKE else (lambda n: n)
    out = dict(mode=mode, gate=gate, T_ns=1e3 * T, N=N, seed=seed, theta=theta, c=c)
    if mode == "target":
        for wf, n in ((200.0, 800), (2e4, 800), (1e5, 1200)):
            X, inf = gate_stage(M, X, N, T, psis, paths, 0.0, 0.0, wf, 0.99999, it(n))
        out.update(F_T_D60=inf["F_T"], C_D60=inf["C"], eval=evaluate(X, gate, T), X=X.tolist())
    else:
        sched = tracking.SCHEDULE if not SMOKE else tracking.SCHEDULE[:1] + tracking.SCHEDULE[-1:]
        for (nm, frac, thr, w, wf, thf, n, free) in sched:
            X, _ = gate_stage(M, X, N, frac * T, psis, paths, thr, w, wf, thf, it(n), free=False)
        pol = tracking.POLISH
        rungs = []
        for w in (tracking.LADDER if not SMOKE else tracking.LADDER[:2]):
            X, inf = gate_stage(M, X, N, T, psis, paths, pol["thr"], w, pol["w_fin"], pol["thr_fin"], it(pol["iters"]))
            rungs.append(dict(w=w, F_T_D60=inf["F_T"], C_D60=inf["C"], eval=evaluate(X, gate, T), X=X.tolist()))
        out.update(rungs=rungs)
    out["wall_s"] = time.time() - t0
    return out


if __name__ == "__main__":
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 32
    cases = [(mode, gate, Tg, seed) for Tg in (TGS if not SMOKE else TGS[:1]) for mode in ("target", "track")
             for gate in GATES for seed in (SEEDS if not SMOKE else SEEDS[:1])]
    cases.sort(key=lambda c: (-c[2], c[0] != "track"))           # longest jobs first
    out = OUT if not SMOKE else OUT.with_name("flip_gate_smoke.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and not SMOKE:
        for line in open(out):
            r = json.loads(line)
            done.add((r["mode"], r["gate"], round(r["T_ns"]), r["seed"]))
    cases = [c for c in cases if (c[0], c[1], round(1e3 * c[2] / G_SQ), c[3]) not in done]
    print(f"{len(cases)} cases", flush=True)
    with mp.Pool(W) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(run, cases):
            f.write(json.dumps(r) + "\n")
            f.flush()
            ev = r["eval"] if r["mode"] == "target" else r["rungs"][-1]["eval"]
            print(r["mode"], r["gate"], round(r["T_ns"]), r["seed"], f"{r['wall_s']:.0f}s",
                  f"avg6_opt lind={ev['lind']['avg6_opt']:.4f} closed={ev['closed']['avg6_opt']:.4f}", flush=True)
    if not SMOKE:
        open(OUT.with_suffix(".done"), "w").write("done\n")
