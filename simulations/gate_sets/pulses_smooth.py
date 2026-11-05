"""Smoothing robustness of the piecewise-constant reference pulses (data for the pulse section of the SI).

The pulses of the paper are piecewise constant on N slices of length Delta t = T/N (SI Sec. VII), a modelling choice,
whereas a real drive has a finite bandwidth.  This script tests how much the results depend on the steps.  Every
Cartesian control c(t) of a pulse (Omega_x, Omega_y and the real and imaginary parts of the two branch amplitudes; for
SNAP the cavity drive) is convolved with a normalized Gaussian kernel of standard deviation sigma = 1, 2, 4 ns,
    c_sigma(t) = int dt' K_sigma(t - t') c(t'),     K_sigma(u) = exp(-u^2 / 2 sigma^2) / (sqrt(2 pi) sigma),
with zero drive before t = 0 and after T.  The pulse keeps its duration T, so the convolution also lowers the integrated
drive within a few sigma of both ends.  The convolution of the step pulse is evaluated in closed form and averaged over
each of M sub-slices per slice (sub-slice length h = Delta t / M), which gives a piecewise-constant pulse with N M
sub-slices; the same sub-slice grid with sigma = 0 reproduces the original pulse exactly.  Each sub-slice value is a
combination of the slice values with non-negative weights of sum <= 1, so every smoothed control stays inside its bound
disk (|Omega| <= 10 g, |branch amplitude| <= g); this is checked numerically for every pulse.  The smoothed pulses are
re-evaluated with the propagators of the repository at Fock cutoff 90:
  (a) track   the Fig. 4 tracking pulses (GKP |0_L> along the brachistochrone; CS, ECD, SNAP; T = 412 and 824 ns; the
              20 starts of each, run selection of load.groups as in plot_fig4.py): F_T and the closeness C, under the
              Lindblad equation (common/lindblad.evolve, Yale rates, 8 record points per slice, as traces.py) and closed
              (CSModel.propagate);
  (b) target  the target-only pulses of Letter Table I at 412 ns (sims/capability_pulses.py: CS, ECD, SNAP with
              chi_d = g and 4 g, seeds 1000-1007): F_T under the Lindblad equation and closed, as capability_noisy.py;
  (c) flip    the target-only flip gate pulses at 824 ns (sims/flip_gate.py, mode 'target'; CS, ECD, SNAP; seeds
              3000-3007): the gate metrics of flip_gate.evaluate (six-state average fidelity with X_L after the best
              virtual Z_L update, Lindblad and closed, independent integrator of analytics_audit/review_oct28).
Diagnostic variant (edge 'hold'): the first and last slice values are continued outside [0, T] instead of zero, so the
integrated drive at the ends is kept and only the finite bandwidth acts.
Writes sims/results/pulses/pulses_smooth.jsonl (one line per pulse, sigma and edge; resumable).
  OMP_NUM_THREADS=1 python simulations/gate_sets/pulses_smooth.py --workers 32
        [--only track,flip,target] [--no-hold]
  ... --smoke   checks of the convolution, grid convergence and timing on a few pulses -> pulses_smooth_smoke.json
  Sub-slice length h = 0.2575 ns (M = 32 sub-slices at 412 ns, 64 at 824 ns), chosen from --smoke (grid convergence
  of the smoothed F_T and gate metric at h = 1.03, 0.515, 0.2575, 0.129 ns; numbers in pulses_facts_B.json).
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

HERE = _L.vfile(__file__).parent
MF = HERE.parent
MS = MF.parent / "manuscript"
for p in (MF / "analytics_audit" / "review_oct28", MS / "fig4_cs_path_following", MS, HERE):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from common import grape, lindblad, tracking  # noqa: E402
from common.params import G_SQ  # noqa: E402
import compute as CP  # noqa: E402
import load as LOAD  # noqa: E402
import capability_noisy as CN  # noqa: E402
import flip_gate as FG  # noqa: E402

RES = MF / "sims" / "results"
OUTDIR = RES / "pulses"
OUT = OUTDIR / "pulses_smooth.jsonl"
JOBS80 = MS / "fig4_cs_path_following" / "results" / "jobs80"
D = 90
SIGMAS = (0.0, 1.0, 2.0, 4.0)            # ns
H_NS = 0.2575                            # target sub-slice length (ns): M = 32 at 412 ns, 64 at 824 ns (see --smoke)
NOISE = lindblad.rates("Yale")
SUBL = 8                                 # record points per slice for the closeness (traces.py)
_CACHE = {}


# ----------------------------------------------------------------------------- the smoothing
def _R(u):
    """G(u) - max(u, 0), with G(u) = int_{-inf}^u Phi = u Phi(u) + phi(u): small and symmetric, phi(u) - |u| Phi(-|u|)"""
    a = np.abs(u)
    return np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi) - a * ndtr(-a)


def kernel_weights(N, M, T, sig, edge="zero"):
    """W (N M, N): cell average over sub-slice j of the Gaussian-smoothed indicator of slice k.
    edge 'zero': zero drive outside [0, T] (the main choice; rows near the ends sum to < 1);
    edge 'hold': the first and last slice values continued outside [0, T] (diagnostic; every row sums to 1), which
    separates the finite bandwidth from the lower integrated drive at the ends"""
    dt = T / N
    h = dt / M
    if sig == 0:
        return np.kron(np.eye(N), np.ones((M, 1)))
    a = np.arange(N * M)[:, None] * h
    b = a + h
    tk = np.arange(N)[None, :] * dt
    tk1 = tk + dt
    # overlap of the cell [a, b] with the slice [t_k, t_k+1] (cells are aligned with the slices)
    ov = (np.minimum(b, tk1) - np.maximum(a, tk)).clip(0) / h
    s = sig
    corr = (s / h) * (_R((b - tk) / s) - _R((a - tk) / s) - _R((b - tk1) / s) + _R((a - tk1) / s))
    W = ov + corr
    if edge == "hold":            # kernel mass left of 0 to slice 0, right of T to slice N-1 (cell averages)
        W[:, 0] += (s / h) * (_R(-a[:, 0] / s) - _R(-b[:, 0] / s))
        W[:, -1] += (s / h) * (_R((b[:, 0] - T) / s) - _R((a[:, 0] - T) / s))
    elif edge != "zero":
        raise ValueError(edge)
    return W


def smooth(C, T, sig_us, M, edge="zero"):
    """Gaussian-smoothed controls on the N M sub-slices (cell averages) and the weights"""
    W = kernel_weights(len(C), M, T, sig_us, edge)
    return W @ C, W


def to_polar(Cf, g=G_SQ):
    """Cartesian controls (n, 6) -> slice variables (r, f_Omega, a_g, f_g, a_e, f_e) of CSModel.controls"""
    X = np.zeros((len(Cf), 6))
    for j, b in ((0, 10 * g), (2, g), (4, g)):
        z = Cf[:, j] + 1j * Cf[:, j + 1]
        X[:, j], X[:, j + 1] = np.abs(z) / b, np.angle(z)
    return X


def bound_ratios(Cf, gate, g=G_SQ):
    """max |Omega| / 10 g and max |branch amplitude| / g (SNAP: the cavity drive only; its e columns are unused)"""
    om = np.abs(Cf[:, 0] + 1j * Cf[:, 1]).max() / (10 * g)
    br = np.abs(Cf[:, 2] + 1j * Cf[:, 3]).max() / g
    if gate in ("cs", "cd"):
        br = max(br, np.abs(Cf[:, 4] + 1j * Cf[:, 5]).max() / g)
    return float(om), float(br)


def rel_change(Cs, C0, gate):
    """relative L2 change of the qubit drive and of the oscillator controls on the sub-slice grid"""
    cols = [2, 3, 4, 5] if gate in ("cs", "cd") else [2, 3]
    out = {}
    for name, cc in (("drive", [0, 1]), ("osc", cols)):
        n0 = np.linalg.norm(C0[:, cc])
        out[name] = float(np.linalg.norm(Cs[:, cc] - C0[:, cc]) / n0) if n0 > 0 else 0.0
    return out


def m_for(T_us, N):
    """sub-slices per slice: the multiple of SUBL closest to Delta t / H_NS"""
    return max(SUBL, int(round(1e3 * T_us / N / H_NS / SUBL)) * SUBL)


def prepare(C, T, N, M, sig_ns, model, gate, edge="zero"):
    """smoothed pulse as slice variables of the model, with the checks"""
    Cs, W = smooth(C, T, 1e-3 * sig_ns, M, edge)
    Xf = to_polar(Cs)
    Cf, _ = model.controls(Xf)
    C0 = np.repeat(C, M, axis=0)
    chk = dict(w_min=float(W.min()), w_rowsum_max=float(W.sum(1).max()), polar_roundtrip=float(np.abs(Cf - Cs).max()),
               bound_Omega=None, bound_osc=None)
    chk["bound_Omega"], chk["bound_osc"] = bound_ratios(Cf, gate)
    chk["rel_dC"] = rel_change(Cf, C0, gate)
    chk["area_ratio_drive"] = float(np.abs(Cf[:, 0] + 1j * Cf[:, 1]).sum() / max(np.abs(C0[:, 0] + 1j * C0[:, 1]).sum(),
                                                                                 1e-300))
    return Xf, Cf, chk


# ----------------------------------------------------------------------------- evaluations
def osc_rho(psi):
    v = psi.reshape(2, D)
    return v[0][:, None] * v[0].conj()[None, :] + v[1][:, None] * v[1].conj()[None, :]


def gkp90():
    if "gkp" not in _CACHE:
        S9, T9 = CP.transfer("gkp", D)
        path9 = tracking.Path(CP.pad(CP.reference_dense("gkp", "geodesic", 80), D))
        _CACHE["gkp"] = (S9, T9, path9)
    return _CACHE["gkp"]


def eval_prep(model, Cf, T, N, M, closeness=True):
    """F_T (Lindblad, closed) and the closeness C (Lindblad, closed) of the smoothed pulse Cf (N M sub-slices)"""
    S9, T9, path9 = gkp90()
    psi0 = grape.joint(S9)
    h = T / (N * M)
    out = {}
    traj, _, _ = model.propagate(Cf, h, psi0)
    out["F_T_closed"] = float(np.real(np.vdot(T9, osc_rho(traj[-1]) @ T9)))
    rec = M // SUBL
    if closeness:
        s = np.arange(N * SUBL + 1) / (N * SUBL)
        G, _ = path9.at(s)
        F = [float(np.real(np.vdot(G[k], osc_rho(traj[k * rec]) @ G[k]))) for k in range(N * SUBL + 1)]
        out["C_closed"] = float(min(F))
        rhoT, recs = lindblad.evolve(model, Cf, h, psi0, NOISE, dt_max=2e-3, record_every=rec)
        FL = [float(np.real(np.vdot(G[k], ro @ G[k]))) for k, ro in enumerate([osc_rho(psi0)] + recs)]
        out["C_noisy"] = float(min(FL))
    else:
        rhoT = lindblad.evolve(model, Cf, h, psi0, NOISE, dt_max=2e-3)
    out["F_T_noisy"] = float(lindblad.osc_fidelity(rhoT, T9))
    return out


def track_cases():
    G = LOAD.groups("traces80")
    cases = []
    for gate in ("cs", "cd", "snap"):
        for k in (1, 2):
            for r in G[("gkp", "geodesic", gate, k)]:
                n, F = np.asarray(r["nbarL"]), np.asarray(r["FTL"])          # mfcommon.trivial, same formula
                triv = bool(np.max(np.abs(n - n[0])) < 1e-3 and np.max(np.abs(F - F[0])) < 1e-4)
                cases.append(dict(set="track", gate=gate, k=k, seed=int(r["seed"]), jid=r["job"]["id"],
                                  T_ns=float(r["T_ns"]), trivial=triv,
                                  stored=dict(F_T_noisy=r["F_T_final_noisy"], C_noisy=r["C_noisy"],
                                              F_T_closed=r["F_T_final"], C_closed=r["C"])))
    return cases


def target_cases():
    p = OUTDIR / "capability_pulses.jsonl"
    rows = [json.loads(l) for l in open(p)] if p.exists() else []
    return [dict(set="target", gate=r["gate"], seed=r["seed"], T_ns=r["T_ns"],
                 stored=dict(F_T_noisy=r["F90_noisy"], F_T_closed=r["F90_closed"])) for r in rows]


def flip_cases():
    out = []
    for l in open(RES / "flip_gate.jsonl"):
        r = json.loads(l)
        if r["mode"] == "target" and abs(r["T_ns"] - 824.0) < 1:
            out.append(dict(set="flip", gate=r["gate"], seed=r["seed"], T_ns=r["T_ns"],
                            stored=dict(lind=r["eval"]["lind"], closed=r["eval"]["closed"])))
    return out


def load_X(case):
    if case["set"] == "track":
        r = json.loads((JOBS80 / f"{case['jid']}.json").read_text())
        return np.array(r["X"]), float(r["T_us"])
    if case["set"] == "target":
        for l in open(OUTDIR / "capability_pulses.jsonl"):
            r = json.loads(l)
            if r["gate"] == case["gate"] and r["seed"] == case["seed"]:
                return np.array(r["X"]), 1e-3 * r["T_ns"]
    for l in open(RES / "flip_gate.jsonl"):
        r = json.loads(l)
        if r["mode"] == "target" and r["gate"] == case["gate"] and r["seed"] == case["seed"] and abs(r["T_ns"] - 824) < 1:
            return np.array(r["X"]), 1e-3 * r["T_ns"]
    raise KeyError(case)


def model_of(case, Dm=D):
    if case["set"] == "target":
        return CN.model(case["gate"], Dm)
    return tracking.make_model(case["gate"], Dm, G_SQ, 10 * G_SQ)


def run(task):
    case, sig, M_force = task[:3]
    edge = task[3] if len(task) > 3 else "zero"
    t0 = time.time()
    X, T = load_X(case)
    N = len(X)
    M = M_force or m_for(T, N)
    model = model_of(case)
    C, _ = model.controls(X)
    Xf, Cf, chk = prepare(C, T, N, M, sig, model, case["gate"], edge)
    out = {k: v for k, v in case.items() if k != "stored"}
    out.update(sigma_ns=sig, edge=edge, M=M, h_ns=1e3 * T / (N * M), N=N, checks=chk)
    if case["set"] == "flip":
        ev = FG.evaluate(X if sig == 0 and not M_force else Xf, case["gate"], T)
        out.update(eval=ev, grid="native N slices (exact per-slice integrator)" if sig == 0 and not M_force else "N M")
    else:
        out.update(eval_prep(model, Cf, T, N, M, closeness=(case["set"] == "track")))
    out["stored"] = case["stored"]
    out["wall_s"] = time.time() - t0
    return out


def key(r):
    return (r["set"], r["gate"], round(r["T_ns"]), r["seed"], r["sigma_ns"], r.get("edge", "zero"))


# ----------------------------------------------------------------------------- smoke: checks, convergence, timing
def smoke():
    res = dict()
    tc = {(c["gate"], c["k"], c["seed"]): c for c in track_cases()}
    # 1. closed-form convolution against a brute-force quadrature (CS 412 ns, representative start s11)
    c = tc[("cs", 1, 11)]
    X, T = load_X(c)
    N = len(X)
    model = model_of(c)
    C, _ = model.controls(X)
    M = m_for(T, N)
    chk = {}
    for sig in (1.0, 2.0, 4.0):
        Cs, W = smooth(C, T, 1e-3 * sig, M)
        tf = np.linspace(0, T, N * M * 40 + 1)                 # 40 quadrature points per sub-slice
        tm = 0.5 * (tf[1:] + tf[:-1])
        step = C[np.minimum((tm / (T / N)).astype(int), N - 1)]
        u = np.arange(-int(8 * sig / (1e3 * (tf[1] - tf[0]))), int(8 * sig / (1e3 * (tf[1] - tf[0]))) + 1) * (tf[1] - tf[0])
        K = np.exp(-0.5 * (u / (1e-3 * sig)) ** 2)
        K /= K.sum()                                           # discrete normalization (zero padding outside [0, T])
        conv = np.stack([np.convolve(step[:, j], K, mode="same") for j in range(6)], 1)
        cell = conv.reshape(N * M, 40, 6).mean(1)
        chk[str(sig)] = dict(max_abs_diff_over_bound=float(np.abs(cell - Cs).max() / G_SQ),
                             w_min=float(W.min()), rowsum_max=float(W.sum(1).max()), rowsum_min=float(W.sum(1).min()))
    res["convolution_check_cs_k1_s11"] = chk
    print("convolution check", chk, flush=True)
    # 2. grid convergence of the evaluation (sigma = 1 ns is the hardest) and timing
    tasks = []
    for (g, k, s) in (("cs", 1, 11), ("cd", 1, 9), ("snap", 2, 5), ("cs", 2, 16)):
        for M in ((16, 32, 64) if k == 1 else (32, 64, 128)):
            for sig in (0.0, 1.0):
                tasks.append((tc[(g, k, s)], sig, M))
    fc = {(c["gate"], c["seed"]): c for c in flip_cases()}
    for M in (16, 32):
        tasks.append((fc[("cs", 3000)], 1.0, M))
    tasks.append((fc[("cs", 3000)], 0.0, 16))                    # repeat grid vs the native evaluation
    tasks.append((fc[("cs", 3000)], 0.0, None))
    with mp.Pool(min(len(tasks), 30)) as pool:
        rows = pool.map(run, tasks)
    res["convergence"] = [{k: v for k, v in r.items() if k not in ("stored",)} | dict(stored=r["stored"]) for r in rows]
    for r in rows:
        if r["set"] == "flip":
            print(r["set"], r["gate"], r["seed"], "M", r["M"], "sig", r["sigma_ns"], f"{r['wall_s']:.0f}s",
                  "lind", r["eval"]["lind"]["avg6_opt"], "closed", r["eval"]["closed"]["avg6_opt"], flush=True)
        else:
            print(r["set"], r["gate"], r["k"], r["seed"], "M", r["M"], "h", round(r["h_ns"], 4), "sig", r["sigma_ns"],
                  f"{r['wall_s']:.0f}s", "FTn", f"{r['F_T_noisy']:.8f}", "FTc", f"{r['F_T_closed']:.8f}",
                  "Cn", f"{r['C_noisy']:.6f}", "stored FTn", f"{r['stored']['F_T_noisy']:.8f}", flush=True)
    (OUTDIR / "pulses_smooth_smoke.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if "--smoke" in sys.argv:
        smoke()
        sys.exit()
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 32
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else ("track", "target", "flip")
    cases = (track_cases() if "track" in only else []) + (target_cases() if "target" in only else []) + \
        (flip_cases() if "flip" in only else [])
    edges = ("zero", "hold") if "--no-hold" not in sys.argv else ("zero",)
    tasks = [(c, s, None, e) for c in cases for e in edges for s in SIGMAS if s > 0 or e == "zero"]
    done = set()
    if OUT.exists():
        done = {key(json.loads(l)) for l in open(OUT)}
    tasks = [t for t in tasks if (t[0]["set"], t[0]["gate"], round(t[0]["T_ns"]), t[0]["seed"], t[1], t[3]) not in done]
    cost = {"flip": 3.0, "track": 1.0, "target": 0.5}
    tasks.sort(key=lambda t: (t[3] != "zero", -(cost[t[0]["set"]] * t[0]["T_ns"] * (t[1] > 0 or t[0]["set"] != "flip"))))
    print(f"{len(tasks)} tasks, {W} workers", flush=True)
    t0 = time.time()
    with mp.Pool(W) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(run, tasks)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            v = (r["eval"]["lind"]["avg6_opt"] if r["set"] == "flip" else r["F_T_noisy"])
            print(f"[{(time.time() - t0) / 60:6.1f} min] {i + 1}/{len(tasks)} {r['set']} {r['gate']} {round(r['T_ns'])} "
                  f"s{r['seed']} sigma={r['sigma_ns']:g} {r['edge']} M={r['M']} {r['wall_s']:.0f}s F={v:.6f} "
                  f"bounds={r['checks']['bound_Omega']:.4f},{r['checks']['bound_osc']:.4f}", flush=True)
    open(OUT.with_suffix(".done"), "w").write("done\n")
