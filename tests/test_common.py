"""Validation of manuscript/common against published numbers and exact identities.  Run: python tests/test_common.py
Each check prints OK/FAIL; the script exits non-zero if any check fails.
  [S]  states against the photon-optimal-path note (Sec. 1) and the protocol deck (slide 8)
  [A]  analytic paths against the note's Tables 1-2
  [V]  squeezer capacity v and T_tan against the deck (slide 10)
  [B]  brachistochrone Hamiltonian: reaches the target at T* = theta0/omega; normal-ordered form exact
  [G]  GRAPE gradient against central finite differences
  [L]  Lindblad evaluator: noiseless limit and first-order loss (Lemma 1 of the note)
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import pathlib
import sys

ROOT = _L.vfile(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from common import brachistochrone as B  # noqa: E402
from common import grape, lindblad, paths, photon_core as pc, states  # noqa: E402
from common.params import G_SQ, OMEGA_REF  # noqa: E402

FAIL = []


def check(tag, value, ref, tol, rel=False):
    err = abs(value - ref) / (abs(ref) if rel else 1)
    ok = err <= tol
    print(f"{'OK  ' if ok else 'FAIL'} [{tag}] {value:.6g} vs {ref:.6g} (tol {tol}{' rel' if rel else ''})")
    if not ok:
        FAIL.append(tag)


D = 80
g0, g1 = states.gkp(0.4, 0, D), states.gkp(0.4, 1, D)
n = np.arange(D)
check("S gkp <n>_0", float(np.sum(abs(g0) ** 2 * n)), 2.439, 2e-3)
check("S gkp <n>_1", float(np.sum(abs(g1) ** 2 * n)), 2.870, 2e-3)
check("S gkp <0_L|1_L>", float(abs(np.vdot(g0, g1))), 0.01416, 2e-4)
check("S gkp c0", float(abs(g0[0])), 0.73763, 2e-5)
check("S gkp theta0", float(np.arccos(abs(g0[0]))), 0.7412, 2e-4)
ct = states.cat(2.0, D)
check("S cat c0", float(abs(ct[0])), 0.19136, 2e-5)
check("S cat theta0", float(np.arccos(abs(ct[0]))), 1.3782, 2e-4)
check("S flip theta", float(np.arccos(abs(np.vdot(g0, g1)))), 1.5566, 2e-4)

# [A] Table 1 / Table 2 of the note
P = paths.Prep(g0)
check("A GKP J_geo", P.J_geo(), 0.6506, 2e-4)
U, J = paths.photon_path_prep(P)
check("A GKP J_num", J, 0.5973, 3e-3, rel=True)
check("A GKP L_num/theta0", pc.length(U) / P.th, 1.067, 3e-3, rel=True)
check("A GKP lower bound B", pc.lower_bound(P.nv, P.uT), 0.5904, 3e-3, rel=True)
for k, ref in ((2, 1.5708), (4, 3.1416)):
    Pf = paths.Prep(states.fock(k, D))
    check(f"A Fock{k} J_geo = n pi/4", Pf.J_geo(), ref, 1e-4)
Lg = pc.Logical(0.4, 80)
check("A flip J_geo", Lg.J_geo(), 3.9217, 2e-4)
det = Lg.detour()
check("A detour J", pc.exposure(pc.resample(det, 30), np.diag(Lg.nv)), 2.1763, 3e-3, rel=True)
Uf, Jf = pc.optimize_path(np.diag(Lg.nv), Lg.u0, Lg.uT, [pc.resample(det, 30)], 30)
check("A flip photon-optimal J (ell = 0)", Jf, 1.9053, 5e-3, rel=True)
check("A flip photon-optimal L (ell = 0)", pc.length(Uf), 2.1066, 5e-3, rel=True)

# [V] squeezer capacity at the start of the path (deck slides 9-10): v = g |<chi|(a^2 - a^dag^2)/2|S>|
a = np.diag(np.sqrt(np.arange(1, D)), 1)
K = 0.5 * (a @ a - a.T @ a.T)
for name, S, T, ref in (("fock2", states.vacuum(D), states.fock(2, D), 0.7071), ("fock4", states.vacuum(D), states.fock(4, D), 0.0),
                        ("cat", states.vacuum(D), ct, 0.3899), ("gkp", states.vacuum(D), g0, 0.3595),
                        ("flip", g0, g1, 0.0061)):
    c0, th, chi = B.geodesic_parts(S, T)
    v = float(abs(np.vdot(chi, K @ S)))
    check(f"V capacity v/g {name}", v, ref, 2e-4)
    if v > 1e-9:
        print(f"      T_tan = {1e3 * th / (v * 15.0):.0f} ns at g = 15 rad/us (deck), {1e3 * th / (v * G_SQ):.0f} ns at g = {G_SQ}")

# [B] brachistochrone
S, T = states.vacuum(D), g0
c0, th, chi = B.geodesic_parts(S, T)
H = B.H_opt(S, T, OMEGA_REF)
psi = expm(-1j * H * th / OMEGA_REF) @ S
check("B H_opt reaches T at T*", float(abs(np.vdot(T, psi)) ** 2), 1.0, 1e-12)
check("B Delta H on |S> = omega", float(np.sqrt(np.real(np.vdot(S, H @ H @ S)) - abs(np.vdot(S, H @ S)) ** 2)), OMEGA_REF, 1e-10)
Ds = 24
Ss, Ts = states.vacuum(Ds), states.gkp(0.4, 0, Ds)
Hs = B.H_opt(Ss, Ts, 1.0)
h = B.normal_ordered(Hs)
check("B normal-ordered reconstruction", float(np.max(np.abs(B.from_normal_ordered(h, Ds) - Hs))), 0.0, 1e-6)  # factorial cancellation

# [G] GRAPE gradient
Dg, N = 10, 5
for model in ("two", "sigmaz", "onesided"):
    M = grape.CSModel(Dg, G_SQ, 10 * G_SQ, model)
    rng = np.random.default_rng(1)
    X = grape.random_start(N, rng)
    psi0 = grape.joint(states.vacuum(Dg))
    Q = grape.osc_projector(states.fock(2, Dg))
    Qrun = [grape.osc_projector(states.fock(2, Dg)) for _ in range(N)]
    dt = 0.03
    c, gx, _ = M.cost_grad(X.ravel(), X.shape, dt, psi0, Q, Qrun, 0.7, 0.3)
    x = X.ravel()
    fd = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = 1e-6
        fd[i] = (M.cost_grad(x + e, X.shape, dt, psi0, Q, Qrun, 0.7, 0.3)[0] - M.cost_grad(x - e, X.shape, dt, psi0, Q, Qrun, 0.7, 0.3)[0]) / 2e-6
    check(f"G gradient max |err| ({model})", float(np.max(np.abs(fd - gx))), 0.0, 1e-7)

# [L] Lindblad evaluator
M = grape.CSModel(30, G_SQ, 10 * G_SQ, "two")
X = grape.random_start(20, np.random.default_rng(3))
C, _ = M.controls(X)
psi0 = grape.joint(states.vacuum(30))
traj, _, _ = M.propagate(C, 0.01, psi0)
rho = lindblad.evolve(M, C, 0.01, psi0, dict(kappa=0, kphi=0, g1=0, gphi=0))
check("L noiseless = pure", float(np.max(np.abs(rho - np.outer(traj[-1], traj[-1].conj())))), 0.0, 1e-10)
kap = 1e-3
rho = lindblad.evolve(M, C, 0.01, psi0, dict(kappa=kap, kphi=0, g1=0, gphi=0), dt_max=2e-4)
F = float(np.real(np.vdot(traj[-1], rho @ traj[-1])))
nb = np.array([np.sum(abs(p.reshape(2, 30)) ** 2 @ np.arange(30)) for p in traj])
first = kap * 0.01 * (0.5 * nb[0] + nb[1:-1].sum() + 0.5 * nb[-1])
check("L first-order loss 1-F ~ kappa int <n>dt", 1 - F, first, 0.05, rel=True)

print("ALL OK" if not FAIL else f"FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
