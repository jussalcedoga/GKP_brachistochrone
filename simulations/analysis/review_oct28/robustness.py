#!/usr/bin/env python3
"""Robustness checks of the tracking benchmark (review Oct 28, second set), with the independent integrator of
lindblad_crosscheck.py (same folder: Hamiltonians from the Supplement, Lindblad Eq. (7), Yale rates, cutoff 90,
per-slice expm_multiply of the vectorized generator).  Stored data read: results/jobs80 (pulses), traces80 (Lindblad
traces x, z, F, w_in), rho80 (states, for a cross-check only); run selection with mfcommon.

 1  gates or state transfers: every stored flip pulse at m = 2.81 (3 paths x 3 gate sets x 20 starts) applied, with the
    qubit in |g>, to the six cardinal code states |0_L>, |1_L>, |+-_L> ~ |0_L> +- |1_L>, |+-i_L> ~ |0_L> +- i|1_L>
    (normalized; <0_L|1_L> = 0.0142).  By linearity the channel on the code span is fixed by the outputs of
    |0><0|, |1><1|, |0><1| (x) |g><g| (three Lindblad runs per pulse; two state vectors for the closed evolution).
    Oscillator fidelity with X_L|psi> (X_L: |0_L> <-> |1_L>, extended linearly), the 6-state average, and the average
    after the best virtual frame update e^{i phi Z_L/2} on the output (equivalently the target e^{-i phi Z_L/2} X_L|psi>,
    Z_L = +1 on |0_L>, -1 on |1_L>, extended linearly, targets renormalized).  Reference: H_opt of the flip
    brachistochrone (oscillator only) at T* and at 2.81 T*.  Also: the photon-number reach of both code inputs and the
    closed F_T and gate metrics at cutoff 120.
 2  quality of the prepared GKP states: final Lindblad oscillator state of every Fig. 4 GKP run (3 gate sets x 412, 824,
    2133 ns x 20 starts): <S_X> = <D(2 alpha)>, <S_Z> = <D(2 beta)>, <Z_L> = <D(beta)>, alpha = sqrt(pi/2), beta = i alpha
    (displacements exponentiated at cutoff 220 and cut to 90), nbar; the stored final states (rho80) as a cross-check.
 3  geometric closeness from the stored Lindblad traces: w_in = 2F - x sin 2v - z cos 2v (v = theta t/T),
    C_geo(t) = max_{v' in [0, theta]} (w_in + x sin 2v' + z cos 2v')/2, C_geo = min_t C_geo(t), against C = min_t F(t).
 4  resources used by the representative pulses (closed evolution, 8 points per slice): Delta H_osc(t) of the
    oscillator part of the Hamiltonian (all terms except the qubit drive; for SNAP the dispersive term is included) in
    the joint state, in units of omega = g/sqrt2; nbar(t), P(n >= 40), P(n >= 70); and the same for the reference path.
 5  noise-channel budget of the Fig. 4 representative GKP pulses (CS, ECD, SNAP at 412, 824, 2133 ns, plus the
    non-trivial SNAP representative at 412 ns): 1 - F_T closed, with the qubit channels only (gamma_1, gamma_phi), the
    cavity channels only (kappa, kappa_phi), and all channels; the final ancilla population P_e and the oscillator
    fidelity conditioned on the ancilla in |g> (and in |e>); since the pulses may end with the ancilla in any state, also
    the purity of the closed final ancilla state, its dominant eigenvector |q*>, and P(q*) and the fidelity conditioned
    on finding |q*> (a final ancilla measurement along q*, i.e. an error flag).
 6  (--item6, appended to robustness.json) the noise rates of the CS proposal (params.NOISE['CSproposal']): noisy F_T
    of every Fig. 4 GKP run (3 gate sets x 412, 824, 2133 ns x 20 starts) and of the flip brachistochrone pulses at
    m = 2.81 (3 gate sets x 20 starts), next to the closed and the Yale-rate F_T.
Writes robustness.json next to this file (nothing with --smoke).
Usage (repository root):
  python simulations/analysis/review_oct28/robustness.py [--workers 30]
"""
from __future__ import annotations
import layout as _L  # repository layout: data/ and output/ (src/layout.py)

import argparse
import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import expm_multiply

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE))
import lindblad_crosscheck as LC  # noqa: E402

RES, D, G = LC.RES, LC.D, LC.G
OMEGA = G / math.sqrt(2)
OUT_JSON = HERE / "robustness.json"
D_BIG = 120                     # closed cutoff check of the flip pulses
PAD = 220                       # cutoff at which the displacement operators are exponentiated
ALPHA = math.sqrt(math.pi / 2)
GATES = ("cs", "cd", "snap")
REFS = ("geodesic", "tradeoff", "photon")


# ----------------------------------------------------------------------------- helpers
def hamiltonian_D(gate, x, Dc):
    """the Hamiltonian of LC.hamiltonian at cutoff Dc (same formulas; checked equal at Dc = 90)"""
    a = np.diag(np.sqrt(np.arange(1, Dc)), 1).astype(complex)
    ad = a.conj().T
    n = np.diag(np.arange(Dc)).astype(complex)
    Pq = [np.diag([1.0, 0.0]).astype(complex), np.diag([0.0, 1.0]).astype(complex)]
    r, phi, a0, f0, a1, f1 = x
    Om = 10 * G * r * np.exp(1j * phi)
    H = sp.kron(LC.csr(0.5 * np.array([[0, np.conj(Om)], [Om, 0]])), LC.csr(np.eye(Dc)))
    if gate == "cs":
        for q, (aq, fq) in enumerate(((a0, f0), (a1, f1))):
            xi = G * aq * np.exp(1j * fq)
            H = H + sp.kron(LC.csr(Pq[q]), LC.csr(0.5j * (np.conj(xi) * a @ a - xi * ad @ ad)))
    elif gate == "cd":
        for q, (aq, fq) in enumerate(((a0, f0), (a1, f1))):
            eps = G * aq * np.exp(1j * fq)
            H = H + sp.kron(LC.csr(Pq[q]), LC.csr((1j / math.sqrt(2)) * (np.conj(eps) * a - eps * ad)))
    else:
        eps = G * a0 * np.exp(1j * f0)
        H = H + sp.kron(LC.csr(np.eye(2)), LC.csr((1j / math.sqrt(2)) * (np.conj(eps) * a - eps * ad)))
        H = H - G * sp.kron(LC.csr(Pq[1]), LC.csr(n))
    return H.tocsr()


_LD = None


def LD():
    global _LD
    if _LD is None:
        _LD = LC.vec_dissipator()
    return _LD


def load_pulse(jid):
    r = json.loads((RES / "jobs80" / f"{jid}.json").read_text())
    return np.array(r["X"]), float(r["T_us"]), r["job"]


def joint(osc, q=0):
    v = np.zeros(2 * osc.size, complex)
    v[q * osc.size:(q + 1) * osc.size] = osc
    return v


_LDSUB = {}
CHANNELS = {"all": (0, 1, 2, 3), "qubit": (2, 3), "cavity": (0, 1)}    # LC.jump_ops order: a, n, sigma_-, sigma_z


def LD_sub(which):
    """vectorized dissipator of a subset of the channels of Eq. (7) (same formula as LC.vec_dissipator)"""
    if which not in _LDSUB:
        ops = LC.jump_ops()
        d = 2 * D
        Id = sp.identity(d, complex, format="csr")
        L = sp.csr_matrix((d * d, d * d), dtype=complex)
        for i in CHANNELS[which]:
            Lop, rate = ops[i]
            LdL = (Lop.conj().T @ Lop).tocsr()
            L = L + rate * (sp.kron(Lop, Lop.conj()) - 0.5 * sp.kron(LdL, Id) - 0.5 * sp.kron(Id, LdL.T))
        _LDSUB[which] = L.tocsr()
    return _LDSUB[which]


def lindblad_final(X, gate, T, V0, LDm=None):
    """vec rho(T) for the columns of V0 (vec of joint operators), method A of lindblad_crosscheck"""
    dt = T / len(X)
    V = V0
    LDm = LD() if LDm is None else LDm
    for x in X:
        Lk = (LC.vec_hamiltonian(LC.hamiltonian(gate, x)) + LDm).tocsr()
        V = expm_multiply((Lk * dt).tocsr(), V)
    return V


def osc_of(v, Dc=D):
    R = v.reshape(2 * Dc, 2 * Dc)
    return R[:Dc, :Dc] + R[Dc:, Dc:]


def pops(psi, Dc=D):
    p = np.abs(psi[:Dc]) ** 2 + np.abs(psi[Dc:]) ** 2
    n = np.arange(Dc)
    return float(p @ n), float(p[40:].sum()), float(p[70:].sum()), float(p[90:].sum()) if Dc > 90 else 0.0


def tdist(A, B):
    M = A - B
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(0.5 * (M + M.conj().T)))))


def ms(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, min=float(v.min()),
                max=float(v.max()), n=int(len(v)))


# ----------------------------------------------------------------------------- item 1: gate metrics
CODE = None


def code():
    global CODE
    if CODE is None:
        B = [LC.gkp(0, D), LC.gkp(1, D)]
        Gm = np.array([[np.vdot(B[k], B[l]) for l in range(2)] for k in range(2)])
        CODE = (B, Gm)
    return CODE


CARD = {"0": (1, 0), "1": (0, 1), "+": (1, 1), "-": (1, -1), "+i": (1, 1j), "-i": (1, -1j)}


def K_from_osc(rho00, rho11, rho01):
    """K[i, j, k, l] = <k| E(|i><j|) |l> on the code kets (E(|1><0|) = E(|0><1|)^dag)"""
    B, _ = code()
    rh = {(0, 0): rho00, (1, 1): rho11, (0, 1): rho01, (1, 0): rho01.conj().T}
    K = np.zeros((2, 2, 2, 2), complex)
    for (i, j), r in rh.items():
        for k in range(2):
            for l in range(2):
                K[i, j, k, l] = np.vdot(B[k], r @ B[l])
    return K


def gate_metrics(K, Gm=None):
    Gm = code()[1] if Gm is None else Gm
    states = {}
    for name, a in CARD.items():
        a = np.array(a, complex)
        a = a / math.sqrt(np.real(np.conj(a) @ Gm @ a))
        states[name] = (a, np.einsum("i,j,ijkl->kl", a, a.conj(), K))

    def avg(phi, per=False):
        out = {}
        for name, (a, M) in states.items():
            b = np.array([a[1] * np.exp(-0.5j * phi), a[0] * np.exp(0.5j * phi)])
            out[name] = float(np.real(np.conj(b) @ M @ b) / np.real(np.conj(b) @ Gm @ b))
        return out if per else float(np.mean(list(out.values())))
    grid = np.linspace(-math.pi, math.pi, 1441)
    vals = np.array([avg(p) for p in grid])
    p0 = grid[int(np.argmax(vals))]
    res = minimize_scalar(lambda p: -avg(p), bounds=(p0 - 0.01, p0 + 0.01), method="bounded",
                          options=dict(xatol=1e-10))
    phi = float((res.x + math.pi) % (2 * math.pi) - math.pi)
    return dict(per_state=avg(0.0, True), avg6=avg(0.0), avg6_opt=avg(phi), phi_opt=phi,
                per_state_opt=avg(phi, True), F_0to1=avg(0.0, True)["0"], F_1to0=avg(0.0, True)["1"])


def flip_task(jid):
    t0 = time.time()
    X, T, jb = load_pulse(jid)
    B, _ = code()
    dt = T / len(X)
    # closed, both code inputs, photon reach (4 points per slice)
    Psi = np.stack([joint(B[0]), joint(B[1])], axis=1)
    reach = np.zeros((2, 3))
    for x in X:
        Y = expm_multiply((-1j * LC.hamiltonian(jb["gate"], x)).tocsr(), Psi, start=0, stop=dt, num=5, endpoint=True)
        for y in Y[1:]:
            for c in range(2):
                reach[c] = np.maximum(reach[c], pops(y[:, c])[:3])
        Psi = Y[-1]
    A = np.array([[[np.vdot(B[k], Psi[q * D:(q + 1) * D, i]) for i in range(2)] for k in range(2)] for q in range(2)])
    Kc = np.einsum("qki,qlj->ijkl", A, A.conj())
    # closed, both code inputs, at cutoff 120 (cutoff check of F_T and of the gate metrics)
    Bb = [LC.gkp(0, D_BIG), LC.gkp(1, D_BIG)]
    Gb = np.array([[np.vdot(Bb[k], Bb[l]) for l in range(2)] for k in range(2)])
    Psi = np.stack([joint(Bb[0]), joint(Bb[1])], axis=1)
    p90 = np.zeros(2)
    for x in X:
        Y = expm_multiply((-1j * hamiltonian_D(jb["gate"], x, D_BIG)).tocsr(), Psi, start=0, stop=dt, num=5,
                          endpoint=True)
        for y in Y[1:]:
            for c in range(2):
                p90[c] = max(p90[c], pops(y[:, c], D_BIG)[3])
        Psi = Y[-1]
    Ab = np.array([[[np.vdot(Bb[k], Psi[q * D_BIG:(q + 1) * D_BIG, i]) for i in range(2)] for k in range(2)]
                   for q in range(2)])
    gm120 = gate_metrics(np.einsum("qki,qlj->ijkl", Ab, Ab.conj()), Gb)
    FT120 = gm120["F_0to1"]
    # Lindblad: |0><0|, |1><1|, |0><1| (x) |g><g|
    u0, u1 = joint(B[0]), joint(B[1])
    V0 = np.stack([np.outer(u0, u0.conj()).ravel(), np.outer(u1, u1.conj()).ravel(), np.outer(u0, u1.conj()).ravel()],
                  axis=1)
    V = lindblad_final(X, jb["gate"], T, V0)
    KL = K_from_osc(osc_of(V[:, 0]), osc_of(V[:, 1]), osc_of(V[:, 2]))
    return dict(kind="flip", id=jid, ref=jb["ref"], gate=jb["gate"], T_ns=1e3 * T, closed=gate_metrics(Kc),
                lind=gate_metrics(KL), reach_in0=dict(nbar_max=reach[0, 0], P40_max=reach[0, 1], P70_max=reach[0, 2]),
                reach_in1=dict(nbar_max=reach[1, 0], P40_max=reach[1, 1], P70_max=reach[1, 2]),
                F_T_closed_D120=FT120, P90_max_D120=float(p90[0]), P90_max_D120_in1=float(p90[1]),
                closed_D120=gm120, wall_s=time.time() - t0)


def hopt_task(m):
    """flip brachistochrone H_opt = i w (|chi><S| - |S><chi|) on the oscillator, w = theta/T, T = m T*"""
    t0 = time.time()
    B, _ = code()
    S, Tt = B
    ov = np.vdot(S, Tt)
    chi = Tt * np.exp(-1j * np.angle(ov)) - abs(ov) * S
    chi /= np.linalg.norm(chi)
    th = math.acos(abs(ov))
    Tstar = th / OMEGA
    T = m * Tstar
    H = 1j * (th / T) * (np.outer(chi, S.conj()) - np.outer(S, chi.conj()))
    U = expm(-1j * H * T)
    ps = [U @ S, U @ Tt]
    Kc = np.zeros((2, 2, 2, 2), complex)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    Kc[i, j, k, l] = np.vdot(B[k], ps[i]) * np.vdot(ps[j], B[l])
    I = sp.identity(D, complex, format="csr")
    Hs = LC.csr(H)
    L = -1j * (sp.kron(Hs, I) - sp.kron(I, Hs.T))
    for Lop, rate in ((LC.A_OSC, LC.RATES["kappa"]), (LC.N_OSC, LC.RATES["kphi"])):
        Ls = LC.csr(Lop)
        LdL = (Ls.conj().T @ Ls).tocsr()
        L = L + rate * (sp.kron(Ls, Ls.conj()) - 0.5 * sp.kron(LdL, I) - 0.5 * sp.kron(I, LdL.T))
    V0 = np.stack([np.outer(S, S.conj()).ravel(), np.outer(Tt, Tt.conj()).ravel(), np.outer(S, Tt.conj()).ravel()],
                  axis=1)
    V = expm_multiply((L * T).tocsr(), V0)
    KL = K_from_osc(*(V[:, c].reshape(D, D) for c in range(3)))
    return dict(kind="hopt", m=m, T_ns=1e3 * T, Tstar_ns=1e3 * Tstar, theta=th, closed=gate_metrics(Kc),
                lind=gate_metrics(KL), wall_s=time.time() - t0)


# ----------------------------------------------------------------------------- item 2: GKP quality
_DISP = {}


def disp_ops(pad=PAD):
    if pad not in _DISP:
        a = np.diag(np.sqrt(np.arange(1, pad)), 1).astype(complex)
        _DISP[pad] = {nm: expm(g * a.conj().T - np.conj(g) * a)[:D, :D]
                      for nm, g in (("S_X", 2 * ALPHA), ("S_Z", 2j * ALPHA), ("Z_L", 1j * ALPHA))}
    return _DISP[pad]


def gkp_quality(rho):
    out = {}
    for nm, Dm in disp_ops().items():
        v = np.trace(rho @ Dm)
        out[nm] = float(np.real(v))
        out[nm + "_imag"] = float(abs(np.imag(v)))
    out["nbar"] = float(np.real(np.sum(np.arange(D) * np.diag(rho))))
    out["F_T"] = float(np.real(np.vdot(code()[0][0], rho @ code()[0][0])))
    return out


def prep_task(jid):
    t0 = time.time()
    X, T, jb = load_pulse(jid)
    u = joint(LC.vac(D))
    V = lindblad_final(X, jb["gate"], T, np.outer(u, u.conj()).ravel()[:, None])
    rho = osc_of(V[:, 0])
    q = gkp_quality(rho)
    z = np.load(RES / "rho80" / f"{jid}.npz")
    q["TD_vs_rho80"] = tdist(rho, z["rhoL"][-1].astype(complex))
    return dict(kind="prep", id=jid, gate=jb["gate"], k=jb.get("k"), T_ns=1e3 * T, **q, wall_s=time.time() - t0)


# ----------------------------------------------------------------------------- item 4: resources
def resource_task(jid):
    t0 = time.time()
    X, T, jb = load_pulse(jid)
    S90 = LC.transfer(jb["name"], D)[0]
    psi = joint(S90)
    dt = T / len(X)
    SUB = 8
    dH, dHf, nb, p40, p70 = [], [], [], [], []
    for x in X:
        H = LC.hamiltonian(jb["gate"], x)
        x0 = np.array(x, float)
        x0[0] = 0.0                                     # r = 0: no qubit drive -> oscillator part only
        Ho = LC.hamiltonian(jb["gate"], x0)
        Y = expm_multiply((-1j * H).tocsr(), psi, start=0, stop=dt, num=SUB + 1, endpoint=True)
        row = [[], [], [], [], []]
        for y in Y:
            for Hm, lst in ((Ho, row[0]), (H, row[1])):
                hy = Hm @ y
                lst.append(math.sqrt(max(float(np.real(np.vdot(hy, hy)) - np.real(np.vdot(y, hy)) ** 2), 0.0)))
            n_, a40, a70, _ = pops(y)
            row[2].append(n_)
            row[3].append(a40)
            row[4].append(a70)
        for lst, r in zip((dH, dHf, nb, p40, p70), row):
            lst.append(r)
        psi = Y[-1]
    dH, dHf, nb, p40, p70 = (np.array(v) for v in (dH, dHf, nb, p40, p70))   # (N, SUB + 1)
    tavg = lambda A: float(np.mean(0.5 * (A[:, :-1] + A[:, 1:])))          # noqa: E731  (trapezoid, equal steps)
    i, j = np.unravel_index(np.argmax(nb), nb.shape)
    return dict(kind="res", id=jid, gate=jb["gate"], name=jb["name"], T_ns=1e3 * T, m=jb.get("m"),
                dH_osc_avg=tavg(dH) / OMEGA, dH_osc_max=float(dH.max()) / OMEGA,
                dH_full_avg=tavg(dHf) / OMEGA, dH_full_max=float(dHf.max()) / OMEGA,
                nbar_max=float(nb.max()), nbar_avg=tavg(nb), t_peak_ns=1e3 * dt * (i + j / SUB),
                P40_at_peak=float(p40[i, j]), P70_at_peak=float(p70[i, j]), P40_max=float(p40.max()),
                P70_max=float(p70.max()), wall_s=time.time() - t0)


def path_resources(name):
    S, chi, th = LC.geodesic(name)
    s = np.linspace(0, 1, 2001)
    P = np.cos(s[:, None] * th) * S[None] + np.sin(s[:, None] * th) * chi[None]
    pr = np.abs(P) ** 2
    nb = pr @ np.arange(D)
    i = int(np.argmax(nb))
    Hn = 1j * OMEGA * (np.outer(chi, S.conj()) - np.outer(S, chi.conj()))     # H_opt at speed omega
    dH = [math.sqrt(max(np.real(np.vdot(Hn @ p, Hn @ p)) - np.real(np.vdot(p, Hn @ p)) ** 2, 0)) / OMEGA for p in P]
    return dict(theta=th, Tstar_ns=1e3 * th / OMEGA, dH_over_omega_min=float(min(dH)), dH_over_omega_max=float(max(dH)),
                nbar_max=float(nb.max()), nbar_avg=float(np.mean(0.5 * (nb[1:] + nb[:-1]))),
                P40_at_peak=float(pr[i, 40:].sum()), P70_at_peak=float(pr[i, 70:].sum()),
                P40_max=float(pr[:, 40:].sum(1).max()), P70_max=float(pr[:, 70:].sum(1).max()))


# ----------------------------------------------------------------------------- item 5: noise budget
def budget_task(jid):
    t0 = time.time()
    X, T, jb = load_pulse(jid)
    tgt = LC.gkp(0, D)
    u = joint(LC.vac(D))
    dt = T / len(X)
    psi = u.copy()
    for x in X:
        psi = expm_multiply((-1j * dt * LC.hamiltonian(jb["gate"], x)).tocsr(), psi)
    # closed final qubit state: rho_q[i, j] = <psi_j|psi_i>_osc; |q*> its dominant eigenvector (the state a final
    # ancilla measurement would expect if no error occurred)
    rq = np.array([[np.vdot(psi[j * D:(j + 1) * D], psi[i * D:(i + 1) * D]) for j in range(2)] for i in range(2)])
    ev, evec = np.linalg.eigh(0.5 * (rq + rq.conj().T))
    qs = evec[:, -1]

    def stats(R):
        blk = lambda i, j: R[i * D:(i + 1) * D, j * D:(j + 1) * D]                    # noqa: E731
        rg, re_ = blk(0, 0), blk(1, 1)
        rs = sum(np.conj(qs[i]) * qs[j] * blk(i, j) for i in range(2) for j in range(2))
        ng, ne, ns = float(np.trace(rg).real), float(np.trace(re_).real), float(np.trace(rs).real)
        F = lambda r: float(np.real(np.vdot(tgt, r @ tgt)))                               # noqa: E731
        return dict(F_T=F(rg) + F(re_), P_e=ne, F_given_g=F(rg) / ng if ng > 1e-12 else None,
                    F_given_e=F(re_) / ne if ne > 1e-12 else None, P_qstar=ns,
                    F_given_qstar=F(rs) / ns if ns > 1e-12 else None)
    out = dict(kind="budget", id=jid, gate=jb["gate"], k=jb.get("k"), T_ns=1e3 * T,
               qubit_closed=dict(purity=float(np.real(np.trace(rq @ rq))), P_e=float(rq[1, 1].real),
                                 qstar=[[float(np.real(c)), float(np.imag(c))] for c in qs]),
               closed=stats(np.outer(psi, psi.conj())))
    for which in ("qubit", "cavity", "all"):
        V = lindblad_final(X, jb["gate"], T, np.outer(u, u.conj()).ravel()[:, None], LD_sub(which))
        out[which] = stats(V[:, 0].reshape(2 * D, 2 * D))
    out["wall_s"] = time.time() - t0
    return out


# ----------------------------------------------------------------------------- item 6: CS-proposal noise
def csp_rates():
    """rates of params.NOISE['CSproposal'] with the conventions of Eq. (7): kappa = 1/T1c, kappa_phi = 2/T_phi,c (0 if
    absent), gamma_1 = 1/T1q, gamma_phi = 1/T_phi,q"""
    from common.params import NOISE                       # constants only (src/src/common/params.py)
    s = NOISE["CSproposal"]
    return dict(T1c=s["T1c"], T1q=s["T1q"], Tphiq=s["Tphiq"], Tphic=s.get("Tphic"), source=s.get("source")), dict(
        kappa=1 / s["T1c"], kphi=2 / s["Tphic"] if s.get("Tphic") else 0.0, g1=1 / s["T1q"], gphi=1 / s["Tphiq"])


_LDR = {}


def LD_rates(r):
    """vectorized dissipator of Eq. (7) at the rates r (operators of LC.jump_ops)"""
    key = tuple(sorted(r.items()))
    if key not in _LDR:
        ops = [L for L, _ in LC.jump_ops()]
        rates = (r["kappa"], r["kphi"], r["g1"], r["gphi"] / 2)
        d = 2 * D
        Id = sp.identity(d, complex, format="csr")
        L = sp.csr_matrix((d * d, d * d), dtype=complex)
        for Lop, rate in zip(ops, rates):
            if rate > 0:
                LdL = (Lop.conj().T @ Lop).tocsr()
                L = L + rate * (sp.kron(Lop, Lop.conj()) - 0.5 * sp.kron(LdL, Id) - 0.5 * sp.kron(Id, LdL.T))
        _LDR[key] = L.tocsr()
    return _LDR[key]


def csp_task(jid):
    t0 = time.time()
    X, T, jb = load_pulse(jid)
    S, tgt = LC.transfer(jb["name"], D)
    u = joint(S)
    dt = T / len(X)
    psi = u.copy()
    for x in X:
        psi = expm_multiply((-1j * dt * LC.hamiltonian(jb["gate"], x)).tocsr(), psi)
    Fc = float(abs(np.vdot(tgt, psi[:D])) ** 2 + abs(np.vdot(tgt, psi[D:])) ** 2)
    V = lindblad_final(X, jb["gate"], T, np.outer(u, u.conj()).ravel()[:, None], LD_rates(csp_rates()[1]))
    rho = osc_of(V[:, 0])
    tr = json.loads((RES / "traces80" / f"{jid}.json").read_text())
    return dict(kind="csp", id=jid, name=jb["name"], gate=jb["gate"], k=jb.get("k"), T_ns=1e3 * T, F_closed=Fc,
                F_csp=float(np.real(np.vdot(tgt, rho @ tgt))), F_csp_stored=tr.get("F_T_final_noisy_CS"),
                wall_s=time.time() - t0)


def item6(workers, t0):
    import mfcommon as mf
    Gr = mf.groups()
    keys = [("gkp", g, k) for g in GATES for k in (1, 2, 3)] + [("flip", g, 2) for g in GATES]
    ids = {key: [r["job"]["id"] for r in Gr[(key[0], "geodesic", key[1], key[2])]] for key in keys}
    tasks = [("csp", i) for v in ids.values() for i in v]
    tasks.sort(key=lambda t: -(2.5 if "_k3_" in t[1] else 1))
    print(f"item 6: {len(tasks)} tasks", flush=True)
    with mp.Pool(min(workers, len(tasks))) as pool:
        res = list(pool.imap_unordered(worker, tasks))
    out = json.loads(OUT_JSON.read_text())
    yale = {}
    for e in out["item2_gkp_quality"].values():
        yale.update({r["id"]: r["F_T"] for r in e["runs"]})
    for e in out["item1_flip_gate"]["groups"].values():
        yale.update({r["id"]: r["lind"]["F_0to1"] for r in e["runs"]})
    q, rt = csp_rates()
    groups = {}
    for key, lst in ids.items():
        rs = [o for o in res if o["id"] in lst]
        for o in rs:
            o["F_yale"] = yale[o["id"]]
        b = lambda f: max(rs, key=lambda o: o[f])                                           # noqa: E731
        groups[f"{key[0]}/{key[1]}/k{key[2]}"] = dict(
            T_ns=rs[0]["T_ns"], n=len(rs), F_closed=ms([o["F_closed"] for o in rs]),
            F_yale=ms([o["F_yale"] for o in rs]), F_csp=ms([o["F_csp"] for o in rs]),
            best=dict(closed=b("F_closed")["F_closed"], yale=b("F_yale")["F_yale"], csp=b("F_csp")["F_csp"],
                      csp_id=b("F_csp")["id"]),
            check_vs_stored_F_T_final_noisy_CS=max(abs(o["F_csp"] - o["F_csp_stored"]) for o in rs
                                                   if o["F_csp_stored"] is not None),
            runs=[{k: o[k] for k in ("id", "F_closed", "F_yale", "F_csp", "F_csp_stored")} for o in rs])
    out["item6_csproposal_noise"] = dict(noise=q, rates_per_us=rt, groups=groups,
                                         check_LD_rates_yale_vs_LC=float(abs(LD_rates(LC.RATES) - LC.vec_dissipator()).max()),
                                         wall_s=time.time() - t0)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"appended item 6 to {OUT_JSON} in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- item 3: geometric closeness
def geo_closeness(r, theta):
    t = np.asarray(r["tL_ns"])
    x, z, F, w = (np.asarray(r[k]) for k in ("xL", "zL", "FL", "wL"))
    v = theta * t / t[-1]
    win = 2 * F - x * np.sin(2 * v) - z * np.cos(2 * v)
    R, psi = np.hypot(x, z), np.arctan2(x, z)                 # x sin 2v' + z cos 2v' = R cos(2v' - psi)
    inside = (psi >= 0) & (psi <= 2 * theta)
    vstar = np.where(inside, psi / 2, np.where(np.cos(-psi) >= np.cos(2 * theta - psi), 0.0, theta))
    best = np.where(inside, R, np.maximum(z, x * np.sin(2 * theta) + z * np.cos(2 * theta)))
    Cg_t = 0.5 * (win + best)
    k = int(np.argmin(F))
    return dict(C=float(F.min()), C_geo=float(Cg_t.min()), t_C_ns=float(t[k]), t_Cgeo_ns=float(t[int(np.argmin(Cg_t))]),
                lag_at_C=float((vstar[k] - v[k]) / theta), win_vs_stored_wL=float(np.max(np.abs(win - w))))


# ----------------------------------------------------------------------------- driver
def worker(task):
    kind, arg = task
    return {"flip": flip_task, "prep": prep_task, "res": resource_task, "hopt": hopt_task,
            "budget": budget_task, "csp": csp_task}[kind](arg)


def checks():
    rng = np.random.default_rng(1)
    out = {}
    for gate in GATES:
        x = rng.uniform(0, 1, 6)
        out[f"hamiltonian_D_vs_LC_{gate}"] = float(abs(hamiltonian_D(gate, x, D) - LC.hamiltonian(gate, x)).max())
    d1, d2 = disp_ops(PAD), disp_ops(320)
    out["disp_pad220_vs_320"] = float(max(np.max(np.abs(d1[k] - d2[k])) for k in d1))
    out["LD_sub_all_vs_LC"] = float(abs(LD_sub("all") - LC.vec_dissipator()).max())
    out["LD_sub_qubit_plus_cavity_vs_all"] = float(abs(LD_sub("qubit") + LD_sub("cavity") - LD_sub("all")).max())
    B, Gm = code()
    out["overlap_0L_1L"] = float(np.real(Gm[0, 1]))
    rho = np.outer(B[0], B[0].conj())
    out["target_0L"] = gkp_quality(rho)
    out["vacuum"] = gkp_quality(np.outer(LC.vac(D), LC.vac(D)))
    out["theta_gkp_prep"] = LC.geodesic("gkp")[2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--item6", action="store_true", help="only item 6, appended to an existing robustness.json")
    a = ap.parse_args()
    t0 = time.time()
    if a.item6:
        item6(a.workers, t0)
        return
    print(json.dumps(checks(), indent=1), flush=True)
    if a.smoke:
        for task in (("hopt", 1.0), ("res", "seed_flip_geodesic_cs_k2_s8"), ("prep", "seed_gkp_geodesic_cs_k1_s11"),
                     ("budget", "seed_gkp_geodesic_cs_k1_s11"), ("flip", "seed_flip_geodesic_cs_k2_s8")):
            o = worker(task)
            print(task, json.dumps({k: v for k, v in o.items() if k not in ("closed", "lind")}),
                  json.dumps(o.get("lind", {}).get("avg6_opt")), json.dumps(o.get("closed", {}).get("per_state")),
                  flush=True)
        return
    import mfcommon as mf
    Gr = mf.groups()
    flip_ids = {(ref, g): [r["job"]["id"] for r in Gr[("flip", ref, g, 2)]] for ref in REFS for g in GATES}
    prep_keys = [(g, k) for g in GATES for k in (1, 2, 3)]
    prep_ids = {key: [r["job"]["id"] for r in Gr[("gkp", "geodesic", key[0], key[1])]] for key in prep_keys}
    triv = {r["job"]["id"]: mf.trivial(r) for lst in Gr.values() for r in lst}
    res_ids = []
    for g in GATES:
        for k in (1, 2):
            res_ids.append(mf.representative(Gr[("gkp", "geodesic", g, k)])["job"]["id"])
    res_ids.insert(3, mf.representative([r for r in Gr[("gkp", "geodesic", "snap", 1)] if not mf.trivial(r)])["job"]["id"])
    res_ids = list(dict.fromkeys(res_ids))
    for g in GATES:
        res_ids.append(mf.representative(Gr[("flip", "geodesic", g, 2)])["job"]["id"])
    bud_ids = [mf.representative(Gr[("gkp", "geodesic", g, k)])["job"]["id"] for g in GATES for k in (1, 2, 3)]
    bud_ids.append(mf.representative([r for r in Gr[("gkp", "geodesic", "snap", 1)] if not mf.trivial(r)])["job"]["id"])
    tasks = [("flip", i) for v in flip_ids.values() for i in v] + [("prep", i) for v in prep_ids.values() for i in v]
    tasks += [("res", i) for i in res_ids] + [("hopt", 1.0), ("hopt", 2.807285306528499)]
    tasks += [("budget", i) for i in bud_ids]
    weight = lambda t: ({"flip": 3, "budget": 3}.get(t[0], 1)) * (2.5 if "_k3_" in str(t[1]) else 1)   # noqa: E731
    tasks.sort(key=lambda t: -weight(t))
    print(f"{len(tasks)} tasks, {a.workers} workers", flush=True)
    results = []
    with mp.Pool(min(a.workers, len(tasks))) as pool:
        for i, o in enumerate(pool.imap_unordered(worker, tasks)):
            results.append(o)
            if i % 20 == 0 or i == len(tasks) - 1:
                print(f"[{time.time() - t0:7.1f}s] {i + 1}/{len(tasks)}", flush=True)
    out = dict(meta=dict(date="2026-10-28", D=D, D_big=D_BIG, pad=PAD, g=G, omega=OMEGA, rates=LC.RATES,
                         wall_s=time.time() - t0), checks=checks())
    # item 1
    stored = {}
    for o in results:
        if o["kind"] == "flip":
            tr = json.loads((RES / "traces80" / f"{o['id']}.json").read_text())
            stored[o["id"]] = (tr["F_T_final"], tr["F_T_final_noisy"])
    it1 = {}
    for (ref, g), ids in flip_ids.items():
        rs = [o for o in results if o["kind"] == "flip" and o["id"] in ids]
        e = dict(n=len(rs))
        for ev in ("closed", "lind"):
            e[ev] = {q: ms([r[ev][q] for r in rs]) for q in ("F_0to1", "F_1to0", "avg6", "avg6_opt")}
            e[ev]["per_state"] = {s: ms([r[ev]["per_state"][s] for r in rs]) for s in CARD}
            e[ev]["per_state_opt"] = {s: ms([r[ev]["per_state_opt"][s] for r in rs]) for s in CARD}
            b = max(rs, key=lambda r: r[ev]["avg6_opt"])
            e[ev]["best"] = dict(id=b["id"], avg6_opt=b[ev]["avg6_opt"], phi_opt=b[ev]["phi_opt"], avg6=b[ev]["avg6"],
                                 F_0to1=b[ev]["F_0to1"], F_1to0=b[ev]["F_1to0"])
            e[ev]["phi_opt_all"] = [r[ev]["phi_opt"] for r in rs]
        e["check_F_0to1_vs_stored"] = dict(
            closed=max(abs(r["closed"]["F_0to1"] - stored[r["id"]][0]) for r in rs),
            lind=max(abs(r["lind"]["F_0to1"] - stored[r["id"]][1]) for r in rs))
        e["reach_in0"] = {q: ms([r["reach_in0"][q] for r in rs]) for q in ("nbar_max", "P40_max", "P70_max")}
        e["reach_in1"] = {q: ms([r["reach_in1"][q] for r in rs]) for q in ("nbar_max", "P40_max", "P70_max")}
        e["F_T_closed_D90_minus_D120"] = ms([r["closed"]["F_0to1"] - r["F_T_closed_D120"] for r in rs])
        e["F_T_closed_D120"] = ms([r["F_T_closed_D120"] for r in rs])
        e["P90_max_D120"] = ms([r["P90_max_D120"] for r in rs])
        e["P90_max_D120_in1"] = ms([r["P90_max_D120_in1"] for r in rs])
        e["closed_D120"] = {q: ms([r["closed_D120"][q] for r in rs]) for q in ("F_0to1", "F_1to0", "avg6", "avg6_opt")}
        e["closed_D90_minus_D120_avg6_opt"] = ms([r["closed"]["avg6_opt"] - r["closed_D120"]["avg6_opt"] for r in rs])
        e["runs"] = [{k: r[k] for k in ("id", "closed", "lind", "reach_in0", "reach_in1", "F_T_closed_D120",
                                        "P90_max_D120", "P90_max_D120_in1", "closed_D120")} for r in rs]
        it1[f"{ref}/{g}"] = e
    out["item1_flip_gate"] = dict(groups=it1, H_opt=[o for o in results if o["kind"] == "hopt"])
    # item 2
    it2 = {}
    for (g, k), ids in prep_ids.items():
        rs = [o for o in results if o["kind"] == "prep" and o["id"] in ids]
        b = max(rs, key=lambda r: r["F_T"])
        it2[f"{g}/k{k}"] = dict(T_ns=rs[0]["T_ns"], n=len(rs), n_trivial=sum(triv[r["id"]] for r in rs),
                                stats={q: ms([r[q] for r in rs]) for q in ("F_T", "S_X", "S_Z", "Z_L", "nbar")},
                                best_by_F_T={q: b[q] for q in ("id", "F_T", "S_X", "S_Z", "Z_L", "nbar")},
                                max_imag=max(max(r[q + "_imag"] for q in ("S_X", "S_Z", "Z_L")) for r in rs),
                                TD_vs_rho80_max=max(r["TD_vs_rho80"] for r in rs),
                                runs=[{q: r[q] for q in ("id", "F_T", "S_X", "S_Z", "Z_L", "nbar", "TD_vs_rho80")}
                                      for r in rs])
    out["item2_gkp_quality"] = it2
    # item 3 (stored traces)
    th = LC.geodesic("gkp")[2]
    it3 = {}
    for g in GATES:
        for k in (1, 2, 3):
            lst = Gr[("gkp", "geodesic", g, k)]
            gc = [dict(id=r["job"]["id"], trivial=mf.trivial(r), **geo_closeness(r, th)) for r in lst]
            it3[f"{g}/k{k}"] = dict(
                T_ns=lst[0]["T_ns"], n=len(gc), n_trivial=sum(c["trivial"] for c in gc),
                C=ms([c["C"] for c in gc]), C_geo=ms([c["C_geo"] for c in gc]),
                gain=ms([c["C_geo"] - c["C"] for c in gc]),
                deficit_removed=ms([(c["C_geo"] - c["C"]) / max(1 - c["C"], 1e-12) for c in gc]),
                lag_at_C=ms([c["lag_at_C"] for c in gc]),
                nontrivial=dict(C=ms([c["C"] for c in gc if not c["trivial"]]),
                                C_geo=ms([c["C_geo"] for c in gc if not c["trivial"]])) if any(
                    not c["trivial"] for c in gc) else None,
                win_check_max=max(c["win_vs_stored_wL"] for c in gc), runs=gc)
    out["item3_geometric_closeness"] = dict(theta=th, groups=it3)
    # item 4
    out["item4_resources"] = dict(
        pulses=[o for o in results if o["kind"] == "res"],
        path=dict(gkp=path_resources("gkp"), flip=path_resources("flip")))
    # item 5
    bud = []
    for o in results:
        if o["kind"] == "budget":
            o = dict(o, trivial=triv[o["id"]])
            o["infidelity"] = {ev: 1 - o[ev]["F_T"] for ev in ("closed", "qubit", "cavity", "all")}
            o["added_by"] = dict(qubit=o["qubit"]["F_T"] - o["closed"]["F_T"],
                                 cavity=o["cavity"]["F_T"] - o["closed"]["F_T"],
                                 all=o["all"]["F_T"] - o["closed"]["F_T"])
            bud.append(o)
    bud.sort(key=lambda o: (GATES.index(o["gate"]), o["T_ns"], o["trivial"]))
    out["item5_noise_budget"] = bud
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_JSON} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
