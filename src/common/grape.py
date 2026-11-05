"""GRAPE for a qubit coupled to an oscillator with qubit-conditioned squeezing (exact gradients, numpy only).

Model (hbar = 1, rotating frame, piecewise-constant controls on N slices of length dt = T/N):
    H(t) = 1/2 [Omega_x(t) sigma_x + Omega_y(t) sigma_y] (x) 1
           + |g><g| (x) S(xi_g(t)) + |e><e| (x) S(xi_e(t)),
    S(xi) = (i/2)(xi^* a^2 - xi a^dag^2)       (a constant xi for a time t squeezes by r = |xi| t)
with the hardware bounds |Omega| <= Omega_max and |xi_q| <= g.  On the vacuum a squeezer moves the state at the
Fubini-Study speed |xi|/sqrt2, so g/sqrt2 is the reference speed of the analytic paths (params.OMEGA_REF).
Control models: 'two'  independent squeezers on both qubit states (the protocol deck's two-amplitude model);
                'sigmaz'  xi_e = -xi_g, the conditional squeeze exp(G sigma_z) of the group's draft (App. B);
                'onesided'  xi_g = 0, the one-sided gate proposed by Del Grosso et al.
Variables per slice (the six numbers of the deck): rho_Omega, phi_Omega, rho_g, phi_g, rho_e, phi_e with
Omega = Omega_max rho_Omega e^{i phi_Omega}, xi_q = g rho_q e^{i phi_q}, 0 <= rho <= 1 (box bounds of L-BFGS-B).

Cost (minimized):
    C = (1 - F_T) + w_path <1 - F_k>_k + lam_top <P_top>_k ,
    F_T = <T| Tr_q |psi_N><psi_N| |T>                     (oscillator fidelity, any final qubit state),
    F_k = <p(t_k)| Tr_q |psi_k><psi_k| |p(t_k)>          (closeness to the reference path at the same time),
    P_top = weight in the top third of the Fock ladder     (guard against truncation artefacts; weight lam_top w_fin).
Every term is a quadratic form <psi_k|Q_k|psi_k>; the exact gradient follows from one forward and one backward sweep,
with dU/dc = V[(V^dag H_c V) o Gamma]V^dag in the eigenbasis of each slice Hamiltonian (Gamma: divided differences of
exp(-i lambda dt)).  tests/test_common.py checks it against finite differences.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize


class CSModel:
    def __init__(self, D, g, omega_max, model="two", top_frac=2 / 3):
        self.D, self.g, self.om, self.model = D, float(g), float(omega_max), model
        a = np.diag(np.sqrt(np.arange(1, D)), 1).astype(complex)
        ad = a.conj().T
        SA = 0.5j * (a @ a - ad @ ad)          # S(xi) = Re(xi) SA + Im(xi) SB
        SB = 0.5 * (a @ a + ad @ ad)
        I2, ID = np.eye(2), np.eye(D)
        sx = np.array([[0, 1], [1, 0]], complex)
        sy = np.array([[0, -1j], [1j, 0]], complex)
        Pg = np.diag([1.0, 0.0]).astype(complex)
        Pe = np.diag([0.0, 1.0]).astype(complex)
        # Cartesian generators, order: Omega_x, Omega_y, Re xi_g, Im xi_g, Re xi_e, Im xi_e
        self.Hc = [0.5 * np.kron(sx, ID), 0.5 * np.kron(sy, ID), np.kron(Pg, SA), np.kron(Pg, SB),
                   np.kron(Pe, SA), np.kron(Pe, SB)]
        self.Hc_sparse = []
        for M in self.Hc:
            r, c = np.nonzero(np.abs(M) > 0)
            self.Hc_sparse.append((r, c, M[r, c]))
        self.n = np.tile(np.arange(D), 2).astype(float)
        # truncation guard: levels n >= top_frac D (n >= 40 at D = 60); weight there is penalized in the cost
        self.top = (np.tile(np.arange(D), 2) >= int(np.ceil(top_frac * D))).astype(float)
        self.H0 = np.zeros((2 * D, 2 * D), complex)

    # --------------------------------------------------------------------------- variables <-> controls
    def controls(self, X):
        """X (N, 6) polar variables -> C (N, 6) Cartesian controls and the Jacobian dC/dX (N, 6, 6)"""
        N = X.shape[0]
        rO, fO, rg, fg, re, fe = X.T
        C = np.zeros((N, 6))
        J = np.zeros((N, 6, 6))
        C[:, 0], C[:, 1] = self.om * rO * np.cos(fO), self.om * rO * np.sin(fO)
        J[:, 0, 0], J[:, 0, 1] = self.om * np.cos(fO), -self.om * rO * np.sin(fO)
        J[:, 1, 0], J[:, 1, 1] = self.om * np.sin(fO), self.om * rO * np.cos(fO)
        if self.model == "onesided":
            rg = np.zeros_like(rg)
        if self.model in ("two", "onesided"):
            if self.model == "two":
                C[:, 2], C[:, 3] = self.g * rg * np.cos(fg), self.g * rg * np.sin(fg)
                J[:, 2, 2], J[:, 2, 3] = self.g * np.cos(fg), -self.g * rg * np.sin(fg)
                J[:, 3, 2], J[:, 3, 3] = self.g * np.sin(fg), self.g * rg * np.cos(fg)
            C[:, 4], C[:, 5] = self.g * re * np.cos(fe), self.g * re * np.sin(fe)
            J[:, 4, 4], J[:, 4, 5] = self.g * np.cos(fe), -self.g * re * np.sin(fe)
            J[:, 5, 4], J[:, 5, 5] = self.g * np.sin(fe), self.g * re * np.cos(fe)
        elif self.model == "sigmaz":                    # xi_e = -xi_g; variables (rho_e, phi_e) unused
            C[:, 2], C[:, 3] = self.g * rg * np.cos(fg), self.g * rg * np.sin(fg)
            C[:, 4], C[:, 5] = -C[:, 2], -C[:, 3]
            J[:, 2, 2], J[:, 2, 3] = self.g * np.cos(fg), -self.g * rg * np.sin(fg)
            J[:, 3, 2], J[:, 3, 3] = self.g * np.sin(fg), self.g * rg * np.cos(fg)
            J[:, 4], J[:, 5] = -J[:, 2], -J[:, 3]
        else:
            raise ValueError(self.model)
        return C, J

    def hamiltonians(self, C):
        return np.einsum("kj,jab->kab", C, np.array(self.Hc)) + self.H0

    # --------------------------------------------------------------------------- dynamics
    def propagate(self, C, dt, psi0):
        """states psi_0 .. psi_N (N+1, 2D) and the slice eigen-decompositions"""
        H = self.hamiltonians(C)
        lam, V = np.linalg.eigh(H)
        out = np.empty((len(C) + 1, psi0.size), complex)
        out[0] = psi0
        for k in range(len(C)):
            out[k + 1] = V[k] @ (np.exp(-1j * lam[k] * dt) * (V[k].conj().T @ out[k]))
        return out, lam, V

    def cost_grad(self, x, shape, dt, psi0, Qfin, Qrun, w_run, lam_top, w_fin=1.0):
        """C and dC/dx for the flattened polar variables x.  Qfin: projector (2D, 2D) of the final fidelity;
        Qrun: list of N projectors (or None) of the running path fidelity at psi_1 .. psi_N."""
        X = x.reshape(shape)
        N = shape[0]
        Cc, Jac = self.controls(X)
        psi, lam, V = self.propagate(Cc, dt, psi0)
        # cost = w_fin (1 - F_T) + w_run (1/N) sum_k (1 - F_k) + lam_top (1/N) sum_k P_top(psi_k)
        Ff = float(np.real(np.vdot(psi[-1], Qfin @ psi[-1])))
        cost = w_fin * (1.0 - Ff)
        # costate lambda_k = dC/d<psi_k| (Wirtinger): -Qfin psi_N - (w/N) Q_k psi_k + (lam_top/N) Top psi_k
        Fk = np.zeros(N)
        Pt = np.zeros(N)
        inj = np.zeros((N + 1, psi0.size), complex)
        for k in range(1, N + 1):
            if Qrun is not None and w_run > 0:
                qk = Qrun[k - 1] @ psi[k]
                Fk[k - 1] = float(np.real(np.vdot(psi[k], qk)))
                inj[k] -= (w_run / N) * qk
            if lam_top > 0:
                tp = self.top * psi[k]
                Pt[k - 1] = float(np.real(np.vdot(psi[k], tp)))
                inj[k] += (lam_top / N) * tp
        if Qrun is not None and w_run > 0:
            cost += w_run * float(np.mean(1.0 - Fk))
        if lam_top > 0:
            cost += lam_top * float(np.mean(Pt))
        inj[N] -= w_fin * (Qfin @ psi[-1])
        grad_c = np.zeros((N, 6))
        chi = inj[N]
        for k in range(N - 1, -1, -1):
            # dC/dc_kj = 2 Re <chi_{k+1}| dU_k/dc |psi_k>, chi_{k+1} = costate after slice k
            e = np.exp(-1j * lam[k] * dt)
            # Gamma_mn = (e_m - e_n)/(lam_m - lam_n) = -i dt exp(-i(lam_m + lam_n)dt/2) sinc((lam_m - lam_n)dt/2): stable
            xh = 0.5 * dt * (lam[k][:, None] - lam[k][None, :])
            Gam = -1j * dt * np.exp(-0.5j * dt * (lam[k][:, None] + lam[k][None, :])) * np.sinc(xh / np.pi)
            at = V[k].conj().T @ chi
            bt = V[k].conj().T @ psi[k]
            M = np.conj(at)[:, None] * Gam * bt[None, :]
            W = V[k].conj() @ M @ V[k].T
            for j, (r, c, v) in enumerate(self.Hc_sparse):
                grad_c[k, j] = 2.0 * np.real(np.sum(v * W[r, c]))
            chi = V[k] @ (np.conj(e) * (V[k].conj().T @ chi)) + inj[k]
        gx = np.einsum("kj,kjm->km", grad_c, Jac)
        return cost, gx.ravel(), dict(F_T=Ff, F_path=Fk, P_top=Pt)


# ------------------------------------------------------------------------------- helpers
def joint(osc, q=0):
    """|osc> (x) |q> in the (2, D) ordering"""
    D = osc.size
    v = np.zeros(2 * D, complex)
    v[q * D:(q + 1) * D] = osc
    return v


def osc_projector(phi):
    """1_q (x) |phi><phi| on the joint space"""
    P = np.outer(phi, phi.conj())
    return np.kron(np.eye(2), P)


def random_start(N, rng, smooth=8):
    """smooth random polar variables (low-frequency Fourier noise), rho in [0.2, 0.9]"""
    t = np.linspace(0, 1, N)
    X = np.zeros((N, 6))
    for j in range(6):
        y = np.zeros(N)
        for m in range(1, smooth + 1):
            y += rng.normal() * np.sin(np.pi * m * t + rng.uniform(0, 2 * np.pi)) / m
        if j % 2 == 0:
            X[:, j] = 0.2 + 0.7 / (1 + np.exp(-2 * y))
        else:
            X[:, j] = np.pi * y
    return X


def optimize(model, psi0, target_osc, T, N, path_states=None, w_run=0.0, lam_top=3.0, X0=None, seed=0,
             maxiter=1500, deadline=None, w_fin=1.0):
    """GRAPE at fixed duration T (us); returns dict with X, F_T, iterations.  w_fin weights the final infidelity
    (tracking runs use w_fin >> w_run so that F_T stays at the target while the closeness to the path is raised).
    The truncation guard enters with weight lam_top * w_fin, i.e. always lam_top relative to the fidelity term: with
    lam_top = 0.3 (unscaled) pulses reached F_T = 1 at D = 60 by climbing to <n> ~ 15-28, and lost it at D = 80."""
    lam_top = lam_top * w_fin
    rng = np.random.default_rng(seed)
    X0 = random_start(N, rng) if X0 is None else X0
    shape = X0.shape
    dt = T / N
    Qfin = osc_projector(target_osc)
    Qrun = [osc_projector(p) for p in path_states] if path_states is not None else None
    bounds = [(0.0, 1.0) if j % 2 == 0 else (None, None) for _ in range(N) for j in range(6)]
    t0 = time.time()
    hist = []

    def fun(x):
        c, g, info = model.cost_grad(x, shape, dt, psi0, Qfin, Qrun, w_run, lam_top, w_fin)
        hist.append(c)
        if deadline is not None and time.time() > deadline:
            raise TimeoutError
        return c, g
    try:
        res = minimize(fun, X0.ravel(), jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=1e-13, gtol=1e-9, maxcor=30))
        x = res.x
        nit = int(res.nit)
    except TimeoutError:
        x = None
        nit = -1
    if x is None:
        x = X0.ravel()
    X = x.reshape(shape)
    c, _, info = model.cost_grad(X.ravel(), shape, dt, psi0, Qfin, Qrun, w_run if Qrun is not None else 0.0, lam_top, w_fin)
    return dict(X=X, cost=c, F_T=info["F_T"], nit=nit, wall_s=time.time() - t0, hist=hist[::10])


def resample_path(states, n_out):
    """time-matched path states at s_k = k/n_out (k = 1..n_out) along a dense path (rows, constant speed in arclength)"""
    P = np.asarray(states)
    ov = np.abs(np.sum(np.conj(P[:-1]) * P[1:], axis=1))
    seg = np.r_[0, np.cumsum(np.arccos(np.clip(ov, -1, 1)))]
    t = seg[-1] * np.arange(1, n_out + 1) / n_out
    out = []
    for ti in t:
        i = min(max(np.searchsorted(seg, ti) - 1, 0), len(P) - 2)
        u, v = P[i], P[i + 1]
        v = v * np.exp(-1j * np.angle(np.vdot(u, v)))
        th = seg[i + 1] - seg[i]
        a = ti - seg[i]
        if th < 1e-12:
            w = u
        else:
            w = (np.sin(th - a) * u + np.sin(a) * v) / np.sin(th)
        out.append(w / np.linalg.norm(w))
    return np.array(out)
