"""Path tracking with qubit-controlled bosonic gate sets: the protocol of the protocol deck (the 'tangential policy').

The pulse is asked to stay on a reference path at the same fraction of the path at every slice (deck Eqs. (10)-(21)):
    F_k = sum_q |<g(s_k), q|psi_k>|^2,   s_k = k/N (uniform) or sum_{j<=k} d_j / sum_j d_j (free profile, stage D),
    L   = w sum_{k<N} max(0, thr - F_k)^2 + w_fin max(0, thr_fin - F_N)^2  (+ lam_top <P_top>, truncation guard),
minimized over the slice variables x_k = (r, phi, a_0, f_0, a_1, f_1) in the box [0,1] for r, a_q (deck Eq. (16)) with
L-BFGS-B and the exact adjoint gradient, through the stage schedule of Supplementary Table 6; the tracking weight is the largest
w of the ladder W = {10, ..., 2000} whose polished pulse keeps the final fidelity with Lindblad noise >= 0.99 (deck
Eq. (29), level 2: warm starts and two bisections).

Gate sets, with the same bounds, the same slices and the same durations:
    'cs'  qubit-conditioned squeezing     sum_q |q><q| (x) (i/2)(xi_q^* a^2 - xi_q a^dag^2),    |xi_q| <= g  (deck Eq. (1))
    'cd'  qubit-conditioned displacement  sum_q |q><q| (x) (i/sqrt2)(eps_q^* a - eps_q a^dag),  |eps_q| <= g (the ECD
          generator: sigma_z-conditional plus unconditional displacements),
    'snap' SNAP-displacement control (SNAPModel): fixed dispersive shift chi = g, unconditional displacement |eps| <= g,
all with the qubit drive |Omega| <= 10 g.  On the vacuum every oscillator generator moves the state at the Fubini-Study
speed g/sqrt2 at its bound, so all three have the same energy-variance resource.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from . import grape, lindblad

SCHEDULE = (  # Supplementary Table 6: (name, T_m/T, thr, w, w_fin, thr_fin, iterations, free progress profile)
    [(f"A{i + 1}", 0.625, t, 10.0, 200.0, 0.9999, 350, False) for i, t in enumerate((0.6, 0.8, 0.9, 0.955))]
    + [(f"B{i + 1}", 1.0, t, 10.0, 200.0, 0.9999, 400, False) for i, t in enumerate((0.9, 0.955))]
    + [(f"C{i + 1}", 1.0, t, 10.0, 2e4, 0.99995, 300, False) for i, t in enumerate((0.80, 0.85, 0.90, 0.93, 0.95))]
    + [("D", 1.0, 0.90, 10.0, 2e4, 0.99995, 300, True)]
    + [(f"E{i + 1}", 1.0, t, 1000.0, 1000.0, 0.99995, 300, False) for i, t in enumerate((0.97, 0.99, 0.999))])
POLISH = dict(thr=0.97, w_fin=1e5, thr_fin=0.99995, iters=300)          # Supplementary Table 6, 'E, polish'
LADDER = (10.0, 20.0, 40.0, 70.0, 150.0, 300.0, 600.0, 1000.0, 2000.0)   # deck Eq. (29)
F_LOSS_MIN = 0.99


# ----------------------------------------------------------------------------- gate sets
class CDModel(grape.CSModel):
    """qubit drive plus a displacement for each qubit state (the ECD generator), same variables and bounds as CSModel"""

    def __init__(self, D, g, omega_max, top_frac=2 / 3):
        super().__init__(D, g, omega_max, "two", top_frac)
        a = np.diag(np.sqrt(np.arange(1, D)), 1).astype(complex)
        ad = a.conj().T
        DA = (1j / math.sqrt(2)) * (a - ad)         # G(eps) = Re(eps) DA + Im(eps) DB = (i/sqrt2)(eps^* a - eps a^dag)
        DB = (a + ad) / math.sqrt(2)
        ID = np.eye(D)
        sx = np.array([[0, 1], [1, 0]], complex)
        sy = np.array([[0, -1j], [1j, 0]], complex)
        Pg, Pe = np.diag([1.0, 0.0]).astype(complex), np.diag([0.0, 1.0]).astype(complex)
        self.Hc = [0.5 * np.kron(sx, ID), 0.5 * np.kron(sy, ID), np.kron(Pg, DA), np.kron(Pg, DB), np.kron(Pe, DA),
                   np.kron(Pe, DB)]
        self.Hc_sparse = []
        for M in self.Hc:
            r, c = np.nonzero(np.abs(M) > 0)
            self.Hc_sparse.append((r, c, M[r, c]))


class SNAPModel(grape.CSModel):
    """the Hamiltonian of SNAP-displacement control (Krastanov et al. 2015, Heeres et al. 2015): an always-on dispersive
    shift -chi a^dag a |e><e| with chi = g (the same coupling energy scale as the CS and CD bounds), the qubit drive
    |Omega| <= 10 g, and an unconditional cavity drive (i/sqrt2)(eps^* a - eps a^dag) with |eps| <= g, which moves the vacuum
    at the same Fubini-Study speed g/sqrt2 as the CS and CD generators.  Number-selective qubit rotations (the SNAP gates)
    and displacements both emerge from these controls.  Same six variables per slice as CSModel: (rho_Omega, phi_Omega),
    (rho_g, phi_g) -> eps = g rho_g e^{i phi_g}; (rho_e, phi_e) are unused (zero generators): the dispersive coupling is
    fixed, not a control."""

    def __init__(self, D, g, omega_max, top_frac=2 / 3, chi=None):
        super().__init__(D, g, omega_max, "two", top_frac)
        self.chi = float(g if chi is None else chi)
        a = np.diag(np.sqrt(np.arange(1, D)), 1).astype(complex)
        ad = a.conj().T
        DA = (1j / math.sqrt(2)) * (a - ad)
        DB = (a + ad) / math.sqrt(2)
        I2, ID = np.eye(2), np.eye(D)
        sx = np.array([[0, 1], [1, 0]], complex)
        sy = np.array([[0, -1j], [1j, 0]], complex)
        Pe = np.diag([0.0, 1.0]).astype(complex)
        Z = np.zeros((2 * D, 2 * D), complex)
        self.Hc = [0.5 * np.kron(sx, ID), 0.5 * np.kron(sy, ID), np.kron(I2, DA), np.kron(I2, DB), Z, Z.copy()]
        self.Hc_sparse = []
        for M in self.Hc:
            r, c = np.nonzero(np.abs(M) > 0)
            self.Hc_sparse.append((r, c, M[r, c]))
        self.H0 = -self.chi * np.kron(Pe, np.diag(np.arange(D)).astype(complex))


def make_model(gate, D, g, omega_max, top_frac=2 / 3):
    """gate: 'cs' conditional squeezing, 'cd' conditional displacement (ECD), 'snap' dispersive SNAP-displacement;
    top_frac: the truncation guard penalizes the population at n >= top_frac D"""
    if gate == "cs":
        return grape.CSModel(D, g, omega_max, "two", top_frac)
    if gate == "cd":
        return CDModel(D, g, omega_max, top_frac)
    if gate == "snap":
        return SNAPModel(D, g, omega_max, top_frac)
    raise ValueError(gate)


# ----------------------------------------------------------------------------- reference path (piecewise great circle)
class Path:
    """a dense path from S to T (rows: oscillator states, full Fock vectors), parametrized by the arclength fraction s"""

    def __init__(self, dense):
        P = np.array(dense, complex)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        for i in range(1, len(P)):                      # gauge: consecutive overlaps real and positive
            P[i] *= np.exp(-1j * np.angle(np.vdot(P[i - 1], P[i])))
        th = np.arccos(np.clip(np.abs(np.sum(P[:-1].conj() * P[1:], 1)), -1, 1))
        keep = np.r_[True, th > 1e-12]
        self.P = P[keep]
        self.th = np.arccos(np.clip(np.abs(np.sum(self.P[:-1].conj() * self.P[1:], 1)), -1, 1))
        self.cum = np.r_[0.0, np.cumsum(self.th)]
        self.L = float(self.cum[-1])

    def at(self, s):
        """states g(s) and derivatives dg/ds at the fractions s (array)"""
        sig = np.clip(np.asarray(s, float), 0, 1) * self.L
        i = np.clip(np.searchsorted(self.cum, sig, side="right") - 1, 0, len(self.th) - 1)
        a = (sig - self.cum[i])[:, None]
        th = self.th[i][:, None]
        u, v = self.P[i], self.P[i + 1]
        G = (np.sin(th - a) * u + np.sin(a) * v) / np.sin(th)
        dG = self.L * (-np.cos(th - a) * u + np.cos(a) * v) / np.sin(th)
        return G, dG


# ----------------------------------------------------------------------------- hinge loss and exact gradient
def cost_grad(model, x, N, dt, psi0, path, w, thr, w_fin, thr_fin, lam_top=1e3, free=False):
    """deck Eq. (20) and its gradient with respect to the polar slice variables (and the progress weights d if free)"""
    D = model.D
    Xp = x[:6 * N].reshape(N, 6)
    if free:
        d = x[6 * N:]
        cs = np.cumsum(d)
        s = cs / cs[-1]
    else:
        s = np.arange(1, N + 1) / N
    C, Jac = model.controls(Xp)
    psi, lam, V = model.propagate(C, dt, psi0)
    G, dG = path.at(s)
    Pk = psi[1:].reshape(N, 2, D)
    amp = np.einsum("kd,kqd->kq", G.conj(), Pk)
    F = np.sum(np.abs(amp) ** 2, 1)
    h = np.maximum(0.0, thr - F)
    h[-1] = max(0.0, thr_fin - F[-1])
    wk = np.full(N, w)
    wk[-1] = w_fin
    cost = float(np.sum(wk * h ** 2))
    coef = -2.0 * wk * h                                 # dL/dF_k
    inj = np.zeros((N + 1, 2 * D), complex)
    inj[1:] = (coef[:, None, None] * G[:, None, :] * amp[:, :, None]).reshape(N, 2 * D)
    if lam_top > 0:
        Pt = np.real(np.sum(model.top * np.abs(psi[1:]) ** 2, 1))
        cost += lam_top * float(np.mean(Pt))
        inj[1:] += (lam_top / N) * model.top * psi[1:]
    grad_c = np.zeros((N, 6))
    chi = inj[N]
    for k in range(N - 1, -1, -1):
        e = np.exp(-1j * lam[k] * dt)
        xh = 0.5 * dt * (lam[k][:, None] - lam[k][None, :])
        Gam = -1j * dt * np.exp(-0.5j * dt * (lam[k][:, None] + lam[k][None, :])) * np.sinc(xh / np.pi)
        at = V[k].conj().T @ chi
        bt = V[k].conj().T @ psi[k]
        M = np.conj(at)[:, None] * Gam * bt[None, :]
        W = V[k].conj() @ M @ V[k].T
        for j, (r, c, v) in enumerate(model.Hc_sparse):
            grad_c[k, j] = 2.0 * np.real(np.sum(v * W[r, c]))
        chi = V[k] @ (np.conj(e) * (V[k].conj().T @ chi)) + inj[k]
    gx = np.einsum("kj,kjm->km", grad_c, Jac).ravel()
    if free:                                             # deck Eq. (27): speed profile
        dF = 2.0 * np.real(np.sum(np.conj(amp) * np.einsum("kd,kqd->kq", dG.conj(), Pk), 1))
        dLs = coef * dF
        dLd = (np.cumsum(dLs[::-1])[::-1] - np.sum(s * dLs)) / cs[-1]
        gx = np.r_[gx, dLd]
    return cost, gx, dict(F=F, s=s)


def stage(model, X, N, T, psi0, path, thr, w, w_fin, thr_fin, iters, free=False, lam_top=1e3, d0=None):
    """one minimization of deck Eq. (28) (L-BFGS-B started at X); returns (X, d, info)"""
    dt = T / N
    bounds = [(0.0, 1.0) if j % 2 == 0 else (None, None) for _ in range(N) for j in range(6)]
    x0 = X.ravel()
    if free:
        d0 = np.ones(N) if d0 is None else d0
        x0 = np.r_[x0, d0]
        bounds = bounds + [(1e-3, None)] * N
    fun = lambda x: cost_grad(model, x, N, dt, psi0, path, w, thr, w_fin, thr_fin, lam_top, free)[:2]  # noqa: E731
    r = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                 options=dict(maxiter=iters, maxfun=2 * iters, ftol=1e-14, gtol=1e-10, maxcor=30))
    c, _, info = cost_grad(model, r.x, N, dt, psi0, path, w, thr, w_fin, thr_fin, lam_top, free)
    return r.x[:6 * N].reshape(N, 6), (r.x[6 * N:] if free else None), dict(cost=c, F_T=float(info["F"][-1]),
                                                                           C=float(info["F"].min()), nit=int(r.nit))


# ----------------------------------------------------------------------------- evaluation
def lindblad_F(model, X, T, N, psi0, target, noise, dt_max=2e-3):
    C, _ = model.controls(X)
    rho = lindblad.evolve(model, C, T / N, psi0, noise, dt_max=dt_max)
    return lindblad.osc_fidelity(rho, target)


class PathModel:
    """the reference path itself as a Hamiltonian evolution (qubit idle in |g>), for its fidelity under the same noise:
    M equal arcs traversed at the constant speed L/T, arc j generated by i(L/T)(|phi_j><u_j| - h.c.)"""

    def __init__(self, path, D, M=400):
        self.D = D
        s = np.linspace(0, 1, M + 1)
        G, _ = path.at(s)
        H = []
        for j in range(M):
            u, v = G[j], G[j + 1]
            v = v * np.exp(-1j * np.angle(np.vdot(u, v)))
            th = np.arccos(np.clip(abs(np.vdot(u, v)), -1, 1))
            phi = (v - np.vdot(u, v) * u) / max(np.sin(th), 1e-15)
            Hj = 1j * (np.outer(phi, u.conj()) - np.outer(u, phi.conj())) * th
            H.append(np.kron(np.eye(2), Hj))
        self.Hs = np.array(H)            # per unit time of one arc: multiply by M/T
        self.M = M

    def hamiltonians(self, C):
        return self.Hs * C[:, :1, None]   # C[:, 0] = M/T (arcs per us)


def path_lindblad_F(path, D, T, target, noise, M=400, dt_max=2e-3):
    pm = PathModel(path, D, M)
    C = np.full((M, 6), M / T)
    psi0 = grape.joint(path.P[0][:D])
    rho = lindblad.evolve(pm, C, T / M, psi0, noise, dt_max=dt_max)
    return lindblad.osc_fidelity(rho, target)


def diagnostics(model, X, T, N, psi0, path, target, sub=2):
    """closeness on the slice ends and on a finer grid (sub points per slice), photon number, bounds, pulse speed"""
    from . import geometry
    C, _ = model.controls(X)
    dt = T / N
    traj, _, _ = model.propagate(np.repeat(C, sub, axis=0), dt / sub, psi0)
    s = np.arange(N * sub + 1) / (N * sub)
    G, _ = path.at(s)
    cl = geometry.closeness(traj, G)
    ends = traj[::sub]
    cl_end = geometry.closeness(ends, G[::sub])
    D = model.D
    nb = geometry.nbar_osc(traj, D)
    ov = np.abs(np.sum(traj[:-1].conj() * traj[1:], 1))
    L_pulse = float(np.sum(np.arccos(np.clip(ov, -1, 1))))
    t = T * s
    return dict(C=float(cl["C"]), C_ends=float(cl_end["C"]), F_mean=float(cl["F_mean"]), bound=float(cl["bound"]),
                F_T=float(geometry.target_fidelity(traj[-1:], target)[0]),
                expo_ns=float(1e3 * np.trapezoid(nb, t)) if hasattr(np, "trapezoid") else float(1e3 * np.trapz(nb, t)),
                nbar_max=float(nb.max()), L_pulse=L_pulse, Lambda=L_pulse / path.L,
                t_ns=(1e3 * t).tolist(), F=cl["F"].tolist(), lam_max=cl["lam_max"].tolist(), w_out=cl["w_out"].tolist(),
                nbar=nb.tolist(), F_T_curve=geometry.target_fidelity(traj, target).tolist(),
                F_T_path=[float(abs(np.vdot(target, gg[:D])) ** 2) for gg in G], nbar_path=[
                    float(np.real(np.vdot(gg, np.arange(len(gg)) * gg))) for gg in G])
