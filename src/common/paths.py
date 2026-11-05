"""Analytic and numerical reference paths in state space (time-optimal, photon-optimal, trade-off).

Built on photon_core (the note's numerics, real even-parity sector):
  brachistochrone  geodesic |p(s)> = cos s |S> + sin s |chi>, s in [0, theta_0], H_opt = i omega (|chi><S| - |S><chi|)
  photon-optimal   minimizer of J = int <n> ds (weighted geodesic, Euler-Lagrange Eq. (3) of the note)
  trade-off        minimizer of int (1 + w <n>) ds  <=>  photon_core exposure with N -> n + I/w ;  w = kappa/mu
                   minimizes the first-order error eps = (mu T + kappa int<n>dt) at equal Delta H (T = L/omega)
All paths are traversed at the same energy variance Delta H = omega = params.OMEGA_REF, so T = L/omega.
"""
from __future__ import annotations

import numpy as np

from . import photon_core as pc
from .params import OMEGA_REF, kappa_mu


# ----------------------------------------------------------------------------- problems
class Prep(pc.Prep):
    """|0> -> |T> for any real even target vector (full Fock vector, length D)"""

    def __init__(self, target_full):
        D = target_full.size
        self.nv, self.uT = pc.even_real(target_full.astype(complex), D)
        d = len(self.nv)
        self.u0 = np.eye(d)[0]
        self.N = np.diag(self.nv)
        self.c0 = self.u0 @ self.uT
        self.th = np.arccos(self.c0)
        chi = self.uT - self.c0 * self.u0
        self.chi = chi / np.linalg.norm(chi)
        self.n_chi = self.chi @ (self.nv * self.chi)
        self.sig = np.sqrt(max(self.chi @ (self.nv ** 2 * self.chi) - self.n_chi ** 2, 0.0))
        self.D = D


def to_full(U, D):
    """even-sector real vectors (rows, length ceil(D/2)) -> full Fock vectors (rows, D), complex"""
    U = np.atleast_2d(U)
    out = np.zeros((U.shape[0], D), complex)
    out[:, 0::2] = U
    return out


# ----------------------------------------------------------------------------- path optimizers
def photon_path_prep(P, K=30, starts=4):
    geo = P.geo(np.linspace(0, P.th, K + 2))
    st = [geo] + [pc.noisy_geodesic(geo, s) for s in range(1, starts)]
    return pc.optimize_path(P.N, P.u0, P.uT, st, K)


def tradeoff_path(N, u0, uT, w, starts, K=30):
    """minimizer of int (1 + w <N>) ds = w * int <N + I/w> ds"""
    U, J = pc.optimize_path(N + np.eye(len(N)) / w, u0, uT, starts, K)
    return U


def tradeoff_family(N, u0, uT, geo_nodes, photon_nodes, ws, K=30):
    """trade-off paths for every w in ws, continuation from both ends (geodesic and photon-optimal); at each w the
    lower-cost local optimum is kept.  Returns list of node arrays."""
    ws = np.asarray(ws)
    up, down = {}, {}
    prev = geo_nodes
    for w in np.sort(ws):                              # continuation from the geodesic side
        prev = tradeoff_path(N, u0, uT, w, [prev], K)
        up[w] = prev
    prev = photon_nodes
    for w in np.sort(ws)[::-1]:                        # continuation from the photon-optimal side
        prev = tradeoff_path(N, u0, uT, w, [prev], K)
        down[w] = prev
    out = []
    for w in ws:
        cost = lambda U: pc.length(U) + w * pc.exposure(U, N)   # noqa: E731
        out.append(up[w] if cost(up[w]) <= cost(down[w]) else down[w])
    return out


# ----------------------------------------------------------------------------- diagnostics
def profile(U, N, sub=41, target=None, Pi=None):
    """dense samples of a node path: arclength s, time t (ns) at omega, <n>, Var(n), fidelity with target, leakage"""
    X, s = pc.dense(U, sub)
    nv = np.diag(N)
    nb = np.sum(X ** 2 * nv, 1)
    var = np.sum(X ** 2 * nv ** 2, 1) - nb ** 2
    out = dict(s=s, t_ns=1e3 * s / OMEGA_REF, nbar=nb, var=var, X=X)
    if target is not None:
        out["F_target"] = (X @ target) ** 2
    if Pi is not None:
        out["leak"] = 1 - np.sum(X * (X @ Pi), 1)
    return out


def summary(U, N, geo_dense=None):
    L = pc.length(U)
    J = pc.exposure(U, N)
    out = dict(L=L, J=J, T_ns=1e3 * L / OMEGA_REF, expo_photon_ns=1e3 * J / OMEGA_REF)
    if geo_dense is not None:
        X, _ = pc.dense(U, 41)
        out["D_geo"] = float(pc.dist_to(X, geo_dense).max())
    return out


def first_order_error(L, J, noise):
    """eps = (mu L + kappa J)/omega: first-order error of a path at Delta H = omega (Lemma 1 of the note + mu proxy)"""
    kappa, mu = kappa_mu(noise)
    return (mu * L + kappa * J) / OMEGA_REF
