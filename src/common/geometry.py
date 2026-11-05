"""Comparison of a pulse (joint qubit-oscillator trajectory) with a reference path.

closeness at equal time   F(t_k) = <p(t_k)| rho_osc(t_k) |p(t_k)>,  rho_osc = Tr_q |psi_k><psi_k|
closest approach          C = min_k F(t_k)                     (the protocol deck's metric, Eqs. (10)-(13))
two upper bounds          F <= lambda_max(rho_osc)   (entanglement with the qubit)
                          F <= 1 - w_out, w_out = weight of rho_osc outside span{|S>, |chi>} (leakage off the plane)
Bures angle               Theta(rho, sigma) = arccos sqrt F(rho, sigma)  (pure: arccos|<psi|phi>|)
Path length of a pulse    L = sum_k Theta(psi_k, psi_{k+1}) of the JOINT pure state (Fubini-Study length)
"""
from __future__ import annotations

import numpy as np


def rho_osc(psi2):
    P = np.asarray(psi2).reshape(2, -1)
    return P.T @ P.conj()


def closeness(traj, path):
    """traj (K, 2D) joint states, path (K, D) reference states at the same times -> dict of curves"""
    D = path.shape[1]
    F, lam, wout, pur = [], [], [], []
    S = path[0]
    for psi, p in zip(traj, path):
        r = rho_osc(psi)
        F.append(float(np.real(np.vdot(p, r @ p))))
        ev = np.linalg.eigvalsh(r)
        lam.append(float(ev[-1]))
        pur.append(float(np.real(np.trace(r @ r))))
    # plane of the path: span{S, chi}, chi from the end point
    T = path[-1]
    chi = T - np.vdot(S, T) * S
    chi = chi / np.linalg.norm(chi)
    Pp = np.outer(S, S.conj()) + np.outer(chi, chi.conj())
    for psi in traj:
        r = rho_osc(psi)
        wout.append(float(1 - np.real(np.trace(Pp @ r))))
    F, lam, wout, pur = map(np.array, (F, lam, wout, pur))
    return dict(F=F, lam_max=lam, w_out=wout, purity=pur, C=float(F.min()), F_mean=float(F.mean()),
                bound=float(np.minimum(lam, 1 - wout).min()))


def joint_length(traj):
    P = np.asarray(traj)
    ov = np.abs(np.sum(np.conj(P[:-1]) * P[1:], axis=1))
    return float(np.sum(np.arccos(np.clip(ov, -1, 1))))


def nbar_osc(traj, D):
    n = np.arange(D)
    return np.array([float(np.sum(np.abs(np.asarray(p).reshape(2, D)) ** 2 @ n)) for p in traj])


def target_fidelity(traj, target):
    return np.array([float(np.real(np.vdot(target, rho_osc(p) @ target))) for p in traj])
