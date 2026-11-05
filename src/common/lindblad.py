"""Noisy evaluation of a GRAPE pulse (zero temperature Lindblad master equation, used for evaluation only).

    drho/dt = -i[H(t), rho] + kappa D[a] rho + kappa_phi D[n] rho + gamma_1 D[sigma_-] rho + (gamma_phi/2) D[sigma_z] rho
with D[L]rho = L rho L^dag - {L^dag L, rho}/2.  Each slice (constant H) is split into sub-steps of <= dt_max; each
sub-step applies the exact unitary of the slice and then the exact channel of every dissipator (they commute with each
other to the order kept by the splitting; the splitting error is O(dt_sub^2) and checked in S3 by halving dt_max):
    photon loss          Kraus operators K_m (m <= m_max) of the amplitude-damping channel, p = 1 - exp(-kappa dt)
    cavity dephasing     rho_nm *= exp(-kappa_phi (n - m)^2 dt / 2)
    qubit relaxation     exact amplitude damping, p = 1 - exp(-gamma_1 dt)
    qubit dephasing      rho_ge *= exp(-gamma_phi dt)
Rates from params.NOISE: kappa = 1/T1c, kappa_phi = 2/Tphic (so that rho_01 decays at 1/Tphic), gamma_1 = 1/T1q,
gamma_phi = 1/Tphiq.  Optional self-Kerr (K/2) a^dag^2 a^2 (deck cross-check only).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

from .params import NOISE


def rates(name):
    s = NOISE[name]
    inv = lambda x: 0.0 if not x else 1.0 / x      # noqa: E731
    return dict(kappa=inv(s["T1c"]), kphi=2 * inv(s.get("Tphic")), g1=inv(s["T1q"]), gphi=inv(s["Tphiq"]),
                kerr=s.get("kerr_per_s", 0.0) * 1e-6)                # K in rad/us


def _loss_weights(D, p, m_max):
    """w[m, n] = sqrt(C(n, m) p^m (1-p)^(n-m)) for the Kraus operator K_m |n> = w[m, n] |n - m>"""
    n = np.arange(D)
    W = np.zeros((m_max + 1, D))
    for m in range(m_max + 1):
        ok = n >= m
        lw = gammaln(n[ok] + 1) - gammaln(m + 1) - gammaln(n[ok] - m + 1) + m * np.log(max(p, 1e-300)) + (n[ok] - m) * np.log1p(-p)
        W[m, ok] = np.exp(0.5 * lw)
    return W


def evolve(model, C, dt, psi0, noise, dt_max=1.0e-3, m_max=4, record_every=None):
    """rho(T) (2D, 2D) for the Cartesian controls C (N, 6), slice length dt (us), noise = rates dict;
    record_every: also return the oscillator states rho_osc at the slice ends k = 0, r, 2r, ..."""
    D = model.D
    H = model.hamiltonians(C)
    if noise.get("kerr", 0.0) > 0:
        a = np.diag(np.sqrt(np.arange(1, D)), 1)
        HK = 0.5 * noise["kerr"] * (a.T @ a.T @ a @ a)
        H = H + np.kron(np.eye(2), HK)[None]
    lam, V = np.linalg.eigh(H)
    nsub = max(1, int(math.ceil(dt / dt_max - 1e-9)))
    h = dt / nsub
    W = _loss_weights(D, 1 - math.exp(-noise["kappa"] * h), m_max) if noise["kappa"] > 0 else None
    n = np.arange(D)
    deph = np.exp(-0.5 * noise["kphi"] * h * (n[:, None] - n[None, :]) ** 2) if noise["kphi"] > 0 else None
    p1 = 1 - math.exp(-noise["g1"] * h)
    fphi = math.exp(-noise["gphi"] * h)
    rho = np.outer(psi0, psi0.conj())
    rec = []
    for k in range(len(C)):
        Uh = V[k] @ np.diag(np.exp(-1j * lam[k] * h)) @ V[k].conj().T
        for _ in range(nsub):
            rho = Uh @ rho @ Uh.conj().T
            r = rho.reshape(2, D, 2, D)
            if W is not None:
                y = np.zeros_like(r)
                for m in range(m_max + 1):
                    wm = W[m, m:]
                    y[:, :D - m, :, :D - m] += r[:, m:, :, m:] * wm[None, :, None, None] * wm[None, None, None, :]
                r = y
            if deph is not None:
                r = r * deph[None, :, None, :]
            if p1 > 0:                                             # sigma_- = |g><e| (index 1 -> 0)
                y = r.copy()
                y[0, :, 0, :] = r[0, :, 0, :] + p1 * r[1, :, 1, :]
                y[1, :, 1, :] = (1 - p1) * r[1, :, 1, :]
                y[0, :, 1, :] = math.sqrt(1 - p1) * r[0, :, 1, :]
                y[1, :, 0, :] = math.sqrt(1 - p1) * r[1, :, 0, :]
                r = y
            if fphi < 1:
                r = r.copy()
                r[0, :, 1, :] *= fphi
                r[1, :, 0, :] *= fphi
            rho = r.reshape(2 * D, 2 * D)
        if record_every and (k + 1) % record_every == 0:
            rr = rho.reshape(2, D, 2, D)
            rec.append(rr[0, :, 0, :] + rr[1, :, 1, :])
    return (rho, rec) if record_every else rho


def osc_fidelity(rho_joint, target_osc):
    D = target_osc.size
    r = rho_joint.reshape(2, D, 2, D)
    ro = r[0, :, 0, :] + r[1, :, 1, :]
    return float(np.real(np.vdot(target_osc, ro @ target_osc)))
