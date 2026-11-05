"""The brachistochrone (time-optimal) Hamiltonian between two states and its Fock-space structure.

For |S> -> |T> with c_0 = |<S|T>|, theta_0 = arccos c_0 and |chi> = (|T~> - c_0|S>)/sqrt(1 - c_0^2)
(|T~> = |T> e^{-i arg<S|T>}), the minimum-time evolution at fixed energy variance Delta H = omega is
    H_opt = i omega (|chi><S| - |S><chi|),    |p(t)> = cos(omega t)|S> + sin(omega t)|chi>,    T* = theta_0/omega
(Carlini et al., PRL 96, 060503 (2006); Anandan & Aharonov, PRL 65, 1697 (1990)).  H_opt is rank two, so its
operator norm equals Delta H = omega on |S>.
Normal-ordered form (exact on the truncated space):  |m><n| = sum_p (-1)^p a^dag^(m+p) a^(n+p) / (p! sqrt(m! n!)),
so H_opt = sum_{j,k} h_jk a^dag^j a^k with h_jk = sum_{p<=min(j,k)} (-1)^p A_{j-p,k-p}/(p! sqrt((j-p)!(k-p)!)),
A = i omega(|chi><S| - |S><chi|) in the Fock basis.  The 'order' of a term is j + k (photons created + annihilated).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.special import gammaln


def geodesic_parts(S, T):
    """(c0, theta0, chi) with the Pancharatnam gauge on T"""
    ov = np.vdot(S, T)
    Tg = T * np.exp(-1j * np.angle(ov)) if abs(ov) > 1e-14 else T
    c0 = float(abs(ov))
    chi = Tg - c0 * S
    chi = chi / np.linalg.norm(chi)
    return c0, float(np.arccos(min(c0, 1.0))), chi


def H_opt(S, T, omega=1.0):
    _, _, chi = geodesic_parts(S, T)
    return 1j * omega * (np.outer(chi, S.conj()) - np.outer(S, chi.conj()))


def path_states(S, T, s):
    """geodesic states at arclength s (array)"""
    _, _, chi = geodesic_parts(S, T)
    s = np.atleast_1d(s)
    return np.cos(s)[:, None] * S[None] + np.sin(s)[:, None] * chi[None]


def normal_ordered(A):
    """coefficients h[j, k] of A = sum h_jk a^dag^j a^k (A given in the Fock basis, D x D, truncated)"""
    D = A.shape[0]
    lf = gammaln(np.arange(D) + 1)
    h = np.zeros((D, D), complex)
    for j in range(D):
        for k in range(D):
            acc = 0.0 + 0.0j
            for p in range(min(j, k) + 1):
                acc += (-1) ** p * A[j - p, k - p] * np.exp(-gammaln(p + 1) - 0.5 * (lf[j - p] + lf[k - p]))
            h[j, k] = acc
    return h


def from_normal_ordered(h, D, max_order=None):
    """operator sum_{j+k <= max_order} h_jk a^dag^j a^k on the D-dimensional Fock space"""
    a = np.diag(np.sqrt(np.arange(1, D)), 1).astype(complex)
    ad = a.conj().T
    apow = [np.eye(D, dtype=complex)]
    adpow = [np.eye(D, dtype=complex)]
    for _ in range(1, D):
        apow.append(apow[-1] @ a)
        adpow.append(adpow[-1] @ ad)
    M = np.zeros((D, D), complex)
    for j in range(D):
        for k in range(D):
            if max_order is not None and j + k > max_order:
                continue
            if abs(h[j, k]) > 0:
                M += h[j, k] * adpow[j] @ apow[k]
    return M


def truncated_order_fidelity(S, T, orders, omega=1.0):
    """target fidelity |<T|exp(-i H_K T*)|S>|^2 when the normal-ordered expansion of H_opt is truncated at total order
    j + k <= K (K in orders); the truncated operator is Hermitized, (H_K + H_K^dag)/2"""
    D = S.size
    c0, th, _ = geodesic_parts(S, T)
    H = H_opt(S, T, omega)
    h = normal_ordered(H)
    out = {}
    for K in orders:
        HK = from_normal_ordered(h, D, K)
        HK = 0.5 * (HK + HK.conj().T)
        psi = expm(-1j * HK * th / omega) @ S
        out[int(K)] = float(abs(np.vdot(T, psi)) ** 2)
    return out
