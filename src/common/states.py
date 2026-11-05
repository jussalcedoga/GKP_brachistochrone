"""Bosonic target states in the Fock basis.

Conventions (identical to the photon-optimal-path note): [q, p] = i, D(gamma) = exp(gamma a^dag - gamma^* a),
finite-energy square-lattice GKP code words
    |mu_L>  ~  exp(-Delta^2 n) sum_{m = mu (mod 2), l} exp(-i pi m l / 2) D(m alpha + l beta) |0>,
    alpha = sqrt(pi/2), beta = i alpha,  |m|, |l| <= 10,
even cats |C_alpha> ~ |alpha> + |-alpha>, Fock states |n>, and the lowest binomial ('kitten') code word
(|0> + |4>)/sqrt2 for the universality outlook.  All targets used here are real and parity-even.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def coherent(g, D):
    """Fock amplitudes of the coherent state |g>, n = 0 .. D-1"""
    n = np.arange(D)
    c = np.zeros(D, complex)
    if abs(g) < 1e-14:
        c[0] = 1.0
        return c
    return np.exp(-abs(g) ** 2 / 2 + n * np.log(abs(g)) - 0.5 * gammaln(n + 1) + 1j * n * np.angle(g))


def gkp(Delta, mu=0, D=80, M=10, Dbig=260):
    """finite-energy GKP code word |mu_L> (mu = 0: +Z, 1: -Z), truncated to D and normalized"""
    a = np.sqrt(np.pi / 2)
    psi = np.zeros(Dbig, complex)
    for m in range(-M, M + 1):
        if m % 2 != mu:
            continue
        for l in range(-M, M + 1):
            psi += np.exp(-1j * np.pi * m * l / 2) * coherent(m * a + 1j * l * a, Dbig)
    psi *= np.exp(-Delta ** 2 * np.arange(Dbig))
    psi = psi[:D]
    psi = psi / np.linalg.norm(psi)
    return _real_gauge(psi)


def cat(alpha, D=80, parity=+1):
    """even (parity=+1) or odd (-1) cat ~ |alpha> + parity |-alpha>"""
    psi = coherent(alpha, D) + parity * coherent(-alpha, D)
    return _real_gauge(psi / np.linalg.norm(psi))


def fock(n, D=80):
    psi = np.zeros(D, complex)
    psi[n] = 1.0
    return psi


def binomial_kitten(D=80):
    psi = np.zeros(D, complex)
    psi[0] = psi[4] = 1 / np.sqrt(2)
    return psi


def vacuum(D=80):
    return fock(0, D)


def _real_gauge(psi):
    """fix the global phase so that the first non-negligible amplitude is real positive (all targets are real)"""
    k = int(np.argmax(np.abs(psi) > 1e-12))
    return psi * np.exp(-1j * np.angle(psi[k]))


def target(name, D=80, **kw):
    """named targets of the paper: 'fock2', 'fock4', 'cat', 'gkp', 'binomial'; kw override Delta / alpha"""
    if name.startswith("fock"):
        return fock(int(name[4:]), D)
    if name == "cat":
        return cat(kw.get("alpha", 2.0), D)
    if name == "gkp":
        return gkp(kw.get("Delta", 0.4), 0, D)
    if name == "gkp1":
        return gkp(kw.get("Delta", 0.4), 1, D)
    if name == "binomial":
        return binomial_kitten(D)
    raise KeyError(name)


LABELS = {"fock2": r"Fock $|2\rangle$", "fock4": r"Fock $|4\rangle$", "cat": r"cat $|C_2^+\rangle$",
          "gkp": r"GKP $|0_L\rangle$", "flip": r"flip $|0_L\rangle\!\to\!|1_L\rangle$", "binomial": "binomial"}
PLAIN = {"fock2": "Fock |2>", "fock4": "Fock |4>", "cat": "even cat (alpha = 2)", "gkp": "GKP |0_L> (Delta = 0.4)",
         "flip": "flip |0_L> -> |1_L>", "binomial": "binomial (|0>+|4>)/sqrt2"}
