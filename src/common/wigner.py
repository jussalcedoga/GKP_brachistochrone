"""Wigner function on a square phase-space grid (Hermite-function method).

W(x, p) = (1/pi) int dy psi*(x + y) psi(x - y) exp(2 i p y),  x = (a + a^dag)/sqrt2, p = i(a^dag - a)/sqrt2.
The y-integral uses the same grid (spacing h).  A density matrix is handled through its eigen-decomposition.
Same method as the in-house csqrl.physics.WignerGrid (re-implemented here so the manuscript is self-contained).
"""
from __future__ import annotations

import functools

import numpy as np

W_LIM = 1.0 / np.pi          # fixed colour limits +- 1/pi for every Wigner panel


class WignerGrid:
    def __init__(self, D, L=5.0, npts=161):
        self.D, self.L = D, L
        x = np.linspace(-L, L, npts)
        self.x = self.p = x
        h = x[1] - x[0]
        self.h = h
        H = np.zeros((D, npts))
        H[0] = np.pi ** (-0.25) * np.exp(-x ** 2 / 2)
        if D > 1:
            H[1] = np.sqrt(2) * x * H[0]
        for k in range(2, D):
            H[k] = np.sqrt(2 / k) * x * H[k - 1] - np.sqrt((k - 1) / k) * H[k - 2]
        self.H = H
        kk = np.arange(-(npts - 1), npts)
        self.Eph = np.exp(2j * np.outer(x, kk * h))
        ii = np.arange(npts)
        ip, im = ii[None, :] + kk[:, None], ii[None, :] - kk[:, None]
        self.valid = (ip >= 0) & (ip < npts) & (im >= 0) & (im < npts)
        self.ipc, self.imc = np.clip(ip, 0, npts - 1), np.clip(im, 0, npts - 1)

    def of_vectors(self, V):
        """sum of the Wigner functions of the (unnormalized) Fock vectors V (k, D) -> array (p, x)"""
        W = np.zeros((len(self.p), len(self.x)))
        for v in np.atleast_2d(V):
            f = v[: self.D] @ self.H
            W += np.real(self.Eph @ (np.conj(f[self.ipc]) * f[self.imc] * self.valid))
        return W * self.h / np.pi

    def of_rho(self, rho, tol=1e-8):
        lam, U = np.linalg.eigh(rho)
        keep = lam > tol
        return self.of_vectors(np.sqrt(lam[keep])[:, None] * U[:, keep].T)

    def of_joint(self, psi2):
        """Wigner function of the oscillator state Tr_q |psi><psi| of a joint vector psi2 (2, D)"""
        return self.of_vectors(np.asarray(psi2).reshape(2, -1))


@functools.lru_cache(maxsize=8)
def grid(D, L=5.0, npts=161):
    return WignerGrid(D, L, npts)
