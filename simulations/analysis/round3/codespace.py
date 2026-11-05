"""Round 3 (Oct 28): code-space averages for logical operations.

If an oscillator evolution U(t) keeps the code space C invariant, U(t) C = C at every t, then for any state functional
f the Haar average over code states, <f(U(t) psi)>_C, is independent of t (Haar invariance on C).  Hence the
code-averaged first-order error of a code-preserving logical operation of duration T is
    eps_C = T [Gamma_q + kappa nbar_C + kappa_phi V_C],
    nbar_C = Tr(Pi n)/2,   V_C = <Var(n)>_C = Tr(Pi n^2)/2 - [Tr(Pi n)^2 + Tr((Pi n Pi)^2)]/6   (qubit Haar moments).
This script evaluates nbar_C, V_C (closed form and Monte Carlo over Haar states of C), the flip brachistochrone T*, the
code-averaged first-order error at T* for the Yale and CS-proposal rates, and checks the constancy along the flip
brachistochrone numerically (Haar average of nbar and Var(n) at several t).
  python simulations/analysis/round3/codespace.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib

import numpy as np

from common import photon_core as pc
from common.params import NOISE, OMEGA_REF

HERE = _L.vfile(__file__).parent
L = pc.Logical(0.4, 80)
n = L.nv
Q = np.linalg.qr(np.array([L.u0, L.uT]).T)[0]                 # orthonormal basis of C (real)
A = Q.T @ np.diag(n) @ Q
A2 = Q.T @ np.diag(n ** 2) @ Q
nbar_C = np.trace(A) / 2
V_C = np.trace(A2) / 2 - (np.trace(A) ** 2 + np.trace(A @ A)) / 6
rng = np.random.default_rng(1)
Z = rng.standard_normal((200000, 2)) + 1j * rng.standard_normal((200000, 2))
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
m1 = np.real(np.einsum("ki,ij,kj->k", Z.conj(), A, Z))
m2 = np.real(np.einsum("ki,ij,kj->k", Z.conj(), A2, Z))
mc = dict(nbar=float(m1.mean()), var=float((m2 - m1 ** 2).mean()))
# constancy along the brachistochrone: U(t) = exp(-i omega t sigma_y) in the basis (|0_L>, |1_L^perp>) of C
e1 = (L.uT - L.c * L.u0) / np.sqrt(1 - L.c ** 2)
B = np.array([L.u0, e1]).T
Ab = B.T @ np.diag(n) @ B
A2b = B.T @ np.diag(n ** 2) @ B
sy = np.array([[0, -1j], [1j, 0]])
along = []
for f in np.linspace(0, 1, 6):
    U = np.cos(f * L.th) * np.eye(2) - 1j * np.sin(f * L.th) * sy
    W = Z @ U.T
    a1 = np.real(np.einsum("ki,ij,kj->k", W.conj(), Ab, W))
    a2 = np.real(np.einsum("ki,ij,kj->k", W.conj(), A2b, W))
    along.append(dict(s=float(f), nbar=float(a1.mean()), var=float((a2 - a1 ** 2).mean())))
Tstar = L.th / OMEGA_REF                                       # us
out = dict(nbar_C=float(nbar_C), V_C=float(V_C), monte_carlo=mc, along_brachistochrone=along, T_star_ns=1e3 * Tstar,
           exposure_C_ns=float(1e3 * nbar_C * Tstar), n0=float(L.n0), n1=float(L.n1))
for name in ("Yale", "CSproposal"):
    s = NOISE[name]
    Gq = 1 / (2 * s["Tphiq"]) + 1 / (4 * s["T1q"])
    k = 1 / s["T1c"]
    kp = 2 / s["Tphic"] if s.get("Tphic") else 0.0
    out[f"eps_C_{name}"] = dict(total=float(Tstar * (Gq + k * nbar_C + kp * V_C)), ancilla=float(Tstar * Gq),
                                loss=float(Tstar * k * nbar_C), dephasing=float(Tstar * kp * V_C))
print(json.dumps(out, indent=1))
(HERE / "codespace.json").write_text(json.dumps(out, indent=1))
