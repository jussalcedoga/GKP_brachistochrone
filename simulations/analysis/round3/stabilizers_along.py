"""Round 3: stabilizer expectation values <S_X> = <D(2 alpha)>, <S_Z> = <D(2 beta)> (alpha = sqrt(pi/2), beta = i alpha)
along the flip brachistochrone and along the photon-optimal path between the code words (Delta = 0.4, cutoff 80; the
displacements are exponentiated at cutoff 220 and cut to 80).  A logical operation leaves the stabilizer subsystem
unchanged (Shaw, Doherty, Grimsmo, PRX Quantum 5, 010331 (2024)); a path through the vacuum does not.
  python simulations/analysis/round3/stabilizers_along.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import math
import pathlib

import numpy as np
from scipy.linalg import expm

from common import photon_core as pc

HERE = _L.vfile(__file__).parent
D, PAD = 80, 220
a = np.diag(np.sqrt(np.arange(1, PAD)), 1)


def disp(z):
    return expm(z * a.conj().T - np.conj(z) * a)[:D, :D]


al = math.sqrt(math.pi / 2)
SX, SZ = disp(2 * al), disp(2j * al)
L = pc.Logical(0.4, D)
full = lambda X: np.array([[x[i // 2] if i % 2 == 0 else 0.0 for i in range(D)] for x in X])  # noqa: E731


def stab(X):
    F = full(X)
    return np.real(np.einsum("ki,ij,kj->k", F, SX, F)), np.real(np.einsum("ki,ij,kj->k", F, SZ, F))


geo = L.geo(np.linspace(0, L.th, 201))
U, _ = pc.optimize_path(np.diag(L.nv), L.u0, L.uT, [pc.resample(L.detour(), 60)], 60, maxiter=20000)
ph = pc.dense(U, 11)[0]
out = {}
for name, X in (("brachistochrone", geo), ("photon_optimal", ph)):
    sx, sz = stab(X)
    out[name] = dict(SX_min=float(sx.min()), SX_max=float(sx.max()), SZ_min=float(sz.min()), SZ_max=float(sz.max()))
    print(name, {k: round(v, 4) for k, v in out[name].items()})
out["vacuum_exact"] = math.exp(-math.pi)
(HERE / "stabilizers_along.json").write_text(json.dumps(out, indent=1))
