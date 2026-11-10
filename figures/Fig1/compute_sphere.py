"""Fig. 1(a, b) on one common sphere (Nov 9, author and advisor review: both panels on the same footing).

The sphere is the unit sphere of the real three-dimensional subspace spanned by |0>, |0_L> and |1_L>, so (a) and (b)
share the contours of the mean photon number and the positions of the three states. The paths are those of
simulations/paths/fig1_brachistochrone.py, recomputed here with the same calls (real even sector, cutoff 80) and checked
against its stored per-panel projections. The brachistochrones lie in the subspace exactly. The photon-optimal and
trade-off paths leave it slightly, so the figure shows their normalized projections, and their smallest weight in the
subspace is recorded (w_in_min).
Writes fig1_sphere.json in this folder.   python figures/Fig1/compute_sphere.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))
import mfcommon as M  # noqa: E402,I001  (puts the shared library of manuscript/ on the path)

import numpy as np  # noqa: E402

from common import paths, photon_core as pc, states  # noqa: E402
from common.params import GKP_DELTA, kappa_mu  # noqa: E402

D = 80
STORED = json.loads((M.MS / "fig1_brachistochrone" / "results" / "fig1.json").read_text())

# the paths, exactly as in simulations/paths/fig1_brachistochrone.py
P = paths.Prep(states.gkp(GKP_DELTA, 0, D))
geo = P.geo(np.linspace(0, P.th, 200))
Uph, _ = paths.photon_path_prep(P)
Xph, _ = pc.dense(Uph, 41)
Lg = pc.Logical(GKP_DELTA, D)
Nf = np.diag(Lg.nv)
geo_f = Lg.geo(np.linspace(0, Lg.th, 200))
Uf, _ = pc.optimize_path(Nf, Lg.u0, Lg.uT, [pc.resample(Lg.detour(), 30)], 30)
kq, mq = kappa_mu("Yale")
Ut = paths.tradeoff_family(Nf, Lg.u0, Lg.uT, Lg.geo(np.linspace(0, Lg.th, 32)), Uf, [kq / mq])[0]
PATHS = {"sphere_prep": {"geodesic": geo, "photon": Xph},
         "sphere_flip": {"geodesic": geo_f, "photon": pc.dense(Uf, 41)[0], "tradeoff": pc.dense(Ut, 41)[0]}}
assert np.array_equal(P.nv, Lg.nv), "the two problems use different sectors"
assert np.allclose(P.uT, Lg.u0, atol=1e-12), "|0_L> differs between the two problems"
nv = Lg.nv
vac = np.eye(len(nv))[0]


def panel_basis(u0, uT, Xph):
    """the per-panel basis of compute.py (start, path direction, main out-of-plane direction of the photon path)"""
    e1 = u0 / np.linalg.norm(u0)
    e2 = uT - (uT @ e1) * e1
    e2 /= np.linalg.norm(e2)
    resid = Xph - np.outer(Xph @ e1, e1) - np.outer(Xph @ e2, e2)
    e3 = np.linalg.svd(resid, full_matrices=False)[2][0]
    e3 = e3 - (e3 @ e1) * e1 - (e3 @ e2) * e2
    e3 /= np.linalg.norm(e3)
    E = np.array([e1, e2, e3])
    if np.mean(Xph @ e3) < 0:
        E[2] *= -1
    return E


def project(E, X):
    c = X @ E.T
    w = np.sum(c ** 2, 1)
    return c / np.sqrt(w)[:, None], float(w.min())


# 1. the recomputed paths are those of the stored figure data
check = {}
for key, (u0, uT) in (("sphere_prep", (P.u0, P.uT)), ("sphere_flip", (Lg.u0, Lg.uT))):
    E = panel_basis(u0, uT, PATHS[key]["photon"])
    for k, X in PATHS[key].items():
        xyz, _ = project(E, X)
        check[f"{key}/{k}"] = float(np.max(np.abs(xyz - np.array(STORED[key]["paths"][k]["xyz"]))))
print("recomputed paths against the stored per-panel projections (max abs difference):",
      {k: f"{v:.1e}" for k, v in check.items()}, flush=True)
assert max(check.values()) < 1e-6, check

# 2. the common sphere: span{|0_L>, |1_L>, |0>}, orthonormalized in this order, the vacuum on the positive third axis
e1 = Lg.u0 / np.linalg.norm(Lg.u0)
e2 = Lg.uT - (Lg.uT @ e1) * e1
e2 /= np.linalg.norm(e2)
e3 = vac - (vac @ e1) * e1 - (vac @ e2) * e2
e3 /= np.linalg.norm(e3)
E = np.array([e1, e2, e3])
N3 = E @ (nv[:, None] * E.T)
pts = {name: (E @ v / np.linalg.norm(E @ v)).tolist() for name, v in (("vacuum", vac), ("0L", Lg.u0), ("1L", Lg.uT))}
res = dict(note=__doc__.split("\n\n")[0], basis="|0_L>, |1_L>, |0> (Gram-Schmidt)", check=check, states=pts)
for key in PATHS:
    out = dict(N3=N3.tolist(), paths={}, vacuum=dict(xyz=pts["vacuum"], w_in=1.0))
    for k, X in PATHS[key].items():
        xyz, wmin = project(E, X)
        out["paths"][k] = dict(xyz=xyz.tolist(), w_in_min=wmin)
    res[key] = out
    print(key, "smallest weight in the common subspace:", {k: round(v["w_in_min"], 4) for k, v in out["paths"].items()},
          flush=True)
ang = {f"{a}-{b}": float(np.degrees(np.arccos(np.clip(np.dot(pts[a], pts[b]), -1, 1))))
       for a, b in (("vacuum", "0L"), ("vacuum", "1L"), ("0L", "1L"))}
res["angles_deg"] = ang
print("angles between the three states (deg):", {k: round(v, 2) for k, v in ang.items()})
(HERE / "fig1_sphere.json").write_text(json.dumps(res))
print("wrote", HERE / "fig1_sphere.json")
