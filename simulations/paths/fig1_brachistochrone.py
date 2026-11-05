"""Fig. 1 data: the brachistochrone between bosonic states and its Hamiltonian.

For every transfer (vacuum -> Fock |2>, |4>, even cat alpha = 2, GKP |0_L> Delta = 0.4; logical flip |0_L> -> |1_L>):
  c0, theta0, T* = theta0/omega (omega = g/sqrt2, g = 7.5 rad/us), the Fock content |<n|chi>|^2 of the path direction
  (the levels H_opt couples to the start), the fidelity reached when the normal-ordered expansion of H_opt is truncated
  at total order j + k <= K, and Wigner functions of the path state at t/T* = 0, 1/4, 1/2, 3/4, 1.
Writes results/fig1.json and results/fig1_wigner.npz.  python simulations/paths/fig1_brachistochrone.py   (~1 min)
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from common import brachistochrone as B, states  # noqa: E402
from common.params import G_SQ, GKP_DELTA, OMEGA_REF  # noqa: E402
from common.wigner import grid  # noqa: E402

D = 80
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
vac = states.vacuum(D)
g0, g1 = states.gkp(GKP_DELTA, 0, D), states.gkp(GKP_DELTA, 1, D)
TRANSFERS = {"fock2": (vac, states.fock(2, D)), "fock4": (vac, states.fock(4, D)), "cat": (vac, states.cat(2.0, D)),
             "gkp": (vac, g0), "flip": (g0, g1)}

res = dict(params=dict(D=D, g=G_SQ, omega=OMEGA_REF, Delta=GKP_DELTA, alpha=2.0), transfers={})
for name, (S, T) in TRANSFERS.items():
    c0, th, chi = B.geodesic_parts(S, T)
    H = B.H_opt(S, T, OMEGA_REF)
    nchi = float(np.sum(abs(chi) ** 2 * np.arange(D)))
    res["transfers"][name] = dict(c0=c0, theta0=th, T_star_ns=1e3 * th / OMEGA_REF, chi2=(abs(chi[:41]) ** 2).tolist(),
                                  n_chi=nchi, H_norm=float(np.linalg.norm(H, 2)),
                                  n_S=float(np.sum(abs(S) ** 2 * np.arange(D))), n_T=float(np.sum(abs(T) ** 2 * np.arange(D))))
    print(f"{name}: c0 = {c0:.5f}, theta0 = {th:.4f} rad, T* = {1e3 * th / OMEGA_REF:.1f} ns, <n>_chi = {nchi:.3f}, "
          f"||H_opt|| = {res['transfers'][name]['H_norm']:.4f} (= omega {OMEGA_REF:.4f})", flush=True)

# operator order needed: truncated normal-ordered expansion (cutoff 40: all targets converged there)
Ds = 40
orders = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 56, 78]
TR_S = {"fock4": (states.vacuum(Ds), states.fock(4, Ds)), "cat": (states.vacuum(Ds), states.cat(2.0, Ds)),
        "gkp": (states.vacuum(Ds), states.gkp(GKP_DELTA, 0, Ds)), "flip": (states.gkp(GKP_DELTA, 0, Ds), states.gkp(GKP_DELTA, 1, Ds)),
        "fock2": (states.vacuum(Ds), states.fock(2, Ds))}
res["order"] = dict(orders=orders, D=Ds)
for name, (S, T) in TR_S.items():
    f = B.truncated_order_fidelity(S, T, orders, 1.0)
    res["order"][name] = [f[k] for k in orders]
    print(f"order {name}: " + ", ".join(f"K={k}: {f[k]:.4f}" for k in orders), flush=True)

# Wigner snapshots along the brachistochrone
G = grid(D, 5.0, 161)
fr = [0.0, 0.25, 0.5, 0.75, 1.0]
W = {}
for name in ("gkp", "flip", "cat", "fock2", "fock4"):
    S, T = TRANSFERS[name]
    th = res["transfers"][name]["theta0"]
    P = B.path_states(S, T, np.array(fr) * th)
    W[name] = np.array([G.of_vectors(p[None]) for p in P])
    res["transfers"][name]["F_target_snap"] = [float(abs(np.vdot(T, p)) ** 2) for p in P]
np.savez_compressed(OUT / "fig1_wigner.npz", x=G.x, fr=np.array(fr), **W)


# state-space picture (panels a, b): the real 3D subspace spanned by the start, the path direction chi and the main
# direction in which the photon-optimal path leaves the brachistochrone plane (first right-singular vector of its
# out-of-plane part); <n> on that unit sphere and the coordinates of the paths
def sphere(u0, uT, nv, others):
    e1 = u0 / np.linalg.norm(u0)
    e2 = uT - (uT @ e1) * e1
    e2 /= np.linalg.norm(e2)
    Xph = others["photon"]
    resid = Xph - np.outer(Xph @ e1, e1) - np.outer(Xph @ e2, e2)
    e3 = np.linalg.svd(resid, full_matrices=False)[2][0]
    e3 = e3 - (e3 @ e1) * e1 - (e3 @ e2) * e2
    e3 /= np.linalg.norm(e3)
    E = np.array([e1, e2, e3])
    if np.mean(Xph @ e3) < 0:                       # orient e3 toward the photon-optimal path
        E[2] *= -1
    N3 = E @ (nv[:, None] * E.T)
    vac = np.eye(len(nv))[0]
    cv = E @ vac
    out = dict(N3=N3.tolist(), paths={}, vacuum=dict(xyz=(cv / np.linalg.norm(cv)).tolist(), w_in=float(cv @ cv)))
    for k, X in others.items():
        c = X @ E.T
        w = np.sum(c ** 2, 1)
        out["paths"][k] = dict(xyz=(c / np.sqrt(w)[:, None]).tolist(), w_in_min=float(w.min()))
    return out


from common import paths, photon_core as pc  # noqa: E402,I001
from common.params import kappa_mu  # noqa: E402

P = paths.Prep(g0)
geo = P.geo(np.linspace(0, P.th, 200))
Uph, _ = paths.photon_path_prep(P)
Xph, _ = pc.dense(Uph, 41)
res["sphere_prep"] = sphere(P.u0, P.uT, P.nv, {"geodesic": geo, "photon": Xph})
Lg = pc.Logical(GKP_DELTA, D)
Nf = np.diag(Lg.nv)
geo_f = Lg.geo(np.linspace(0, Lg.th, 200))
Uf, _ = pc.optimize_path(Nf, Lg.u0, Lg.uT, [pc.resample(Lg.detour(), 30)], 30)
kq, mq = kappa_mu("Yale")
Ut = paths.tradeoff_family(Nf, Lg.u0, Lg.uT, Lg.geo(np.linspace(0, Lg.th, 32)), Uf, [kq / mq])[0]
res["sphere_flip"] = sphere(Lg.u0, Lg.uT, Lg.nv, {"geodesic": geo_f, "photon": pc.dense(Uf, 41)[0],
                                                  "tradeoff": pc.dense(Ut, 41)[0]})
for key in ("sphere_prep", "sphere_flip"):
    print(key, {k: round(v["w_in_min"], 4) for k, v in res[key]["paths"].items()}, flush=True)
(OUT / "fig1.json").write_text(json.dumps(res, indent=1))
print("wrote", OUT / "fig1.json")
