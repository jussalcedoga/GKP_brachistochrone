"""Fig. 2 data: the optimal photon path, analytic against numerical.

All paths at the same energy variance Delta H = omega = g/sqrt2 (t = s/omega).
  preparation |0> -> GKP |0_L> (Delta = 0.4), even cat (alpha = 2), Fock |2>, |4>:
     brachistochrone (Prop. 3: <n> = n_chi sin^2 s), numerical optimal photon path (30 free nodes, L-BFGS-B, exact arc
     integrals; best of 4 starts), closed-form first-order path (Prop. 4, hypergeometric), lower bound B (Thm. 2)
  logical flip |0_L> -> |1_L>: brachistochrone (Prop. 5), numerical optimal photon path (ell = 0, start: detour),
     closed-form detour through the vacuum (Prop. 6), lower bound max{B(0_L), B(1_L)}
Writes results/fig2.json.  python simulations/paths/fig2_photon_optimal.py   (~2 min)
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from common import paths, photon_core as pc, states  # noqa: E402
from common.params import GKP_DELTA, OMEGA_REF  # noqa: E402

D = 80
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def prof(U, N, target, geo_dense, sub=41):
    X, s = pc.dense(U, sub)
    nv = np.diag(N)
    nb = np.sum(X ** 2 * nv, 1)
    dt = np.diff(s, prepend=0.0) / OMEGA_REF                          # us
    expo = np.cumsum(np.r_[0, 0.5 * (nb[1:] + nb[:-1]) * np.diff(s)]) / OMEGA_REF * 1e3   # photon ns
    del dt
    return dict(t_ns=(1e3 * s / OMEGA_REF).tolist(), nbar=nb.tolist(), F_T=((X @ target) ** 2).tolist(),
                expo=expo.tolist(), D_geo=pc.dist_to(X, geo_dense).tolist(), L=pc.length(U), J=pc.exposure(U, N),
                T_ns=1e3 * pc.length(U) / OMEGA_REF, J_ns=1e3 * pc.exposure(U, N) / OMEGA_REF)


res = dict(params=dict(D=D, omega=OMEGA_REF, Delta=GKP_DELTA), prep={}, flip={})
for name, vec in (("gkp", states.gkp(GKP_DELTA, 0, D)), ("cat", states.cat(2.0, D)), ("fock2", states.fock(2, D)),
                  ("fock4", states.fock(4, D))):
    P = paths.Prep(vec)
    geo = P.geo(np.linspace(0, P.th, 32))
    Xg, _ = pc.dense(geo, 41)
    U, J = paths.photon_path_prep(P)
    Bnd = float(int(name[4:]) * np.pi / 4) if name.startswith("fock") else pc.lower_bound(P.nv, P.uT)  # Thm. 1 / 2
    r = dict(theta0=P.th, n_chi=P.n_chi, sigma_chi=P.sig, J_geo_closed=P.J_geo(), B=Bnd,
             geodesic=prof(geo, P.N, P.uT, Xg), photon=prof(U, P.N, P.uT, Xg))
    if P.sig > 1e-9:                                                  # closed form (Prop. 4) exists for sigma_chi > 0
        Xc, info = pc.closed_form_path(P)
        Uc = pc.resample(Xc, 60)
        r["closed"] = prof(Uc, P.N, P.uT, Xg)
        r["closed"]["a_s"] = info["s"].tolist()
        r["closed"]["a"] = info["a"].tolist()
        r["closed"]["beta"], r["closed"]["mu"] = float(info["beta"]), float(info["mu"])
        # numerical deviation a(s) of the photon path from the geodesic, signed along xi (Fig. 3 of the note)
        xi = (P.nv * P.chi - P.n_chi * P.chi) / P.sig
        Xn, sn = pc.dense(U, 41)
        sg = np.clip(np.arctan2(Xn @ P.chi, Xn @ P.u0), 0, P.th)       # position along the geodesic
        r["photon"]["a_proj_s"] = sg.tolist()
        r["photon"]["a_proj"] = np.arcsin(np.clip(Xn @ xi, -1, 1)).tolist()
    res["prep"][name] = r
    print(f"{name}: theta0 = {P.th:.4f}, J_geo = {P.J_geo():.4f} (closed form), J_num = {J:.4f} "
          f"({100 * (1 - J / P.J_geo()):.1f}% saving), L_num/theta0 = {pc.length(U) / P.th:.4f}, B = {r['B']:.4f}"
          + (f", closed form J = {r['closed']['J']:.4f}" if "closed" in r else ""), flush=True)

L = pc.Logical(GKP_DELTA, D)
N = np.diag(L.nv)
geo = L.geo(np.linspace(0, L.th, 32))
Xg, _ = pc.dense(geo, 41)
det = L.detour()
Ud = pc.resample(det, 60)
U0, J0 = pc.optimize_path(N, L.u0, L.uT, [pc.resample(det, 30)], 30)
B0 = max(pc.lower_bound(L.nv, L.u0), pc.lower_bound(L.nv, L.uT))
# Prop. 6 in closed form: legs |0_L> -> |0> -> |1_L>, J_det = th_a J(th_a, n0) + th_b J(th_b, n1)
Jl = lambda th, n: n / np.sin(th) ** 2 * (0.5 - np.sin(2 * th) / (4 * th))          # noqa: E731
tha, thb = np.arccos(abs(L.u0[0])), np.arccos(abs(L.uT[0]))
det_closed = dict(theta_a=float(tha), theta_b=float(thb), L=float(tha + thb), J=float(tha * Jl(tha, L.n0) + thb * Jl(thb, L.n1)),
                  T_ns=float(1e3 * (tha + thb) / OMEGA_REF))
res["flip"] = dict(theta=L.th, n0=L.n0, n1=L.n1, N01=L.N01, J_geo_closed=L.J_geo(), B=B0, detour_closed=det_closed,
                   geodesic=prof(geo, N, L.uT, Xg), photon=prof(U0, N, L.uT, Xg), detour=prof(Ud, N, L.uT, Xg))
for k in ("geodesic", "photon", "detour"):
    X, s = pc.dense(geo if k == "geodesic" else (U0 if k == "photon" else Ud), 41)
    res["flip"][k]["leak"] = (1 - np.sum(X * (X @ L.Pi), 1)).tolist()
    res["flip"][k]["vac"] = (X[:, 0] ** 2).tolist()
print(f"flip: theta = {L.th:.4f}, J_geo = {L.J_geo():.4f}, J_num(ell=0) = {J0:.4f}, L = {pc.length(U0):.4f}, "
      f"detour (closed form) J = {det_closed['J']:.4f}, L = {det_closed['L']:.4f}, B = {B0:.4f}", flush=True)
(OUT / "fig2.json").write_text(json.dumps(res))
print("wrote", OUT / "fig2.json")
