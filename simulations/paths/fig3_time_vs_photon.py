"""Fig. 3 data: how close are the optimal time and optimal photon paths?

1. Target scan (all at Delta H = omega = g/sqrt2): preparation from the vacuum of Fock |2>, |4>, |6>, even cats
   alpha = 1, 1.5, 2, 2.5, GKP |0_L> Delta = 0.30, 0.34, 0.40, 0.45, 0.50, the binomial word (|0> + |4>)/sqrt2;
   logical flip |0_L> -> |1_L> (Delta = 0.4) without (ell = 0) and with the code-space penalty ell = 3, 10, 30, 100
   (Eq. (2) of the note: N = n + ell(1 - Pi), needed for a gate that must act on both code words).
   Reported: time cost T_ph/T* - 1 (= L/theta0 - 1), exposure saving 1 - J_ph/J_geo, largest Bures distance D_geo.
2. Time-photon trade-off family: minimizers of int (1 + w<n>) ds for w in [1e-2, 1e3] (w = 0: brachistochrone,
   w -> infinity: photon-optimal path), for the flip and for GKP preparation; first-order error
   eps(w) = (mu L + kappa J)/omega for the two literature noise sets; sweet spot w* = kappa/mu.
3. Wigner functions at the middle of the flip paths (brachistochrone, trade-off at w*_Yale, photon-optimal) and of the
   GKP preparation paths.
Writes results/fig3.json, results/fig3_wigner.npz.  python simulations/paths/fig3_time_vs_photon.py   (~10 min)
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from common import paths, photon_core as pc, states  # noqa: E402
from common.params import GKP_DELTA, NOISE, OMEGA_REF, kappa_mu  # noqa: E402
from common.wigner import grid  # noqa: E402

D = 80
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
res = dict(params=dict(D=D, omega=OMEGA_REF), scan=[], tradeoff={}, noise={})
for k in ("Yale", "CSproposal"):
    kap, mu = kappa_mu(k)
    res["noise"][k] = dict(kappa=kap, mu=mu, w_star=kap / mu, source=NOISE[k]["source"])
    print(f"noise {k}: kappa = {kap:.3e}/us, mu = {mu:.3e}/us, w* = kappa/mu = {kap / mu:.3f}")


def prep_entry(family, par, vec):
    P = paths.Prep(vec)
    geo = P.geo(np.linspace(0, P.th, 32))
    Xg, _ = pc.dense(geo, 41)
    U, J = paths.photon_path_prep(P)
    X, _ = pc.dense(U, 41)
    e = dict(kind="prep", family=family, par=par, theta0=P.th, L_ph=pc.length(U), J_ph=J, J_geo=P.J_geo(),
             time_cost=pc.length(U) / P.th - 1, saving=1 - J / P.J_geo(), D_geo=float(pc.dist_to(X, Xg).max()))
    print(f"{family} {par}: theta0 = {P.th:.4f}, time cost {100 * e['time_cost']:.2f}%, saving {100 * e['saving']:.2f}%, "
          f"D_geo = {e['D_geo']:.4f} rad", flush=True)
    return e, P, geo, U


for n in (2, 4, 6):
    res["scan"].append(prep_entry("fock", n, states.fock(n, D))[0])
for a in (1.0, 1.5, 2.0, 2.5):
    res["scan"].append(prep_entry("cat", a, states.cat(a, D))[0])
for d in (0.30, 0.34, 0.40, 0.45, 0.50):
    e, P_gkp, geo_gkp, U_gkp = prep_entry("gkp", d, states.gkp(d, 0, D))
    res["scan"].append(e)
    if abs(d - GKP_DELTA) < 1e-9:
        keep_gkp = (P_gkp, geo_gkp, U_gkp)
res["scan"].append(prep_entry("binomial", 0, states.binomial_kitten(D))[0])

# logical flip with and without the code-space penalty (continuation from large ell down to 0, detour start at 0)
L = pc.Logical(GKP_DELTA, D)
N = np.diag(L.nv)
geo_f = L.geo(np.linspace(0, L.th, 32))
Xgf, _ = pc.dense(geo_f, 41)
prev = geo_f
flip_paths = {}
for ell in (100.0, 30.0, 10.0, 3.0, 0.0):
    starts = [prev] + ([pc.resample(L.detour(), 30)] if ell <= 3 else [])
    U, _ = pc.optimize_path(L.Nl(ell), L.u0, L.uT, starts, 30)
    prev = U
    flip_paths[ell] = U
    X, _ = pc.dense(U, 41)
    J = pc.exposure(U, N)
    leak = float(np.max(1 - np.sum(X * (X @ L.Pi), 1)))
    e = dict(kind="flip", family="flip", par=ell, theta0=L.th, L_ph=pc.length(U), J_ph=J, J_geo=L.J_geo(),
             time_cost=pc.length(U) / L.th - 1, saving=1 - J / L.J_geo(), D_geo=float(pc.dist_to(X, Xgf).max()), peak_leak=leak)
    res["scan"].append(e)
    print(f"flip ell = {ell}: time cost {100 * e['time_cost']:.2f}%, saving {100 * e['saving']:.2f}%, D_geo = {e['D_geo']:.4f}, "
          f"peak leakage {100 * leak:.2f}%", flush=True)

# trade-off family
ws = np.logspace(-2, 3, 26)
for key, (Nm, u0, uT, geo, Uph, th, Jg) in {
        "flip": (N, L.u0, L.uT, geo_f, flip_paths[0.0], L.th, L.J_geo()),
        "gkp": (keep_gkp[0].N, keep_gkp[0].u0, keep_gkp[0].uT, keep_gkp[1], keep_gkp[2], keep_gkp[0].th, keep_gkp[0].J_geo())}.items():
    fam = paths.tradeoff_family(Nm, u0, uT, geo, Uph, ws)
    Ls = [pc.length(U) for U in fam]
    Js = [pc.exposure(U, Nm) for U in fam]
    Xg, _ = pc.dense(geo, 41)
    Dg = [float(pc.dist_to(pc.dense(U, 41)[0], Xg).max()) for U in fam]
    mn = [float(np.min(np.sum(pc.dense(U, 41)[0] ** 2 * np.diag(Nm), 1))) for U in fam]
    r = dict(w=ws.tolist(), L=Ls, J=Js, T_ns=[1e3 * x / OMEGA_REF for x in Ls], expo_ns=[1e3 * x / OMEGA_REF for x in Js],
             D_geo=Dg, nbar_min=mn, theta0=th, J_geo=Jg, L_photon=pc.length(Uph), J_photon=pc.exposure(Uph, Nm), eps={})
    for nk in ("Yale", "CSproposal"):
        e0 = paths.first_order_error(th, Jg, nk)
        r["eps"][nk] = dict(eps_geo=e0, eps=[paths.first_order_error(a, b, nk) for a, b in zip(Ls, Js)],
                            eps_photon=paths.first_order_error(r["L_photon"], r["J_photon"], nk))
        ws_ = res["noise"][nk]["w_star"]
        Ustar = paths.tradeoff_family(Nm, u0, uT, geo, Uph, [ws_])[0]
        r["eps"][nk]["at_w_star"] = dict(w=ws_, L=pc.length(Ustar), J=pc.exposure(Ustar, Nm),
                                         eps=paths.first_order_error(pc.length(Ustar), pc.exposure(Ustar, Nm), nk),
                                         nbar_min=float(np.min(np.sum(pc.dense(Ustar, 41)[0] ** 2 * np.diag(Nm), 1))),
                                         D_geo=float(pc.dist_to(pc.dense(Ustar, 41)[0], Xg).max()))
        print(f"{key} {nk}: w* = {ws_:.3f}: eps(geo) = {e0:.3e}, eps(w*) = {r['eps'][nk]['at_w_star']['eps']:.3e} "
              f"({100 * (r['eps'][nk]['at_w_star']['eps'] / e0 - 1):+.1f}%), eps(photon) = {r['eps'][nk]['eps_photon']:.3e}, "
              f"T(w*) = {1e3 * r['eps'][nk]['at_w_star']['L'] / OMEGA_REF:.0f} ns, min n = {r['eps'][nk]['at_w_star']['nbar_min']:.3f}",
              flush=True)
        if key == "flip" and nk == "Yale":
            U_star_flip = Ustar
    res["tradeoff"][key] = r

# Wigner functions at the middle of the paths
G = grid(D, 5.0, 161)


def mid(U):
    X, s = pc.dense(U, 41)
    j = int(np.argmin(abs(s - s[-1] / 2)))
    return paths.to_full(X[j], D)[0]


W = dict(flip_geo=G.of_vectors(mid(geo_f)[None]), flip_star=G.of_vectors(mid(U_star_flip)[None]),
         flip_photon=G.of_vectors(mid(flip_paths[0.0])[None]), gkp_geo=G.of_vectors(mid(keep_gkp[1])[None]),
         gkp_photon=G.of_vectors(mid(keep_gkp[2])[None]))
np.savez_compressed(OUT / "fig3_wigner.npz", x=G.x, **W)
(OUT / "fig3.json").write_text(json.dumps(res, indent=1))
print("wrote", OUT / "fig3.json")
