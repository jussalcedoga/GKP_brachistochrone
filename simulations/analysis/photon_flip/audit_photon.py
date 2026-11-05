"""Audit of the photon-optimal analytics (Oct 28, round 3).

Checks, for the preparations |0> -> |0_L> and |0> -> |1_L> (Delta = 0.4, cutoff 80, real even sector) and for the
logical flip |0_L> -> |1_L>:
  1. the single-mode closed form a(theta) with the consistent coefficient eta = 2 n_xi/n_chi (second-order expansion of
     J) and with eta_EL = eta - 4 sigma^2/n_chi^2 (linearization of the n-divided Euler-Lagrange equation): ODE residual,
     boundary values, the second-order prediction J2 against the exact exposure of the path, exposure and length;
  2. the vector second-order solution (all transverse directions, C = 2M/n_chi);
  3. numerical optima on K = 30, 60, 120 great-circle nodes (exact arc integrals), started from the geodesic and the
     closed form;
  4. the flip: the free optimum on K = 30, 60, 120 nodes from several starts (two brachistochrone legs through the
     vacuum, two photon-optimal legs, the geodesic, a path that avoids the vacuum), its closest approach to the vacuum,
     and the decomposition J*_flip = J*(0_L) + J*(1_L);
  5. the angle between the directions in which the two photon-optimal legs leave the vacuum.
Writes results/audit_photon.json.  python simulations/analysis/photon_flip/audit_photon.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.special import hyp2f1

from common import paths, photon_core as pc, states
from common.params import GKP_DELTA, OMEGA_REF

HERE = _L.vfile(__file__).parent
(HERE / "results").mkdir(exist_ok=True)
D = 80
NS = 1e3 / OMEGA_REF                   # rad -> ns at Delta H = omega


def single_mode(P, kind="J2", m=6001):
    th, nv, chi = P.th, P.nv, P.chi
    xi = (nv - P.n_chi) * chi / P.sig
    n_xi = float(xi @ (nv * xi))
    eta = 2 * n_xi / P.n_chi - (4 * P.sig ** 2 / P.n_chi ** 2 if kind == "EL" else 0.0)
    nu = 0.5 * (1 + np.sqrt(1 + 4 * eta))
    u1 = lambda s: np.sin(s) ** nu * hyp2f1((nu + 2) / 2, (nu - 2) / 2, nu + 0.5, np.sin(s) ** 2)  # noqa: E731
    A = 2 * P.sig / (P.n_chi * (2 - eta))
    s = np.linspace(0, th, m)
    a = np.zeros(m)
    si = s[1:-1]
    a[1:-1] = A * (np.sin(si) - np.sin(th) ** 2 * u1(si) / (np.sin(si) * u1(th)))
    # ODE residual of a'' + 2 cot a' + (3 - eta/sin^2) a = 2 sigma/(n_chi sin)
    h = s[1] - s[0]
    ap = np.gradient(a, h)
    app = np.gradient(ap, h)
    k = slice(m // 50, -m // 50)
    rhs = 2 * P.sig / (P.n_chi * np.sin(s[k]))
    res = app[k] + 2 / np.tan(s[k]) * ap[k] + (3 - eta / np.sin(s[k]) ** 2) * a[k] - rhs
    # second-order prediction of J for this a(s) and the exact exposure of the path
    J2 = np.trapezoid(P.n_chi * np.sin(s) ** 2 + 2 * P.sig * a * np.sin(s) + (n_xi - 1.5 * P.n_chi * np.sin(s) ** 2) * a ** 2
                      + 0.5 * P.n_chi * np.sin(s) ** 2 * ap ** 2, s)
    X = np.cos(a)[:, None] * P.geo(s) + np.sin(a)[:, None] * xi
    return X, dict(eta=float(eta), nu=float(nu), n_xi=n_xi, A=float(A), a_max=float(np.max(np.abs(a))),
                   ode_res_rel=float(np.max(np.abs(res)) / np.max(np.abs(rhs))), a_end=float(a[-2]),
                   J2=float(J2), J=pc.exposure(X, P.N), L=pc.length(X))


def vector_mode(P, Mg=1200):
    Bp = np.linalg.svd(np.vstack([P.u0, P.chi]))[2][2:].T
    m = Bp.shape[1]
    v = Bp.T @ (P.nv * P.chi)
    Mm = Bp.T @ (P.nv[:, None] * Bp)
    C = 2 * Mm / P.n_chi
    s = np.linspace(0, P.th, Mg + 1)
    h = s[1]
    si = s[1:-1]
    A = (sp.kron(sp.diags([1., -2., 1.], [-1, 0, 1], shape=(Mg - 1, Mg - 1)) / h ** 2, sp.eye(m))
         + sp.block_diag([4 * np.eye(m) - C / np.sin(x) ** 2 for x in si]))
    u = spsolve(A.tocsc(), np.tile(2 * v / P.n_chi, Mg - 1)).reshape(Mg - 1, m)
    a = np.zeros((Mg + 1, m))
    a[1:-1] = u / np.sin(si)[:, None]
    X = pc._build(P.geo(s), Bp, a)
    return X, dict(a_max=float(np.max(np.linalg.norm(a, axis=1))), J=pc.exposure(X, P.N), L=pc.length(X))


def optimum(N, u0, uT, starts, K):
    U, J = pc.optimize_path(N, u0, uT, [pc.resample(S, K) for S in starts], K, maxiter=20000)
    return U, J


def leg_info(P):
    out = dict(theta=float(P.th), c=float(P.c0), n_chi=float(P.n_chi), sigma=float(P.sig),
               sigma_over_n=float(P.sig / P.n_chi), J_geo=float(P.J_geo()), L_geo=float(P.th))
    Xj, out["closed_J2"] = single_mode(P, "J2")
    Xe, out["closed_EL"] = single_mode(P, "EL")
    Xv, out["vector"] = vector_mode(P)
    geo = P.geo(np.linspace(0, P.th, 400))
    opt = {}
    U = None
    for K in (30, 60, 120):
        st = [geo, Xj, Xv] + ([pc.dense(U, 9)[0]] if U is not None else [])
        U, J = optimum(P.N, P.u0, P.uT, st, K)
        opt[K] = dict(J=J, L=pc.length(U))
    out["optimum"] = opt
    out["B"] = pc.lower_bound(P.nv, P.uT)
    return out, dict(closed=Xj, vector=Xv, opt=U)


def main():
    res = {}
    T0 = states.gkp(GKP_DELTA, 0, D)
    T1 = states.gkp(GKP_DELTA, 1, D)
    P0, P1 = paths.Prep(T0), paths.Prep(T1)
    res["prep_0L"], X0 = leg_info(P0)
    res["prep_1L"], X1 = leg_info(P1)
    for k in ("prep_0L", "prep_1L"):
        r = res[k]
        print(f"{k}: theta={r['theta']:.4f} c={r['c']:.4f} n_chi={r['n_chi']:.3f} sigma={r['sigma']:.3f} "
              f"sigma/n={r['sigma_over_n']:.3f}  J_geo={r['J_geo'] * NS:.1f}")
        for name in ("closed_J2", "closed_EL"):
            c = r[name]
            print(f"   {name}: eta={c['eta']:.3f} nu={c['nu']:.3f} a_max={c['a_max']:.3f} res={c['ode_res_rel']:.1e} "
                  f"J2={c['J2'] * NS:.2f} J={c['J'] * NS:.2f} T={c['L'] * NS:.1f}")
        print(f"   vector: a_max={r['vector']['a_max']:.3f} J={r['vector']['J'] * NS:.2f} T={r['vector']['L'] * NS:.1f}")
        for K, o in r["optimum"].items():
            print(f"   optimum K={K}: J={o['J'] * NS:.3f} T={o['L'] * NS:.2f}")
        print(f"   bound B={r['B'] * NS:.2f}", flush=True)

    # ---------------------------------------------------------------- flip
    L = pc.Logical(GKP_DELTA, D)
    e0 = np.eye(L.d)[0]
    sa, sb = np.sign(L.u0 @ e0), np.sign(L.uT @ e0)
    flip = dict(theta=float(L.th), c=float(L.c), sign_u0=float(sa), sign_uT=float(sb))
    assert np.allclose(np.abs(P0.uT), np.abs(L.u0), atol=1e-12) and np.allclose(np.abs(P1.uT), np.abs(L.uT), atol=1e-12)
    N = np.diag(L.nv)
    rev = lambda X: X[::-1]                                                   # noqa: E731
    det = L.detour()
    legs_closed = np.vstack([sa * rev(X0["closed"]), sb * X1["closed"][1:]])
    legs_vector = np.vstack([sa * rev(X0["vector"]), sb * X1["vector"][1:]])
    legs_opt = np.vstack([sa * rev(pc.dense(X0["opt"], 9)[0]), sb * pc.dense(X1["opt"], 9)[0][1:]])
    for name, X in (("detour_geo_legs", det), ("closed_legs", legs_closed), ("vector_legs", legs_vector),
                    ("optimal_legs", legs_opt)):
        assert np.allclose(X[0], L.u0) and np.allclose(X[-1], L.uT)
        flip[name] = dict(J=pc.exposure(X, N), L=pc.length(X))
    geo = L.geo(np.linspace(0, L.th, 400))
    # a start that avoids the vacuum: the w = 3 trade-off-like path, approximated by pulling the geodesic toward the vacuum
    pull = geo + 0.6 * np.sin(np.linspace(0, np.pi, 400))[:, None] * e0[None, :]
    pull /= np.linalg.norm(pull, axis=1, keepdims=True)
    starts = dict(detour=det, closed_legs=legs_closed, geodesic=geo, pulled=pull)
    flip["optimum"] = {}
    for K in (30, 60, 120):
        row = {}
        for name, S in starts.items():
            U, J = optimum(N, L.u0, L.uT, [S], K)
            Xd, _ = pc.dense(U, 21)
            vac = Xd[:, 0] ** 2
            row[name] = dict(J=J, L=pc.length(U), vac_max=float(vac.max()), s_at_vac=float(np.argmax(vac) / (len(vac) - 1)))
        flip["optimum"][K] = row
        print(f"flip K={K}: " + "; ".join(f"{n}: J={r['J'] * NS:.2f} T={r['L'] * NS:.1f} vac={r['vac_max']:.6f}"
                                          for n, r in row.items()), flush=True)
    # directions in which the photon-optimal legs leave the vacuum (unit vectors in the complement of |0>)
    def out_dir(U):
        X, _ = pc.dense(U, 41)
        d = X[3] - (X[3] @ e0) * e0
        return d / np.linalg.norm(d)
    d0, d1 = sa * out_dir(X0["opt"]), sb * out_dir(X1["opt"])
    flip["leg_angle_rad"] = float(np.arccos(np.clip(d0 @ d1, -1, 1)))
    chi0, chi1 = sa * P0.chi, sb * P1.chi
    flip["chi_angle_rad"] = float(np.arccos(np.clip(chi0 @ chi1, -1, 1)))
    print(f"flip: theta={L.th:.4f}; detour(geo legs) J={flip['detour_geo_legs']['J'] * NS:.2f} T={flip['detour_geo_legs']['L'] * NS:.1f}; "
          f"closed legs J={flip['closed_legs']['J'] * NS:.2f} T={flip['closed_legs']['L'] * NS:.1f}; "
          f"vector legs J={flip['vector_legs']['J'] * NS:.2f} T={flip['vector_legs']['L'] * NS:.1f}; "
          f"optimal legs J={flip['optimal_legs']['J'] * NS:.2f} T={flip['optimal_legs']['L'] * NS:.1f}; "
          f"angle between leg directions {flip['leg_angle_rad']:.3f} rad, between chi0, chi1 {flip['chi_angle_rad']:.3f} rad")
    res["flip"] = flip
    res["NS_per_rad"] = NS
    (HERE / "results" / "audit_photon.json").write_text(json.dumps(res, indent=1))
    np.savez(HERE / "results" / "paths.npz", closed0=X0["closed"], closed1=X1["closed"], vector0=X0["vector"],
             vector1=X1["vector"], opt0=X0["opt"], opt1=X1["opt"], legs_closed=legs_closed, legs_vector=legs_vector)
    print("wrote", HERE / "results" / "audit_photon.json")


if __name__ == "__main__":
    main()
