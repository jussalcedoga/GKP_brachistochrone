"""Analytic inputs of Fig. 2 that are not in manuscript/fig2_photon_optimal/results/fig2.json (read only):
 1. the first-order (closed-form) photon path of GKP preparation with the consistent expansion coefficient
    beta = 2 nu / n_chi (analytics audit C06; fig2.json holds the curve of the superseded beta = 2.30):
      a'' + 2 cot s a' + (3 - beta/sin^2 s) a = 2 sigma_chi/(n_chi sin s),  a(0) = a(theta0) = 0,
      a(s) = 2 sigma/(n (2 - beta)) [sin s - sin^2 theta0 u1(s)/(sin s u1(theta0))],
      u1 = sin^mu s 2F1((mu+2)/2, (mu-2)/2; mu + 1/2; sin^2 s),  mu = (1 + sqrt(1 + 4 beta))/2,
    with psi(s) = cos a psi_g(s) + sin a |xi>, |xi> = (n - n_chi)|chi>/sigma_chi, so that
      <n>(s) = n_chi sin^2 s cos^2 a + nu sin^2 a + sigma_chi sin s sin 2a,  F_T(s) = cos^2 a cos^2(theta0 - s),
    traversed at the speed limit omega: dt = sqrt(cos^2 a + a'^2) ds / omega;
 2. the minimal photon number at given target fidelity, m(F) = min{<n> : |<T|psi>|^2 >= F}: the parametric curve of
    psi_lambda = (n + lambda)^{-1}|T> / norm, lambda > 0 (F from |<0|T>|^2 to 1), and m = 0 below |<0|T>|^2,
    for T = |0_L> (preparation) and T = |1_L> (flip target);
 3. self-checks: the ODE residual of a(s), the exposure of the closed form against the numerical optimum;
 4. the analytic photon-optimal flip (round 3): the free photon-optimal flip passes through the vacuum and equals the
    time-reversed photon-optimal preparation of |0_L> followed by that of |1_L> (analytics_audit/photon_flip), so its
    analytic counterpart is the closed form on each leg (replaces the two brachistochrone arcs used before).
Writes results/fig2_extra.json.   python figures/Fig2/compute_fig2.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))
import mfcommon as M  # noqa: E402,I001

import numpy as np  # noqa: E402
from scipy.special import hyp2f1  # noqa: E402

from common import params as P, paths, photon_core as pc, states  # noqa: E402

D = 80
OM = P.OMEGA_REF
F2 = json.loads((M.MS / "fig2_photon_optimal" / "results" / "fig2.json").read_text())


def closed_form(T):
    """first-order photon path of the preparation |0> -> T (real T, <0|T> > 0)"""
    nv = np.arange(len(T))
    c0 = T[0]
    th = float(np.arccos(c0))
    chi = T.copy()
    chi[0] = 0.0
    chi /= np.linalg.norm(chi)
    n_chi = float(chi @ (nv * chi))
    sig = float(np.sqrt(chi @ (nv ** 2 * chi) - n_chi ** 2))
    xi = (nv - n_chi) * chi / sig
    nu = float(xi @ (nv * xi))
    beta = 2 * nu / n_chi
    mu = (1 + np.sqrt(1 + 4 * beta)) / 2
    s = np.linspace(1e-6, th - 1e-9, 20001)
    u1 = lambda x: np.sin(x) ** mu * hyp2f1((mu + 2) / 2, (mu - 2) / 2, mu + 0.5, np.sin(x) ** 2)  # noqa: E731
    A = 2 * sig / (n_chi * (2 - beta))
    a = A * (np.sin(s) - np.sin(th) ** 2 * u1(s) / (np.sin(s) * u1(th)))
    ap = np.gradient(a, s)
    app = np.gradient(ap, s)
    res = app + 2 / np.tan(s) * ap + (3 - beta / np.sin(s) ** 2) * a - 2 * sig / (n_chi * np.sin(s))
    inner = slice(200, -200)
    ode_res = float(np.max(np.abs(res[inner])) / np.max(np.abs(2 * sig / (n_chi * np.sin(s[inner])))))
    w = n_chi * np.sin(s) ** 2 * np.cos(a) ** 2 + nu * np.sin(a) ** 2 + sig * np.sin(s) * np.sin(2 * a)
    FT = np.cos(a) ** 2 * np.cos(th - s) ** 2
    dl = np.sqrt(np.cos(a) ** 2 + ap ** 2)
    ell = np.r_[0, np.cumsum(0.5 * (dl[1:] + dl[:-1]) * np.diff(s))]
    t_ns = 1e3 * ell / OM
    expo = np.r_[0, np.cumsum(0.5 * (w[1:] * dl[1:] + w[:-1] * dl[:-1]) * np.diff(s))] * 1e3 / OM
    k = slice(None, None, 10)
    return dict(theta0=th, n_chi=n_chi, sigma_chi=sig, nu=nu, beta=beta, mu=float(mu), ode_residual_rel=ode_res,
                a_max=float(np.max(np.abs(a))), L=float(ell[-1]), T_ns=float(t_ns[-1]), J=float(ell[-1] * 0 + expo[-1] * OM / 1e3),
                J_ns=float(expo[-1]), s=s[k].tolist(), a=a[k].tolist(), t_ns=t_ns[k].tolist(), nbar=w[k].tolist(),
                F_T=FT[k].tolist(), expo=expo[k].tolist())


def min_photons(T, npts=600):
    """lower boundary m(F) of <n> at target fidelity F: psi_lambda = (n + lambda)^{-1} T, normalized"""
    nv = np.arange(len(T))
    lam = np.logspace(-4, 4, npts)
    F, m = [], []
    for L in lam:
        psi = T / (nv + L)
        psi /= np.linalg.norm(psi)
        F.append(float((T @ psi) ** 2))
        m.append(float(psi @ (nv * psi)))
    F, m = np.array(F), np.array(m)
    o = np.argsort(F)
    F0 = float(T[0] ** 2)
    return dict(F=np.r_[0.0, F0, F[o]].tolist(), m=np.r_[0.0, 0.0, m[o]].tolist(), F_vacuum=F0,
                n_target=float(T @ (nv * T)))


def two_sided_bound(m0, m1, theta):
    """J >= B2 = min_{L >= theta} int_0^L max{m0(cos^2 s), m1(cos^2(L - s))} ds (cos^2 set to 0 beyond pi/2): at
    arclength s from |0_L> the state has |<0_L|psi>| >= cos s, at remaining length L - s it has |<1_L|psi>| >= cos(L - s)"""
    f0 = lambda F: np.interp(F, m0["F"], m0["m"])          # noqa: E731
    f1 = lambda F: np.interp(F, m1["F"], m1["m"])          # noqa: E731
    best = (np.inf, None)
    for L in np.linspace(theta, theta + 1.5, 301):
        s = np.linspace(0, L, 4001)
        c0 = np.where(s < np.pi / 2, np.cos(np.minimum(s, np.pi / 2)) ** 2, 0.0)
        c1 = np.where(L - s < np.pi / 2, np.cos(np.minimum(L - s, np.pi / 2)) ** 2, 0.0)
        val = float(np.trapezoid(np.maximum(f0(c0), f1(c1)), s))
        best = min(best, (val, float(L)))
    return dict(B2=best[0], L_at_min=best[1], B2_ns=1e3 * best[0] / OM)


def closed_leg(P, m=6001):
    """closed form of Eq. (6) for the preparation |0> -> P.uT in the real even sector (rows: states)"""
    xi = (P.nv - P.n_chi) * P.chi / P.sig
    eta = 2 * float(xi @ (P.nv * xi)) / P.n_chi
    nu = (1 + np.sqrt(1 + 4 * eta)) / 2
    u1 = lambda x: np.sin(x) ** nu * hyp2f1((nu + 2) / 2, (nu - 2) / 2, nu + 0.5, np.sin(x) ** 2)  # noqa: E731
    s = np.linspace(0, P.th, m)
    a = np.zeros(m)
    a[1:-1] = 2 * P.sig / (P.n_chi * (2 - eta)) * (np.sin(s[1:-1]) - np.sin(P.th) ** 2 * u1(s[1:-1])
                                                    / (np.sin(s[1:-1]) * u1(P.th)))
    return np.cos(a)[:, None] * P.geo(s) + np.sin(a)[:, None] * xi


def modal_leg(P, m=6001):
    """modal closed form of the preparation |0> -> P.uT (round 4): the exact minimizer of the second-order functional,
    a sum over the eigenmodes n|_V e_k = m_k e_k of the photon number on V = span{|0>, |chi>}^perp, each term the
    hypergeometric solution with eta_k = 2 m_k / n_chi and amplitude 2 <e_k|n|chi> / (n_chi (2 - eta_k))"""
    nv, chi, n_chi, th = P.nv, P.chi, P.n_chi, P.th
    e0 = np.eye(len(nv))[0]
    Q = np.linalg.qr(np.column_stack([e0, chi, np.eye(len(nv))]))[0][:, 2:]
    mk, Ev = np.linalg.eigh(Q.T @ (nv[:, None] * Q))
    E = Q @ Ev
    ck = E.T @ (nv * chi)                                   # <e_k|n|chi>
    s = np.linspace(0, th, m)
    A = np.zeros((m, len(nv)))
    for k in range(len(mk)):
        eta = 2 * mk[k] / n_chi + (1e-9 if abs(2 * mk[k] / n_chi - 2) < 1e-9 else 0.0)
        nu = 0.5 + np.sqrt(0.25 + eta)
        u1 = lambda x: np.sin(x) ** nu * hyp2f1((nu + 2) / 2, (nu - 2) / 2, nu + 0.5, np.sin(x) ** 2)  # noqa: E731
        ak = np.zeros(m)
        ak[1:-1] = 2 * ck[k] / (n_chi * (2 - eta)) * (np.sin(s[1:-1]) - np.sin(th) ** 2 * u1(s[1:-1])
                                                      / (np.sin(s[1:-1]) * u1(th)))
        A += ak[:, None] * E[:, k][None, :]
    na = np.linalg.norm(A, axis=1)
    dirn = A / np.maximum(na, 1e-300)[:, None]
    return np.cos(na)[:, None] * P.geo(s) + np.sin(na)[:, None] * dirn


def curve(X, nv, target, k=10):
    """t (ns, at the speed limit), nbar, target fidelity and cumulative exposure along a dense path"""
    seg = np.arccos(np.clip(np.sum(X[:-1] * X[1:], 1), -1, 1))
    s = np.r_[0, np.cumsum(seg)]
    nb = np.sum(X ** 2 * nv, 1)
    expo = np.r_[0, np.cumsum(0.5 * (nb[1:] + nb[:-1]) * seg)] * 1e3 / OM
    sl = slice(None, None, k)
    return dict(t_ns=(1e3 * s / OM)[sl].tolist(), nbar=nb[sl].tolist(), F_T=((X @ target) ** 2)[sl].tolist(),
                expo=expo[sl].tolist(), T_ns=float(1e3 * s[-1] / OM), J_ns=float(1e3 * pc.exposure(X, np.diag(nv)) / OM))


def flip_modal():
    """analytic photon-optimal flip with the modal closed form on both legs through the vacuum"""
    L = pc.Logical(0.4, D)
    P0, P1 = paths.Prep(states.gkp(0.4, 0, D)), paths.Prep(states.gkp(0.4, 1, D))
    e0 = np.eye(L.d)[0]
    X = np.vstack([np.sign(L.u0 @ e0) * modal_leg(P0)[::-1], np.sign(L.uT @ e0) * modal_leg(P1)[1:]])
    assert np.allclose(X[0], L.u0) and np.allclose(X[-1], L.uT)
    return curve(X, L.nv, L.uT)


def numeric_geodesic(u0, uT, nv, seed=1, K=60):
    """the time-optimal path found numerically: minimize the Fubini-Study length (the exact arc integral of Eq. J with
    n -> 1, i.e. the sum of the arc angles) over K free nodes from a randomly perturbed start; it must reproduce the
    great circle of Eq. (1)"""
    th = float(np.arccos(np.clip(u0 @ uT, -1, 1)))
    geo = np.array([(np.sin(th - t) * u0 + np.sin(t) * uT) / np.sin(th) for t in np.linspace(0, th, 2001)])
    start = pc.noisy_geodesic(pc.resample(geo, K), seed, 0.3)
    U, Lmin = pc.optimize_path(np.eye(len(nv)), u0, uT, [start], K, maxiter=20000)
    X = pc.dense(U, 41)[0]
    out = curve(X, nv, uT)
    Q = np.linalg.qr(np.column_stack([u0, uT]))[0]          # exact distance from the plane of the great circle
    off = np.linalg.norm(X - (X @ Q) @ Q.T, axis=1)
    out.update(L_minus_theta=float(Lmin - th), start_excess=float(pc.length(start) / th - 1),
               max_dev_rad=float(np.arcsin(min(off.max(), 1.0))))
    return out


def flip_closed():
    """analytic photon-optimal flip: the photon-optimal flip passes through the vacuum (analytics_audit/photon_flip), so
    it is the time-reversed photon-optimal preparation of |0_L> followed by that of |1_L>; each leg in closed form"""
    L = pc.Logical(0.4, D)
    P0, P1 = paths.Prep(states.gkp(0.4, 0, D)), paths.Prep(states.gkp(0.4, 1, D))
    e0 = np.eye(L.d)[0]
    X = np.vstack([np.sign(L.u0 @ e0) * closed_leg(P0)[::-1], np.sign(L.uT @ e0) * closed_leg(P1)[1:]])
    assert np.allclose(X[0], L.u0) and np.allclose(X[-1], L.uT)
    seg = np.arccos(np.clip(np.sum(X[:-1] * X[1:], 1), -1, 1))
    s = np.r_[0, np.cumsum(seg)]
    nb = np.sum(X ** 2 * L.nv, 1)
    expo = np.r_[0, np.cumsum(0.5 * (nb[1:] + nb[:-1]) * seg)] * 1e3 / OM
    k = slice(None, None, 10)
    return dict(t_ns=(1e3 * s / OM)[k].tolist(), nbar=nb[k].tolist(), F_T=((X @ L.uT) ** 2)[k].tolist(),
                expo=expo[k].tolist(), T_ns=float(1e3 * s[-1] / OM), J_ns=float(1e3 * pc.exposure(X, np.diag(L.nv)) / OM),
                vac=float(np.max(X[:, 0] ** 2)))


def main():
    (HERE / "results").mkdir(exist_ok=True)
    T0 = np.real(states.gkp(0.4, 0, D))
    T1 = np.real(states.gkp(0.4, 1, D))
    T0 *= np.sign(T0[0])
    T1 *= np.sign(T1[0])
    cf = closed_form(T0)
    num = F2["prep"]["gkp"]["photon"]
    cf["excess_vs_numerical"] = cf["J_ns"] / num["J_ns"] - 1
    out = dict(closed_gkp=cf, m_0L=min_photons(T0), m_1L=min_photons(T1), flip_closed=flip_closed())
    P0 = paths.Prep(states.gkp(0.4, 0, D))
    out["modal_gkp"] = curve(modal_leg(P0), P0.nv, P0.uT)
    out["flip_modal"] = flip_modal()
    Lg = pc.Logical(0.4, D)
    out["geo_num_gkp"] = numeric_geodesic(P0.u0, P0.uT, P0.nv)
    out["geo_num_flip"] = numeric_geodesic(Lg.u0, Lg.uT, Lg.nv)
    for k in ("geo_num_gkp", "geo_num_flip"):
        g = out[k]
        print(f"numerical time-optimal path ({k}): L - theta = {g['L_minus_theta']:.1e} rad (start {100 * g['start_excess']:.0f}% "
              f"longer), largest distance from the great circle {g['max_dev_rad']:.1e} rad, T = {g['T_ns']:.2f} ns, "
              f"J = {g['J_ns']:.2f} photon ns")
    print(f"modal closed form: GKP J = {out['modal_gkp']['J_ns']:.2f} photon ns in {out['modal_gkp']['T_ns']:.1f} ns; "
          f"flip J = {out['flip_modal']['J_ns']:.2f} photon ns in {out['flip_modal']['T_ns']:.1f} ns")
    fc = out["flip_closed"]
    print(f"flip, closed form on both legs: T = {fc['T_ns']:.1f} ns, J = {fc['J_ns']:.2f} photon ns "
          f"(numerical optimum {F2['flip']['photon']['J_ns']:.2f}, excess {100 * (fc['J_ns'] / F2['flip']['photon']['J_ns'] - 1):.2f}%), "
          f"vacuum weight at the junction {fc['vac']:.6f}")
    out["flip_bound"] = two_sided_bound(out["m_0L"], out["m_1L"], F2["flip"]["theta"])
    out["gkp_bound_ns"] = 1e3 * F2["prep"]["gkp"]["B"] / OM
    print(f"flip two-sided bound B2 = {out['flip_bound']['B2']:.4f} rad = {out['flip_bound']['B2_ns']:.1f} photon ns "
          f"(at L = {out['flip_bound']['L_at_min']:.4f}; numerical optimum {F2['flip']['photon']['J_ns']:.1f}); "
          f"GKP bound {out['gkp_bound_ns']:.1f} photon ns")
    (HERE / "results" / "fig2_extra.json").write_text(json.dumps(out))
    print(f"closed form GKP: beta={cf['beta']:.3f} mu={cf['mu']:.3f} nu={cf['nu']:.3f} |a|max={cf['a_max']:.3f} rad "
          f"T={cf['T_ns']:.1f} ns J/omega={cf['J_ns']:.1f} photon ns (numerical {num['J_ns']:.1f}; excess "
          f"{100 * cf['excess_vs_numerical']:.2f}%)  ODE residual {cf['ode_residual_rel']:.1e}")
    print(f"m(F): |<0|0_L>|^2={out['m_0L']['F_vacuum']:.4f}, |<0|1_L>|^2={out['m_1L']['F_vacuum']:.4f}")


if __name__ == "__main__":
    main()
