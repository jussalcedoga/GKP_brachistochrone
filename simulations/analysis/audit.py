"""Analytics audit of the draft: one function per analytic claim of manuscript/paper/{main,main_cs,supplement,sm_tracking}.tex.
Each claim function prints and records its values, the tolerance used and a verdict:
    OK         the statement is correct as printed (to the stated precision)
    CORRECTED  the statement is wrong or imprecise; AUDIT.md gives the corrected LaTeX
Writes audit_results.json and analytics_numbers.tex (one \\newcommand per number the draft should quote) next to this
file.  Expensive optimized paths are cached in cache/ (delete it to recompute everything).
    python simulations/analysis/audit.py          (~10 min on the compute server, single core)
Reads (never writes) manuscript/: the shared library common/ and the stored results of fig1-fig3 and si/.
"""
from __future__ import annotations
import layout as _L  # repository layout: data/ and output/ (src/layout.py)

import json
import pathlib
import pickle
import re
import sys
import time

HERE = _L.vfile(__file__).parent
MS = HERE.parents[1] / "manuscript"
sys.path.insert(0, str(MS))

import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import gammaln, hyp2f1  # noqa: E402

from common import brachistochrone as B, paths, photon_core as pc, states  # noqa: E402
from common.params import G_SQ, GKP_DELTA, NOISE, OMEGA_REF, kappa_mu  # noqa: E402

D = 80
W = OMEGA_REF                                   # rad/us
trapz = getattr(np, "trapezoid", None) or np.trapz
OUT, NUM = {}, {}
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
FIG2 = json.loads((MS / "fig2_photon_optimal" / "results" / "fig2.json").read_text())
FIG3 = json.loads((MS / "fig3_time_vs_photon" / "results" / "fig3.json").read_text())
FIG1 = json.loads((MS / "fig1_brachistochrone" / "results" / "fig1.json").read_text())
GAL = json.loads((MS / "si" / "state_gallery" / "results" / "state_gallery.json").read_text())
CAP = json.loads((MS / "si" / "squeezer_capacity" / "results" / "squeezer_capacity.json").read_text())


def ns(L):
    return 1e3 * L / W


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    return v


def rec(cid, title, verdict, **values):
    OUT[cid] = dict(title=title, verdict=verdict, values=jsonable(values))
    print(f"\n[{cid}] {verdict}: {title}", flush=True)
    for k, v in values.items():
        print(f"    {k}: {jsonable(v)}", flush=True)


def mac(name, val, fmt=None, note=""):
    NUM[name] = (val if isinstance(val, str) else format(val, fmt), note)


def cached(name, fn):
    p = CACHE / f"{name}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    t0 = time.time()
    v = fn()
    p.write_bytes(pickle.dumps(v))
    print(f"  (computed {name} in {time.time() - t0:.0f} s)", flush=True)
    return v


# ----------------------------------------------------------------------------------------------- shared objects
VEC = {"gkp": lambda: states.gkp(GKP_DELTA, 0, D), "cat": lambda: states.cat(2.0, D), "fock2": lambda: states.fock(2, D),
       "fock4": lambda: states.fock(4, D), "binomial": lambda: states.binomial_kitten(D)}


def prep(name):
    return paths.Prep(VEC[name]())


def photon_prep(name):
    return cached(f"photon_{name}", lambda: paths.photon_path_prep(prep(name)))          # (U, J), 4 starts, K = 30


LG = pc.Logical(GKP_DELTA, D)
NF = np.diag(LG.nv)


def flip_photon():
    return cached("photon_flip", lambda: pc.optimize_path(NF, LG.u0, LG.uT, [pc.resample(LG.detour(), 30)], 30))


def flip_tradeoff(noise="Yale"):
    k, m = kappa_mu(noise)

    def f():
        geo = LG.geo(np.linspace(0, LG.th, 32))
        return paths.tradeoff_family(NF, LG.u0, LG.uT, geo, flip_photon()[0], [k / m])[0]
    return cached(f"tradeoff_flip_{noise}", f)


def refined(name, N, u0, uT, U30, K):
    X, _ = pc.dense(U30, 41)
    return cached(f"refined_{name}_K{K}", lambda: pc.optimize_path(N, u0, uT, [pc.resample(X, K)], K, maxiter=30000))


def ops(Dm=D):
    a = np.diag(np.sqrt(np.arange(1, Dm)), 1).astype(complex)
    return a, a.conj().T


def dH(G, psi):
    m = np.vdot(psi, G @ psi)
    return float(np.sqrt(max(np.real(np.vdot(psi, G @ (G @ psi))) - abs(m) ** 2, 0.0)))


# ----------------------------------------------------------------------------------------------- claims
def c01_speeds():
    a, ad = ops()
    vac = states.vacuum(D)
    g = G_SQ
    v = {}
    for lab, xi in (("CS xi=g", g), ("CS xi=ig", 1j * g)):
        v[lab] = dH(0.5j * (np.conj(xi) * a @ a - xi * ad @ ad), vac)
    for lab, e in (("CD/SNAP drive eps=g", g), ("CD/SNAP drive eps=ig", 1j * g)):
        v[lab] = dH((1j / np.sqrt(2)) * (np.conj(e) * a - e * ad), vac)
    v["SNAP dispersive -chi n (chi=g) on vacuum"] = dH(-g * ad @ a, vac)
    G = 0.5j * g * (a @ a - ad @ ad)
    fock_ratio = {n: dH(G, states.fock(n, D)) / (W * np.sqrt(n * n + n + 1)) for n in range(6)}
    err = max(abs(x - W) for k, x in v.items() if "dispersive" not in k)
    ok = err < 1e-9 and v["SNAP dispersive -chi n (chi=g) on vacuum"] < 1e-12 and max(abs(r - 1) for r in fock_ratio.values()) < 1e-9
    rec("C01", "Fubini-Study speed g/sqrt2 of the CS, CD and SNAP-drive generators on the vacuum; Delta S on Fock n",
        "OK" if ok else "CHECK", omega=W, speeds=v, max_err=err, DeltaS_over_omega_sqrt_n2_n_1=fock_ratio)
    mac("omegaRef", W, ".3f", "omega = g/sqrt2 (1/us)")
    mac("gRate", G_SQ, ".1f", "g (1/us)")


def transfers():
    vac = states.vacuum(D)
    g0, g1 = states.gkp(GKP_DELTA, 0, D), states.gkp(GKP_DELTA, 1, D)
    return {"gkp": (vac, g0), "cat": (vac, states.cat(2.0, D)), "fock2": (vac, states.fock(2, D)),
            "fock4": (vac, states.fock(4, D)), "binomial": (vac, states.binomial_kitten(D)), "flip": (g0, g1)}


def c02_brachistochrone():
    out, ok = {}, True
    for name, (S, T) in transfers().items():
        c0, th, chi = B.geodesic_parts(S, T)
        H = B.H_opt(S, T, W)
        ev = np.linalg.eigvalsh(H)
        rank = int(np.sum(np.abs(ev) > 1e-9))
        t = 0.37 * th / W
        p_exp = expm(-1j * H * t) @ S
        p_an = B.path_states(S, T, [W * t])[0]
        ts = np.linspace(0, th / W, 7)
        FT = np.array([abs(np.vdot(T, B.path_states(S, T, [W * x])[0])) ** 2 for x in ts])
        FT_pred = np.cos(th - W * ts) ** 2
        r = dict(c0=c0, theta0=th, T_star_ns=ns(th), rank=rank, norm=float(np.max(np.abs(ev))), DeltaH_S=dH(H, S),
                 path_err=float(np.linalg.norm(p_exp - p_an)), FT_err=float(np.max(np.abs(FT - FT_pred))),
                 n_S=float(np.real(np.vdot(S, np.arange(D) * S))), n_T=float(np.real(np.vdot(T, np.arange(D) * T))))
        ok &= rank == 2 and abs(r["norm"] - W) < 1e-9 and abs(r["DeltaH_S"] - W) < 1e-9 and r["path_err"] < 1e-9 and r["FT_err"] < 1e-9
        out[name] = r
    S, T = transfers()["flip"]
    out["overlaps"] = dict(vac_0L=float(abs(transfers()["gkp"][1][0])), c_flip=float(abs(np.vdot(S, T))),
                           vac_1L=float(abs(T[0])))
    rec("C02", "brachistochrone: rank two, ||H|| = Delta H = omega, exp(-iHt)|S> = p(t), F_T = cos^2(theta0 - omega t), T*",
        "OK" if ok else "CHECK", **out)
    for k, nm in (("gkp", "GKP"), ("cat", "Cat"), ("fock2", "Fock"), ("flip", "Flip"), ("binomial", "Binomial")):
        mac(f"Tstar{nm}", out[k]["T_star_ns"], ".0f", f"T* (ns) {k}")
        mac(f"thetaZero{nm}", out[k]["theta0"], ".4f", f"theta0 (rad) {k}")
    mac("cFlip", out["overlaps"]["c_flip"], ".4f", "<0_L|1_L>")
    mac("vacZeroL", out["overlaps"]["vac_0L"], ".5f", "<0|0_L>")
    mac("nZeroL", out["gkp"]["n_T"], ".3f", "<n> of |0_L>")
    mac("nOneL", out["flip"]["n_T"], ".3f", "<n> of |1_L>")


def c03_normal_order():
    Ds = 40
    vac = states.vacuum(Ds)
    g0, g1 = states.gkp(GKP_DELTA, 0, Ds), states.gkp(GKP_DELTA, 1, Ds)
    res, ok = {}, True
    for name, (S, T) in (("gkp", (vac, g0)), ("flip", (g0, g1))):
        H = B.H_opt(S, T, 1.0)
        h = B.normal_ordered(H)
        M = B.from_normal_ordered(h, Ds)
        # Eq. (2) of main.tex written out: H = i omega sum_{n>=1} chi_n/sqrt(n!) sum_k (-1)^k/k! a^dag^(n+k) a^k - h.c.
        r = dict(normal_order_roundtrip_err=float(np.max(np.abs(M - H))))
        if name == "gkp":
            _, _, chi = B.geodesic_parts(S, T)
            a, ad = ops(Ds)
            apow = [np.eye(Ds, dtype=complex)]
            adpow = [np.eye(Ds, dtype=complex)]
            for _ in range(1, Ds):
                apow.append(apow[-1] @ a)
                adpow.append(adpow[-1] @ ad)
            A = np.zeros((Ds, Ds), complex)
            for n in range(1, Ds):
                for k in range(0, Ds - n):
                    A += chi[n] / np.exp(0.5 * gammaln(n + 1)) * (-1) ** k / np.exp(gammaln(k + 1)) * adpow[n + k] @ apow[k]
            H2 = 1j * (A - A.conj().T)
            r["eq2_err"] = float(np.max(np.abs(H2 - H)))
        f = B.truncated_order_fidelity(S, T, [2, 40], 1.0)
        r["F_K2"], r["F_K40"] = f[2], f[40]
        res[name] = r
        ok &= r["normal_order_roundtrip_err"] < 1e-8 and r.get("eq2_err", 0) < 1e-8
    table = {"gkp": (0.730, 0.992), "flip": (0.299, 0.991)}
    ok &= all(abs(res[k]["F_K2"] - table[k][0]) < 6e-4 and abs(res[k]["F_K40"] - table[k][1]) < 6e-4 for k in table)
    rec("C03", "normal-ordered form Eq. (2) is exact on the truncated space; K-truncation table (cutoff 40)",
        "OK" if ok else "OK as identity; double-precision round-off at high order (see C03b)", **res, table_printed=table,
        note="the identity |n><m| = sum_p (-1)^p a^dag^(n+p) a^(m+p)/(p! sqrt(n! m!)) is exact on the truncated space "
             "(each term that would leave it vanishes individually); the round-trip error is floating-point "
             "cancellation between coefficients ~1/sqrt(j! k!) and matrix elements ~sqrt(j! k!)")


def truncated_fidelity_mp(S, T, Ks, dps=60):
    """fidelity |<T|exp(-i H_K T*)|S>|^2 with the normal-ordered expansion of H_opt truncated at j + k <= K, the
    truncated operator built in mpmath (dps digits) and only then rounded: double precision loses everything at high
    order (the coefficients h_jk alternate in sign and multiply matrix elements of order sqrt(j! k!))"""
    import mpmath as mp
    mp.mp.dps = dps
    Ds = S.size
    c0, th, chi = B.geodesic_parts(S, T)
    assert np.max(np.abs(np.imag(S))) < 1e-12 and np.max(np.abs(np.imag(chi))) < 1e-12
    s_, x_ = np.real(S), np.real(chi)
    Bm = [[mp.mpf(float(x_[n] * s_[m] - s_[n] * x_[m])) for m in range(Ds)] for n in range(Ds)]   # H = i omega Bm
    fac = [mp.factorial(k) for k in range(Ds)]
    sq = [mp.sqrt(f) for f in fac]
    b = [[mp.mpf(0)] * Ds for _ in range(Ds)]
    for j in range(Ds):
        for k in range(Ds):
            acc = mp.mpf(0)
            for p in range(min(j, k) + 1):
                if Bm[j - p][k - p] != 0:
                    acc += (-1) ** p * Bm[j - p][k - p] / (fac[p] * sq[j - p] * sq[k - p])
            b[j][k] = acc
    out = {}
    for K in Ks:
        BK = np.zeros((Ds, Ds))
        for r in range(Ds):
            for c in range(Ds):
                acc = mp.mpf(0)
                for j in range(r + 1):
                    k = c - r + j
                    if k < 0 or k > c or j + k > K:
                        continue
                    acc += b[j][k] * sq[c] * sq[r] / fac[c - k]
                BK[r, c] = float(acc)
        HK = 1j * 0.5 * (BK - BK.T)                      # Hermitian part of i B_K (omega = 1)
        psi = expm(-1j * HK * th) @ S
        out[int(K)] = float(abs(np.vdot(T, psi)) ** 2)
    return out


def c03b_order_table_mp():
    Ds = 40
    vac = states.vacuum(Ds)
    g0, g1 = states.gkp(GKP_DELTA, 0, Ds), states.gkp(GKP_DELTA, 1, Ds)
    Ks = [2, 4, 6, 8, 12, 16, 24, 40, 56, 78]
    printed = {"fock2": [0.104, 0.938, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
               "fock4": [0.000, 0.771, 0.966, 0.997, 1.000, 1.000, 1.000, 1.000, 1.000],
               "cat": [0.531, 0.692, 0.534, 0.639, 0.977, 0.931, 1.000, 1.000, 1.000],
               "gkp": [0.730, 0.758, 0.764, 0.598, 0.842, 0.813, 0.979, 0.992, 1.000],
               "flip": [0.299, 0.261, 0.719, 0.311, 0.821, 0.610, 0.970, 0.991, 1.000]}
    res, worst = {}, 0.0
    for name, (S, T) in (("fock2", (vac, states.fock(2, Ds))), ("fock4", (vac, states.fock(4, Ds))),
                         ("cat", (vac, states.cat(2.0, Ds))), ("gkp", (vac, g0)), ("flip", (g0, g1))):
        mp_ = truncated_fidelity_mp(S, T, Ks)
        dbl = B.truncated_order_fidelity(S, T, Ks, 1.0)
        dev = [abs(round(mp_[K], 3) - v) for K, v in zip(Ks[:-1], printed[name])]
        worst = max(worst, max(dev))
        res[name] = dict(extended_precision={str(K): mp_[K] for K in Ks}, double_precision={str(K): dbl[K] for K in Ks},
                         printed=dict(zip(map(str, Ks[:-1]), printed[name])), max_dev_printed=max(dev))
    rec("C03b", "K-truncation table recomputed with the truncated operator built in extended precision (60 digits)",
        "OK" if worst < 1.5e-3 else "CORRECTED", worst_deviation_from_printed=worst, **res)


def lindblad_F(S, T, kappa, steps=3000):
    a, ad = ops()
    n = ad @ a
    H = B.H_opt(S, T, W)
    c0, th, chi = B.geodesic_parts(S, T)
    Ts = th / W
    dt = Ts / steps
    rho = np.outer(S, S.conj())

    def L(r):
        return -1j * (H @ r - r @ H) + kappa * (a @ r @ ad - 0.5 * (n @ r + r @ n))
    for _ in range(steps):
        k1 = L(rho)
        k2 = L(rho + 0.5 * dt * k1)
        k3 = L(rho + 0.5 * dt * k2)
        k4 = L(rho + dt * k3)
        rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    p = B.path_states(S, T, [th])[0]
    ss = np.linspace(0, th, 2001)
    P = B.path_states(S, T, ss)
    nb = np.real(np.einsum("si,i,si->s", P.conj(), np.arange(D), P))
    am = np.einsum("si,ij,sj->s", P.conj(), a, P)
    I1 = trapz(nb - np.abs(am) ** 2, ss) / W
    return 1 - float(np.real(np.vdot(p, rho @ p))), kappa * I1, I1


def c04_first_order_loss():
    res = {}
    ok = True
    for name in ("gkp", "flip"):
        S, T = transfers()[name]
        r = {}
        for lab, kap in (("Yale", 1 / 606.0), ("10xYale", 10 / 606.0)):
            inF, pred, I1 = lindblad_F(S, T, kap)
            r[lab] = dict(kappa=kap, one_minus_F=inF, kappa_int=pred, rel_dev=inF / pred - 1)
        r["int_n_dt_us"] = I1
        res[name] = r
        ok &= abs(r["Yale"]["rel_dev"]) < 2e-3 and abs(r["10xYale"]["rel_dev"]) > abs(r["Yale"]["rel_dev"])
    rec("C04", "Lemma (first-order loss): 1 - F = kappa int(<n> - |<a>|^2) dt + O(kappa^2) on the brachistochrone",
        "OK" if ok else "CHECK", **res)


def discrete(U):
    rows = []
    for k in range(1, len(U) - 1):
        pm, p, pp = U[k - 1], U[k], U[k + 1]
        h1 = np.arccos(np.clip(pm @ p, -1, 1))
        h2 = np.arccos(np.clip(p @ pp, -1, 1))
        d2 = 2 * ((pp - p) / h2 - (p - pm) / h1) / (h1 + h2)
        t = pp - pm
        t = t - (t @ p) * p
        rows.append((p, t / np.linalg.norm(t), d2))
    return rows


def el_stats(U, nv, edge=3):
    R = []
    for p, t, d2 in discrete(U)[edge:-edge]:
        Np = nv * p
        w = p @ Np
        ok_rhs = -p + 2 * (Np / w - p - (t @ Np / w) * t)
        pr_rhs = -p + 2 * (Np / w - p - ((t @ (nv * t)) / w) * p)

        def Pn(v):
            return v - (v @ p) * p - (v @ t) * t
        cn, co = Pn(d2), Pn(ok_rhs)
        R.append([np.linalg.norm(cn - co) / np.linalg.norm(co), float(cn @ Pn(Np)) > 0, p @ d2, t @ d2,
                  p @ ok_rhs, t @ ok_rhs, p @ pr_rhs, t @ pr_rhs, np.linalg.norm(co)])
    R = np.array(R, float)
    return dict(n_nodes=len(R), normal_rel_err_median=float(np.median(R[:, 0])), normal_rel_err_p90=float(np.percentile(R[:, 0], 90)),
                frac_curvature_toward_higher_n=float(R[:, 1].mean()),
                num_psi_dot_psi2=[float(R[:, 2].min()), float(R[:, 2].max())],
                num_t_dot_psi2=[float(R[:, 3].min()), float(R[:, 3].max())],
                correct_psi_dot_rhs=[float(R[:, 4].min()), float(R[:, 4].max())],
                correct_t_dot_rhs=[float(R[:, 5].min()), float(R[:, 5].max())],
                printed_psi_dot_rhs=[float(R[:, 6].min()), float(R[:, 6].max())],
                printed_t_dot_rhs=[float(R[:, 7].min()), float(R[:, 7].max())],
                geodesic_curvature_median=float(np.median(R[:, 8])))


def c05_euler_lagrange():
    res = {}
    Pg = prep("gkp")
    for K in (60, 120):
        U, J = refined("gkp", Pg.N, Pg.u0, Pg.uT, photon_prep("gkp")[0], K)
        res[f"gkp_K{K}"] = dict(J=J, **el_stats(U, Pg.nv))
        U, J = refined("flip", NF, LG.u0, LG.uT, flip_photon()[0], K)
        res[f"flip_K{K}"] = dict(J=J, **el_stats(U, LG.nv))
    good = all(res[k]["normal_rel_err_median"] < 0.05 for k in res if k.endswith("K120"))
    printed_bad = all(max(abs(res[k]["printed_psi_dot_rhs"][0] + 1), abs(res[k]["printed_psi_dot_rhs"][1] + 1)) > 0.1 for k in res)
    rec("C05", "Euler-Lagrange equation of J: printed form vs psi'' = -psi + 2[N psi/w - psi - (psi'^T N psi/w) psi']",
        "CORRECTED" if (good and printed_bad) else "CHECK", **res,
        note="psi.psi'' = -1 and psi'.psi'' = 0 hold for every unit-speed curve on the sphere; the printed RHS violates "
             "both, the corrected one satisfies them identically; the normal (curvature) component, common to both, is "
             "reproduced by the numerical optimum. The curvature vector points toward HIGHER <n> (grad_perp ln w): the "
             "path bulges toward low photon number, like a light ray bending toward higher refractive index.")


def closed_a(s, th, n, sig, beta):
    mu = 0.5 * (1 + np.sqrt(1 + 4 * beta))

    def u1(x):
        return np.sin(x) ** mu * hyp2f1((mu + 2) / 2, (mu - 2) / 2, mu + 0.5, np.sin(x) ** 2)
    a = np.zeros_like(s)
    i = (s > 0) & (s < th)
    si = s[i]
    a[i] = 2 * sig / (n * (2 - beta)) * (np.sin(si) - np.sin(th) ** 2 / np.sin(si) * u1(si) / u1(th))
    return a, mu, u1


def ansatz_J(x, s, n, nu, sig, grad=True):
    a = np.r_[0.0, x, 0.0]
    h = np.diff(s)
    am, ap, sm = 0.5 * (a[1:] + a[:-1]), np.diff(a) / h, 0.5 * (s[1:] + s[:-1])
    w = n * np.sin(sm) ** 2 * np.cos(am) ** 2 + nu * np.sin(am) ** 2 + sig * np.sin(sm) * np.sin(2 * am)
    q = np.sqrt(np.cos(am) ** 2 + ap ** 2)
    J = float(np.sum(w * q * h))
    if not grad:
        return J
    dw = -n * np.sin(sm) ** 2 * np.sin(2 * am) + nu * np.sin(2 * am) + 2 * sig * np.sin(sm) * np.cos(2 * am)
    fa = dw * q + w * (-np.sin(am) * np.cos(am) / q)
    fp = w * ap / q
    g = np.zeros(len(a))
    g[:-1] += h * (0.5 * fa) - fp
    g[1:] += h * (0.5 * fa) + fp
    return J, g[1:-1]


def ode_residual(u, s, beta, rhs):
    h = s[1] - s[0]
    upp = (u[2:] - 2 * u[1:-1] + u[:-2]) / h ** 2
    r = upp + (4 - beta / np.sin(s[1:-1]) ** 2) * u[1:-1] - rhs
    return float(np.max(np.abs(r[5:-5])) / max(1e-12, np.max(np.abs(upp[5:-5]))))


def closed_form_study(name):
    P = prep(name)
    th, n, sig = P.th, P.n_chi, P.sig
    xi = (P.nv * P.chi - n * P.chi) / sig
    nu = float(xi @ (P.nv * xi))
    beta_paper = 2 * (nu * n - 2 * sig ** 2) / n ** 2
    beta_alt = 2 * nu / n
    s = np.linspace(0, th, 4001)
    out = dict(theta0=th, n_chi=n, sigma_chi=sig, nu=nu, beta_paper=beta_paper, beta_alt=beta_alt)
    U, Jph = photon_prep(name)
    out["J_full_opt"] = Jph
    out["J_geo"] = P.J_geo()
    X, _ = pc.dense(U, 41)
    sg = np.clip(np.arctan2(X @ P.chi, X @ P.u0), 0, th)
    a_full = np.arcsin(np.clip(X @ xi, -1, 1))
    out["a_max_full_opt"] = float(np.max(np.abs(a_full)))
    out["s_at_a_max_full_opt"] = float(sg[np.argmax(np.abs(a_full))] / th)
    out["full_opt_weight_outside_ansatz_max"] = float(np.max(1 - (X @ P.u0) ** 2 - (X @ P.chi) ** 2 - (X @ xi) ** 2))
    for lab, beta in (("paper", beta_paper), ("alt", beta_alt)):
        a, mu, u1 = closed_a(s, th, n, sig, beta)
        Jq = ansatz_J(a[1:-1], s, n, nu, sig, grad=False)
        Xc = np.cos(a)[:, None] * P.geo(s) + np.sin(a)[:, None] * xi
        Jpc = pc.exposure(pc.resample(Xc, 60), P.N)
        A = 2 * sig / (n * (2 - beta))
        out[lab] = dict(beta=beta, mu=mu, a_max=float(np.max(np.abs(a))), s_at_a_max=float(s[np.argmax(np.abs(a))] / th),
                        J_quadrature=Jq, J_resampled60=Jpc, excess_vs_full_opt=Jpc / Jph - 1,
                        ode_res_homogeneous=ode_residual(u1(s), s, beta, 0.0),
                        ode_res_particular=ode_residual(A * np.sin(s) ** 2, s, beta, 2 * sig / n))
    r = minimize(ansatz_J, closed_a(s, th, n, sig, beta_paper)[0][1:-1], args=(s, n, nu, sig), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=20000, ftol=1e-15, gtol=1e-11))
    a_ex = np.r_[0.0, r.x, 0.0]
    out["ansatz_exact"] = dict(J=float(r.fun), a_max=float(np.max(np.abs(a_ex))), s_at_a_max=float(s[np.argmax(np.abs(a_ex))] / th),
                               excess_vs_full_opt=float(r.fun) / Jph - 1, success=bool(r.success))
    for lab in ("paper", "alt"):
        ac, _, _ = closed_a(s, th, n, sig, out[lab]["beta"])
        out[lab]["rms_dev_from_ansatz_exact"] = float(np.sqrt(np.mean((ac - a_ex) ** 2)))
        out[lab]["rel_amax_dev_from_ansatz_exact"] = out[lab]["a_max"] / out["ansatz_exact"]["a_max"] - 1
    return out


def first_order_vector(P, consistent, Mg=600):
    """multi-mode first-order photon path (pc.first_order_prep with a choice of the coupling matrix):
    u'' + (4 - C/sin^2 s) u = 2 v/n_chi, a = u/sin s, with C = 2(M - 2 v v^T/n_chi)/n_chi as in photon_core (the
    linearization of the w-normalized Euler-Lagrange equation) or C = 2M/n_chi (the consistent second-order expansion
    of J about the geodesic, in which the product of the zeroth-order force and the first-order change of w is dropped)"""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    Bp = np.linalg.svd(np.vstack([P.u0, P.chi]))[2][2:].T
    m = Bp.shape[1]
    v = Bp.T @ (P.nv * P.chi)
    Mm = Bp.T @ (P.nv[:, None] * Bp)
    C = 2 * Mm / P.n_chi if consistent else 2 * (Mm - 2 * np.outer(v, v) / P.n_chi) / P.n_chi
    s = np.linspace(0, P.th, Mg + 1)
    h = s[1]
    si = s[1:-1]
    A = sp.kron(sp.diags([1., -2., 1.], [-1, 0, 1], shape=(Mg - 1, Mg - 1)) / h ** 2, sp.eye(m)) + \
        sp.block_diag([4 * np.eye(m) - C / np.sin(x) ** 2 for x in si])
    u = spsolve(A.tocsc(), np.tile(2 * v / P.n_chi, Mg - 1)).reshape(Mg - 1, m)
    a = np.zeros((Mg + 1, m))
    a[1:-1] = u / np.sin(si)[:, None]
    X = pc._build(P.geo(s), Bp, a)
    return X, np.linalg.norm(a, axis=1)


def c06b_vector_first_order():
    res = {}
    for name in ("gkp", "cat"):
        P = prep(name)
        U, J = photon_prep(name)
        Xn, _ = pc.dense(U, 41)
        dev_full = np.arcsin(np.sqrt(np.clip(1 - (Xn @ P.u0) ** 2 - (Xn @ P.chi) ** 2, 0, 1)))
        r = dict(J_full_opt=J, max_deviation_full_opt=float(dev_full.max()))
        for lab, cons in (("photon_core_C", False), ("consistent_C", True)):
            X, an = first_order_vector(P, cons)
            Jv = pc.exposure(pc.resample(X, 60), P.N)
            r[lab] = dict(J=Jv, excess_vs_full_opt=Jv / J - 1, max_deviation=float(an.max()))
        res[name] = r
    rec("C06b", "multi-mode first-order photon path: photon_core's coupling matrix vs the consistent one", "SEE AUDIT.md", **res)
    for k, nm in (("gkp", "GKP"), ("cat", "Cat")):
        mac(f"vectorExcessCons{nm}", 100 * res[k]["consistent_C"]["excess_vs_full_opt"], ".2f", f"multi-mode first order (consistent) excess % {k}")
        mac(f"vectorExcessPaper{nm}", 100 * res[k]["photon_core_C"]["excess_vs_full_opt"], ".1f", f"multi-mode first order (photon_core C) excess % {k}")
        mac(f"devMaxFull{nm}", res[k]["max_deviation_full_opt"], ".3f", f"max departure of the numerical optimum from the geodesic plane (rad) {k}")
        mac(f"devMaxVectorCons{nm}", res[k]["consistent_C"]["max_deviation"], ".3f", f"max departure, consistent multi-mode first order (rad) {k}")


def gallery_vectors():
    def squeezed_vacuum(r):
        psi = np.zeros(D, complex)
        k = np.arange(D // 2)
        psi[0::2] = (-np.tanh(r)) ** k * np.exp(0.5 * gammaln(2 * k + 1) - k * np.log(2) - gammaln(k + 1)) / np.sqrt(np.cosh(r))
        return psi / np.linalg.norm(psi)

    def four_cat(al):
        psi = sum(states.coherent(al * 1j ** k, D) for k in range(4))
        return states._real_gauge(psi / np.linalg.norm(psi))

    def plus_L(dl):
        p = states.gkp(dl, 0, D) + states.gkp(dl, 1, D)
        return p / np.linalg.norm(p)
    return {"sq1": squeezed_vacuum(1.0), "cat15": states.cat(1.5, D), "cat25": states.cat(2.5, D), "cat4": four_cat(2.0),
            "gkp03": states.gkp(0.3, 0, D), "gkp05": states.gkp(0.5, 0, D), "gkpplus": plus_L(0.4)}


def c06_closed_form():
    res = {name: closed_form_study(name) for name in ("gkp", "cat")}
    gal = {}
    for key, vec in gallery_vectors().items():
        P = paths.Prep(vec)
        e = GAL["targets"][key]
        if P.sig < 1e-9:
            continue
        xi = (P.nv * P.chi - P.n_chi * P.chi) / P.sig
        nu = float(xi @ (P.nv * xi))
        s = np.linspace(0, P.th, 4001)
        row = {}
        for lab, beta in (("paper", 2 * (nu * P.n_chi - 2 * P.sig ** 2) / P.n_chi ** 2), ("alt", 2 * nu / P.n_chi)):
            a, mu, _ = closed_a(s, P.th, P.n_chi, P.sig, beta)
            Xc = np.cos(a)[:, None] * P.geo(s) + np.sin(a)[:, None] * xi
            row[lab] = dict(beta=beta, excess=pc.exposure(pc.resample(Xc, 60), P.N) / e["J_ph"] - 1)
        row["stored_excess"] = e.get("closed_excess")
        gal[key] = row
    res["gallery"] = gal
    res["gallery_max_excess_paper"] = max(v["paper"]["excess"] for v in gal.values())
    res["gallery_max_excess_alt"] = max(v["alt"]["excess"] for v in gal.values())
    better = "paper" if all(abs(res[k]["paper"]["excess_vs_full_opt"]) <= abs(res[k]["alt"]["excess_vs_full_opt"]) for k in ("gkp", "cat")) else "alt"
    res["more_accurate_beta"] = better
    rec("C06", "closed-form first-order photon path (beta of the paper vs beta = 2 nu/n_chi), against the exact ansatz "
        "optimum and the full numerical optimum", "SEE AUDIT.md", **res)
    for k, nm in (("gkp", "GKP"), ("cat", "Cat")):
        mac(f"closedExcess{nm}", 100 * res[k]["paper"]["excess_vs_full_opt"], ".1f", f"closed form (paper beta) exposure excess % {k}")
    mac("closedExcessGalleryMax", 100 * res["gallery_max_excess_paper"], ".1f", "max excess % over the gallery (paper beta)")
    for k, nm in (("gkp", "GKP"), ("cat", "Cat")):
        mac(f"betaCons{nm}", res[k]["alt"]["beta"], ".2f", f"consistent beta = 2 nu/n_chi, {k}")
        mac(f"muCons{nm}", res[k]["alt"]["mu"], ".2f", f"mu = (1 + sqrt(1 + 4 beta))/2 with the consistent beta, {k}")
        mac(f"closedExcessCons{nm}", 100 * res[k]["alt"]["excess_vs_full_opt"], ".1f", f"closed form (consistent beta) excess % {k}")
        mac(f"nChi{nm}", res[k]["n_chi"], ".3f", f"n_chi {k}")
        mac(f"sigmaChi{nm}", res[k]["sigma_chi"], ".3f", f"sigma_chi {k}")
        mac(f"nuXi{nm}", res[k]["nu"], ".3f", f"nu = <xi|n|xi> {k}")
    mac("closedExcessConsGalleryMax", 100 * res["gallery_max_excess_alt"], ".1f", "max excess % over the gallery (consistent beta)")
    return res


def m_curve(nv, T):
    F, nb = pc.min_photon_curve(nv, T)
    return F, nb


def m_of(F, Fc, nbc, c0sq):
    F = np.asarray(F, float)
    return np.where(F <= c0sq, 0.0, np.interp(F, Fc, nbc, left=0.0))


def two_sided_flip_bound():
    """J >= min_L int_0^L max(m_0(cos^2 sigma), m_1(cos^2(L - sigma))) d sigma: at arclength sigma along any path of length
    L from |0_L> to |1_L>, the Bures distance to |0_L> is <= sigma and to |1_L> is <= L - sigma"""
    F0, n0 = m_curve(LG.nv, LG.u0)
    F1, n1 = m_curve(LG.nv, LG.uT)
    c0a, c0b = LG.u0[0] ** 2, LG.uT[0] ** 2

    def I(L, M=6001):
        sg = np.linspace(0, L, M)
        f0 = np.where(sg <= np.pi / 2, m_of(np.cos(np.minimum(sg, np.pi / 2)) ** 2, F0, n0, c0a), 0.0)
        r = L - sg
        f1 = np.where(r <= np.pi / 2, m_of(np.cos(np.minimum(r, np.pi / 2)) ** 2, F1, n1, c0b), 0.0)
        return float(trapz(np.maximum(f0, f1), sg))
    Ls = np.linspace(LG.th, 4.0, 400)
    Is = np.array([I(L) for L in Ls])
    j = int(np.argmin(Is))
    B0, B1 = pc.lower_bound(LG.nv, LG.u0), pc.lower_bound(LG.nv, LG.uT)
    return dict(B_two_sided=float(Is[j]), L_at_min=float(Ls[j]), B_0L=B0, B_1L=B1, B_sum=B0 + B1, I_at_theta=float(Is[0]))


def c07_lower_bound():
    res = {}
    for name in ("gkp", "cat"):
        P = prep(name)
        Bv = pc.lower_bound(P.nv, P.uT)
        J = photon_prep(name)[1]
        F, nb = m_curve(P.nv, P.uT)
        chk = {}
        for Fq in (0.6, 0.8, 0.95):
            if Fq > P.c0 ** 2:
                chk[str(Fq)] = dict(curve=float(np.interp(Fq, F, nb)), slsqp=float(pc.min_photon_numeric(P.nv, P.uT, Fq)))
        o = np.argsort(F)
        keep = F[o] < 1 - 1e-9                           # the lambda -> infinity end is pure round-off (F = 1 - O(1e-16))
        dn = np.diff(nb[o][keep])
        res[name] = dict(B=Bv, B_photon_ns=ns(Bv), J_opt=J, J_opt_over_B_minus_1=J / Bv - 1, m_curve_vs_bruteforce=chk,
                         m_curve_largest_decrease=float(-dn.min()) if dn.size else 0.0,
                         m_curve_F_range=[float(F.min()), float(F.max())], c0_squared=float(P.c0 ** 2))
    fl = two_sided_flip_bound()
    Jf = flip_photon()[1]
    res["flip"] = dict(**fl, B_max_printed=1.2638, J_opt=Jf, gap_one_sided=Jf / max(fl["B_0L"], fl["B_1L"]) - 1,
                       gap_two_sided=Jf / fl["B_two_sided"] - 1)
    ok = (abs(res["gkp"]["B"] - 0.5904) < 2e-4 and abs(100 * res["gkp"]["J_opt_over_B_minus_1"] - 1.2) < 0.06
          and abs(100 * res["cat"]["J_opt_over_B_minus_1"] - 2.6) < 0.06 and abs(max(fl["B_0L"], fl["B_1L"]) - 1.2638) < 2e-4)
    rec("C07", "lower bound B on the exposure (rigorous; numbers); flip: one-sided max{B(0_L),B(1_L)} and a two-sided "
        "bound", "OK (numbers); proof and flip tightening in AUDIT.md" if ok else "CHECK", **res)
    mac("BboundGKP", ns(res["gkp"]["B"]), ".1f", "B/omega GKP (photon ns)")
    mac("gapBoundGKP", 100 * res["gkp"]["J_opt_over_B_minus_1"], ".1f", "J_opt/B - 1 GKP (%)")
    mac("gapBoundCat", 100 * res["cat"]["J_opt_over_B_minus_1"], ".1f", "J_opt/B - 1 cat (%)")
    mac("BboundFlipTwoSided", ns(fl["B_two_sided"]), ".0f", "two-sided flip bound (photon ns)")
    mac("gapBoundFlipTwoSided", 100 * res["flip"]["gap_two_sided"], ".0f", "J_opt/B2 - 1 flip (%)")


def c08_fock_theorem():
    res = {}
    for name in ("fock2", "fock4", "binomial"):
        P = prep(name)
        U, J = photon_prep(name)
        # pc.lower_bound needs <0|T> != 0 (its lambda-curve degenerates for a Fock target); there m(F) = nF exactly,
        # so B = int_0^{pi/2} n cos^2 r dr = n pi/4 (the Fock theorem)
        Bv = int(name[4:]) * np.pi / 4 if name.startswith("fock") else pc.lower_bound(P.nv, P.uT)
        chi_is_fock = bool(np.sum(np.abs(P.chi) > 1e-12) == 1)
        res[name] = dict(theta0=P.th, chi_is_Fock=chi_is_fock, J_geo=P.J_geo(), B=Bv, J_num_opt=J,
                         J_num_minus_J_geo=J - P.J_geo(), B_minus_J_geo=Bv - P.J_geo(),
                         n_pi_over_4=(int(name[4:]) * np.pi / 4 if name.startswith("fock") else None))
    P = prep("gkp")
    res["gkp_counterexample"] = dict(chi_is_Fock=False, J_geo=P.J_geo(), J_num_opt=photon_prep("gkp")[1])
    ok = all(abs(res[k]["J_num_minus_J_geo"]) < 1e-5 and abs(res[k]["B_minus_J_geo"]) < 1e-4 for k in ("fock2", "fock4", "binomial"))
    rec("C08", "Fock theorem J >= n pi/4 and its generalization: the brachistochrone is the unique photon-optimal path "
        "whenever chi is a Fock state", "OK (statement sharpened, see AUDIT.md)" if ok else "CHECK", **res)


def c09_path_numbers():
    res = {}
    P = prep("gkp")
    geo = P.geo(np.linspace(0, P.th, 32))
    U, J = photon_prep("gkp")
    Xg, _ = pc.dense(geo, 41)
    X, _ = pc.dense(U, 41)
    res["gkp"] = dict(J_geo_closed=P.J_geo(), J_geo_numeric=pc.exposure(geo, P.N), T_geo_ns=ns(P.th),
                      T_ph_ns=ns(pc.length(U)), expo_geo_ns=ns(P.J_geo()), expo_ph_ns=ns(J), saving=1 - J / P.J_geo(),
                      time_cost=pc.length(U) / P.th - 1, D_geo=float(pc.dist_to(X, Xg).max()))
    geof = LG.geo(np.linspace(0, LG.th, 32))
    Uf, Jf = flip_photon()
    Xgf, _ = pc.dense(geof, 41)
    Xf, sf = pc.dense(Uf, 41)
    ss = np.linspace(0, LG.th, 201)
    G = LG.geo(ss)
    nb_direct = np.sum(G ** 2 * LG.nv, 1)
    nb_formula = (np.sin(LG.th - ss) ** 2 * LG.n0 + np.sin(ss) ** 2 * LG.n1 + 2 * np.sin(LG.th - ss) * np.sin(ss) * LG.N01) / np.sin(LG.th) ** 2
    tha, thb = np.arccos(abs(LG.u0[0])), np.arccos(abs(LG.uT[0]))

    def Jl(t, nn):
        return nn / np.sin(t) ** 2 * (0.5 - np.sin(2 * t) / (4 * t))
    Ud = pc.resample(LG.detour(), 60)
    res["flip"] = dict(theta=LG.th, n0=LG.n0, n1=LG.n1, N01=LG.N01, J_geo_closed=LG.J_geo(), J_geo_numeric=pc.exposure(geof, NF),
                       nbar_formula_err=float(np.max(np.abs(nb_direct - nb_formula))),
                       J_det_closed=float(tha * Jl(tha, LG.n0) + thb * Jl(thb, LG.n1)), J_det_numeric=pc.exposure(Ud, NF),
                       L_det=float(tha + thb), theta_a=tha, theta_b=thb, J_opt=Jf, L_opt=pc.length(Uf), T_geo_ns=ns(LG.th),
                       T_ph_ns=ns(pc.length(Uf)), expo_geo_ns=ns(LG.J_geo()), expo_ph_ns=ns(Jf), saving=1 - Jf / LG.J_geo(),
                       time_cost=pc.length(Uf) / LG.th - 1, D_geo=float(pc.dist_to(Xf, Xgf).max()),
                       nbar_min=float(np.min(np.sum(Xf ** 2 * LG.nv, 1))),
                       fig3_scan_ell0=[e for e in FIG3["scan"] if e["family"] == "flip" and e["par"] == 0.0][0],
                       fig2_photon=dict(J=FIG2["flip"]["photon"]["J"], J_ns=FIG2["flip"]["photon"]["J_ns"], T_ns=FIG2["flip"]["photon"]["T_ns"]))
    scan = FIG3["scan"]
    gk = [e for e in scan if e["family"] == "gkp"]
    ca = [e for e in scan if e["family"] == "cat"]
    res["scan"] = dict(gkp_time_cost_max=max(e["time_cost"] for e in gk), gkp_saving_max=max(e["saving"] for e in gk),
                       cat_time_cost_max=max(e["time_cost"] for e in ca), cat_saving_max=max(e["saving"] for e in ca),
                       gkp=[(e["par"], e["time_cost"], e["saving"], e["D_geo"]) for e in gk],
                       cat=[(e["par"], e["time_cost"], e["saving"], e["D_geo"]) for e in ca])
    res["fig1_sphere_weight_min"] = min(v["w_in_min"] for key in ("sphere_prep", "sphere_flip") for v in FIG1[key]["paths"].values())
    sq = GAL["targets"]["sq1"]
    res["squeezed_vacuum"] = dict(T_native_ns=sq["native"]["T_ns"], T_star_ns=sq["T_star_ns"], ratio=sq["native"]["T_ns"] / sq["T_star_ns"],
                                  J_native_ns=sq["native"]["J_ns"], J_geo_ns=ns(sq["J_geo"]), J_ph_ns=ns(sq["J_ph"]), B_ns=ns(sq["B"]),
                                  native_J_formula=1e3 * (np.sinh(2.0) / 4 - 0.5) / G_SQ)
    ok = (abs(res["gkp"]["J_geo_closed"] - res["gkp"]["J_geo_numeric"]) < 1e-8 and res["flip"]["nbar_formula_err"] < 1e-10
          and abs(res["flip"]["J_geo_closed"] - 3.9217) < 1e-4 and abs(res["flip"]["J_det_closed"] - 2.1763) < 1e-4
          and abs(res["flip"]["L_det"] - 1.9970) < 1e-4)
    rec("C09", "closed forms of J_geo (prep and flip), flip <n>_g(s), detour J_det, and the quoted path numbers",
        "OK (see AUDIT.md for the 362 vs 359 photon ns item)" if ok else "CHECK", **res)
    g, f = res["gkp"], res["flip"]
    mac("TphGKP", g["T_ph_ns"], ".0f", "photon-optimal T (ns) GKP")
    mac("expoGeoGKP", g["expo_geo_ns"], ".0f", "brachistochrone exposure (photon ns) GKP")
    mac("expoPhGKP", g["expo_ph_ns"], ".0f", "photon-optimal exposure (photon ns) GKP")
    mac("savingGKP", 100 * g["saving"], ".1f", "exposure saving % GKP")
    mac("timeCostGKP", 100 * g["time_cost"], ".1f", "extra time % GKP")
    mac("DgeoGKP", g["D_geo"], ".2f", "max departure (rad) GKP")
    mac("TphFlip", f["T_ph_ns"], ".0f", "photon-optimal T (ns) flip")
    mac("expoGeoFlip", f["expo_geo_ns"], ".0f", "brachistochrone exposure (photon ns) flip")
    mac("expoPhFlip", f["expo_ph_ns"], ".0f", "photon-optimal exposure (photon ns) flip")
    mac("savingFlip", 100 * f["saving"], ".0f", "exposure saving % flip")
    mac("timeCostFlip", 100 * f["time_cost"], ".0f", "extra time % flip")
    mac("DgeoFlip", f["D_geo"], ".2f", "max departure (rad) flip")
    mac("JgeoFlip", f["J_geo_closed"], ".4f", "J_geo flip (rad)")
    mac("JdetFlip", f["J_det_closed"], ".4f", "J_det (rad)")
    mac("LdetFlip", f["L_det"], ".4f", "L_det (rad)")
    mac("JoptFlip", f["J_opt"], ".4f", "J numerical optimum flip (rad)")
    mac("LoptFlip", f["L_opt"], ".4f", "L numerical optimum flip (rad)")
    mac("scanTimeMax", 100 * max(res["scan"]["gkp_time_cost_max"], res["scan"]["cat_time_cost_max"]), ".0f", "max extra time % (GKP 0.3-0.5, cats <= 2.5)")
    mac("scanSavingMax", 100 * max(res["scan"]["gkp_saving_max"], res["scan"]["cat_saving_max"]), ".0f", "max saving %")
    mac("sphereWeightMin", 100 * res["fig1_sphere_weight_min"], ".1f", "min weight in the 3D subspace of Fig. 1 (%)")


def var_int(U, N):
    pr = paths.profile(U, N)
    return float(trapz(pr["var"], pr["s"])), float(trapz(pr["nbar"], pr["s"])), float(pr["s"][-1])


def qubit_rates(T1, Tphi):
    g1, gp = 1 / T1, 1 / Tphi
    return dict(equatorial=gp / 2 + g1 / 4, haar_average=g1 / 3 + gp / 3, excited_state=g1,
                max_over_states=max(g1 * p * p + 2 * gp * p * (1 - p) for p in np.linspace(0, 1, 100001)))


def c10_first_order_error():
    res = {}
    for nk in ("Yale", "CSproposal"):
        s = NOISE[nk]
        k, mu = kappa_mu(nk)
        res[nk] = dict(kappa=k, mu=mu, inv_mu_us=1 / mu, w_star=k / mu, qubit_rates=qubit_rates(s["T1q"], s["Tphiq"]))
    kphi = 2 / NOISE["Yale"]["Tphic"]
    kY, muY = kappa_mu("Yale")
    Pg = prep("gkp")
    geo = Pg.geo(np.linspace(0, Pg.th, 32))
    rows = {}
    for lab, (U, N) in {"gkp_geodesic": (geo, Pg.N), "gkp_photon": (photon_prep("gkp")[0], Pg.N),
                        "flip_geodesic": (LG.geo(np.linspace(0, LG.th, 32)), NF), "flip_tradeoff_Yale": (flip_tradeoff("Yale"), NF),
                        "flip_photon": (flip_photon()[0], NF)}.items():
        Iv, Jn, L = var_int(U, N)
        eps_t, eps_n, eps_d = muY * L / W, kY * Jn / W, kphi * Iv / W
        rows[lab] = dict(L=L, J=Jn, T_ns=ns(L), int_var_ds=Iv, eps_time=eps_t, eps_photon=eps_n, eps_cavity_dephasing=eps_d,
                         eps=eps_t + eps_n, eps_with_dephasing=eps_t + eps_n + eps_d, dephasing_share=eps_d / (eps_t + eps_n))
    res["paths_Yale_at_omega"] = rows
    fl = FIG3["tradeoff"]["flip"]["eps"]
    res["fig3_flip"] = {nk: dict(eps_geo=fl[nk]["eps_geo"], eps_w_star=fl[nk]["at_w_star"]["eps"], eps_photon=fl[nk]["eps_photon"],
                                 gain_w_star=1 - fl[nk]["at_w_star"]["eps"] / fl[nk]["eps_geo"],
                                 photon_vs_geo=fl[nk]["eps_photon"] / fl[nk]["eps_geo"] - 1,
                                 T_w_star_ns=ns(fl[nk]["at_w_star"]["L"]), w_star=fl[nk]["at_w_star"]["w"]) for nk in ("Yale", "CSproposal")}
    gk = FIG3["tradeoff"]["gkp"]["eps"]
    res["fig3_gkp_gain_w_star"] = {nk: 1 - gk[nk]["at_w_star"]["eps"] / gk[nk]["eps_geo"] for nk in ("Yale", "CSproposal")}
    # eps_1 printed in Fig. 4(a): the brachistochrone run in T (pace m = T omega / theta0), and Fig. 5 at m = 2.807
    fig4 = {}
    for Tn in (344.0, 412.0, 824.0, 2133.0):
        m = (Tn * 1e-3) * W / Pg.th
        g = rows["gkp_geodesic"]
        fig4[str(int(Tn))] = dict(eps1=m * g["eps"], eps1_with_dephasing=m * g["eps_with_dephasing"])
    m5 = 6.18 / (G_SQ * LG.th / W)
    fig5 = {lab: dict(T_ns=m5 * rows[lab]["T_ns"], eps1=m5 * rows[lab]["eps"], eps1_with_dephasing=m5 * rows[lab]["eps_with_dephasing"])
            for lab in ("flip_geodesic", "flip_tradeoff_Yale", "flip_photon")}
    res["fig4_eps1"], res["fig5_eps1"], res["m_fig5"] = fig4, fig5, m5
    # where does the complete first-order error (with kappa_phi int Var(n) dt) put the sweet spot along the family?
    ws = np.r_[np.logspace(-2, 3, 21), kY / muY]
    fam_res = {}
    for key, (Nm, u0, uT, geo0, Uph) in {
            "flip": (NF, LG.u0, LG.uT, LG.geo(np.linspace(0, LG.th, 32)), flip_photon()[0]),
            "gkp": (Pg.N, Pg.u0, Pg.uT, geo, photon_prep("gkp")[0])}.items():
        fam = cached(f"family_{key}", lambda: paths.tradeoff_family(Nm, u0, uT, geo0, Uph, ws))
        rows_f = []
        for w_, U in zip(ws, fam):
            Iv, Jn, L = var_int(U, Nm)
            rows_f.append(dict(w=float(w_), T_ns=ns(L), eps=(muY * L + kY * Jn) / W, eps_full=(muY * L + kY * Jn + kphi * Iv) / W))
        Iv, Jn, L = var_int(Uph, Nm)
        ph = dict(w=float("inf"), T_ns=ns(L), eps=(muY * L + kY * Jn) / W, eps_full=(muY * L + kY * Jn + kphi * Iv) / W)
        g0r = rows["gkp_geodesic" if key == "gkp" else "flip_geodesic"]
        best = min(rows_f + [ph], key=lambda r: r["eps_full"])
        best0 = min(rows_f + [ph], key=lambda r: r["eps"])
        fam_res[key] = dict(eps_geo=g0r["eps"], eps_full_geo=g0r["eps_with_dephasing"], argmin_eps=best0, argmin_eps_full=best,
                            gain_full_at_best=1 - best["eps_full"] / g0r["eps_with_dephasing"],
                            photon_end=ph, family=rows_f)
    res["family_with_cavity_dephasing_Yale"] = fam_res
    rec("C10", "first-order error eps = mu T + kappa int<n>dt: meaning of mu, omitted cavity dephasing, quoted numbers",
        "CORRECTED (wording of mu; dephasing term to be stated)", **res)
    mac("invMuYale", 1 / muY, ".0f", "1/mu Yale (us)")
    mac("invMuCSP", res["CSproposal"]["inv_mu_us"], ".1f", "1/mu CS proposal (us)")
    mac("wStarYale", res["Yale"]["w_star"], ".2f", "w* Yale")
    mac("wStarCSP", res["CSproposal"]["w_star"], ".2f", "w* CS proposal")
    mac("epsGainYale", 100 * res["fig3_flip"]["Yale"]["gain_w_star"], ".0f", "flip eps gain at w* (Yale) %")
    mac("epsPhotonYale", -100 * res["fig3_flip"]["Yale"]["photon_vs_geo"], ".0f", "flip eps gain of the photon path (Yale) %")
    mac("epsGainCSP", 100 * res["fig3_flip"]["CSproposal"]["gain_w_star"], ".0f", "flip eps gain at w* (CS proposal) %")
    mac("epsPhotonCSP", 100 * res["fig3_flip"]["CSproposal"]["photon_vs_geo"], ".0f", "photon path worse by % (CS proposal)")
    mac("TwStarFlip", res["fig3_flip"]["Yale"]["T_w_star_ns"], ".0f", "trade-off path T (ns), Yale")
    mac("dephShareGKP", 100 * rows["gkp_geodesic"]["dephasing_share"], ".0f", "cavity dephasing / eps, GKP brachistochrone (%)")
    mac("dephShareFlipGeo", 100 * rows["flip_geodesic"]["dephasing_share"], ".0f", "cavity dephasing / eps, flip brachistochrone (%)")
    mac("dephShareFlipPh", 100 * rows["flip_photon"]["dephasing_share"], ".0f", "cavity dephasing / eps, flip photon path (%)")
    mac("dephShareGKPph", 100 * rows["gkp_photon"]["dephasing_share"], ".0f", "cavity dephasing / eps, GKP photon path (%)")
    mac("dephShareFlipTr", 100 * rows["flip_tradeoff_Yale"]["dephasing_share"], ".0f", "cavity dephasing / eps, flip trade-off (%)")
    for lab, nm in (("flip_geodesic", "FlipGeo"), ("flip_tradeoff_Yale", "FlipTr"), ("flip_photon", "FlipPh"),
                    ("gkp_geodesic", "GKPgeo"), ("gkp_photon", "GKPph")):
        mac(f"eps{nm}", 1e3 * rows[lab]["eps"], ".2f", f"eps (1e-3) without cavity dephasing, {lab}")
        mac(f"epsFull{nm}", 1e3 * rows[lab]["eps_with_dephasing"], ".2f", f"eps (1e-3) with cavity dephasing, {lab}")
    fg = rows["flip_geodesic"]["eps_with_dephasing"]
    mac("epsFullGainFlipTr", 100 * (1 - rows["flip_tradeoff_Yale"]["eps_with_dephasing"] / fg), ".0f", "flip, w* path vs geodesic, with dephasing (%)")
    mac("epsFullGainFlipPh", 100 * (1 - rows["flip_photon"]["eps_with_dephasing"] / fg), ".0f", "flip, photon path vs geodesic, with dephasing (%)")
    mac("epsFullGainGKPph", 100 * (1 - rows["gkp_photon"]["eps_with_dephasing"] / rows["gkp_geodesic"]["eps_with_dephasing"]), ".1f",
        "GKP prep, photon path vs geodesic, with dephasing (%)")
    for Tn, nm in (("412", "A"), ("824", "B"), ("2133", "C")):
        mac(f"epsOneFigFour{nm}", 100 * fig4[Tn]["eps1"], ".2f", f"eps1 (%) brachistochrone at {Tn} ns, no cavity dephasing")
        mac(f"epsOneFullFigFour{nm}", 100 * fig4[Tn]["eps1_with_dephasing"], ".2f", f"eps1 (%) brachistochrone at {Tn} ns, with dephasing")
    for lab, nm in (("flip_geodesic", "Geo"), ("flip_tradeoff_Yale", "Tr"), ("flip_photon", "Ph")):
        mac(f"epsOneFigFive{nm}", 100 * fig5[lab]["eps1"], ".2f", f"eps1 (%) flip {lab} at m = 2.81, no cavity dephasing")
        mac(f"epsOneFullFigFive{nm}", 100 * fig5[lab]["eps1_with_dephasing"], ".2f", f"eps1 (%) flip {lab} at m = 2.81, with dephasing")
    ff = fam_res["flip"]["argmin_eps_full"]
    mac("wFullBestFlip", ff["w"], ".1f", "family weight minimizing eps with cavity dephasing (flip, Yale)")
    mac("TFullBestFlip", ff["T_ns"], ".0f", "its duration (ns)")
    mac("epsFullGainFlipBest", 100 * fam_res["flip"]["gain_full_at_best"], ".0f", "its gain vs geodesic (%)")
    mac("haarMuYale", 1e3 * res["Yale"]["qubit_rates"]["haar_average"], ".3f", "Haar-averaged qubit error rate (1e-3/us), Yale")
    mac("eqMuYale", 1e3 * res["Yale"]["mu"], ".3f", "equatorial qubit error rate mu (1e-3/us), Yale")


def c11_capacity():
    a, ad = ops()
    SA, SB = 0.5j * (a @ a - ad @ ad), 0.5 * (a @ a + ad @ ad)
    DA, DB = (1j / np.sqrt(2)) * (a - ad), (a + ad) / np.sqrt(2)
    nop = ad @ a

    def push(O, t, p):
        return float(np.real(np.vdot(t, -1j * (O @ p))))
    res = {}
    for name, (S, T) in transfers().items():
        c0, th, chi = B.geodesic_parts(S, T)
        vs, vd, vsn = [], [], []
        for s in np.linspace(0, th, 41):
            p = np.cos(s) * S + np.sin(s) * chi
            t = -np.sin(s) * S + np.cos(s) * chi
            vs.append(np.hypot(push(SA, t, p), push(SB, t, p)))
            vd.append(np.hypot(push(DA, t, p), push(DB, t, p)))
            vsn.append(abs(push(nop, t, p)))
        vs = np.array(vs)
        v_formula = abs(chi[2]) / np.sqrt(2) if name != "flip" else None
        res[name] = dict(v_over_g_start=vs[0], v_over_g_min=vs.min(), v_over_g_max=vs.max(), v_over_g_formula=v_formula,
                         v_over_omega=vs[0] * G_SQ / W, T_tan_ns=(1e3 * th / (G_SQ * vs[0]) if vs[0] > 1e-12 else None),
                         T_star_ns=ns(th), CD_or_SNAP_drive_push_max=max(vd), SNAP_dispersive_push_max_per_unit_chi=max(vsn))
    res["flip_means_v_over_omega_stored"] = {k: CAP["flip"]["mean_" + k] for k in ("geodesic", "tradeoff", "photon")}
    res["prep_means_stored"] = {n: dict(geo=CAP[n]["mean_geodesic"], photon=CAP[n]["mean_photon"]) for n in ("fock2", "fock4", "cat", "gkp")}
    ok = all(res[n]["v_over_g_max"] - res[n]["v_over_g_min"] < 1e-9 for n in res if isinstance(res[n], dict) and "v_over_g_max" in res[n])
    ok &= all(res[n]["CD_or_SNAP_drive_push_max"] < 1e-12 for n in transfers())
    rec("C11", "tangential capacity: v = g|<2|chi>|/sqrt2 from the vacuum, constant along every brachistochrone of real "
        "states; v = 0 for displacement drives and the dispersive term on parity-even real paths",
        "OK (meaning of T >= T_tan to be stated precisely, see AUDIT.md)" if ok else "CHECK", **res)
    for n, nm in (("gkp", "GKP"), ("cat", "Cat"), ("fock2", "Fock"), ("flip", "Flip")):
        mac(f"vOverG{nm}", res[n]["v_over_g_start"], ".4f" if n == "flip" else ".3f", f"v/g {n}")
        if res[n]["T_tan_ns"] is not None:
            mac(f"Ttan{nm}", res[n]["T_tan_ns"] / (1e3 if n == "flip" else 1), ".0f", f"T_tan {n} ({'us' if n == 'flip' else 'ns'})")
    mac("capFlipGeo", res["flip_means_v_over_omega_stored"]["geodesic"], ".3f", "mean v/omega flip brachistochrone")
    mac("capFlipTradeoff", res["flip_means_v_over_omega_stored"]["tradeoff"], ".2f", "mean v/omega flip trade-off")
    mac("capFlipPhoton", res["flip_means_v_over_omega_stored"]["photon"], ".2f", "mean v/omega flip photon path")


def c12_nogo():
    rec("C12", "no-go theorem (product state => Gaussian oscillator evolution) and F <= min{lambda_max, 1 - w_out}",
        "OK (proof checked analytically; no numerical check run)",
        note="Projection of i d/dt(|q>|phi>) = H|q>|phi> on <q| gives i|phi'> = [sum_k <q|Q_k|q> O_k - i<q|q'>]|phi>, a "
             "quadratic Hamiltonian plus a real scalar: Gaussian unitary up to a phase. Covers CS, CD and SNAP (the "
             "dispersive term n|e><e| and the drives are of degree <= 2). For preparation from the vacuum (Gaussian) of a "
             "non-Gaussian target, entanglement is therefore necessary. For the flip (non-Gaussian start) the theorem "
             "only restricts a product-state evolution to a Gaussian map of |0_L>, which cannot reach |1_L> exactly at "
             "finite energy. F = <p|rho|p> <= lambda_max(rho) and <= tr(P rho) = 1 - w_out for P the projector on the "
             "plane of the path.")


def c13_arithmetic():
    r = {}
    for khz, a0 in ((32.8, 30), (46.5, 26)):
        chi = 2 * np.pi * khz * 1e3 * 1e-6            # rad/us
        r[f"{khz} kHz, |alpha0|={a0}"] = dict(chi_rad_per_us=chi, per_branch_rate=chi * a0 / 2)
    r["g_over_sqrt2"] = W
    r["pi_rotation_ns"] = 1e3 * np.pi / (10 * G_SQ)
    r["durations_ns"] = {str(x): 1e3 * x / G_SQ for x in (2.58, 3.09, 6.18, 16.0)}
    r["Tphiq_ratio_Yale_over_CSP"] = NOISE["Yale"]["Tphiq"] / NOISE["CSproposal"]["Tphiq"]
    r["J_rad_to_photon_ns"] = 1e3 / W
    ok = abs(r["32.8 kHz, |alpha0|=30"]["per_branch_rate"] - 3.09) < 0.01 and abs(r["46.5 kHz, |alpha0|=26"]["per_branch_rate"] - 3.80) < 0.01
    rec("C13", "ECD rate chi|alpha0|/2 (from H = -(chi/2) sigma_z a^dag a displaced by alpha0), pi-rotation time, durations",
        "OK" if ok else "CHECK", **r)
    for x, nm in ((2.58, "A"), (3.09, "B"), (6.18, "C"), (16.0, "D")):
        mac(f"duration{nm}", 1e3 * x / G_SQ, ".0f", f"duration gT = {x} (ns)")
    mac("piRotationNs", r["pi_rotation_ns"], ".1f", "pi rotation at |Omega| = 10 g (ns)")


def c14_language():
    hits = []
    for f in ("main.tex", "main_cs.tex", "supplement.tex", "sm_tracking.tex"):
        for i, line in enumerate((MS / "paper" / f).read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"energy[- ]variance", line):
                hits.append(dict(file=f, line=i, text=line.strip()[:160]))
    rec("C14", "'energy variance' used for Delta H (the energy uncertainty, a standard deviation)",
        "CORRECTED" if hits else "OK", occurrences=hits)


def write_numbers():
    lines = ["% Numbers quoted by the draft, generated by simulations/analysis/audit.py from the data "
             "(do not edit by hand)."]
    for k in sorted(NUM):
        v, note = NUM[k]
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}" + (f"  % {note}" if note else ""))
    (HERE / "analytics_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    t0 = time.time()
    for fn in (c01_speeds, c02_brachistochrone, c03_normal_order, c03b_order_table_mp, c04_first_order_loss,
               c05_euler_lagrange, c06_closed_form, c06b_vector_first_order, c07_lower_bound, c08_fock_theorem, c09_path_numbers, c10_first_order_error, c11_capacity, c12_nogo,
               c13_arithmetic, c14_language):
        try:
            fn()
        except Exception as e:                       # keep going; the failing claim is recorded
            import traceback
            traceback.print_exc()
            OUT[fn.__name__] = dict(title=fn.__name__, verdict="ERROR", values=dict(error=repr(e)))
        (HERE / "audit_results.json").write_text(json.dumps(OUT, indent=1))
        write_numbers()
    print(f"\ndone in {time.time() - t0:.0f} s; wrote audit_results.json and analytics_numbers.tex", flush=True)


if __name__ == "__main__":
    main()
