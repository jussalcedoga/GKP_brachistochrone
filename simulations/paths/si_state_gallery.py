"""SM, state gallery: brachistochrone and optimal photon path for more bosonic targets (universality outlook).

Preparation from the vacuum, all at the energy variance Delta H = omega = g/sqrt2 (t = s/omega), Fock cutoff 80.
Targets (all real and parity even, so the even-sector machinery of common/photon_core applies):
  squeezed vacuum S(r)|0>, r = 1 (8.7 dB)   Gaussian; also reached by the native squeezer itself in T_sq = r/g
  Fock |2>, |4>, |6>;  binomial kitten (|0> + |4>)/sqrt2
  even cats alpha = 1.5, 2.5;  four-component cat |C_4> ~ sum_k |i^k alpha>, alpha = 2
  GKP |0_L> Delta = 0.3, 0.5;  GKP |+_L> ~ |0_L> + |1_L> (Delta = 0.4, each word in the real gauge <0|mu_L> >= 0)
Per target: theta0, T* = theta0/omega, n_chi, sigma_chi, J_geo (Prop. 3, closed form), numerical optimum J and L (best of
4 starts), closed-form first-order path (Prop. 4, when sigma_chi > 0), lower bound B (Thm. 1 for Fock-like targets,
Thm. 2 otherwise), <n>(t) on every path, Wigner function of the target.
Writes results/state_gallery.json, results/state_gallery_wigner.npz.   python simulations/paths/si_state_gallery.py   (~5 min)
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
from scipy.special import gammaln  # noqa: E402

from common import paths, photon_core as pc, states  # noqa: E402
from common.params import G_SQ, OMEGA_REF  # noqa: E402
from common.wigner import grid  # noqa: E402

D = 80
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def squeezed_vacuum(r, D):
    """S(r)|0> = cosh(r)^-1/2 sum_n (-tanh r)^n sqrt((2n)!)/(2^n n!) |2n>  (squeezing along x for r > 0)"""
    psi = np.zeros(D, complex)
    n = np.arange(D // 2)
    psi[0::2] = (-np.tanh(r)) ** n * np.exp(0.5 * gammaln(2 * n + 1) - n * np.log(2) - gammaln(n + 1)) / np.sqrt(np.cosh(r))
    return psi / np.linalg.norm(psi)


def four_cat(alpha, D):
    """|C_4> ~ |alpha> + |i alpha> + |-alpha> + |-i alpha>: support n = 0 mod 4, real amplitudes"""
    psi = sum(states.coherent(alpha * 1j ** k, D) for k in range(4))
    return states._real_gauge(psi / np.linalg.norm(psi))


def plus_L(Delta, D):
    p = states.gkp(Delta, 0, D) + states.gkp(Delta, 1, D)
    return p / np.linalg.norm(p)


TARGETS = [  # key, label (LaTeX), plain description, vector, Fock-like (paths coincide, Thm. 1)
    ("sq1", r"squeezed vac.\ $r=1$", "squeezed vacuum r = 1 (8.7 dB)", squeezed_vacuum(1.0, D), False),
    ("fock2", r"Fock $|2\rangle$", "Fock |2>", states.fock(2, D), True),
    ("fock4", r"Fock $|4\rangle$", "Fock |4>", states.fock(4, D), True),
    ("fock6", r"Fock $|6\rangle$", "Fock |6>", states.fock(6, D), True),
    ("binomial", r"binomial $\frac{|0\rangle+|4\rangle}{\sqrt2}$", "binomial (|0>+|4>)/sqrt2", states.binomial_kitten(D), True),
    ("cat15", r"cat $|C^+_{1.5}\rangle$", "even cat alpha = 1.5", states.cat(1.5, D), False),
    ("cat25", r"cat $|C^+_{2.5}\rangle$", "even cat alpha = 2.5", states.cat(2.5, D), False),
    ("cat4", r"4-cat $|C_{4,\alpha=2}\rangle$", "four-component cat alpha = 2", four_cat(2.0, D), False),
    ("gkp03", r"GKP $|0_L\rangle$, $\Delta=0.3$", "GKP |0_L> Delta = 0.3", states.gkp(0.3, 0, D), False),
    ("gkp05", r"GKP $|0_L\rangle$, $\Delta=0.5$", "GKP |0_L> Delta = 0.5", states.gkp(0.5, 0, D), False),
    ("gkpplus", r"GKP $|{+}_L\rangle$, $\Delta=0.4$", "GKP |+_L> Delta = 0.4", plus_L(0.4, D), False),
]


def prof(U, N, target, sub=41):
    X, s = pc.dense(U, sub)
    nb = np.sum(X ** 2 * np.diag(N), 1)
    return dict(t_ns=(1e3 * s / OMEGA_REF).tolist(), nbar=nb.tolist(), F_T=((X @ target) ** 2).tolist())


G = grid(D, 5.0, 121)
res = dict(params=dict(D=D, omega=OMEGA_REF, g=G_SQ), targets={})
W = {}
for key, lab, plain, vec, focklike in TARGETS:
    P = paths.Prep(vec)
    geo = P.geo(np.linspace(0, P.th, 32))
    Xg, _ = pc.dense(geo, 41)
    U, J = paths.photon_path_prep(P)
    X, _ = pc.dense(U, 41)
    nT = float(P.uT @ (P.nv * P.uT))
    B = float(nT * np.pi / 4) if key.startswith("fock") else pc.lower_bound(P.nv, P.uT)
    e = dict(label=lab, plain=plain, focklike=focklike, nbar_T=nT, c0=float(P.c0), theta0=float(P.th),
             T_star_ns=1e3 * P.th / OMEGA_REF, n_chi=float(P.n_chi), sigma_chi=float(P.sig), J_geo=float(P.J_geo()),
             J_ph=float(J), L_ph=float(pc.length(U)), B=B, time_cost=float(pc.length(U) / P.th - 1),
             saving=float(1 - J / P.J_geo()), D_geo=float(pc.dist_to(X, Xg).max()),
             geodesic=prof(geo, P.N, P.uT), photon=prof(U, P.N, P.uT))
    if P.sig > 1e-9:                                                     # Prop. 4 needs sigma_chi > 0
        Xc, info = pc.closed_form_path(P)
        Uc = pc.resample(Xc, 60)
        e["closed"] = prof(Uc, P.N, P.uT)
        e["J_closed"] = float(pc.exposure(Uc, P.N))
        e["closed_excess"] = float(e["J_closed"] / J - 1)
        xi = (P.nv * P.chi - P.n_chi * P.chi) / P.sig
        e["nu"], e["beta"], e["mu"] = float(xi @ (P.nv * xi)), float(info["beta"]), float(info["mu"])
    if key == "sq1":                                                     # the native squeezer: <n> = sinh^2(g t)
        r = 1.0
        t = np.linspace(0, r / G_SQ, 201)
        e["native"] = dict(t_ns=(1e3 * t).tolist(), nbar=(np.sinh(G_SQ * t) ** 2).tolist(), T_ns=1e3 * r / G_SQ,
                           J_ns=float(1e3 * (np.sinh(2 * r) / 4 - r / 2) / G_SQ))
    res["targets"][key] = e
    W[key] = G.of_vectors(vec[None])
    print(f"{key:8s} theta0 {P.th:.4f}  T* {e['T_star_ns']:6.1f} ns  n_chi {P.n_chi:.3f}  sigma {P.sig:.3f}  "
          f"J_geo {P.J_geo():.4f}  J_ph {J:.4f}  B {B:.4f}  time +{100 * e['time_cost']:.2f}%  "
          f"saving {100 * e['saving']:.2f}%  D_geo {e['D_geo']:.3f}"
          + (f"  closed form +{100 * e['closed_excess']:.2f}%" if "closed" in e else ""), flush=True)
sq = res["targets"]["sq1"]
print(f"native squeezer to r = 1: {sq['native']['T_ns']:.1f} ns = {sq['native']['T_ns'] / sq['T_star_ns']:.3f} T*, "
      f"exposure {sq['native']['J_ns']:.1f} photon ns vs brachistochrone {1e3 * sq['J_geo'] / OMEGA_REF:.1f}")
(OUT / "state_gallery.json").write_text(json.dumps(res))
np.savez_compressed(OUT / "state_gallery_wigner.npz", x=G.x, **W)
print("wrote", OUT / "state_gallery.json")
