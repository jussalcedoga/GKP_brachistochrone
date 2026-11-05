"""Audit of the photon-optimal analytics, part 2 (Oct 28, round 3): one principle for preparations, flips and transfers.

The photon-optimal path between any two real even-parity states is the geodesic of the photon metric n |d psi| (Eq. EL).
The metric vanishes only at the vacuum.  Every sub-arc of a minimizer is a minimizer, so a minimizer that passes
through the vacuum is the time-reversed photon-optimal preparation of psi_i followed by that of psi_f, and
J*(psi_i -> psi_f) = J*(0 -> psi_i) + J*(0 -> psi_f).  This script tests, for several transfers, whether the free
minimizer (started from paths that avoid the vacuum and from the vacuum route) passes through the vacuum and attains the
sum, and records the geometry at the vacuum:
  transfers  flip |0_L> -> |1_L> at Delta = 0.3, 0.4, 0.5; |0_L> -> |+_L> (Delta = 0.4); cat (alpha = 2) -> |0_L>
  for each   free optimum (K = 60 nodes; starts: geodesic, geodesic pulled toward the vacuum, vacuum route), max vacuum
             weight along it, J and length; J*(0 -> psi_i) + J*(0 -> psi_f) from the preparation optima; the closed form
             of Eq. (6) on both sides of the vacuum; the angle by which the path turns at the vacuum; the overlap of the
             directions in which the two halves leave the vacuum with Fock |2>
  code space for the flip (Delta = 0.4): n_C = lambda_min(Pi n Pi), the bound J >= n_C theta for any path inside the code
             space, the exposure of the brachistochrone, and the weight outside the code space along the photon optimum
Writes results/audit_photon2.json.  python simulations/analysis/photon_flip/audit_photon2.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

import numpy as np

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE))
from audit_photon import NS, single_mode  # noqa: E402

from common import paths, photon_core as pc, states  # noqa: E402

D = 80


def even(v):
    nv, u = pc.even_real(np.asarray(v, complex), D)
    return nv, u


def prep_opt(nv, T, K=60):
    P = paths.Prep(full(nv, T))
    U, J = pc.optimize_path(P.N, P.u0, P.uT, [pc.resample(P.geo(np.linspace(0, P.th, 400)), K)], K, maxiter=20000)
    return P, U, J


def full(nv, u):
    v = np.zeros(D, complex)
    v[nv.astype(int)] = u
    return v


def leave_dir(X, e0):
    """unit direction in which a dense path X (starting at the vacuum) leaves it"""
    d = X[2] - (X[2] @ e0) * e0
    return d / np.linalg.norm(d)


def transfer(name, ui, uf, nv):
    e0 = np.eye(len(nv))[0]
    N = np.diag(nv)
    if ui @ uf < 0:
        uf = -uf
    th = float(np.arccos(np.clip(ui @ uf, -1, 1)))
    geo = np.array([(np.sin(th - s) * ui + np.sin(s) * uf) / np.sin(th) for s in np.linspace(0, th, 400)])
    pull = geo + 0.6 * np.sin(np.linspace(0, np.pi, 400))[:, None] * e0[None, :]
    pull /= np.linalg.norm(pull, axis=1, keepdims=True)
    si, sf = np.sign(ui @ e0), np.sign(uf @ e0)
    Pi, Ui, Ji = prep_opt(nv, si * ui)
    Pf, Uf, Jf = prep_opt(nv, sf * uf)
    Xi, Xf = pc.dense(Ui, 41)[0], pc.dense(Uf, 41)[0]
    route = np.vstack([si * Xi[::-1], sf * Xf[1:]])
    Ci, _ = single_mode(Pi, "J2")
    Cf, _ = single_mode(Pf, "J2")
    closed = np.vstack([si * Ci[::-1], sf * Cf[1:]])
    best = None
    rows = {}
    for nm, S in (("geodesic", geo), ("pulled", pull), ("vacuum_route", route)):
        U, J = pc.optimize_path(N, ui, uf, [pc.resample(S, 60)], 60, maxiter=20000)
        vac = float(np.max(pc.dense(U, 21)[0][:, 0] ** 2))
        rows[nm] = dict(J_ns=J * NS, T_ns=pc.length(U) * NS, vac_max=vac)
        if best is None or J < best[1]:
            best = (U, J)
    di, df = si * leave_dir(Xi, e0), sf * leave_dir(Xf, e0)
    i2 = int(np.where(nv == 2)[0][0])
    out = dict(name=name, theta=th, J_geo_ns=pc.exposure(geo, N) * NS, starts=rows, J_free_ns=best[1] * NS,
               J_sum_ns=(Ji + Jf) * NS, T_sum_ns=(pc.length(Ui) + pc.length(Uf)) * NS,
               J_closed_ns=pc.exposure(closed, N) * NS, T_closed_ns=pc.length(closed) * NS,
               J_prep_i_ns=Ji * NS, J_prep_f_ns=Jf * NS,
               turn_deg=float(np.degrees(np.pi - np.arccos(np.clip(di @ df, -1, 1)))),
               dir_overlap_2=[float(di[i2]), float(df[i2])])
    print(f"{name}: theta={th:.3f}  J_geo={out['J_geo_ns']:.1f}  free optimum J={out['J_free_ns']:.2f} "
          f"(starts: " + ", ".join(f"{k} {v['J_ns']:.2f}/vac {v['vac_max']:.5f}" for k, v in rows.items()) + ")  "
          f"J*(0->i)+J*(0->f)={out['J_sum_ns']:.2f}  closed form both sides={out['J_closed_ns']:.2f}  "
          f"turn at vacuum {out['turn_deg']:.1f} deg  <2|leave dirs>={out['dir_overlap_2'][0]:.3f},{out['dir_overlap_2'][1]:.3f}",
          flush=True)
    return out


def main():
    res = {}
    for Delta in (0.3, 0.4, 0.5):
        nv, u0 = even(pc.gkp(Delta, 0))
        _, u1 = even(pc.gkp(Delta, 1))
        res[f"flip_D{Delta}"] = transfer(f"flip Delta={Delta}", u0, u1, nv)
        if Delta == 0.4:
            plus = u0 + (u1 if u0 @ u1 >= 0 else -u1)
            res["0L_to_plusL"] = transfer("|0_L> -> |+_L>", u0, plus / np.linalg.norm(plus), nv)
            nvc, cat = pc.target("Cat", 2.0, D)
            res["cat_to_0L"] = transfer("cat(2) -> |0_L>", cat, u0, nv)
            # code space of the flip
            uT = u1 if u0 @ u1 >= 0 else -u1
            Q = np.linalg.qr(np.array([u0, uT]).T)[0]
            nC = np.linalg.eigvalsh(Q.T @ np.diag(nv) @ Q)
            th = float(np.arccos(u0 @ uT))
            geo = np.array([(np.sin(th - s) * u0 + np.sin(s) * uT) / np.sin(th) for s in np.linspace(0, th, 2000)])
            N = np.diag(nv)
            U, _ = pc.optimize_path(N, u0, uT, [pc.resample(np.vstack([geo]), 60)], 60, maxiter=20000)
            X = pc.dense(U, 21)[0]
            leak = 1 - np.sum((X @ Q) ** 2, 1)
            res["code_space"] = dict(n_C_min=float(nC[0]), n_C_max=float(nC[1]), theta=th,
                                     bound_ns=float(nC[0] * th * NS), J_geo_ns=float(pc.exposure(geo, N) * NS),
                                     geo_leak_max=float(np.max(1 - np.sum((geo @ Q) ** 2, 1))),
                                     photon_opt_leak_max=float(leak.max()), vac_weight_in_C=float(np.sum(Q[0] ** 2)))
            print("code space:", res["code_space"], flush=True)
    (HERE / "results" / "audit_photon2.json").write_text(json.dumps(res, indent=1))
    print("wrote", HERE / "results" / "audit_photon2.json")


if __name__ == "__main__":
    main()
