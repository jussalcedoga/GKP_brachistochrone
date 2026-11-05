"""SM, squeezer capacity along the paths (why conditional squeezing follows some paths better than others).

With the qubit in a sigma_z eigenstate, the gate set (4.1) moves the oscillator only along -i S(xi)|p>, S = Re(xi) S_A +
Im(xi) S_B (S_A = (i/2)(a^2 - a^dag^2), S_B = (a^2 + a^dag^2)/2), |xi| <= g.  The largest speed it can give ALONG the
unit tangent t(s) of a path at the point p(s) is (deck, slide 9, generalized from s = 0 to the whole path)
    v(s) = g sqrt( Re<t|-i S_A|p>^2 + Re<t|-i S_B|p>^2 ).
Following the path at the brachistochrone pace needs v >= omega = g/sqrt2; v(s)/omega < 1 is the fraction that must come
from entangle-disentangle cycles (qubit rotations between conditional squeezes; the no-go statement of Fig. 4).
Paths: brachistochrone and photon-optimal path for every preparation; for the flip also the trade-off path (w*_Yale).
Writes results/squeezer_capacity.json and squeezer_capacity.png.   python simulations/paths/si_squeezer_capacity.py [--plot]
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parents[0]
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from common import paths, photon_core as pc, states  # noqa: E402
from common.params import kappa_mu  # noqa: E402

D = 80
a = np.diag(np.sqrt(np.arange(1, D)), 1)
SA = 0.5j * (a @ a - a.T @ a.T)
SB = 0.5 * (a @ a + a.T @ a.T)


def refine(N, u0, uT, U, K=120):
    """re-optimize a 30-node path with K nodes (the capacity is constant on every great-circle arc, so a coarse node
    path shows steps; the refined path approximates the smooth optimum)"""
    X, _ = pc.dense(U, 41)
    return pc.optimize_path(N, u0, uT, [pc.resample(X, K)], K)[0]


def capacity(nodes_even, n_s=161):
    """v(s)/(g/sqrt2) along a piecewise great-circle path given by even-sector real nodes"""
    X, s = pc.dense(nodes_even, 11)
    F = paths.to_full(X, D)
    L = s[-1]
    out_s, out_v = [], []
    for j in range(1, len(F) - 1):
        p = F[j]
        t = F[j + 1] - F[j - 1]
        t = t - np.vdot(p, t) * p
        nt = np.linalg.norm(t)
        if nt < 1e-12:
            continue
        t = t / nt
        cA = np.real(np.vdot(t, -1j * (SA @ p)))
        cB = np.real(np.vdot(t, -1j * (SB @ p)))
        out_s.append(s[j] / L)
        out_v.append(float(np.sqrt(cA ** 2 + cB ** 2) * np.sqrt(2)))        # v / (g/sqrt2)
    o = np.linspace(0, 1, n_s)
    return o, np.interp(o, out_s, out_v)


def compute():
    res = {}
    for name, vec in (("fock2", states.fock(2, D)), ("fock4", states.fock(4, D)), ("cat", states.cat(2.0, D)),
                      ("gkp", states.gkp(0.4, 0, D))):
        P = paths.Prep(vec)
        geo = P.geo(np.linspace(0, P.th, 32))
        U, _ = paths.photon_path_prep(P)
        U = refine(P.N, P.u0, P.uT, U)
        s, vg = capacity(geo)
        _, vp = capacity(U)
        res[name] = dict(s=s.tolist(), geodesic=vg.tolist(), photon=vp.tolist(), mean_geodesic=float(vg.mean()),
                         mean_photon=float(vp.mean()))
        print(f"{name}: <v/omega> geodesic {vg.mean():.3f} (start {vg[0]:.3f}), photon {vp.mean():.3f}", flush=True)
    L = pc.Logical(0.4, D)
    N = np.diag(L.nv)
    geo = L.geo(np.linspace(0, L.th, 32))
    Uph, _ = pc.optimize_path(N, L.u0, L.uT, [pc.resample(L.detour(), 30)], 30)
    k, m = kappa_mu("Yale")
    Ut = paths.tradeoff_family(N, L.u0, L.uT, geo, Uph, [k / m])[0]
    Uph = refine(N, L.u0, L.uT, Uph)
    Xt, _ = pc.dense(Ut, 41)
    Ut = pc.optimize_path(N + np.eye(len(N)) / (k / m), L.u0, L.uT, [pc.resample(Xt, 120)], 120)[0]
    r = {}
    for lab, U in (("geodesic", geo), ("tradeoff", Ut), ("photon", Uph)):
        s, v = capacity(U)
        r[lab] = v.tolist()
        r["mean_" + lab] = float(v.mean())
        print(f"flip {lab}: <v/omega> = {v.mean():.3f}, max {v.max():.3f}", flush=True)
    r["s"] = s.tolist()
    res["flip"] = r
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "squeezer_capacity.json").write_text(json.dumps(res, indent=1))
    return res


# --plot re-draws from results/squeezer_capacity.json without recomputing
res = json.loads((HERE / "results" / "squeezer_capacity.json").read_text()) if "--plot" in sys.argv else compute()

from common import style as S  # noqa: E402,I001
import matplotlib.pyplot as plt  # noqa: E402

S.style(usetex=True)
S.big()
YMAX = 1.05 * max(max(max(res[n]["geodesic"]), max(res[n]["photon"])) for n in ("fock2", "fock4", "cat", "gkp"))
fig, axs = plt.subplots(1, 2, figsize=(14.0, 6.2))
fig.subplots_adjust(wspace=0.30, left=0.08, right=0.99, top=0.88, bottom=0.14)
ax = axs[0]
LBL = {"fock2": r"Fock $|2\rangle$", "fock4": r"Fock $|4\rangle$", "cat": r"cat $|C_2^+\rangle$", "gkp": r"GKP $|0_L\rangle$"}
for name in ("fock2", "fock4", "cat", "gkp"):
    ax.plot(res[name]["s"], res[name]["geodesic"], "-", color=S.TARGET_COLORS[name], lw=3.0, label=LBL[name])
    if name in ("cat", "gkp"):
        ax.plot(res[name]["s"], res[name]["photon"], ls=(0, (4, 2)), color=S.TARGET_COLORS[name], lw=2.4)
ax.axhline(1, color=S.DARK, lw=1.6, ls=(0, (1, 1.5)))
ax.set_xlabel(r"position along the path $s/L$")
ax.set_ylabel(r"squeezer capacity $v(s)/\omega$")
ax.set_ylim(-0.03, YMAX)
ax.set_xlim(0, 1)
ax.set_title(r"preparation: brachistochrone (solid), photon path (dashed)", fontsize=S.FS["ann"])
ax.legend(loc="upper left", fontsize=S.FS["small"], ncol=2, columnspacing=1.0)
S.panel_label(ax, "a")
ax = axs[1]
for lab, sty, name in (("geodesic", S.PATH_STYLE["geodesic"], "brachistochrone"), ("tradeoff", S.PATH_STYLE["tradeoff"],
                       r"trade-off $w^*$ (Yale)"), ("photon", S.PATH_STYLE["photon"], "photon-optimal")):
    ax.plot(res["flip"]["s"], res["flip"][lab], **sty, label=name + rf", mean ${res['flip']['mean_' + lab]:.2f}$")
ax.axhline(1, color=S.DARK, lw=1.6, ls=(0, (1, 1.5)))
ax.set_xlabel(r"position along the path $s/L$")
ax.set_ylabel(r"squeezer capacity $v(s)/\omega$")
ax.set_ylim(-0.03, YMAX)
ax.set_xlim(0, 1)
ax.set_title(r"logical flip $|0_L\rangle\to|1_L\rangle$", fontsize=S.FS["ann"])
ax.legend(loc="upper left", fontsize=S.FS["small"])
S.panel_label(ax, "b")
S.save(fig, HERE / "squeezer_capacity")
print("wrote", HERE / "squeezer_capacity.png")
