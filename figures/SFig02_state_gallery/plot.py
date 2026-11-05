"""Fig. S1, state gallery: photon number against time at the speed limit Delta H = omega along the brachistochrone
(analytic), the numerically optimal photon path and the closed form of Eq. (closed) (round 4: the sum over all normal
modes, manuscript_figures/fig2/compute_fig2.modal_leg), for eleven targets prepared from the vacuum.  Style of main
Fig. 2: analytic paths are wide translucent solid bands, numerical paths narrow dashed lines on top in the same colour
(time optimal royal blue, photon optimal crimson).  For Fock targets and the binomial word |chi> is a Fock state, the
closed form is the brachistochrone and so is the numerical optimum: the three curves coincide.
Brachistochrone, numerical optimum, native squeezer and numbers: manuscript/si/state_gallery/results (read only).
Wigner insets: recomputed here for the target vectors below, exact on the stored grid |q|,|p| <= 5 (review Nov 1, L1-F1).  Closed form: recomputed here for the target vectors of analytics_audit/audit.gallery_vectors and the Fock
and binomial vectors of simulations/paths/si_state_gallery.py (same cutoff 80; the Bures angle is checked against the
stored one).  The printout checks the numbers of every panel against the stored curves.
python figures/SFig02_state_gallery/plot.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import importlib.util
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

import mfcommon as M  # noqa: E402

S = M.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

from common import states  # noqa: E402


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CF = module("compute_fig2", _L.REPO / "figures" / "Fig2" / "compute_fig2.py")
AU = module("audit", _L.REPO / "simulations" / "analysis" / "audit.py")
D = 80
VEC = dict(AU.gallery_vectors(), fock2=states.fock(2, D), fock4=states.fock(4, D), fock6=states.fock(6, D),
           binomial=states.binomial_kitten(D))
FS = S.FS
SRC = M.MS / "si" / "state_gallery" / "results"
R = json.loads((SRC / "state_gallery.json").read_text())["targets"]
CLOSED = {}
for key, vec in VEC.items():
    P = CF.paths.Prep(np.asarray(vec, complex))
    assert abs(P.th - R[key]["theta0"]) < 1e-6, key                  # same target as the stored gallery
    X = CF.modal_leg(P)                                               # Eq. (closed): sum over all normal modes
    assert np.all(np.isfinite(X)), key
    c = CF.curve(X, P.nv, P.uT)
    c["J"] = float(CF.pc.exposure(X, P.N))
    c["excess"] = c["J"] / R[key]["J_ph"] - 1
    c["dev_geo"] = float(np.max(np.arccos(np.clip(np.abs(np.sum(X * P.geo(np.linspace(0, P.th, len(X))), 1)), 0, 1))))
    CLOSED[key] = c
    # the time-optimal path found numerically (length minimized from a randomly perturbed start, compute_fig2)
    c["geo_num"] = CF.numeric_geodesic(P.u0, P.uT, P.nv)
ORDER = ["sq1", "fock2", "fock4", "fock6", "binomial", "cat15", "cat25", "cat4", "gkp03", "gkp05", "gkpplus"]
# Wigner insets: the target vectors above (the targets of the stored gallery), exact on the stored grid |q|,|p| <= 5
# (mfcommon.wigner_grid; the stored state_gallery_wigner.npz drops the parts of cat25, cat4, gkp03 and gkpplus beyond the
# window, review Nov 1, L1-F1); saved to S01_state_gallery_wigner.npz in this folder
_Z0 = np.load(SRC / "state_gallery_wigner.npz")            # grid only
_WG = M.wigner_grid(D, len(_Z0["x"]), float(_Z0["x"][-1]))
assert np.allclose(_WG.x, _Z0["x"])
Wz = dict(x=_WG.x, **{key: _WG.of_vectors(np.asarray(VEC[key])[None]) for key in ORDER})
np.savez_compressed(HERE / "S01_state_gallery_wigner.npz", **Wz)
# as in main Fig. 2: analytic paths wide translucent solid bands, numerical paths narrow dashed lines on top
GEO = dict(color=S.ROYALBLUE, ls="-", lw=8.0, alpha=0.32, solid_capstyle="butt", zorder=2)
AN = dict(color=S.CRIMSON, ls="-", lw=8.0, alpha=0.32, solid_capstyle="butt", zorder=2)
PH = dict(color=S.CRIMSON, ls=(0, (3.2, 1.8)), lw=2.8, zorder=4)
GEONUM = dict(color=S.ROYALBLUE, ls=(0, (3.2, 1.8)), lw=2.8, zorder=4)
NATIVE = dict(color=S.INK, lw=3.0, ls=(0, (6, 1.5, 1.5, 1.5)), zorder=3)


def dot(ax, t, n, color):
    ax.plot(t, n, "o", ms=10, color=color, mec="white", mew=1.4, zorder=6, clip_on=False)


fig, axs = plt.subplots(3, 4, figsize=(15.0, 12.8))
fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.07, wspace=0.36, hspace=0.50)
xw = Wz["x"]
for k, key in enumerate(ORDER):
    ax = axs.flat[k]
    e = R[key]
    g, p, c = e["geodesic"], e["photon"], CLOSED[key]
    ax.plot(c["t_ns"], c["nbar"], **AN)
    ax.plot(g["t_ns"], g["nbar"], **GEO)
    ax.plot(c["geo_num"]["t_ns"], c["geo_num"]["nbar"], **GEONUM)
    ax.plot(p["t_ns"], p["nbar"], **PH)
    dot(ax, g["t_ns"][-1], g["nbar"][-1], S.ROYALBLUE)
    dot(ax, c["t_ns"][-1], c["nbar"][-1], S.CRIMSON)
    dot(ax, p["t_ns"][-1], p["nbar"][-1], S.CRIMSON)
    tmax = max(g["t_ns"][-1], p["t_ns"][-1], c["t_ns"][-1])
    ymax = max(max(g["nbar"]), max(p["nbar"]), max(c["nbar"]))
    if "native" in e:
        ax.plot(e["native"]["t_ns"], e["native"]["nbar"], **NATIVE)
        tmax = max(tmax, e["native"]["t_ns"][-1])
    ax.set_xlim(0, 1.03 * tmax)
    ax.set_ylim(0, 1.95 * ymax)                     # headroom for the inset and the numbers above the curves
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))   # integer labels: no "0.0" meeting the x label "0"
    # the stored label of the four-component cat uses alpha, the GKP lattice symbol; the paper writes cat amplitudes as gamma
    ax.set_title(e["label"].replace(r"|C_{4,\alpha=2}\rangle", r"|C_4\rangle$, $\gamma=2"), fontsize=FS["ann"], pad=8)
    if k % 4 == 0:
        ax.set_ylabel(r"$\bar n$")
    if k >= 7:
        ax.set_xlabel(r"time $t$ (ns)")
    if e["focklike"]:
        txt = rf"$T^*={e['T_star_ns']:.0f}$ ns" + "\n" + r"$\Delta J=\Delta T=0$"
    else:                                           # changes of the photon optimum relative to the brachistochrone
        txt = (rf"$T^*={e['T_star_ns']:.0f}$ ns" + "\n" + rf"$\Delta J=-{100 * e['saving']:.1f}\%$" + "\n"
               + rf"$\Delta T=+{100 * e['time_cost']:.1f}\%$")
    if "native" in e:
        txt += "\n" + rf"$T_{{\rm sq}}={e['native']['T_ns']:.0f}$ ns"
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top", fontsize=FS["small"] - 1, linespacing=1.15)
    ins = ax.inset_axes([0.02, 0.60, 0.29, 0.38])
    ins.imshow(Wz[key], origin="lower", extent=[xw[0], xw[-1], xw[0], xw[-1]], cmap=S.RBCMAP, vmin=-S.W_LIM, vmax=S.W_LIM)
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.4)
    S.panel_label(ax, "abcdefghijk"[k], x=-0.24, y=1.04)
ax = axs.flat[11]
ax.axis("off")
# one entry per path (the caption says which curve is analytic and which numerical), as in main Fig. 2
hs = [(Line2D([], [], **GEO), Line2D([], [], **GEONUM)), (Line2D([], [], **AN), Line2D([], [], **PH)),
      Line2D([], [], **NATIVE)]
ax.legend(hs, ["time optimal", "photon optimal", r"squeezer, (a)"], loc="center left", fontsize=FS["small"],
          handlelength=2.6, labelspacing=0.8, frameon=False, bbox_to_anchor=(-0.12, 0.55), borderpad=0.1,
          handletextpad=0.6, handler_map={tuple: HandlerTuple(ndivide=None)})
S.save(fig, HERE / "S01_state_gallery")

# ---------------------------------------------------------------- checks of the numbers shown in the panels
trapz = getattr(np, "trapezoid", None) or np.trapz
print("Wigner inset grid: |q|,|p| <=", float(np.max(np.abs(xw))))
OM = json.loads((SRC / "state_gallery.json").read_text())["params"]["omega"]
for key in ORDER:
    e = R[key]
    g, p, c = e["geodesic"], e["photon"], CLOSED[key]
    T_geo, T_ph = g["t_ns"][-1], p["t_ns"][-1]
    J_geo_c = trapz(g["nbar"], g["t_ns"]) * OM / 1e3            # exposure from the stored curves (rad)
    J_ph_c = trapz(p["nbar"], p["t_ns"]) * OM / 1e3
    print(f"{key:9s} T*={e['T_star_ns']:.1f} (curve {T_geo:.1f}, theta/omega {1e3 * e['theta0'] / OM:.1f}) ns  "
          f"dJ shown -{100 * e['saving']:.1f}% (curves {100 * (J_ph_c / J_geo_c - 1):+.2f}%)  "
          f"dT shown +{100 * e['time_cost']:.1f}% (curves {100 * (T_ph / T_geo - 1):+.2f}%)"
          + (f"  T_sq {e['native']['T_ns']:.1f} ns" if "native" in e else ""))
    print(f"          closed form (modal): J={c['J']:.5f} vs J_ph={e['J_ph']:.5f}: excess {100 * c['excess']:+.4f}%, "
          f"T={c['T_ns']:.1f} ns (numerical {T_ph:.1f}), largest angle from the brachistochrone {c['dev_geo']:.2e} rad")
ex = {k: CLOSED[k]["excess"] for k in ORDER if not R[k]["focklike"]}
print(f"closed-form excess over the non-Fock targets: {100 * min(ex.values()):.4f}% ({min(ex, key=ex.get)}) to "
      f"{100 * max(ex.values()):.4f}% ({max(ex, key=ex.get)})")
print("wrote", HERE / "S01_state_gallery.png")
