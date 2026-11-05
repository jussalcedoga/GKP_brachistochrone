"""Fig. S2, the photon-optimal path, analytic against numerical (the panels moved out of main Fig. 2), in the style of
main Fig. 2 (analytic paths: wide translucent solid bands; numerical paths: narrow dashed lines on top in the same
colour; time optimal royal blue, photon optimal crimson):
(a) Fock targets, where the brachistochrone is the unique photon-optimal path (<n> = n sin^2 omega t, J = n pi/4) and the
    closed form of Eq. (closed) reduces to it;
(b, c) departure of the photon-optimal path from the plane of the brachistochrone, for GKP |0_L> (Delta = 0.4) and the
    even cat (alpha = 2): the closed form of Eq. (closed) (round 4: first order in all transverse directions, the sum
    over all normal modes, compute_fig2.modal_leg; band), the numerical optimum (total departure, dashed; its component
    along |d>, thin dotted, unless --no-d) and the one-mode reduction along |d>, Eq. (closed1) (compute_fig2.closed_form,
    thin purple dotted).
Sources (read only): manuscript/fig2_photon_optimal/results/fig2.json (Fock curves), the cached numerical optima of the
audit (analytics_audit/cache/photon_{gkp,cat}.pkl), figures/Fig2/compute_fig2.py (modal_leg, closed_form);
analytics_audit/audit.first_order_vector (finite-difference solution of the same first-order equations) is used only
as a check of modal_leg.
python figures/SFig01_photon_path_analytic/plot.py [--no-d]"""
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
import numpy as np  # noqa: E402

from common import paths, photon_core as pc, states  # noqa: E402


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CF = module("compute_fig2", _L.REPO / "figures" / "Fig2" / "compute_fig2.py")          # modal_leg, closed_form, curve
AU = module("audit", _L.REPO / "simulations" / "analysis" / "audit.py")             # photon_prep (cached), first_order_vector
FS = S.FS
D = 80
SHOW_D = "--no-d" not in sys.argv
R = json.loads((M.MS / "fig2_photon_optimal" / "results" / "fig2.json").read_text())
P2 = R["prep"]
# as in main Fig. 2
GEO = dict(color=S.ROYALBLUE, ls="-", lw=8.0, alpha=0.32, solid_capstyle="butt", zorder=2)
AN = dict(color=S.CRIMSON, ls="-", lw=8.0, alpha=0.32, solid_capstyle="butt", zorder=2)
PH = dict(color=S.CRIMSON, ls=(0, (3.2, 1.8)), lw=2.8, zorder=4)
GEONUM = dict(color=S.ROYALBLUE, ls=(0, (3.2, 1.8)), lw=2.8, zorder=4)
ONE = dict(color=S.PURPLE, ls=(0, (1.0, 1.6)), lw=2.8, zorder=5)          # one-mode reduction (analytic approximation)
PHX = dict(color=S.CRIMSON, ls=(0, (1.0, 1.6)), lw=2.0, zorder=4)         # numerical component along |d>
TARGET = {"gkp": lambda: states.gkp(0.4, 0, D), "cat": lambda: states.cat(2.0, D)}


def full_real(v):
    v = np.asarray(v)
    assert np.max(np.abs(np.imag(v))) < 1e-10
    v = np.real(v).copy()
    return v * np.sign(v[0])


def dot(ax, t, n, color):
    ax.plot(t, n, "o", ms=10, color=color, mec="white", mew=1.4, zorder=6, clip_on=False)


def departures(name):
    """departure angle from span{|0>, |chi>} against the position s/theta0 along the brachistochrone"""
    P = paths.Prep(TARGET[name]())
    d = (P.nv - P.n_chi) * P.chi / P.sig
    pos = lambda X: np.arctan2(X @ P.chi, X @ P.u0) / P.th          # noqa: E731
    dev = lambda X: np.arcsin(np.sqrt(np.clip(1 - (X @ P.u0) ** 2 - (X @ P.chi) ** 2, 0, 1)))   # noqa: E731
    U, J = AU.photon_prep(name)                     # numerical optimum (cached, 30 nodes, best of 4 starts)
    Xn, _ = pc.dense(U, 41)
    num = dict(s=pos(Xn), total=dev(Xn), d=np.abs(np.arcsin(np.clip(Xn @ d, -1, 1))), J=J)
    Xm = CF.modal_leg(P)                            # Eq. (closed): first order in all transverse directions
    cl = dict(s=pos(Xm), total=dev(Xm), J=pc.exposure(Xm, P.N))
    Xv, an = AU.first_order_vector(P, True)         # check: finite-difference solution of the same equations
    chk = dict(dmax=float(np.max(np.abs(np.interp(pos(Xv), cl["s"], cl["total"]) - dev(Xv)))), max=float(an.max()),
               J=pc.exposure(pc.resample(Xv, 60), P.N))
    cf = CF.closed_form(full_real(TARGET[name]()))  # Eq. (closed1): one mode, along |d> only
    one = dict(s=np.array(cf["s"]) / cf["theta0"], a=np.abs(np.array(cf["a"])), J=cf["J"], beta=cf["beta"])
    return P, num, cl, one, chk


fig, axs = plt.subplots(1, 3, figsize=(15.6, 5.9))
fig.subplots_adjust(left=0.065, right=0.995, top=0.78 if SHOW_D else 0.84, bottom=0.16, wspace=0.30)

# (a) Fock targets: brachistochrone, closed form and numerical optimum coincide
ax = axs[0]
fock = {}
for name in ("fock2", "fock4"):
    r = P2[name]
    n0 = int(name[4:])
    P = paths.Prep(states.fock(n0, D))
    c = CF.curve(CF.modal_leg(P), P.nv, P.uT)
    fock[name] = c
    ax.plot(c["t_ns"], c["nbar"], **AN)
    ax.plot(r["geodesic"]["t_ns"], r["geodesic"]["nbar"], **GEO)
    gn = CF.numeric_geodesic(P.u0, P.uT, P.nv)                # time-optimal path minimized numerically
    ax.plot(gn["t_ns"], gn["nbar"], **GEONUM)
    ax.plot(r["photon"]["t_ns"], r["photon"]["nbar"], **PH)
    dot(ax, r["geodesic"]["t_ns"][-1], r["geodesic"]["nbar"][-1], S.ROYALBLUE)
    dot(ax, r["photon"]["t_ns"][-1], r["photon"]["nbar"][-1], S.CRIMSON)
    ax.text(r["geodesic"]["t_ns"][-1] - 6, n0 + 0.22, rf"$|{n0}\rangle$", ha="right", va="bottom", fontsize=FS["ann"])
dev_fock = max(float(np.max(np.abs(np.interp(P2[n]["geodesic"]["t_ns"], P2[n]["photon"]["t_ns"], P2[n]["photon"]["nbar"])
                                   - np.array(P2[n]["geodesic"]["nbar"])))) for n in ("fock2", "fock4"))
dev_fock_cl = max(float(np.max(np.abs(np.interp(P2[n]["geodesic"]["t_ns"], fock[n]["t_ns"], fock[n]["nbar"])
                                      - np.array(P2[n]["geodesic"]["nbar"])))) for n in ("fock2", "fock4"))
ax.text(0.06, 0.97, r"$\bar n=n\sin^2\omega t$" + "\n" + r"$J=n\pi/4$", transform=ax.transAxes, va="top",
        ha="left", fontsize=FS["small"], linespacing=1.3)
ax.set_xlim(0, 310)
ax.set_ylim(0, 5.0)
ax.set_xlabel(r"time $t$ (ns)")
ax.set_ylabel(r"$\bar n$")
ax.set_title(r"Fock $|0\rangle\to|n\rangle$", fontsize=FS["title"])
S.panel_label(ax, "a", x=-0.21, y=1.04)

# (b, c) departure from the plane of the brachistochrone
info = {}
for j, (name, title) in enumerate((("gkp", r"GKP $|0_L\rangle$, $\Delta=0.4$"), ("cat", r"cat $|C_2^+\rangle$"))):
    ax = axs[j + 1]
    P, num, cl, one, chk = departures(name)
    info[name] = (P, num, cl, one, chk)
    o = np.argsort(num["s"])
    ax.axhline(0, **GEO)
    gn = CF.numeric_geodesic(P.u0, P.uT, P.nv)                # numerical brachistochrone: its departure from the plane
    ax.plot([0, 1], [gn["max_dev_rad"]] * 2, **GEONUM)      # is at most ~1e-5 rad, drawn at that value
    print(f"{name}: numerical brachistochrone departs from the plane by at most {gn['max_dev_rad']:.1e} rad")
    ax.plot(cl["s"], cl["total"], **AN)
    ax.plot(num["s"][o], num["total"][o], **PH)
    if SHOW_D:
        ax.plot(num["s"][o], num["d"][o], **PHX)
    ax.plot(one["s"], one["a"], **ONE)
    top = 1.22 * max(num["total"].max(), cl["total"].max(), one["a"].max())
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.008 * top / 0.15, top)
    ax.set_xlabel(r"path fraction $s=\vartheta/\theta$")
    ax.set_title(title, fontsize=FS["title"])
    S.panel_label(ax, "bc"[j], x=-0.21, y=1.04)
axs[1].set_ylabel(r"departure (rad)")

# one entry per path, as in main Fig. 2 (band: analytic, dashed: numerical; the caption says so)
hs = [(Line2D([], [], **GEO), Line2D([], [], **GEONUM)), (Line2D([], [], **AN), Line2D([], [], **PH)),
      Line2D([], [], **ONE)]
labs = ["time optimal", "photon optimal", r"one mode, along $|d\rangle$ (b, c)"]
if SHOW_D:
    hs.append(Line2D([], [], **PHX))
    labs.append(r"numerical, along $|d\rangle$ (b, c)")
fig.legend(hs, labs, loc="upper center", ncol=2 if SHOW_D else 3, fontsize=FS["legend"] - 1, frameon=False,
           bbox_to_anchor=(0.53, 1.0), columnspacing=2.0, handlelength=2.6, borderpad=0.05, borderaxespad=0.0,
           handler_map={tuple: HandlerTuple(ndivide=None)})
S.save(fig, HERE / "S02_photon_path_analytic")
print(f"Fock: largest |<n>_photon - <n>_geo| = {dev_fock:.2e}, |<n>_closed - <n>_geo| = {dev_fock_cl:.2e}")
for name, (P, num, cl, one, chk) in info.items():
    Jf = R["prep"][name]["photon"]["J"]
    print(f"{name}: n_chi={P.n_chi:.3f} sigma={P.sig:.3f} eta={one['beta']:.3f}  numerical J={num['J']:.5f} "
          f"(fig2.json {Jf:.5f}) B={R['prep'][name]['B']:.5f} (optimum {100 * (num['J'] / R['prep'][name]['B'] - 1):.2f}% above)")
    print(f"   max departure: numerical total {num['total'].max():.3f}, along d {num['d'].max():.3f}; closed form "
          f"(modal) {cl['total'].max():.3f}; one-mode {one['a'].max():.3f} rad")
    print(f"   exposure excess over the numerical optimum: closed form {100 * (cl['J'] / num['J'] - 1):.3f}%, "
          f"one-mode {100 * (one['J'] / num['J'] - 1):.2f}%")
    print(f"   check, finite-difference first order: max departure {chk['max']:.3f}, largest difference from modal_leg "
          f"{chk['dmax']:.1e} rad, excess {100 * (chk['J'] / num['J'] - 1):.3f}%")
print("wrote", HERE / "S02_photon_path_analytic.png")
