"""Fig. S5, first-order error budget along the family of paths that minimize int(1 + w<n>) ds, from the brachistochrone
(w = 0) to the photon-optimal path (w -> infinity), for GKP preparation and the transfer |0_L> -> |1_L>.
Complete first-order error (analytics audit C10):
    eps = mu T + kappa int<n>dt + kappa_phi int Var(n) dt,
mu = 1/(2 T_phi,q) + 1/(4 T_1,q) (ancilla in an equatorial superposition), kappa = 1/T_1c, kappa_phi = 2/T_phi,c.
Measured rates (device of Sivak et al., "Yale"): all three terms; proposal rates (Del Grosso et al., "CSproposal"): no cavity
dephasing is specified, eps has two terms.  The member w* = kappa/mu minimizes the two-term error; it is marked on the
complete eps (filled star) and on the two-term eps (open star).  Arrowheads: direction of increasing w (T = L/omega).
Family path states: the cached family of the audit (manuscript_figures/analytics_audit/cache/family_{gkp,flip}.pkl,
w = logspace(-2, 3, 21) and w* = kappa/mu (Yale), from manuscript/common/paths.tradeoff_family) and the cached
photon-optimal paths; Var(n) from manuscript/common/paths.profile.  The proposal-rate panels use the stored family of
manuscript/fig3_time_vs_photon/results/fig3.json (26 weights, including w* = 0.098).
python figures/SFig04_error_budget/plot.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import pickle
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

import mfcommon as M  # noqa: E402

S = M.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from common import paths, photon_core as pc  # noqa: E402
from common.params import GKP_DELTA, NOISE, OMEGA_REF, kappa_mu  # noqa: E402
from common import states  # noqa: E402

FS = S.FS
W = OMEGA_REF
NS = 1e3 / W                                        # rad -> ns; rad x photons -> photon ns
CACHE = M.MF / "analytics_audit" / "cache"
R3 = json.loads((M.MS / "fig3_time_vs_photon" / "results" / "fig3.json").read_text())
kY, mY = kappa_mu("Yale")
kC, mC = kappa_mu("CSproposal")
KPHI = 2 / NOISE["Yale"]["Tphic"]                   # 1/us
DEPH = S.INK                                        # cavity-dephasing term (no gray, teal, green, orange or yellow)
DEPH_LS = (0, (6, 2, 1.5, 2))
MEAS, PROP = "Sivak et al. rates", "CS-proposal rates"   # Nov 9 (advisor): name the sources (panel titles, short)
MEAS_KEY, PROP_KEY = "rates of Sivak et al.", "rates of the CS proposal of Del Grosso et al."   # the key, in full
FLIP = r"$|0_L\rangle\to|1_L\rangle$"
PREP = r"$|0\rangle\to|0_L\rangle$"
trapz = getattr(np, "trapezoid", None) or np.trapz
D = 80


def load(name):
    return pickle.loads((CACHE / f"{name}.pkl").read_bytes())


def integrals(U, N):
    pr = paths.profile(U, N)
    return float(pr["s"][-1]), float(trapz(pr["nbar"], pr["s"])), float(trapz(pr["var"], pr["s"]))   # L, J, int Var ds


def yale_family(key):
    """rows (w, T ns, E photon ns, V photon^2 ns, eps terms) from the brachistochrone through the family to the photon path"""
    if key == "flip":
        LG = pc.Logical(GKP_DELTA, D)
        N, geo = np.diag(LG.nv), LG.geo(np.linspace(0, LG.th, 32))
    else:
        P = paths.Prep(states.gkp(GKP_DELTA, 0, D))
        N, geo = P.N, P.geo(np.linspace(0, P.th, 32))
    ws = np.r_[np.logspace(-2, 3, 21), kY / mY]
    fam = load(f"family_{key}")
    assert len(fam) == len(ws), (len(fam), len(ws))
    members = [(0.0, geo)] + sorted(zip(ws, fam), key=lambda x: x[0]) + [(np.inf, load(f"photon_{key}")[0])]
    rows = []
    for w, U in members:
        L, J, V = integrals(U, N)
        rows.append(dict(w=float(w), T=NS * L, E=NS * J, V=NS * V, time=mY * L / W, photon=kY * J / W,
                         deph=KPHI * V / W))
    for r in rows:
        r["eps2"] = r["time"] + r["photon"]
        r["eps3"] = r["eps2"] + r["deph"]
    return rows


def cs_family(key):
    r = R3["tradeoff"][key]
    T = NS * np.r_[r["theta0"], r["L"], r["L_photon"]]
    E = NS * np.r_[r["J_geo"], r["J"], r["J_photon"]]
    o = np.argsort(T, kind="stable")
    return r, T[o], E[o]


YF = {k: yale_family(k) for k in ("gkp", "flip")}
fig = plt.figure(figsize=(15.0, 12.2))
gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.42, top=0.82, bottom=0.07, left=0.075, right=0.99)
LET = iter("abcdef")
STY = {"brach": dict(marker="o", ms=17, color=S.ROYALBLUE, mec="k", mew=1.1),
       "photon": dict(marker="o", ms=15, mfc="white", mec=S.CRIMSON, mew=2.6),
       "wstar": dict(marker="*", ms=26, color=S.TRADEOFF, mec="k", mew=1.2),
       "wstar2": dict(marker="*", ms=24, mfc="white", mec=S.TRADEOFF, mew=2.0),
       "wstarC": dict(marker="P", ms=14, color=S.TRADEOFF, mec="k", mew=1.1),
       "best": dict(marker="D", ms=14, color=S.INK, mec="white", mew=1.4)}
ZO = {"brach": 8, "photon": 8, "wstar": 9, "wstar2": 9, "wstarC": 10, "best": 9}   # the w* plus sits inside the circle


def mark(ax, x, y, kind):
    ax.plot(x, y, ls="none", zorder=ZO[kind], clip_on=False, **STY[kind])


def warrow(ax, x, y, frac=0.5, color=S.TRADEOFF):
    """arrowhead on the curve (x, y) at the fraction `frac` of its drawn length, pointing toward increasing w (the order
    of the points); call after the axis limits are set"""
    P = ax.transData.transform(np.c_[x, y])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    t = frac * d[-1]
    inv = ax.transData.inverted()
    p1 = inv.transform([np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])])
    p0 = inv.transform([np.interp(t - 14, d, P[:, 0]), np.interp(t - 14, d, P[:, 1])])
    ax.annotate("", xy=p1, xytext=p0, zorder=7,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=34, shrinkA=0, shrinkB=0))


# (a, b) measured rates: complete three-term budget
for col, key in enumerate(("gkp", "flip")):
    ax = fig.add_subplot(gs[0, col])
    rows = YF[key]
    T = np.array([r["T"] for r in rows])
    t1, t2, t3 = (100 * np.array([r[k] for r in rows]) for k in ("time", "photon", "deph"))
    e2, e3 = t1 + t2, t1 + t2 + t3
    ax.plot(T, t1, color=S.ROYALBLUE, lw=3.4, ls=(0, (5, 2)), zorder=3)
    ax.plot(T, t2, color=S.CRIMSON, lw=3.4, ls=(0, (5, 2)), zorder=3)
    ax.plot(T, t3, color=DEPH, lw=3.0, ls=DEPH_LS, zorder=3)
    ax.plot(T, e2, color=S.TRADEOFF, lw=2.6, ls=(0, (2, 1.5)), zorder=4)
    ax.plot(T, e3, color=S.TRADEOFF, lw=M.LW + 0.4, zorder=5)
    iw = next(i for i, r in enumerate(rows) if abs(r["w"] - kY / mY) < 1e-12)
    ib = int(np.argmin(e3))
    mark(ax, T[0], e3[0], "brach")
    mark(ax, T[-1], e3[-1], "photon")
    mark(ax, T[iw], e3[iw], "wstar")
    mark(ax, T[iw], e2[iw], "wstar2")
    mark(ax, T[ib], e3[ib], "best")
    span = T[-1] - T[0]
    ax.set_xlim(T[0] - 0.07 * span, T[-1] + 0.07 * span)
    ax.set_ylim(0, 1.25 * e3.max())
    warrow(ax, T, e3, 0.62 if key == "gkp" else 0.55)
    ax.set_xlabel(r"duration $T$ (ns)")
    if col == 0:
        ax.set_ylabel(r"first-order error (\%)")
    ax.set_title(f"{PREP if key == 'gkp' else FLIP}, {MEAS}", fontsize=FS["ann"], pad=8)
    S.panel_label(ax, next(LET), x=-0.30, y=1.04)

# (c) measured rates: complete and two-term eps relative to the brachistochrone, against the weight w
ax = fig.add_subplot(gs[0, 2])
XINF = 1e4
for key, col in (("flip", S.RASPBERRY), ("gkp", S.NAVY)):
    rows = YF[key]
    w = np.array([r["w"] for r in rows[1:-1]])
    e2 = np.array([r["eps2"] for r in rows]) / rows[0]["eps2"]
    e3 = np.array([r["eps3"] for r in rows]) / rows[0]["eps3"]
    ax.plot(w, e3[1:-1], color=col, lw=M.LW - 0.4, zorder=4)
    ax.plot(w, e2[1:-1], color=col, lw=2.6, ls=(0, (2, 1.5)), zorder=4)
    ax.plot([w[-1], XINF], [e3[-2], e3[-1]], color=col, lw=1.6, ls=(0, (1, 2)), zorder=3)
    ax.plot([w[-1], XINF], [e2[-2], e2[-1]], color=col, lw=1.6, ls=(0, (1, 2)), zorder=3)
    mark(ax, XINF, e3[-1], "photon")
    iw = next(i for i, r in enumerate(rows) if abs(r["w"] - kY / mY) < 1e-12)
    ib = int(np.argmin(e3))
    mark(ax, rows[iw]["w"], e3[iw], "wstar")
    mark(ax, rows[iw]["w"], e2[iw], "wstar2")
    mark(ax, rows[ib]["w"] if np.isfinite(rows[ib]["w"]) else XINF, e3[ib], "best")
    if key == "flip":                                 # shading: the flip only
        flat = [r["w"] for r, e in zip(rows[1:-1], e3[1:-1]) if e <= 1.01 * e3.min()]
        ax.axvspan(min(flat), XINF, color=col, alpha=0.08, lw=0, zorder=0)
ax.axhline(1.0, color=S.INK, lw=1.4, ls=(0, (1, 2)))
ax.set_xscale("log")
ax.set_xlim(1e-2, 1.6 * XINF)
ax.set_xticks([1e-2, 1e0, 1e2, XINF])
ax.set_xticklabels([r"$10^{-2}$", r"$1$", r"$10^{2}$", r"$\infty$"])
ax.minorticks_off()
ax.set_ylim(0.52, 1.08)
ax.text(1.3e-2, 1.07, "GKP", fontsize=FS["small"] - 2, color=S.NAVY, ha="left", va="top")
ax.text(60, 0.665, FLIP, fontsize=FS["small"] - 2, color=S.RASPBERRY, ha="center", va="center")
ax.set_xlabel(r"weight $w$")
ax.set_ylabel(r"$\epsilon_1/\epsilon_1^{\rm geo}$")
ax.set_title(rf"{MEAS}, against $w$", fontsize=FS["ann"], pad=8)
S.panel_label(ax, next(LET), x=-0.30, y=1.04)

# (d, e) proposal rates: two-term budget (no cavity dephasing in this parameter set)
for col, key in enumerate(("gkp", "flip")):
    ax = fig.add_subplot(gs[1, col])
    r, T, E = cs_family(key)
    t1, t2 = 100 * mC * T * 1e-3, 100 * kC * E * 1e-3
    tot = t1 + t2
    ax.plot(T, t1, color=S.ROYALBLUE, lw=3.4, ls=(0, (5, 2)), zorder=3)
    ax.plot(T, t2, color=S.CRIMSON, lw=3.4, ls=(0, (5, 2)), zorder=3)
    ax.plot(T, tot, color=S.TRADEOFF, lw=M.LW + 0.4, zorder=5)
    a = r["eps"]["CSproposal"]["at_w_star"]
    mark(ax, T[0], tot[0], "brach")
    mark(ax, T[-1], tot[-1], "photon")
    mark(ax, NS * a["L"], 100 * a["eps"], "wstarC")
    span = T[-1] - T[0]
    ax.set_xlim(T[0] - 0.07 * span, T[-1] + 0.07 * span)
    ax.set_ylim(0, 1.30 * tot.max())
    warrow(ax, T, tot, 0.55)
    ax.set_xlabel(r"duration $T$ (ns)")
    if col == 0:
        ax.set_ylabel(r"first-order error (\%)")
    ax.set_title(f"{PREP if key == 'gkp' else FLIP}, {PROP}", fontsize=FS["ann"], pad=8)
    S.panel_label(ax, next(LET), x=-0.30, y=1.04)

# (f) proposal rates: the families in the duration-exposure plane with lines of constant eps (two terms = complete)
ax = fig.add_subplot(gs[1, 2])
ax.set_xlim(100, 420)
ax.set_ylim(0, 900)
for key, col in (("flip", S.RASPBERRY), ("gkp", S.NAVY)):
    r, T, E = cs_family(key)
    ax.plot(T, E, "-", color=col, lw=M.LW, zorder=4)
    mark(ax, T[0], E[0], "brach")
    mark(ax, T[-1], E[-1], "photon")
    a = r["eps"]["CSproposal"]["at_w_star"]
    mark(ax, NS * a["L"], NS * a["J"], "wstarC")
    if key == "flip":
        warrow(ax, T, E, 0.62, color=col)
ren = fig.canvas.get_renderer()
EMAX = 790                                          # the lines stop below the label of the transfer
for eps, El in ((0.8e-2, 360), (1.2e-2, 360), (1.6e-2, 360), (2.0e-2, 180)):   # E at the inline label
    def T_of(E, eps=eps):
        return (eps - kC * np.asarray(E) * 1e-3) / mC * 1e3
    p = ax.transData.transform([[T_of(El - 50), El - 50], [T_of(El + 50), El + 50]])
    ang = np.degrees(np.arctan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0]))
    ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
    t = ax.text(T_of(El), El, rf"${eps * 1e2:.1f}\%$", rotation=ang, rotation_mode="anchor", ha="center", va="center",
                fontsize=FS["small"] - 2, color=S.INK, zorder=6)
    half = t.get_window_extent(ren).height / 2 + 16                      # display px along y, plus a margin
    dE = half / abs(np.diff(ax.transData.transform([[0, 0], [0, 1]])[:, 1])[0])
    for E0, E1 in ((0, El - dE), (El + dE, EMAX)):
        Eg = np.linspace(E0, E1, 50)
        ax.plot(T_of(Eg), Eg, color=S.INK, lw=1.6, ls=(0, (5, 3)), zorder=1)
ax.text(190, 92, "GKP", fontsize=FS["small"] - 2, color=S.NAVY, ha="center", va="center")   # between the 0.8, 1.2% lines
ax.text(294, 812, FLIP, fontsize=FS["small"] - 2, color=S.RASPBERRY, ha="center", va="bottom")
ax.set_xlabel(r"duration $T$ (ns)")
ax.set_ylabel(r"$\int\bar n\,dt$ (photon ns)")
ax.set_title(rf"constant $\epsilon_1$, {PROP}", fontsize=FS["ann"], pad=8)
S.panel_label(ax, next(LET), x=-0.30, y=1.04)

L2 = lambda **k: Line2D([], [], lw=0, **k)  # noqa: E731
hs = [Line2D([], [], color=S.ROYALBLUE, lw=3.4, ls=(0, (5, 2))), Line2D([], [], color=S.CRIMSON, lw=3.4, ls=(0, (5, 2))),
      Line2D([], [], color=DEPH, lw=3.0, ls=DEPH_LS), Line2D([], [], color=S.TRADEOFF, lw=M.LW),
      Line2D([], [], color=S.TRADEOFF, lw=2.6, ls=(0, (2, 1.5))),
      L2(**dict(STY["brach"], ms=15)), L2(**STY["photon"]), L2(marker=">", ms=15, color=S.TRADEOFF),
      (L2(**dict(STY["wstar"], ms=22)), L2(**dict(STY["wstar2"], ms=20))),
      L2(**dict(STY["wstarC"], ms=15)), L2(**dict(STY["best"], ms=13))]
fig.legend(hs, [r"time $\Gamma_qT$", r"photons $\kappa\!\int\!\bar n\,dt$",
                r"cavity dephasing $\kappa_\phi\!\int\!{\rm Var}(\hat n)\,dt$", r"complete $\epsilon_1$",
                r"$\epsilon_1$ without cavity dephasing", r"brachistochrone ($w=0$)", r"photon optimum ($w\to\infty$)",
                r"increasing $w$", rf"$w^*=\kappa/\Gamma_q={kY / mY:.2f}$ ({MEAS_KEY})",
                rf"$w^*=\kappa/\Gamma_q={kC / mC:.2f}$ ({PROP_KEY})", r"minimum of the complete $\epsilon_1$"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
           loc="upper center", ncol=3, fontsize=FS["small"] - 1, bbox_to_anchor=(0.50, 1.0), columnspacing=1.6,
           handlelength=1.8, handletextpad=0.5, borderpad=0.05, borderaxespad=0.0, labelspacing=0.3)
S.save(fig, HERE / "S05_error_budget")

# numbers quoted by the caption, and consistency checks
print(f"kappa_phi = {KPHI:.3e}/us; Yale 1/kappa = {1 / kY:.0f} us, 1/mu = {1 / mY:.1f} us, w* = {kY / mY:.3f}; "
      f"CS proposal 1/kappa = {1 / kC:.0f} us, 1/mu = {1 / mC:.1f} us, w* = {kC / mC:.4f}")
for key in ("gkp", "flip"):
    rows = YF[key]
    g, p = rows[0], rows[-1]
    iw = next(i for i, r in enumerate(rows) if abs(r["w"] - kY / mY) < 1e-12)
    ib = int(np.argmin([r["eps3"] for r in rows]))
    flat = [r["w"] for r in rows[1:-1] if r["eps3"] <= 1.01 * min(x["eps3"] for x in rows)]
    print(f"{key} (Yale): brach T={g['T']:.0f} ns eps2={1e3 * g['eps2']:.3f} eps3={1e3 * g['eps3']:.3f} (deph share "
          f"{100 * g['deph'] / g['eps2']:.0f}%); w*: T={rows[iw]['T']:.0f} eps2={1e3 * rows[iw]['eps2']:.3f} "
          f"({100 * (1 - rows[iw]['eps2'] / g['eps2']):.1f}% below) eps3={1e3 * rows[iw]['eps3']:.3f} "
          f"({100 * (1 - rows[iw]['eps3'] / g['eps3']):.1f}% below); photon: T={p['T']:.0f} eps2={1e3 * p['eps2']:.3f} "
          f"({100 * (p['eps2'] / g['eps2'] - 1):+.1f}%) eps3={1e3 * p['eps3']:.3f} ({100 * (p['eps3'] / g['eps3'] - 1):+.1f}%, "
          f"deph share {100 * p['deph'] / p['eps2']:.0f}%); min eps3 at w={rows[ib]['w']:.2f} T={rows[ib]['T']:.0f} ns "
          f"eps3={1e3 * rows[ib]['eps3']:.3f} ({100 * (1 - rows[ib]['eps3'] / g['eps3']):.1f}% below); within 1% of the "
          f"minimum for w >= {min(flat) if flat else float('nan'):.2f}; min eps2 at w="
          f"{rows[int(np.argmin([r['eps2'] for r in rows]))]['w']:.3f}")
    # two-term eps of the audit family against the stored family of fig3.json (same paths, independent numerics)
    r3 = R3["tradeoff"][key]
    T3 = NS * np.r_[r3["theta0"], r3["L"], r3["L_photon"]]
    e3 = np.r_[r3["eps"]["Yale"]["eps_geo"], r3["eps"]["Yale"]["eps"], r3["eps"]["Yale"]["eps_photon"]]
    o = np.argsort(T3)
    dev = max(abs(np.interp(r["T"], T3[o], e3[o]) / r["eps2"] - 1) for r in rows)
    print(f"   two-term eps vs fig3.json family at equal T: max relative difference {100 * dev:.2f}%")
for key in ("gkp", "flip"):
    e = R3["tradeoff"][key]["eps"]["CSproposal"]
    T = cs_family(key)[1]
    print(f"{key} (CS proposal): eps_brach={100 * e['eps_geo']:.3f}% eps*={100 * e['at_w_star']['eps']:.3f}% "
          f"({100 * (1 - e['at_w_star']['eps'] / e['eps_geo']):.1f}% below, T={NS * e['at_w_star']['L']:.1f} ns vs "
          f"T_brach={T[0]:.1f} ns) eps_photon={100 * e['eps_photon']:.3f}% "
          f"({100 * (e['eps_photon'] / e['eps_geo'] - 1):+.1f}%)")
print("wrote", HERE / "S05_error_budget.png")
