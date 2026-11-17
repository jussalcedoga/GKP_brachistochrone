"""Fig. 2: time-optimal against photon-optimal paths of GKP preparation and of the logical flip, analytic and numerical,
all traversed at the speed limit Delta H = omega = g/sqrt2.  Three candidates (one column wide):
  A  fig2_A  (a, b) photon number <n>(t) along the paths; (c, d) cumulative photon exposure int<n>dt, with the lower
             bound on the total exposure of any path at the speed limit
  B  fig2_B  A plus (e, f) the target fidelity F_T(t) = |<T|psi(t)>|^2 along each path
  C  fig2_C  (a, b) <n> against the target fidelity F_T: below m(F) (shaded) no state with that fidelity exists, so every
             path runs above it and the photon-optimal one hugs it; (c, d) cumulative exposure with the bound
The chosen candidate is also written as fig2.(pdf|png).  Curves: brachistochrone (analytic), photon-optimal path
(numerical optimum, photon_core), its analytic counterpart (round 4: the modal closed form, the exact minimizer of the second-order functional; flip:
the modal closed form on each of the two legs through the vacuum, compute_fig2.flip_modal).  Inputs: manuscript/fig2_photon_optimal/results/fig2.json (read only) and
results/fig2_extra.json (compute_fig2.py).   python figures/Fig2/plot_fig2.py [--choose C]"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))
import mfcommon as M  # noqa: E402,I001

# Nov 16 (author): same style as Fig. 1 -- drawn at the same 14 in width with the same 1.25x font sizes, line widths, boxed
# axes with outward ticks and panel-letter/title placement, so both figures print identically at the column width
M.FS.update({k_: round(1.25 * v_) for k_, v_ in M.FS.items()})
S = M.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

FS = M.FS
CHOOSE = sys.argv[sys.argv.index("--choose") + 1] if "--choose" in sys.argv else "B"
R = json.loads((M.MS / "fig2_photon_optimal" / "results" / "fig2.json").read_text())
X = json.loads((HERE / "results" / "fig2_extra.json").read_text())
OM = R["params"]["omega"]
G, F = R["prep"]["gkp"], R["flip"]
CF = X["modal_gkp"]
# Nov 16 (author): the path styles of Fig. 1 (time optimal solid blue, photon optimal dashed crimson); the analytic paths
# are the lines, the numerical optima are open circles of the same colour sampled along them (no translucent bands)
GEO = dict(color=S.ROYALBLUE, lw=6.0, ls="-", solid_capstyle="round", zorder=2)
AN = dict(color=S.CRIMSON, lw=5.4, ls=(0, (3.5, 1.4)), zorder=2)
NUMK = dict(ls="none", marker="o", ms=15, mfc="white", mew=3.0, zorder=4, clip_on=False)
GEONUM = dict(NUMK, color=S.ROYALBLUE)
PH = dict(NUMK, color=S.CRIMSON)
BOUND = dict(color=S.INK, lw=3.0, ls=(0, (1.0, 2.0)))
FORBID = "#E5E7EB"
N_MARK = 8                                  # numerical samples per path (interior points, equally spaced in time)
# (analytic, numerical) per path
PATHS = {"gkp": [((G["geodesic"], GEO), (X["geo_num_gkp"], GEONUM)), ((CF, AN), (G["photon"], PH))],
         "flip": [((F["geodesic"], GEO), (X["geo_num_flip"], GEONUM)), ((X["flip_modal"], AN), (F["photon"], PH))]}
TITLE = {"gkp": "Preparation", "flip": "Logical Flip"}          # Nov 16: the panel titles of Fig. 1
BNS = {"gkp": X["gkp_bound_ns"], "flip": X["flip_bound"]["B2_ns"]}
TMAX = {"gkp": 156, "flip": 410}
NBAR_MAX = 3.2                              # Nov 16: (a) and (b) share the n axis


def frame(ax):
    """the boxed axes of the Wigner panels of Fig. 1"""
    for sp in ax.spines.values():
        sp.set_visible(True)
    ax.tick_params(direction="out", length=8)


def draw(ax, key, field):
    """analytic line, numerical optimum as open circles (offset by half a spacing between the two paths so they never
    sit on each other), and a filled dot at the end of each path (the paths have different durations)"""
    for j, ((da, sa), (dn, sn)) in enumerate(PATHS[key]):
        ax.plot(da["t_ns"], da[field], **sa)
        t = np.asarray(dn["t_ns"])
        tk = t[-1] * (np.arange(N_MARK) + 0.5 + 0.5 * j) / (N_MARK + 1)
        ax.plot(tk, np.interp(tk, t, dn[field]), **sn)
        ax.plot(da["t_ns"][-1], da[field][-1], "o", ms=17, color=sa["color"], mec="white", mew=2.0, zorder=6,
                clip_on=False)


def label(ax, letter, key=None):
    """panel letter at the top left; on the top row the short title centred over the panel (Nov 16, author)"""
    ax.text(-0.29, 1.02, f"({letter})", transform=ax.transAxes, fontsize=FS["panel"], ha="left", va="bottom")
    if key is not None:
        ax.text(0.5, 1.03, TITLE[key], transform=ax.transAxes, fontsize=FS["title"] - 1, ha="center", va="bottom")


def n_of_t(ax, key, letter, title=True):
    draw(ax, key, "nbar")
    if key == "flip":
        i = int(np.argmin(F["photon"]["nbar"]))
        ax.annotate("vacuum", xy=(F["photon"]["t_ns"][i], 0.08), xytext=(F["photon"]["t_ns"][i], 1.25),
                    fontsize=FS["small"], ha="center", va="bottom", arrowprops=dict(arrowstyle="->", lw=2.8, color=S.INK))
    ax.set_xlim(0, TMAX[key])
    ax.set_ylim(0, NBAR_MAX)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel(r"$\bar n$")
    frame(ax)
    label(ax, letter, key if title else None)


def exposure(ax, key, letter):
    draw(ax, key, "expo")
    B = BNS[key]
    ax.axhline(B, **BOUND, zorder=1)
    # Nov 16 (author): "lower bound" in both panels; free space above the line on the left (GKP), below it mid-panel (flip)
    if key == "gkp":
        ax.text(0.03 * TMAX[key], B * 1.04, "lower bound", fontsize=FS["small"] - 1, color=S.INK, va="bottom",
                ha="left", style="italic")
    else:
        ax.text(0.58 * TMAX[key], B * 0.95, "lower bound", fontsize=FS["small"] - 1, color=S.INK, va="top",
                ha="center", style="italic")
    ax.set_xlim(0, TMAX[key])
    ax.set_ylim(0, 1.18 * max(max(d[0]["expo"]) for pair in PATHS[key] for d in pair))
    ax.set_ylabel("exposure (ns)")                 # Nov 16 (author): compact; "photon exposure int_0^t n dt'" in the caption           # Nov 16 (author); the integral int_0^t n dt' is in the caption
    frame(ax)
    label(ax, letter)


def fid_of_t(ax, key, letter):
    draw(ax, key, "F_T")
    ax.set_xlim(0, TMAX[key])
    ax.set_ylim(0, 1.05)                           # Nov 16 (author): (e) and (f) on the same 0-1 axis (headroom for the end dots)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("target fidelity")               # Nov 16 (author): written out (F_T = |<T|psi(t)>|^2 in the caption)
    frame(ax)
    label(ax, letter)


def n_of_F(ax, key, letter, title=False):
    """n against the target fidelity: below m(F) (shaded) no state with that fidelity exists"""
    m = X["m_0L"] if key == "gkp" else X["m_1L"]
    Fm, mm = np.array(m["F"]), np.array(m["m"])
    ax.fill_between(Fm, 0, mm, color=FORBID, lw=0, zorder=0)
    ax.plot(Fm, mm, **BOUND, zorder=1)
    for j, ((da, sa), (dn, sn)) in enumerate(PATHS[key]):
        ax.plot(da["F_T"], da["nbar"], **sa)
        t = np.asarray(dn["t_ns"])
        tk = t[-1] * (np.arange(N_MARK) + 0.5 + 0.5 * j) / (N_MARK + 1)
        ax.plot(np.interp(tk, t, dn["F_T"]), np.interp(tk, t, dn["nbar"]), **sn)
        ax.plot(da["F_T"][-1], da["nbar"][-1], "o", ms=17, color=sa["color"], mec="white", mew=2.0, zorder=6, clip_on=False)
    if key == "flip":                              # GKP: the excluded region is a thin sliver under the curves (caption)
        ax.text(0.96, 0.05, "no state", transform=ax.transAxes, fontsize=FS["small"] - 1, color=S.INK, ha="right",
                va="bottom", style="italic")
    ax.set_xlim(0.5 if key == "gkp" else 0.0, 1.0)     # preparation starts at |<0|0_L>|^2 = 0.54
    ax.set_ylim(0, NBAR_MAX)
    ax.set_xticks([0.5, 0.75, 1.0] if key == "gkp" else [0, 0.5, 1.0])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel(r"target fidelity $\mathcal{F}_T$")
    ax.set_ylabel(r"$\bar n$")
    frame(ax)
    label(ax, letter, key if title else None)


def legend(fig, y):
    """paths by colour and line style (as in Fig. 1)"""
    # Nov 16 (author): no "lower bound" entry -- the bound is labelled in (c, d)
    hs = [Line2D([], [], **GEO), Line2D([], [], **AN)]
    lg = fig.legend(hs, ["time optimal", "photon optimal"], loc="upper center", ncol=2,
                    bbox_to_anchor=(0.5, y), fontsize=FS["legend"] + 1, columnspacing=1.0, handlelength=2.0, frameon=False)
    return lg       # Nov 16 (author): lines = analytic, circles = numerical optimum, explained in the caption


def make(cand):
    rows = 3 if cand in ("B", "D") else 2
    # Nov 16 (author, PRL one column): shorter panels, less white space; rows that share the time axis of the row below
    # carry no time tick labels
    H = 4.35 * rows + (1.75 if cand != "D" else 2.4)
    fig, axs = plt.subplots(rows, 2, figsize=(14.0, H))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=1.2 / H, top=1 - 1.55 / H, hspace=0.30 if cand != "D" else 0.48, wspace=0.36)
    if cand in ("A", "B", "D"):
        n_of_t(axs[0, 0], "gkp", "a")
        n_of_t(axs[0, 1], "flip", "b")
    else:
        n_of_F(axs[0, 0], "gkp", "a")
        n_of_F(axs[0, 1], "flip", "b")
    exposure(axs[1, 0], "gkp", "c")
    exposure(axs[1, 1], "flip", "d")
    if cand == "B":
        fid_of_t(axs[2, 0], "gkp", "e")
        fid_of_t(axs[2, 1], "flip", "f")
    if cand == "D":                                # Nov 16: proposed bottom row, n against F_T with the bound m(F)
        n_of_F(axs[2, 0], "gkp", "e")
        n_of_F(axs[2, 1], "flip", "f")
    time_row = rows - 2 if cand == "D" else rows - 1   # Nov 16 (author): time label (and ticks) on the last time row only
    for ax in axs[time_row]:
        ax.set_xlabel(r"time $t$ (ns)")
    for r in range(time_row):
        for ax in axs[r]:
            ax.tick_params(labelbottom=False)
    legend(fig, 1.0)
    (HERE / "candidates").mkdir(exist_ok=True)
    S.save(fig, HERE / "candidates" / f"fig2_{cand}")
    if cand == CHOOSE:
        fig2 = HERE / "fig2"
        for ext in ("png", "pdf"):
            (fig2.with_suffix("." + ext)).write_bytes((HERE / "candidates" / f"fig2_{cand}.{ext}").read_bytes())
    print("wrote", HERE / "candidates" / f"fig2_{cand}.png")


for c in ("A", "B", "D"):
    make(c)
print(f"chosen: {CHOOSE}; T (ns): GKP {G['geodesic']['T_ns']:.1f}/{G['photon']['T_ns']:.1f}/{CF['T_ns']:.1f}, flip "
      f"{F['geodesic']['T_ns']:.1f}/{F['photon']['T_ns']:.1f}/{X['flip_modal']['T_ns']:.1f}; "
      f"exposure (photon ns): GKP {G['geodesic']['J_ns']:.1f}/{G['photon']['J_ns']:.1f}/{CF['J_ns']:.1f} bound {BNS['gkp']:.1f}; "
      f"flip {F['geodesic']['J_ns']:.1f}/{F['photon']['J_ns']:.1f} bound {BNS['flip']:.1f}")
