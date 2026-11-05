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

S = M.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

FS = dict(M.FS, tick=19, label=20, title=19, legend=17, ann=17, small=16, panel=24)   # one-column figure, 7.4 in drawn
plt.rcParams.update({"xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "axes.labelsize": FS["label"],
                     "axes.linewidth": 2.4, "xtick.major.size": 9, "ytick.major.size": 9, "xtick.major.width": 2.4,
                     "ytick.major.width": 2.4})
CHOOSE = sys.argv[sys.argv.index("--choose") + 1] if "--choose" in sys.argv else "B"
R = json.loads((M.MS / "fig2_photon_optimal" / "results" / "fig2.json").read_text())
X = json.loads((HERE / "results" / "fig2_extra.json").read_text())
OM = R["params"]["omega"]
G, F = R["prep"]["gkp"], R["flip"]
CF = X["modal_gkp"]
# analytic paths: wide translucent solid bands; numerical optima: narrow dashed lines drawn on top (same colour per path)
GEO = dict(color=S.ROYALBLUE, ls="-", lw=8.0, alpha=0.32, solid_capstyle="butt", zorder=2)
AN = dict(color=S.CRIMSON, ls="-", lw=8.0, alpha=0.32, solid_capstyle="butt", zorder=2)
PH = dict(color=S.CRIMSON, ls=(0, (3.2, 1.8)), lw=2.8, zorder=4)
BOUND = dict(color=S.INK, lw=2.0, ls=(0, (1.2, 2.4)))
FORBID = "#E5E7EB"
GEONUM = dict(color=S.ROYALBLUE, ls=(0, (3.2, 1.8)), lw=2.8, zorder=4)
# plotted in reverse order: the analytic bands first, the dashed numerical optima on top
CUR = {"gkp": [(X["geo_num_gkp"], GEONUM), (G["photon"], PH), (G["geodesic"], GEO), (CF, AN)],
       "flip": [(X["geo_num_flip"], GEONUM), (F["photon"], PH), (F["geodesic"], GEO), (X["flip_modal"], AN)]}
TITLE = {"gkp": r"GKP preparation", "flip": r"$|0_L\rangle\to|1_L\rangle$"}
BNS = {"gkp": X["gkp_bound_ns"], "flip": X["flip_bound"]["B2_ns"]}
TMAX = {"gkp": 156, "flip": 410}


def ends(ax, key, field):
    """a dot at the end of every path: the paths have different durations in absolute time"""
    for d, sty in CUR[key][::-1]:
        ax.plot(d["t_ns"][-1], d[field][-1], "o", ms=10, color=sty["color"], mec="white", mew=1.4, zorder=6, clip_on=False)


def n_of_t(ax, key, letter, title=True):
    for d, sty in CUR[key][::-1]:
        ax.plot(d["t_ns"], d["nbar"], **sty)
    ends(ax, key, "nbar")
    if key == "flip":
        i = int(np.argmin(F["photon"]["nbar"]))
        ax.annotate("vacuum", xy=(F["photon"]["t_ns"][i], 0.06), xytext=(F["photon"]["t_ns"][i], 1.25),
                    fontsize=FS["small"], ha="center", va="bottom", arrowprops=dict(arrowstyle="->", lw=2.2, color=S.INK))
    ax.set_xlim(0, TMAX[key])
    ax.set_ylim(0, 2.9 if key == "gkp" else 4.0)
    ax.set_ylabel(r"$\bar n$")
    if title:
        ax.set_title(TITLE[key], fontsize=FS["title"], pad=8)
    S.panel_label(ax, letter, x=-0.40, y=1.02, size=FS["panel"])


def exposure(ax, key, letter):
    for d, sty in CUR[key][::-1]:
        ax.plot(d["t_ns"], d["expo"], **sty)
    ends(ax, key, "expo")
    B = BNS[key]
    ax.axhline(B, **BOUND, zorder=1)
    if key == "gkp":                               # free space: above the line on the left (GKP), below it mid-panel (flip)
        ax.text(0.03 * TMAX[key], B * 1.04, "lower bound", fontsize=FS["small"] - 1, color=S.INK, va="bottom",
                ha="left", style="italic")
    else:                                          # the curves leave no room for the long label: "bound", left, above
        ax.text(0.03 * TMAX[key], B * 1.04, "bound", fontsize=FS["small"] - 1, color=S.INK, va="bottom",
                ha="left", style="italic")
    ax.set_xlim(0, TMAX[key])
    ax.set_ylim(0, 1.18 * max(max(d["expo"]) for d, _ in CUR[key]))
    ax.set_xlabel(r"time $t$ (ns)")
    ax.set_ylabel(r"$\int_0^t\!\bar n\,dt'$ (ns)")
    S.panel_label(ax, letter, x=-0.40, y=1.02, size=FS["panel"])


def fid_of_t(ax, key, letter):
    for d, sty in CUR[key][::-1]:
        ax.plot(d["t_ns"], d["F_T"], **sty)
    ends(ax, key, "F_T")
    ax.set_xlim(0, TMAX[key])
    if key == "gkp":                               # preparation starts at |<0|0_L>|^2 = 0.54: zoom on the rise (user, Oct 29)
        lo = min(min(d["F_T"]) for d, _ in CUR[key])
        assert lo >= 0.5, f"F_T below the axis floor: {lo}"
        ax.set_ylim(0.5, 1.02)
        ax.set_yticks([0.6, 0.8, 1.0])                 # one decimal, as in (f)
    else:
        ax.set_ylim(0, 1.04)
    ax.set_xlabel(r"time $t$ (ns)")
    ax.set_ylabel(r"$F_T$")
    S.panel_label(ax, letter, x=-0.40, y=1.02, size=FS["panel"])


def n_of_F(ax, key, letter):
    m = X["m_0L"] if key == "gkp" else X["m_1L"]
    Fm, mm = np.array(m["F"]), np.array(m["m"])
    ax.fill_between(Fm, 0, mm, color=FORBID, lw=0, zorder=0)
    ax.plot(Fm, mm, color=S.INK, lw=2.0, zorder=1)
    for d, sty in CUR[key][::-1]:
        ax.plot(d["F_T"], d["nbar"], **sty, zorder=3)
    ax.text(0.96, 0.04, "excluded", transform=ax.transAxes, fontsize=FS["small"], color=S.INK, ha="right", va="bottom")
    ax.set_xlim(0.5 if key == "gkp" else 0.0, 1.0)
    ax.set_ylim(0, 2.9 if key == "gkp" else 4.0)
    ax.set_xticks([0.5, 0.75, 1.0] if key == "gkp" else [0, 0.5, 1.0])
    ax.set_xlabel(r"target fidelity $F_T$")
    ax.set_ylabel(r"$\bar n$")
    ax.set_title(TITLE[key], fontsize=FS["title"], pad=8)
    S.panel_label(ax, letter, x=-0.40, y=1.02, size=FS["panel"])


def legend(fig, y, key_an=True):
    # one entry per path: the analytic band with the dashed numerical optimum on top (the caption says which is which)
    hs = [(Line2D([], [], **GEO), Line2D([], [], **GEONUM)), (Line2D([], [], **AN), Line2D([], [], **PH)),
          Line2D([], [], **BOUND)]
    labs = ["time optimal", "photon optimal", "lower bound"]
    fig.legend(hs, labs, loc="upper center", ncol=3, bbox_to_anchor=(0.56, y), fontsize=FS["legend"],
               columnspacing=1.4, handlelength=2.6, frameon=False, handler_map={tuple: HandlerTuple(ndivide=None)})


def make(cand):
    rows = 3 if cand == "B" else 2
    H = 3.05 * rows + 1.05
    fig, axs = plt.subplots(rows, 2, figsize=(7.4, H))
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.62 / H, top=1 - 1.12 / H, hspace=0.62, wspace=0.50)
    if cand in ("A", "B"):
        n_of_t(axs[0, 0], "gkp", "a")
        n_of_t(axs[0, 1], "flip", "b")
        for ax in axs[0]:
            ax.set_xlabel(r"time $t$ (ns)")
    else:
        n_of_F(axs[0, 0], "gkp", "a")
        n_of_F(axs[0, 1], "flip", "b")
    exposure(axs[1, 0], "gkp", "c")
    exposure(axs[1, 1], "flip", "d")
    if cand == "B":
        fid_of_t(axs[2, 0], "gkp", "e")
        fid_of_t(axs[2, 1], "flip", "f")
    legend(fig, 1.0)
    (HERE / "candidates").mkdir(exist_ok=True)
    S.save(fig, HERE / "candidates" / f"fig2_{cand}")
    if cand == CHOOSE:
        fig2 = HERE / "fig2"
        for ext in ("png", "pdf"):
            (fig2.with_suffix("." + ext)).write_bytes((HERE / "candidates" / f"fig2_{cand}.{ext}").read_bytes())
    print("wrote", HERE / "candidates" / f"fig2_{cand}.png")


for c in ("A", "B"):
    make(c)
print(f"chosen: {CHOOSE}; T (ns): GKP {G['geodesic']['T_ns']:.1f}/{G['photon']['T_ns']:.1f}/{CF['T_ns']:.1f}, flip "
      f"{F['geodesic']['T_ns']:.1f}/{F['photon']['T_ns']:.1f}/{X['flip_modal']['T_ns']:.1f}; "
      f"exposure (photon ns): GKP {G['geodesic']['J_ns']:.1f}/{G['photon']['J_ns']:.1f}/{CF['J_ns']:.1f} bound {BNS['gkp']:.1f}; "
      f"flip {F['geodesic']['J_ns']:.1f}/{F['photon']['J_ns']:.1f} bound {BNS['flip']:.1f}")
