"""Fig. S3, time against photons for every target, relative to the brachistochrone, at the same energy uncertainty
Delta H = omega = g/sqrt2: extra time L/theta - 1 of the photon-saving path (L its length, theta the Bures angle; T = L/omega)
against the exposure saving 1 - J/J_geo.  Preparations: the photon-optimal path.  Transfer |0_L> -> |1_L>: the minimizers
of J_ell = int <n + ell (1 - Pi)> dvartheta for the code-space penalty ell (ell = infinity: brachistochrone, at the origin;
ell = 0: photon optimum).  (a) all targets; (b) the preparations, enlarged.
Data: manuscript/fig3_time_vs_photon/results/fig3.json (scan; read only).
python figures/SFig03_time_photon_comparison/plot.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

import mfcommon as M  # noqa: E402

S = M.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

FS = S.FS
R = json.loads((M.MS / "fig3_time_vs_photon" / "results" / "fig3.json").read_text())
scan = R["scan"]
NS = 1e3 / R["params"]["omega"]
FAM = {"gkp": (S.NAVY, "o", r"GKP $|0_L\rangle$, $\Delta=0.5\to0.3$"),
       "cat": (S.PURPLE, "s", r"cat $|C_\gamma^+\rangle$, $\gamma=1\to2.5$"),
       "flip": (S.RASPBERRY, "D", r"$|0_L\rangle\to|1_L\rangle$, penalty $\ell$"),
       "fock": (S.INK, "^", r"Fock $|2\rangle,|4\rangle,|6\rangle$; binomial")}
BOX = (-0.5, -0.8, 11.0, 13.3)                       # region shown enlarged in (b): x0, y0, width, height
XLAB = r"extra time $L/\theta-1$ (\%)"
YLAB = r"exposure saving $1-J/J_{\rm geo}$ (\%)"


def pts(fam):
    out = [e for e in scan if e["family"] == fam]
    if fam == "fock":
        out += [e for e in scan if e["family"] == "binomial"]
    key = {"gkp": lambda e: -e["par"], "flip": lambda e: -e["par"]}.get(fam, lambda e: e["par"])
    return sorted(out, key=key)                      # GKP: Delta decreasing; flip: ell decreasing


def xy(P):
    return [100 * e["time_cost"] for e in P], [100 * e["saving"] for e in P]


def draw(ax, fam, ms):
    col, mk, lab = FAM[fam]
    P = pts(fam)
    x, y = xy(P)
    if fam != "fock":
        ax.plot(x, y, "-", color=col, lw=2.4, zorder=3)
    ax.plot(x, y, mk, ms=ms + (2 if fam == "fock" else 0), color=col, mec="k" if fam != "fock" else col, mew=1.1,
            zorder=6 if fam == "fock" else 5, clip_on=False, label=lab)
    return P, x, y


fig, axs = plt.subplots(1, 2, figsize=(14.0, 6.6))
fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.14, wspace=0.30)
# (a) all targets
ax = axs[0]
for fam in ("gkp", "cat", "flip", "fock"):
    P, x, y = draw(ax, fam, 13)
    if fam == "flip":
        col = FAM[fam][0]
        for e, xx, yy in zip(P, x, y):
            if e["par"] == 100:                     # inside the preparation cluster: label with a leader line
                ax.annotate(r"$\ell=100$", (xx, yy), xytext=(16.0, 22.5), fontsize=FS["small"] - 2, color=col, ha="left",
                            va="center", arrowprops=dict(arrowstyle="-", color=col, lw=1.2, shrinkB=6))
            else:
                ax.annotate(rf"$\ell={e['par']:g}$", (xx, yy), xytext=(11, -2), textcoords="offset points",
                            fontsize=FS["small"] - 2, color=col, ha="left", va="center")
            if e["par"] == 0:
                ax.annotate("photon optimum", (xx, yy), xytext=(0, 13), textcoords="offset points",
                            fontsize=FS["small"] - 2, color=col, ha="center", va="bottom")
ax.add_patch(Rectangle(BOX[:2], BOX[2], BOX[3], fill=False, ec=S.INK, lw=1.6, ls=(0, (4, 3)), zorder=1))
ax.text(BOX[0] + BOX[2] + 0.6, BOX[1] + 0.5 * BOX[3], "(b)", fontsize=FS["small"] - 2, color=S.INK, ha="left",
        va="center")
ax.set_xlim(-1.5, 46)
ax.set_ylim(-2.5, 60)
ax.set_xlabel(XLAB)
ax.set_ylabel(YLAB)
ax.legend(loc="lower right", bbox_to_anchor=(1.03, 0.045), fontsize=FS["small"] - 2, handlelength=1.0, borderpad=0.2,
          labelspacing=0.35, handletextpad=0.5, frameon=False)
ax.set_title("all targets", fontsize=FS["title"])
S.panel_label(ax, "a", x=-0.19, y=1.03)

# (b) state preparation, enlarged and labelled
ax = axs[1]
LBL = {}
for fam in ("gkp", "cat", "fock"):
    P, x, y = draw(ax, fam, 14)
    for e, xx, yy in zip(P, x, y):
        LBL[(fam, round(float(e["par"]), 2))] = (xx, yy)
TXT = dict(fontsize=FS["small"] - 2)
G, C = FAM["gkp"][0], FAM["cat"][0]
# GKP: the three lowest points share a column of labels below the line (they are closer than a label is wide)
for (d, ylab) in ((0.4, 7.15), (0.45, 6.3), (0.5, 5.45)):
    ax.annotate(rf"$\Delta={d:g}$", LBL[("gkp", d)], xytext=(7.55, ylab), color=G, ha="left", va="center", **TXT,
                arrowprops=dict(arrowstyle="-", color=G, lw=1.1, shrinkA=2, shrinkB=9))
ax.annotate(r"$\Delta=0.34$", LBL[("gkp", 0.34)], xytext=(-10, 6), textcoords="offset points", color=G, ha="right",
            va="bottom", **TXT)
ax.annotate(r"$\Delta=0.3$", LBL[("gkp", 0.3)], xytext=(-12, 0), textcoords="offset points", color=G, ha="right",
            va="center", **TXT)
for g, o in ((1.0, (11, -2, "left", "center")), (1.5, (11, -4, "left", "top")), (2.0, (-12, 0, "right", "center")),
             (2.5, (0, -14, "center", "top"))):
    ax.annotate(rf"$\gamma={g:g}$", LBL[("cat", g)], xytext=o[:2], textcoords="offset points", color=C, ha=o[2],
                va=o[3], **TXT)
ax.annotate("Fock, binomial:\npaths coincide", (0, 0), xytext=(3.6, 1.2), color=S.INK, ha="left", va="center", **TXT,
            arrowprops=dict(arrowstyle="-", color=S.INK, lw=1.2, shrinkB=9))
ax.set_xlim(BOX[0], BOX[0] + BOX[2])
ax.set_ylim(BOX[1], BOX[1] + BOX[3])
ax.set_xlabel(XLAB)
ax.set_ylabel(YLAB)
ax.set_title("state preparation (enlarged)", fontsize=FS["title"])
S.panel_label(ax, "b", x=-0.19, y=1.03)
S.save(fig, HERE / "S03_time_photon_comparison")
for e in scan:
    print(f"{e['family']:8s} par={e['par']:<6g} theta0={e['theta0']:.4f} T*={NS * e['theta0']:.0f} ns  "
          f"T_ph={NS * e['L_ph']:.0f} ns  time +{100 * e['time_cost']:.1f}%  saving {100 * e['saving']:.1f}%  "
          f"D_geo={e['D_geo']:.3f} rad  E_geo={NS * e['J_geo']:.0f} E_ph={NS * e['J_ph']:.0f} photon ns")
print("wrote", HERE / "S03_time_photon_comparison.png")
