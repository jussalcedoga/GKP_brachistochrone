"""Fig. 3 (one column): how a gate set is asked to follow a path, and what the time- and photon-optimal paths pass through.
(b) the tracking policy, schematic (no data): in the plane of |0> and the path direction |chi>, x = 2 Re<0|rho|chi>,
    z = <0|rho|0> - <chi|rho|chi>; pure states of span{|0>, |chi>} lie on the unit circle (cos s|0> + sin s|chi> at the
    polar angle 2s), mixed or leaked states inside.  The path (royal blue) is sliced at t_k = kT/N; the pulse (crimson, as
    the CS runs of Fig. 4) must keep the closeness F_k above the threshold (shaded caps); circles filled 1 -> N.
(a) Wigner functions of the path states: GKP preparation halfway along the time-optimal and the photon-optimal path;
    logical flip halfway along the time-optimal path (close to |+_L>) and at the vacuum point of the photon-optimal path
    (its photon-number minimum).  Path states from the reference paths of the tracking runs (cutoff 80; cached by the SI
    three-path script in supplementary_information/_cache), Wigner grid |q|,|p| <= 5, exact there (mfcommon.wigner_grid,
    review Nov 1, L1-F1); the arrays are saved to fig3_wigner.npz in this folder.
python figures/Fig3/plot_fig3.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))
import mfcommon as M  # noqa: E402,I001

# Nov 16 (author): the style of Fig. 1 -- drawn at the same 14 in width with the same 1.25x font sizes and line widths,
# Wigner panels boxed with outward ticks and q/p labels as Fig. 1 (c, d), titles "Preparation"/"Logical Flip"
M.FS.update({k_: round(1.25 * v_) for k_, v_ in M.FS.items()})
S = M.setup()
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Polygon  # noqa: E402
import numpy as np  # noqa: E402

from common import states, tracking  # noqa: E402

FS = M.FS
F2 = json.loads((M.MS / "fig2_photon_optimal" / "results" / "fig2.json").read_text())
TH0 = F2["prep"]["gkp"]["theta0"]
THR = 0.97
ALPHAS = (0.15, 0.35, 0.55, 0.78, 1.0)
PATHC, PULSEC = S.ROYALBLUE, M.GATE_COLOR["cs"]
CACHE = M.MF / "supplementary_information" / "_cache"
# geometry in inches on the 14 in canvas of Fig. 1: (a) four Wigner panels in one row (preparation pair, flip pair) and
# a colour bar, over (b), the path-following schematic at equal aspect
W_IN = 14.0
PW = 2.55                                        # Wigner panel side
XA, G1, G2 = 1.35, 0.14, 0.55                    # left edge of the first panel, gap within a pair, gap between pairs
XP = [XA, XA + PW + G1, XA + 2 * PW + G1 + G2, XA + 3 * PW + 2 * G1 + G2]
Y_W = 1.75                                       # top of the Wigner row (letter, pair title and panel titles above)
GAP_WT = 1.55                                    # Wigner row -> schematic (tick labels and q label in between)
X0 = 2.6                                         # left edge of (b): room for its z label and tick labels
AW = 9.0                                         # (b) width
AH = AW * 1.19 / 1.18                            # = y range / x range of (b), equal aspect
BOT = 1.25                                       # x tick labels and x label of (b)
Y_T = Y_W + PW + GAP_WT                          # top of the schematic
H_IN = Y_T + AH + BOT
fig = plt.figure(figsize=(W_IN, H_IN))
LW_PATH, LW_PULSE, MS_PATH, MS_PULSE = 6.0, 4.6, 17, 15        # Fig. 1 path width and snapshot markers


def ax_at(x0, y_top, w, h):
    return fig.add_axes([x0 / W_IN, 1 - (y_top + h) / H_IN, w / W_IN, h / H_IN])


def rgba(c, a):
    return (*matplotlib.colors.to_rgb(c), a)


def letter(x_in, y_top_in, l):
    fig.text(x_in / W_IN, 1 - y_top_in / H_IN, f"({l})", fontsize=FS["panel"], ha="left", va="top")


# ------------------------------------------------------------------------------------------------ (b) the policy
ax = ax_at(X0, Y_T, AW, AH)
th = np.linspace(-0.06, np.pi / 2 + 0.06, 300)    # the unit circle of pure states of the plane, to the axis edges
ax.plot(np.sin(th), np.cos(th), color=S.INK, lw=2.2, ls=(0, (2, 2)), zorder=1)
s = np.linspace(0, TH0, 300)
ax.plot(np.sin(2 * s), np.cos(2 * s), color=PATHC, lw=LW_PATH, zorder=3, solid_capstyle="round")
NK = 5
sk = TH0 * np.arange(1, NK + 1) / NK
nk = np.c_[np.sin(2 * sk), np.cos(2 * sk)]
# F >= thr  <=>  r . n_k >= 2 thr - 1 (states of the plane with w_in = 1): the circular segment of half-angle phi.
# Eq. (S loss) uses thr = 0.97 at the slices k < N (polish stage, Supplementary Table "schedule") and the much stricter
# thr_fin = 0.99995 at the last slice, whose segment (half-angle 0.014 rad) is too thin to draw: no cap at the target.
for a in 2 * sk[:-1]:
    phi = np.arccos(2 * THR - 1)
    arc = np.linspace(a - phi, a + phi, 60)
    ax.add_patch(Polygon(np.c_[np.sin(arc), np.cos(arc)], closed=True, fc=PATHC, alpha=0.13, ec=PATHC, lw=1.4, zorder=1))
u = np.linspace(0, 1, 300)                       # schematic state rho(t) under the pulse: lags inside the circle while entangled
ang = 2 * TH0 * (u - 0.035 * np.sin(2 * np.pi * u))
rad = 1 - 0.075 * np.sin(np.pi * u) ** 1.3 - 0.035 * np.exp(-((u - 0.6) / 0.08) ** 2)
rx, rz = rad * np.sin(ang), rad * np.cos(ang)
ax.plot(rx, rz, color=PULSEC, lw=LW_PULSE, zorder=4)
ik = [int(round(x * (len(u) - 1))) for x in np.arange(1, NK + 1) / NK]
for j, i in enumerate(ik):
    ax.plot([nk[j, 0], rx[i]], [nk[j, 1], rz[i]], color=PULSEC, lw=2.2, zorder=5)
    ax.plot(rx[i], rz[i], "o", ms=MS_PULSE, mfc=rgba(PULSEC, ALPHAS[j]), mec=PULSEC, mew=2.4, zorder=6)
    ax.plot(*nk[j], "o", ms=MS_PATH, mfc=rgba(PATHC, ALPHAS[j]), mec=PATHC, mew=2.6, zorder=7)
ax.plot(0, 1, "o", ms=14, color=S.INK, mec="white", mew=1.8, zorder=8)       # as the state dots of Fig. 1
ax.text(0.0, 1.04, r"$|0\rangle$", fontsize=FS["label"] + 2, ha="center", va="bottom")
# Nov 16 (author): just the target ket |psi_F>
ax.text(nk[-1, 0] - 0.035, nk[-1, 1] - 0.025, r"$|\psi_F\rangle$", fontsize=FS["label"] + 2, ha="right",
        va="top")                                  # below-left: to the right it would cross the frame
J = 1
pk = nk[J]
tang = np.array([np.cos(2 * sk[J]), -np.sin(2 * sk[J])])
ax.add_patch(FancyArrowPatch(pk, pk + 0.26 * tang, arrowstyle="-|>", mutation_scale=30, lw=3.4, color=PATHC, zorder=9))
ax.text(*(pk + 0.26 * tang + np.array([0.03, 0.02])), r"$\dot p(t_k)$", fontsize=FS["ann"], color=PATHC, zorder=9,
        ha="left", va="bottom")
ax.annotate(r"$p(t_k)$", pk, xytext=(14, 12), textcoords="offset points", fontsize=FS["ann"], color=PATHC)
ax.annotate(r"$\rho(t_k)$", (rx[ik[J]], rz[ik[J]]), xytext=(-80, -14), textcoords="offset points", fontsize=FS["ann"],
            color=PULSEC)
ax.set_xlim(-0.06, 1.12)
ax.set_ylim(-0.05, 1.14)
ax.set_aspect("equal")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
for sp in ax.spines.values():                    # boxed, outward ticks (Fig. 1)
    sp.set_visible(True)
ax.tick_params(direction="out", length=8)
# the coordinates spelled out on the axes (Oct 30, author): coherence and population difference of |0> and |chi>,
# as in Supplementary Fig. 8, so the caption need not define them
ax.set_xlabel(r"$x=2\,\mathrm{Re}\,\langle0|\rho|\chi\rangle$", labelpad=4)
ax.set_ylabel(r"$z=\langle0|\rho|0\rangle-\langle\chi|\rho|\chi\rangle$", labelpad=6)
letter(0.05, Y_T - 0.25, "b")
hk = [Line2D([], [], color=PATHC, lw=LW_PATH),
      Line2D([], [], ls="none", marker="o", ms=MS_PATH, mfc=rgba(PATHC, 0.55), mec=PATHC, mew=2.6),
      Patch(fc=PATHC, alpha=0.13, ec=PATHC),
      Line2D([], [], color=PULSEC, lw=LW_PULSE),
      Line2D([], [], ls="none", marker="o", ms=MS_PULSE, mfc=rgba(PULSEC, 0.55), mec=PULSEC, mew=2.4)]
ax.legend(hk, [r"path $p(t)$", r"$p(t_k)$", rf"$F_k\geq{THR}$", r"state $\rho(t)$", r"$\rho(t_k)$"], loc="lower left",
          fontsize=FS["legend"] + 1, frameon=False, handlelength=1.8, labelspacing=0.55, borderaxespad=0.6,
          handletextpad=0.7)
# Nov 16 (author): the crimson curve is the state rho(t) the pulse produces, not the pulse: labelled "state rho(t)"
# (the lower-left interior of the quarter circle is empty: the key sits there, inside (b))

# ------------------------------------------------------------------------------------------------ (a) states on the paths
D = 80
Wg = M.wigner_grid(D, 161)                       # |q|,|p| <= 5, exact on the shown window
WSAVE = {}


def path_state(name, ref, f=None, at_vacuum=False):
    X = np.load(CACHE / f"path_{name}_{ref}.npz")["X"][:, :D]
    P = tracking.Path(X)
    if at_vacuum:                                 # the photon-number minimum of the path
        ff = np.linspace(0, 1, 401)
        G, _ = P.at(ff)
        nb = np.real(np.einsum("ij,j,ij->i", G.conj(), np.arange(D), G))
        f = float(ff[int(np.argmin(nb))])
    G, _ = P.at(np.array([f]))
    v = G[0] / np.linalg.norm(G[0])
    return v, f


plus = states.gkp(0.4, 0, D) + states.gkp(0.4, 1, D)
plus /= np.linalg.norm(plus)
cells = [("gkp", "geodesic", dict(f=0.5), "time optimal", PATHC),
         ("gkp", "photon", dict(f=0.5), "photon optimal", S.CRIMSON),
         ("flip", "geodesic", dict(f=0.5), "time optimal", PATHC),
         ("flip", "photon", dict(at_vacuum=True), "photon optimal", S.CRIMSON)]
info = {}
ext = [Wg.x[0], Wg.x[-1], Wg.x[0], Wg.x[-1]]
for c, (name, ref, kw, title, col) in enumerate(cells):
    v, f = path_state(name, ref, **kw)
    W = Wg.of_vectors(v[None])
    WSAVE[f"{name}_{ref}"] = W
    nb = float(np.real(np.vdot(v, np.arange(D) * v)))
    a_ = ax_at(XP[c], Y_W, PW, PW)
    im = a_.imshow(W, origin="lower", extent=ext, cmap=S.RBCMAP, vmin=-S.W_LIM, vmax=S.W_LIM)
    a_.set_xticks([-4, 0, 4])                    # as Fig. 1 (c, d)
    a_.set_yticks([-4, 0, 4])
    a_.tick_params(direction="out", length=8)
    for sp in a_.spines.values():
        sp.set_visible(True)
    a_.set_xlabel(r"$q$", labelpad=2, fontsize=FS["label"] + 3)
    if c == 0:
        a_.set_ylabel(r"$p$", labelpad=2, fontsize=FS["label"] + 3)
    else:
        a_.set_yticklabels([])
    a_.set_title(title, fontsize=FS["small"], color=col, pad=8)      # which path, in its colour
    # Nov 16 (author): photon number in every panel (no |<+_L|p>|^2 label; F_plus is still written to fig3_facts.json),
    # labels without a background box
    a_.text(0.95, 0.94, rf"$t={f:.2f}\,T$", transform=a_.transAxes, ha="right", va="top", fontsize=FS["small"] - 3,
            zorder=2)
    a_.text(0.05, 0.05, rf"$\bar n={nb:.2f}$", transform=a_.transAxes, ha="left", va="bottom",
            fontsize=FS["small"] - 3, zorder=2)
    info[f"{name}_{ref}"] = dict(f=f, nbar=nb, F_plus=float(abs(np.vdot(plus, v)) ** 2))
letter(0.05, 0.25, "a")
for k_, lab in ((0, "Preparation"), (2, "Logical Flip")):         # one title over each pair (Fig. 1 names)
    fig.text((XP[k_] + PW + 0.5 * G1) / W_IN, 1 - (Y_W - 0.78) / H_IN, lab, ha="center", va="bottom",
             fontsize=FS["title"] - 1)
cax = ax_at(XP[3] + PW + 0.18, Y_W, 0.17, PW)
cb = fig.colorbar(im, cax=cax)
cb.set_ticks([-S.W_LIM, 0, S.W_LIM])
cb.set_ticklabels([r"$-1/\pi$", r"$0$", r"$1/\pi$"])
cb.ax.tick_params(labelsize=FS["tick"])
cb.set_label(r"$W(q,p)$", fontsize=FS["label"] + 3, labelpad=-18)   # as Fig. 1
S.save(fig, HERE / "fig3")
np.savez_compressed(HERE / "fig3_wigner.npz", x=Wg.x, **WSAVE)
(HERE / "fig3_facts.json").write_text(json.dumps(info, indent=1))
print("wrote", HERE / "fig3.png", "|", {k: {kk: round(vv, 4) for kk, vv in d.items()} for k, d in info.items()})
