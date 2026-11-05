"""Fig. 3 (one column): how a gate set is asked to follow a path, and what the time- and photon-optimal paths pass through.
(a) the tracking policy, schematic (no data): in the plane of |0> and the path direction |chi>, x = 2 Re<0|rho|chi>,
    z = <0|rho|0> - <chi|rho|chi>; pure states of span{|0>, |chi>} lie on the unit circle (cos s|0> + sin s|chi> at the
    polar angle 2s), mixed or leaked states inside.  The path (royal blue) is sliced at t_k = kT/N; the pulse (crimson, as
    the CS runs of Fig. 4) must keep the closeness F_k above the threshold (shaded caps); circles filled 1 -> N.
(b) Wigner functions of the path states: GKP preparation halfway along the time-optimal and the photon-optimal path;
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

S = M.setup()
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Polygon  # noqa: E402
import numpy as np  # noqa: E402

from common import states, tracking  # noqa: E402

FS = dict(M.FS, tick=19, label=20, title=19, legend=17, ann=17, small=16, panel=24)   # as Fig. 2 (one column, 7.4 in)
plt.rcParams.update({"xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "axes.labelsize": FS["label"],
                     "axes.linewidth": 2.4, "xtick.major.size": 9, "ytick.major.size": 9, "xtick.major.width": 2.4,
                     "ytick.major.width": 2.4})
F2 = json.loads((M.MS / "fig2_photon_optimal" / "results" / "fig2.json").read_text())
TH0 = F2["prep"]["gkp"]["theta0"]
THR = 0.97
ALPHAS = (0.15, 0.35, 0.55, 0.78, 1.0)
PATHC, PULSEC = S.ROYALBLUE, M.GATE_COLOR["cs"]
CACHE = M.MF / "supplementary_information" / "_cache"
# Nov 4 (user): less tall at full column width (as Fig. 2): (a) the four Wigner panels in one row, preparation (left
# pair) and the flip (right pair), over (b), the path-following schematic, at equal aspect and as wide as fits
PW = 1.55                                        # Wigner panel side in (a)
XA, G1, G2 = 0.05, 0.05, 0.16                    # left edge of (a), gap within a pair, gap between the pairs
XP = [XA, XA + PW + G1, XA + 2 * PW + G1 + G2, XA + 3 * PW + 2 * G1 + G2]
X0 = 1.25                                        # left edge of (b), room for its z label and tick labels
FB = dict(ann=22, legend=20, label=24, tick=21)  # Nov 5 (user): larger annotations, legend and axis labels in (b)
FA = dict(pair=20, title=17, inner=15, cbtick=17, cblab=18)   # (a): pair titles, panel titles, in-panel labels, bar
AW = 5.10                                        # (b) width, room on the right for "target |psi_f>"
AH = AW * 1.19 / 1.18                            # = y range / x range of (b), equal aspect
Y_W, GAP_WT, BOT = 0.82, 0.62, 0.90              # top of the Wigner row (two title rows above), row -> (b), x labels
Y_T = Y_W + PW + GAP_WT                          # top of the schematic
W_IN, H_IN = 7.4, Y_T + AH + BOT
fig = plt.figure(figsize=(W_IN, H_IN))


def ax_at(x0, y_top, w, h):
    return fig.add_axes([x0 / W_IN, 1 - (y_top + h) / H_IN, w / W_IN, h / H_IN])


def rgba(c, a):
    return (*matplotlib.colors.to_rgb(c), a)


# ------------------------------------------------------------------------------------------------ (a) the policy
ax = ax_at(X0, Y_T, AW, AH)
th = np.linspace(-0.06, np.pi / 2 + 0.06, 300)    # the unit circle of pure states of the plane, to the axis edges
ax.plot(np.sin(th), np.cos(th), color=S.INK, lw=1.4, ls=(0, (2, 2)), zorder=1)
s = np.linspace(0, TH0, 300)
ax.plot(np.sin(2 * s), np.cos(2 * s), color=PATHC, lw=6.0, zorder=3, solid_capstyle="round")
NK = 5
sk = TH0 * np.arange(1, NK + 1) / NK
nk = np.c_[np.sin(2 * sk), np.cos(2 * sk)]
# F >= thr  <=>  r . n_k >= 2 thr - 1 (states of the plane with w_in = 1): the circular segment of half-angle phi.
# Eq. (S loss) uses thr = 0.97 at the slices k < N (polish stage, Supplementary Table "schedule") and the much stricter
# thr_fin = 0.99995 at the last slice, whose segment (half-angle 0.014 rad) is too thin to draw: no cap at the target.
for a in 2 * sk[:-1]:
    phi = np.arccos(2 * THR - 1)
    arc = np.linspace(a - phi, a + phi, 60)
    ax.add_patch(Polygon(np.c_[np.sin(arc), np.cos(arc)], closed=True, fc=PATHC, alpha=0.13, ec=PATHC, lw=1.0, zorder=1))
u = np.linspace(0, 1, 300)                       # a schematic pulse: lags inside the circle while entangled
ang = 2 * TH0 * (u - 0.035 * np.sin(2 * np.pi * u))
rad = 1 - 0.075 * np.sin(np.pi * u) ** 1.3 - 0.035 * np.exp(-((u - 0.6) / 0.08) ** 2)
rx, rz = rad * np.sin(ang), rad * np.cos(ang)
ax.plot(rx, rz, color=PULSEC, lw=3.6, zorder=4)
ik = [int(round(x * (len(u) - 1))) for x in np.arange(1, NK + 1) / NK]
for j, i in enumerate(ik):
    ax.plot([nk[j, 0], rx[i]], [nk[j, 1], rz[i]], color=PULSEC, lw=1.6, zorder=5)
    ax.plot(rx[i], rz[i], "o", ms=12, mfc=rgba(PULSEC, ALPHAS[j]), mec=PULSEC, mew=1.8, zorder=6)
    ax.plot(*nk[j], "o", ms=13, mfc=rgba(PATHC, ALPHAS[j]), mec=PATHC, mew=2.2, zorder=7)
ax.plot(0, 1, "o", ms=12, color=S.INK, zorder=8)
ax.text(0.0, 1.045, r"$|0\rangle$", fontsize=FB["ann"], ha="center", va="bottom")
ax.text(nk[-1, 0] + 0.05, nk[-1, 1] + 0.02, r"target $|\psi_f\rangle$", fontsize=FB["ann"], ha="left", va="center")
J = 1
pk = nk[J]
tang = np.array([np.cos(2 * sk[J]), -np.sin(2 * sk[J])])
ax.add_patch(FancyArrowPatch(pk, pk + 0.28 * tang, arrowstyle="-|>", mutation_scale=22, lw=2.8, color=PATHC, zorder=9))
ax.text(*(pk + 0.28 * tang + np.array([0.03, 0.02])), r"$\dot p(t_k)$", fontsize=FB["ann"], color=PATHC, zorder=9,
        ha="left", va="bottom")
ax.annotate(r"$p(t_k)$", pk, xytext=(12, 10), textcoords="offset points", fontsize=FB["ann"], color=PATHC)
ax.annotate(r"$\rho(t_k)$", (rx[ik[J]], rz[ik[J]]), xytext=(-72, -12), textcoords="offset points", fontsize=FB["ann"],
            color=PULSEC)
ax.set_xlim(-0.06, 1.12)
ax.set_ylim(-0.05, 1.14)
ax.set_aspect("equal")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.tick_params(labelsize=FB["tick"])
# the coordinates spelled out on the axes (Oct 30, author): coherence and population difference of |0> and |chi>,
# as in Supplementary Fig. 8, so the caption need not define them
ax.set_xlabel(r"$x=2\,\mathrm{Re}\,\langle0|\rho|\chi\rangle$", labelpad=2, fontsize=FB["label"])
ax.set_ylabel(r"$z=\langle0|\rho|0\rangle-\langle\chi|\rho|\chi\rangle$", labelpad=4, fontsize=FB["label"])
S.panel_label(ax, "b", x=-X0 / AW, y=0.99, size=FS["panel"])   # at the left edge of the figure, as (a)
hk = [Line2D([], [], color=PATHC, lw=6.0),
      Line2D([], [], ls="none", marker="o", ms=13, mfc=rgba(PATHC, 0.55), mec=PATHC, mew=2.2),
      Patch(fc=PATHC, alpha=0.13, ec=PATHC),
      Line2D([], [], color=PULSEC, lw=3.6),
      Line2D([], [], ls="none", marker="o", ms=12, mfc=rgba(PULSEC, 0.55), mec=PULSEC, mew=1.8)]
ax.legend(hk, [r"path $p(t)$", r"$p(t_k)$", rf"$F_k\geq{THR}$", r"pulse", r"$\rho(t_k)$"], loc="lower left",
          fontsize=FB["legend"], frameon=False, handlelength=1.6, labelspacing=0.6, borderaxespad=0.6, handletextpad=0.6)
# (the lower-left interior of the quarter circle is empty: the key sits there, inside (a))

# ------------------------------------------------------------------------------------------------ (b) states on the paths
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
for c, (name, ref, kw, title, col) in enumerate(cells):
    v, f = path_state(name, ref, **kw)
    W = Wg.of_vectors(v[None])
    WSAVE[f"{name}_{ref}"] = W
    nb = float(np.real(np.vdot(v, np.arange(D) * v)))
    a_ = ax_at(XP[c], Y_W, PW, PW)
    im = a_.imshow(W, origin="lower", extent=[Wg.x[0], Wg.x[-1], Wg.x[0], Wg.x[-1]], cmap=S.RBCMAP, vmin=-S.W_LIM,
                   vmax=S.W_LIM)
    a_.set_xticks([])
    a_.set_yticks([])
    for sp in a_.spines.values():
        sp.set_visible(True)
        sp.set_color(col)
        sp.set_linewidth(3.2)
    a_.set_title(title, fontsize=FA["title"], color=col, pad=4)
    # in-panel labels inset from the frame and drawn under it (spines zorder 2.5), so no label box covers the outline
    a_.text(0.93, 0.92, rf"$t={f:.2f}\,T$", transform=a_.transAxes, ha="right", va="top", fontsize=FA["inner"],
            zorder=2, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    if name == "flip" and ref == "geodesic":
        lab = rf"$|\langle+_L|p\rangle|^2={abs(np.vdot(plus, v)) ** 2:.2f}$"
    else:
        lab = rf"$\bar n={nb:.2f}$"
    a_.text(0.05, 0.05, lab, transform=a_.transAxes, ha="left", va="bottom", fontsize=FA["inner"], zorder=2,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    info[f"{name}_{ref}"] = dict(f=f, nbar=nb, F_plus=float(abs(np.vdot(plus, v)) ** 2))
    if c == 0:
        S.panel_label(a_, "a", x=-XA / PW, y=1.0 + 0.40 / PW, size=FS["panel"])   # at the left edge, as (b)
for k_, lab in ((0, "GKP preparation"), (2, r"$|0_L\rangle\to|1_L\rangle$")):    # one title over each pair
    fig.text((XP[k_] + PW + 0.5 * G1) / W_IN, 1 - (Y_W - 0.40) / H_IN, lab, ha="center", va="bottom", fontsize=FA["pair"])
cax = ax_at(XP[3] + PW + 0.10, Y_W, 0.11, PW)
cb = fig.colorbar(im, cax=cax)
cb.set_ticks([-S.W_LIM, 0, S.W_LIM])
cb.set_ticklabels([r"$-1/\pi$", r"$0$", r"$1/\pi$"])
cb.ax.tick_params(labelsize=FA["cbtick"])
cb.set_label(r"$W(q,p)$", fontsize=FA["cblab"])
# the label sits beside the narrow middle tick "0" instead of beyond the wide "-1/pi" (less white space to the right)
fig.canvas.draw()
bb0 = [t for t in cb.ax.get_yticklabels() if t.get_text() == r"$0$"][0].get_window_extent(fig.canvas.get_renderer())
x0_ax = cb.ax.transAxes.inverted().transform((bb0.x1 + 6 * fig.dpi / 72, 0))[0]
cb.ax.yaxis.set_label_coords(x0_ax, 0.5)
S.save(fig, HERE / "fig3")
np.savez_compressed(HERE / "fig3_wigner.npz", x=Wg.x, **WSAVE)
(HERE / "fig3_facts.json").write_text(json.dumps(info, indent=1))
print("wrote", HERE / "fig3.png", "|", {k: {kk: round(vv, 4) for kk, vv in d.items()} for k, d in info.items()})
