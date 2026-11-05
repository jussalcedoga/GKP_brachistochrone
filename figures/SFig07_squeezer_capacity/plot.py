"""SI figure S7: how qubit-conditioned squeezing pushes the oscillator along a path, and how much it can.
The construction assumes a product qubit-oscillator state and is first order in dt (a local measure, Prop. cap);
following a non-Gaussian path also requires qubit-oscillator entanglement (Thm. nogo).
(a) the path to follow: the brachistochrone in the plane of |0> and |chi> (x, z of Eq. plane), the state p(t) and the
    tangent; at T = 412 ns the path must advance at theta/T = 0.34 omega, the squeezers can push at most v = 0.51 omega
    (arrows drawn to scale).
(b) the qubit sets the weights: a rotation by |Omega| dt from |g> gives P_g = cos^2(|Omega| dt/2), P_e = 1 - P_g.
(c) the squeezers give the push: in the plane of the controls u = (u_A, u_B) = (Re xi, Im xi) in units of g (|u_j| <= g,
    disk), each branch has a control u_j; to first order the oscillator moves under the population-weighted average
    u_eff = P_g u_g + P_e u_e, and its speed along the path is u_eff . tau with tau_l = Re<pdot|-i K_l|p>; the largest is
    v = g|tau| (star: u = g tau-hat; for real paths tau lies along u_A).  Keeping pace with the path needs
    u_eff . tau = theta/T exactly (dashed line: controls that supply exactly theta/T); a larger push runs ahead of the path
    state, so the shaded set holds the controls that can supply theta/T by lowering their amplitude, not pulses that follow.
    Numbers of GKP preparation at T = 412 ns (theta/T = 0.339 omega, v = 0.509 omega).
(d, e) the capacity v(s)/omega along the paths of preparation and of the transfer |0_L> -> |1_L>
    (manuscript/si/squeezer_capacity/results/squeezer_capacity.json, cutoff 80, read only).  Fock |4> and the binomial
    word (|0> + |4>)/sqrt2 have chi = |4>, so v = g|<2|chi>|/sqrt2 = 0 along both paths (which coincide).  The trade-off
    path of (e) is the member w* = kappa/Gamma_q = 0.786 (rates of Sivak et al.); the photon-optimal path reaches the vacuum
    at s = 0.375 and leaves it along Fock |2> (v ~ omega for s = 0.375-0.6).
python figures/SFig07_squeezer_capacity/plot.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import mfcommon as M  # noqa: E402,I001

S = M.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

FS = M.FS
res = json.loads((M.MS / "si" / "squeezer_capacity" / "results" / "squeezer_capacity.json").read_text())
F2 = json.loads((M.MS / "fig2_photon_optimal" / "results" / "fig2.json").read_text())
TH0 = F2["prep"]["gkp"]["theta0"]
T_STAR = F2["prep"]["gkp"]["geodesic"]["T_ns"]
V_OM = float(np.mean(res["gkp"]["geodesic"]))      # v/omega of CS along the GKP brachistochrone (constant)
NEED_OM = T_STAR / 412.0                            # theta/T in units of omega at T = 412 ns
S_VAC = 0.375                                       # vacuum crossing of the photon-optimal flip path (max |<0|p>| at s = 0.376)
QG, QE = S.ROYALBLUE, S.CRIMSON                     # qubit branches |g>, |e>
CSC, PATHC = M.GATE_COLOR["cs"], S.ROYALBLUE
PUSH = S.INK
AXC = S.NAVY                                        # construction lines (no gray)
W_IN, H_IN = 14.0, 11.6
fig = plt.figure(figsize=(W_IN, H_IN))


def ax_at(x0, y_top, w, h):
    return fig.add_axes([x0 / W_IN, 1 - (y_top + h) / H_IN, w / W_IN, h / H_IN])


def title(ax, letter, text):
    ax.text(0.0, 1.06, rf"({letter})\enspace " + text, transform=ax.transAxes, fontsize=FS["title"],
            ha="left", va="bottom")


def arrow(ax, a, b, color, lw=3.0, ms=24, z=7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0))


# (a) the path to follow ------------------------------------------------------------------------------------
ax = ax_at(0.55, 0.75, 4.0, 4.0)
th = np.linspace(0, np.pi / 2, 200)
ax.plot(np.sin(th), np.cos(th), color=AXC, lw=1.2, ls=(0, (1.5, 2.5)))
s = np.linspace(0, TH0, 200)
ax.plot(np.sin(2 * s), np.cos(2 * s), color=PATHC, lw=6.0, solid_capstyle="round")
sp = 0.45 * TH0
p = np.array([np.sin(2 * sp), np.cos(2 * sp)])
tang = np.array([np.cos(2 * sp), -np.sin(2 * sp)])
nrm = np.array([-tang[1], tang[0]])                  # outward normal of the arc
K = 0.90                                             # arrow length per unit of omega (both arrows to the same scale)
ax.plot(*p, "o", ms=14, color=PATHC, mec="k", mew=1.2, zorder=6)
arrow(ax, p, p + K * NEED_OM * tang, PATHC)                              # needed speed theta/T, along the tangent
off = 0.09 * nrm
arrow(ax, p + off, p + off + K * V_OM * tang, CSC)                       # largest push v, drawn beside it
ax.text(0.84, 0.82, rf"push $\leq v$" + "\n" + rf"$={V_OM:.2f}\,\omega$", color=CSC, fontsize=FS["small"], ha="left",
        va="bottom", linespacing=1.15)
mid = p + 0.5 * K * NEED_OM * tang
ax.annotate(r"needed:" + "\n" + rf"$\theta/T={NEED_OM:.2f}\,\omega$", mid, xytext=(0.30, 0.40), color=PATHC,
            fontsize=FS["small"], ha="center", va="center", linespacing=1.15,
            arrowprops=dict(arrowstyle="-", color=PATHC, lw=1.2, shrinkA=4, shrinkB=4))
ax.annotate(r"$p(t)$", p, xytext=(-62, -6), textcoords="offset points", fontsize=FS["ann"], color=PATHC)
ax.text(0.0, 1.04, r"$|0\rangle$", fontsize=FS["ann"], ha="center", va="bottom")
ax.text(np.sin(2 * TH0) + 0.04, np.cos(2 * TH0), r"$|0_L\rangle$", fontsize=FS["ann"], ha="left", va="center")
ax.set_xlim(-0.08, 1.28)
ax.set_ylim(-0.06, 1.22)
ax.set_aspect("equal")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$z$")
title(ax, "a", "path")

# (b) the qubit sets the weights ----------------------------------------------------------------------------
ax = ax_at(5.15, 0.75, 3.6, 4.0)
ax.add_patch(Circle((0, 0), 1, fc="white", ec=AXC, lw=2.2))
ax.plot([0, 0], [-1, 1], color=AXC, lw=1.0)
ax.plot([-1, 1], [0, 0], color=AXC, lw=1.0, ls=(0, (1, 2)))
thq = np.deg2rad(60)                                  # P_g = cos^2(30 deg) = 0.75
qv = np.array([np.sin(thq), np.cos(thq)])
ax.add_patch(FancyArrowPatch((0, 0), 0.97 * qv, arrowstyle="-|>", mutation_scale=24, lw=3.2, color=PUSH))
arc = np.linspace(0, thq, 40)
ax.plot(0.32 * np.sin(arc), 0.32 * np.cos(arc), color=PUSH, lw=1.8)
ax.text(-0.06, 0.30, r"$|\Omega|\Delta t$", fontsize=FS["ann"], ha="right", va="center")   # left of the arc
ax.plot(0, 1, "o", ms=13, color=QG, zorder=5)
ax.plot(0, -1, "o", ms=13, color=QE, zorder=5)
ax.text(0, 1.10, r"$|g\rangle$", color=QG, fontsize=FS["ann"], ha="center", va="bottom")
ax.text(0, -1.12, r"$|e\rangle$", color=QE, fontsize=FS["ann"], ha="center", va="top")
pg, pe = np.cos(thq / 2) ** 2, np.sin(thq / 2) ** 2
ax.add_patch(Rectangle((-1.0, -1.95), 2 * pg, 0.30, fc=QG, alpha=0.30, ec=QG, lw=1.8, clip_on=False))
ax.add_patch(Rectangle((-1.0 + 2 * pg, -1.95), 2 * pe, 0.30, fc=QE, alpha=0.30, ec=QE, lw=1.8, clip_on=False))
ax.text(-1.0 + pg, -1.80, rf"$P_g={pg:.2f}$", color=QG, fontsize=FS["small"] - 2, ha="center", va="center")
ax.text(-1.0 + 2 * pg + pe, -2.05, rf"$P_e={pe:.2f}$", color=QE, fontsize=FS["small"] - 2, ha="center", va="top")
ax.text(0, -2.42, r"rotation by $|\Omega|\,\Delta t$", fontsize=FS["small"], ha="center", va="top")
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-2.75, 1.35)
ax.set_aspect("equal")
ax.axis("off")
title(ax, "b", "qubit weights")

# (c) the squeezers give the push: control plane (u_A, u_B) in units of g; tau-hat along u_A for real paths ---------
ax = ax_at(9.30, 0.75, 4.3, 4.0)
need = NEED_OM / V_OM                                 # push needed, as a fraction of the largest push v (units of g along tau)
yy = np.linspace(-np.sqrt(1 - need ** 2), np.sqrt(1 - need ** 2), 100)
ax.add_patch(Polygon(np.r_[np.c_[np.full_like(yy, need), yy], np.c_[np.sqrt(1 - yy[::-1] ** 2), yy[::-1]]], closed=True,
                     fc=CSC, alpha=0.15, ec="none"))
ax.add_patch(Circle((0, 0), 1, fc="none", ec=AXC, lw=2.2))
arrow(ax, (-1.14, 0), (1.26, 0), AXC, lw=1.1, ms=14, z=2)            # u_A axis
arrow(ax, (0, -1.14), (0, 1.20), AXC, lw=1.1, ms=14, z=2)            # u_B axis
ax.text(1.28, -0.05, r"$u_A/g$", color=AXC, fontsize=FS["small"], ha="left", va="top")
ax.text(-0.07, 1.17, r"$u_B/g$", color=AXC, fontsize=FS["small"], ha="right", va="center")
ax.text(-0.73, 0.79, r"$|\bm u|\leq g$", color=AXC, fontsize=FS["small"], ha="right", va="bottom")
arrow(ax, (-0.92, 0.13), (-0.50, 0.13), PUSH, lw=2.0, ms=16)          # direction of the path tangent
ax.text(-0.71, 0.19, r"$\hat{\bm\tau}$", color=PUSH, fontsize=FS["ann"], ha="center", va="bottom")
ax.plot([need, need], [-np.sqrt(1 - need ** 2), np.sqrt(1 - need ** 2)], color=CSC, lw=2.4, ls=(0, (4, 2)))
ug = np.array([np.cos(np.deg2rad(-20)), np.sin(np.deg2rad(-20))])
ue = np.array([np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60))])
uef = pg * ug + pe * ue
ax.plot(*np.c_[ug, ue], color=PUSH, lw=1.6, ls=(0, (1, 1.8)))
ax.plot(*ug, "o", ms=13, color=QG, zorder=6)
ax.plot(*ue, "o", ms=13, color=QE, zorder=6)
ax.plot(*uef, "s", ms=13, color=PUSH, zorder=7)
ax.plot([uef[0], uef[0]], [uef[1], 0], color=PUSH, lw=1.4, ls=(0, (1, 1.5)))
arrow(ax, (0, 0), (uef[0], 0), PUSH, lw=3.4, ms=22, z=5)
ax.plot(1, 0, "*", ms=22, color=CSC, mec="k", mew=1.0, zorder=8)
ax.text(ug[0] + 0.05, ug[1] - 0.06, r"$\bm u_g$", color=QG, fontsize=FS["ann"], ha="left", va="top")
ax.text(ue[0] + 0.06, ue[1] + 0.02, r"$\bm u_e$", color=QE, fontsize=FS["ann"], ha="left", va="bottom")
ax.annotate(r"$\bm u_{\rm eff}$", xy=uef, xytext=(0.36, 0.44), color=PUSH, fontsize=FS["ann"], ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=PUSH, lw=1.2, shrinkA=6, shrinkB=9))
ax.text(0.04, -0.07, "push", fontsize=FS["small"], ha="left", va="top")
ax.text(1.06, 0.08, r"best: $v$", color=CSC, fontsize=FS["small"], ha="left", va="bottom")
ax.text(need - 0.03, -1.02, r"needed: $\theta/T$", color=CSC, fontsize=FS["small"], ha="center", va="top")
ax.set_xlim(-1.15, 1.55)
ax.set_ylim(-1.30, 1.25)
ax.set_aspect("equal")
ax.axis("off")
title(ax, "c", r"squeezer push")
fig.text(0.5, 1 - 5.70 / H_IN,
         r"$\displaystyle\bm u_{\rm eff}\!\cdot\!\bm\tau\;=\;\frac{\theta}{T}\;\leq\;g|\bm\tau|=v,\qquad"
         r"\bm u_{\rm eff}=P_g\bm u_g+P_e\bm u_e,\qquad \tau_{A,B}=\mathrm{Re}\,\langle\dot p|{-i}K_{A,B}|p\rangle$",
         ha="center", va="center", fontsize=FS["ann"] + 2)

# (d, e) the capacity along the paths -----------------------------------------------------------------------
TC = {"fock2": S.ROYALBLUE, "fock4": S.INK, "cat": S.PURPLE, "gkp": S.NAVY}
LBL = {"fock2": r"Fock $|2\rangle$", "fock4": r"Fock $|4\rangle$, binomial", "cat": r"cat $|C_2^+\rangle$",
       "gkp": r"GKP $|0_L\rangle$"}
YMAX = 2.7                                          # room for the three-row keys above the curves (cat: up to 1.84)
DASH = (0, (4, 2))
axd = ax_at(1.25, 6.85, 5.55, 3.85)
hd = {}
for name in ("fock2", "fock4", "cat", "gkp"):
    hd[name], = axd.plot(res[name]["s"], res[name]["geodesic"], "-", color=TC[name], lw=M.LW, label=LBL[name], clip_on=False,
                   zorder=4)
    if name in ("cat", "gkp"):                        # Fock and binomial: the photon-optimal path is the brachistochrone
        axd.plot(res[name]["s"], res[name]["photon"], ls=DASH, color=TC[name], lw=M.LW - 1.2, zorder=4)
axd.axhline(1, color=S.INK, lw=1.6, ls=(0, (1, 1.8)), zorder=5)
axd.set_xlabel(r"path fraction $s=\vartheta/L$")
axd.set_ylabel(r"capacity $v/\omega$")
axd.set_ylim(-0.04, YMAX)
axd.set_xlim(0, 1)
ORD = ("fock2", "cat", "gkp", "fock4")              # two columns: targets | Fock 4 and the line-style key
axd.legend([hd[n] for n in ORD] + [Line2D([], [], color=S.INK, lw=M.LW - 1.2),
                                   Line2D([], [], color=S.INK, lw=M.LW - 1.2, ls=DASH)],
           [LBL[n] for n in ORD] + ["brachistochrone", "photon optimum"],
           loc="upper left", fontsize=FS["small"], ncol=2, columnspacing=1.0, handlelength=1.6, labelspacing=0.3)
axd.text(0.0, 1.04, r"(d)\enspace preparation", transform=axd.transAxes, fontsize=FS["title"], ha="left",
         va="bottom")
axe = ax_at(8.15, 6.85, 5.55, 3.85)
for lab, sty, name in (("geodesic", S.PATH_STYLE["geodesic"], "brachistochrone"),
                       ("tradeoff", S.PATH_STYLE["tradeoff"], r"trade-off, $w^*=0.786$"),
                       ("photon", S.PATH_STYLE["photon"], "photon optimum")):
    axe.plot(res["flip"]["s"], res["flip"][lab], **dict(sty, lw=M.LW), label=name + rf", mean ${res['flip']['mean_' + lab]:.2f}$")
axe.axhline(1, color=S.INK, lw=1.6, ls=(0, (1, 1.8)))
sF, vF = np.asarray(res["flip"]["s"]), np.asarray(res["flip"]["photon"])
vvac = float(np.interp(S_VAC, sF, vF))
axe.plot(S_VAC, vvac, "o", ms=13, mfc="white", mec=S.INK, mew=2.2, zorder=8)
axe.annotate("vacuum", (S_VAC, vvac), xytext=(0.20, 1.27), color=S.INK, fontsize=FS["small"], ha="center", va="center",
             arrowprops=dict(arrowstyle="-", color=S.INK, lw=1.2, shrinkA=4, shrinkB=8))
axe.text(0.49, 1.05, r"along $|2\rangle$", color=S.CRIMSON, fontsize=FS["small"], ha="center", va="bottom")
axe.text(0.99, 1.04, r"$v=\omega$", color=S.INK, fontsize=FS["small"], ha="right", va="bottom")
axe.set_xlabel(r"path fraction $s=\vartheta/L$")
axe.set_ylabel(r"capacity $v/\omega$")
axe.set_ylim(-0.04, YMAX)
axe.set_xlim(0, 1)
axe.legend(loc="upper left", fontsize=FS["small"], handlelength=2.2)
axe.text(0.0, 1.04, r"(e)\enspace $|0_L\rangle\to|1_L\rangle$", transform=axe.transAxes, fontsize=FS["title"], ha="left",
         va="bottom")
S.save(fig, HERE / HERE.name)
i_vac = int(np.argmin(np.abs(sF - S_VAC)))
print("wrote", HERE / f"{HERE.name}.png", f"| GKP v/omega={V_OM:.3f}, needed at 412 ns={NEED_OM:.3f}, "
      f"u_eff={uef.round(3)}, push/v={uef[0]:.3f} vs needed/v={need:.3f}; max cat photon v/omega="
      f"{max(res['cat']['photon']):.2f}; flip photon v/omega at s={sF[i_vac]:.3f}: {vF[i_vac]:.3f}, "
      f"s in [0.375, 0.6]: {vF[(sF >= 0.375) & (sF <= 0.6)].min():.3f}-{vF[(sF >= 0.375) & (sF <= 0.6)].max():.3f}")
