"""Figure of the standalone flip-coincidence note (manuscript_review/flip_coincidence/note.tex): the content of S28.png
re-laid out for a text-width float on an A4 or letter page.

S28.png (plot.py) is one row of three panels drawn 17.8 in wide, which prints at 0.36 of its size at a text width of
6.5 in (annotations below 8 pt).  This version stacks the panels in two rows, drawn 13 in wide so that it prints at
one half, like every manuscript figure (mfcommon: drawn at twice the printed width), and moves the legend next to the
Bloch sphere.  Differences in content from S28.png:
  (a) the thin lines are the six cardinal states of the ORTHONORMAL basis {|0_L>, |1_L^perp>} (an exact 2-design), so
      the crimson dashed line is exactly their mean, instead of the SI cardinal states built from the non-orthogonal
      logical states (whose mean deviates from nbar_C by up to 0.003, see the note)
  (c) the point on the rotation axis is the eigenstate |-i> of Y_L
  legend: the three cases A, B, C of the note
Data: manuscript_review/flip_coincidence/num_facts.json and num_curves.npz (num_core.py).  Writes S28_note.png and
caption_note.tex (one line, every number from num_facts.json) into this folder.  PNG only.
  python figures/SFig18_flip_coincidence/plot.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import math
import pathlib
import re
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import mfcommon as M  # noqa: E402

S = M.setup()
import matplotlib  # noqa: E402
from matplotlib import patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

FS = M.FS
SRC = M.ROOT / "manuscript_review" / "flip_coincidence"
F = json.loads((SRC / "num_facts.json").read_text())
Z = np.load(SRC / "num_curves.npz")


def val(sec, key):
    v = F[sec][key]
    return v["value"] if isinstance(v, dict) and "value" in v else v


nC, b, th, TS = float(Z["nC"]), np.asarray(Z["b"], float), float(Z["theta"]), float(Z["TS_ns"])
# ---------------------------------------------------------------- checks of the numbers the figure and caption rely on
assert abs(nC - 2.658) < 5e-4 and abs(float(Z["VC"]) - 9.60) < 5e-3 and abs(TS - 293.5) < 0.1
FL = val("C", "flips")
SUM = val("C", "summary")
assert SUM["max_abs_rel_dev_E"] < 1e-13 and SUM["min_T_over_Tstar"] > 1 - 1e-12 and SUM["max_end_map_residual"] < 1e-12
assert SUM["max_dH0_over_omega"] < 1 + 1e-12 and SUM["n_flips"] == len(FL)
TILT = val("D", "tilted")
VAC = val("E", "vacuum_transfer")
assert abs(VAC["T_ns"] - 397.2) < 0.1 and abs(VAC["J_0L_ns"] - 359.3) < 0.1 and abs(VAC["F_1to0"] - 0.599) < 1e-3
FSM = val("F", "transfer_0L_to_1L_smooth")
assert FSM["hessian_min_eig"] > 0 and FSM["min_J_minus_Jgc"] > -1e-12          # great circle = single-input optimum in C
FDR = val("F", "discrete_optimum_resampled")
assert FDR["minus_Jgc_exact"] > 0                                              # the discrete best is not below it
t03 = next(r for r in TILT if abs(r["lam"] - 0.3) < 1e-9)
labs_o = ["0", "1p", "+", "-", "+i", "-i"]                                     # order of octahedron() in num_core.py
NBO = np.asarray(Z["nb_octa"])
assert NBO.shape[0] == 6 and np.abs(NBO.mean(0) - np.asarray(Z["octa_avg"])).max() < 1e-14
assert np.abs(NBO.mean(0) - nC).max() < 1e-13                                  # the exact 2-design average is flat
assert np.abs(NBO[4] - nC).max() < 1e-12 and np.abs(NBO[5] - nC).max() < 1e-12  # |+-i> (eigenstates of Y_L) do not move
nlo, nhi = nC - np.linalg.norm(b), nC + np.linalg.norm(b)
assert abs(NBO.min() - nlo) < 1e-6 and abs(NBO.max() - nhi) < 1e-6

GEO = dict(color=S.ROYALBLUE, ls="-", lw=9.0, alpha=0.32, solid_capstyle="butt")
PH = dict(color=S.CRIMSON, ls=(0, (3.2, 1.8)), lw=2.8)
THIN = dict(color=S.INK, lw=2.0)
MK_ON = dict(ls="none", mfc="white", mec=S.INK, mew=2.0)
STAR = dict(ls="none", marker="*", ms=19, mec=S.INK, mew=1.6)
DIAM = dict(marker="D", ms=9.5, color=S.INK, mfc=S.INK, mec="white", mew=1.0)
halo = [patheffects.withStroke(linewidth=5, foreground="white")]

W, H = 13.0, 10.3                                                # drawn size (in); printed at 6.5 in = one half
fig = plt.figure(figsize=(W, H))


def box(l_in, b_in, w_in, h_in):
    return [l_in / W, b_in / H, w_in / W, h_in / H]


axa = fig.add_axes(box(1.20, 6.05, 4.35, 3.55))
axb = fig.add_axes(box(7.10, 6.05, 5.75, 3.55))
axc = fig.add_axes(box(0.55, 0.10, 4.55, 4.55))
cax = fig.add_axes(box(5.05, 0.95, 0.20, 3.00))
axl = fig.add_axes(box(6.55, 0.35, 6.40, 4.10))
axl.axis("off")

# ================================================================= (a) photon number along the flip brachistochrone
t = np.asarray(Z["tB_ns"])
KETS = {"0": r"$|0_L\rangle$", "1p": r"$|1_L^\perp\rangle$", "+": r"$|{+}\rangle$", "-": r"$|{-}\rangle$"}
for lab, y in zip(labs_o, NBO):
    axa.plot(t, y, zorder=3 if lab not in ("+i", "-i") else 1, **THIN)
axa.plot(t, np.full_like(t, nC), zorder=2, **GEO)
axa.plot(t, np.asarray(Z["octa_avg"]), zorder=4, **PH)
for lab, y in zip(labs_o, NBO):                                 # labels at the extremes of each curve
    if lab in ("0", "-"):
        k = int(np.argmin(y))
        axa.annotate(KETS[lab], (t[k], y[k]), xytext=(0, -14), textcoords="offset points", ha="center", va="top",
                     fontsize=FS["ann"])
    elif lab in ("1p", "+"):
        k = int(np.argmax(y))
        axa.annotate(KETS[lab], (t[k], y[k]), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
                     fontsize=FS["ann"])
axa.annotate(r"$|{\pm i}\rangle$", (0.5 * TS, nC), xytext=(0, 14), textcoords="offset points", ha="center",
             va="bottom", fontsize=FS["ann"], path_effects=halo)
axa.text(0.03 * TS, nC + 0.014, r"$\bar n_{\mathcal C}$", fontsize=FS["ann"], ha="left", va="bottom", path_effects=halo)
axa.set_xlim(0, TS)
axa.set_ylim(2.20, 3.12)
axa.set_xticks([0, 100, 200, round(TS)])
axa.set_yticks([2.4, 2.6, 2.8, 3.0])
axa.set_xlabel(r"time $t$ (ns)")
axa.set_ylabel(r"photon number $\bar n$")
axa.set_title(r"brachistochrone $\omega Y_L$", fontsize=FS["title"], pad=10)
S.panel_label(axa, "a", x=-0.27, y=1.03)

# ================================================================= (b) averaged exposure against duration
TMAX = 610.0
TT = np.linspace(TS, TMAX, 50)
axb.plot(TT, nC * TT, color=S.INK, lw=2.6, zorder=2)
TT0 = np.linspace(272, TS, 10)
axb.plot(TT0, nC * TT0, color=S.INK, lw=1.6, ls=(0, (1, 2.2)), zorder=2)
axb.axvline(TS, color=S.INK, lw=1.6, ls=(0, (5, 3)), zorder=1)
axb.text(TS + 5, 1625, r"$T^*$", fontsize=FS["ann"], ha="left", va="top")
TCUT = 303.0                                                     # flips with T* < T <= TCUT go to the inset
on = [f for f in FL if f.get("family") in ("slower", "wiggle", "detour") and f["T_ns"] <= TMAX
      and f["T_over_Tstar"] > 1.0 + 1e-9]
main = [f for f in on if f["T_ns"] > TCUT]
near = [f for f in on if f["T_ns"] <= TCUT]
MARK = {"slower": ("o", 11), "wiggle": ("^", 9.5), "detour": ("s", 8.5)}
for fam, (mk, ms) in MARK.items():
    sel = [f for f in main if f["family"] == fam]
    if sel:
        axb.plot([f["T_ns"] for f in sel], [f["E_avg_ns"] for f in sel], marker=mk, ms=ms, zorder=5, **MK_ON)
# the brachistochrone: time optimal (blue disk) = photon optimal (crimson dashed ring)
axb.plot([TS], [nC * TS], "o", ms=22, color=S.ROYALBLUE, alpha=0.40, mec="none", zorder=6)
axb.scatter([TS], [nC * TS], s=180, facecolors="none", edgecolors=S.CRIMSON, linewidths=2.8, linestyles=(0, (2.2, 1.6)),
            zorder=7)
# case C: exact flips that tilt the code space and return.  The line starts at the brachistochrone (lambda = 0) and runs
# through every lambda; the diamonds mark lambda >= 0.1 (lambda = 0.05 would sit under the disk)
tl = [r for r in TILT if r["lam"] >= 0]
assert abs(tl[0]["lam"]) < 1e-12 and abs(tl[0]["E_avg_ns"] - nC * TS) < 1e-9
axb.plot([r["T_ns"] for r in tl], [r["E_avg_ns"] for r in tl], color=S.INK, lw=1.6, zorder=4)
tlm = [r for r in tl if r["lam"] > 0.1 - 1e-9]
axb.plot([r["T_ns"] for r in tlm], [r["E_avg_ns"] for r in tlm], ls="none", zorder=4, **DIAM)
axb.annotate(r"$\lambda=0.3$", (t03["T_ns"], t03["E_avg_ns"]), xytext=(348, 468), textcoords="data",
             fontsize=FS["small"], ha="center", va="top",
             arrowprops=dict(arrowstyle="-", color=S.INK, lw=1.4, shrinkA=2, shrinkB=6))
# case B: the generators of the transfer of |0_L> through the vacuum
axb.plot([VAC["T_ns"]] * 2, [VAC["J_0L_ns"], VAC["avg_over_code_space_ns"]], color=S.INK, lw=1.4, ls=(0, (1, 2)),
         zorder=3)
axb.plot([VAC["T_ns"]], [VAC["avg_over_code_space_ns"]], mfc=S.INK, zorder=5, **STAR)
axb.plot([VAC["T_ns"]], [VAC["J_0L_ns"]], mfc="white", zorder=5, **STAR)
axb.annotate(r"$|0_L\rangle$ alone", (VAC["T_ns"], VAC["J_0L_ns"]), xytext=(15, 0), textcoords="offset points",
             fontsize=FS["small"], ha="left", va="center")
axb.annotate("averaged", (VAC["T_ns"], VAC["avg_over_code_space_ns"]), xytext=(0, 15), textcoords="offset points",
             fontsize=FS["small"], ha="center", va="bottom")
axb.annotate(r"$\bar n_{\mathcal C}T$", (545, nC * 545), xytext=(-12, 10), textcoords="offset points",
             fontsize=FS["ann"], ha="right", va="bottom")
axb.set_xlim(272, TMAX)
axb.set_ylim(250, 1650)                                          # room below the open star of |0_L> alone
axb.set_xticks([300, 400, 500, 600])
axb.set_yticks([500, 1000, 1500])
axb.set_xlabel(r"duration $T$ (ns)")
axb.set_ylabel(r"$\int\bar n\,dt$ (photon ns)")
axb.set_title(r"averaged over the code space", fontsize=FS["title"], pad=10)
S.panel_label(axb, "b", x=-0.235, y=1.03)
# inset: every flip within 3% of T* (two-leg detours and the gentlest random evolutions)
ins = axb.inset_axes([0.635, 0.215, 0.34, 0.31])
xi = np.linspace(TS, TCUT + 0.5, 10)
ins.plot(xi, nC * xi, color=S.INK, lw=2.0, zorder=2)
for fam, (mk, ms) in MARK.items():
    sel = [f for f in near if f["family"] == fam]
    if sel:
        ins.plot([f["T_ns"] for f in sel], [f["E_avg_ns"] for f in sel], marker=mk, ms=ms - 1.5, zorder=5, **MK_ON)
ins.plot([TS], [nC * TS], "o", ms=20, color=S.ROYALBLUE, alpha=0.40, mec="none", zorder=6)
ins.scatter([TS], [nC * TS], s=150, facecolors="none", edgecolors=S.CRIMSON, linewidths=2.2, linestyles=(0, (2.0, 1.4)),
            zorder=7)
ins.set_xlim(TS - 1.6, TCUT + 0.6)
ins.set_ylim(nC * (TS - 1.6), nC * (TCUT + 0.6))
ins.set_xticks([295, 300])
ins.set_yticks([785, 800])
ins.tick_params(labelsize=FS["small"] - 2, length=6, width=1.8, pad=3)
for sp in ins.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(1.6)
ins.set_title(r"near $T^*$", fontsize=FS["small"] - 1, pad=4)

# ================================================================= (c) Bloch sphere of the code space
AZ = math.radians(38.0)
v = np.array([math.sin(AZ), -math.cos(AZ), 0.0])               # towards the viewer
ey = np.array([0, 0, 1.0])
ex = np.cross(ey, v)
npx = 601
X, Y = np.meshgrid(np.linspace(-1, 1, npx), np.linspace(-1, 1, npx))
R2 = X ** 2 + Y ** 2
Zd = np.sqrt(np.clip(1 - R2, 0, 1))
P = X[..., None] * ex + Y[..., None] * ey + Zd[..., None] * v
NB3 = nC + P @ b
VMIN, VMAX = 2.30, 3.00                                          # the range of nbar on C is 2.353 ... 2.962
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("ncmap_light", S.NCMAP(np.linspace(0, 0.72, 256)))
rgba = CMAP(np.clip((NB3 - VMIN) / (VMAX - VMIN), 0, 1))
rgba[..., :3] *= (0.88 + 0.12 * Zd)[..., None]
rgba[..., 3] = (R2 <= 1).astype(float)
axc.imshow(rgba, origin="lower", extent=[-1, 1, -1, 1], interpolation="bilinear", zorder=0)
CONTOUR = "#8E2C48"
cs = axc.contour(X, Y, np.where(R2 <= 0.995, NB3, np.nan), levels=[2.4, 2.6, 2.8], colors=[CONTOUR], linewidths=1.6,
                 alpha=0.75, zorder=1)
tt_ = np.linspace(0, 2 * np.pi, 400)
axc.plot(np.cos(tt_), np.sin(tt_), color=S.INK, lw=2.6, zorder=2)
eq = np.array([[math.cos(a), math.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 400)])   # equator, front half


def proj(q):
    q = np.atleast_2d(q)
    return q @ ex, q @ ey, q @ v


ex_, ey_, dz_ = proj(eq)
front = np.where(dz_ >= 0, 1.0, np.nan)
axc.plot(ex_ * front, ey_ * front, color=S.INK, lw=1.2, ls=(0, (4, 3)), alpha=0.7, zorder=2)
# brachistochrone of |0_L> (great circle, analytic band) and the single-input photon optimum within C (numerical)
ph_ = np.linspace(0, 2 * th, 300)
gc = np.array([np.sin(ph_), 0 * ph_, np.cos(ph_)]).T
gx, gy, gz = proj(gc)
assert np.all(gz >= -1e-12)
band, = axc.plot(gx, gy, zorder=4, **GEO)
band.set_clip_path(matplotlib.patches.Circle((0, 0), 1.0, transform=axc.transData))
Fb = np.asarray(Z["F_best"])
fx, fy, _ = proj(Fb)
axc.plot(fx, fy, zorder=5, **PH)
# another input under the same gate: a small circle about the Y_L axis
Gg = np.asarray(Z["G_gate"])
qx, qy, qz = proj(Gg)
assert np.all(qz > 0)
axc.plot(qx, qy, zorder=3, **THIN)
axc.annotate("", xy=(qx[-1], qy[-1]), xytext=(qx[-12], qy[-12]),
             arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=2.0, mutation_scale=22), zorder=3)
# marked states
pts = {r"$|0_L\rangle$": (np.array([0, 0, 1.0]), (0, 14), "center", "bottom"),
       r"$|1_L\rangle$": (np.asarray(Z["r1"]), (0, -14), "center", "top"),
       r"$|{+_L}\rangle$": (np.array([math.sin(th), 0, math.cos(th)]), (-16, 4), "right", "bottom"),
       r"$|{-i}\rangle$": (np.array([0, -1.0, 0]), (0, -12), "center", "top"),
       r"$|\psi\rangle$": (Gg[0], (10, 6), "left", "bottom")}
for lab, (q, off, ha, va) in pts.items():
    x0, y0, _ = proj(q)
    axc.plot(x0, y0, "o", ms=10, color=S.INK, mec="white", mew=1.5, zorder=8)
    axc.annotate(lab, (x0[0], y0[0]), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=FS["ann"], zorder=9,
                 path_effects=halo)
rmin = -b / np.linalg.norm(b)
x0, y0, _ = proj(rmin)
axc.plot(x0, y0, marker="v", ms=12, color=S.INK, mec="white", mew=1.2, zorder=8, ls="none")
axc.annotate(r"$\bar n_{\min}$", (x0[0], y0[0]), xytext=(-12, 2), textcoords="offset points", ha="right", va="center",
             fontsize=FS["small"], path_effects=halo, zorder=9)
axc.clabel(cs, fmt=r"$%.1f$", fontsize=FS["small"] - 3, inline=True, inline_spacing=4, colors=[CONTOUR])
axc.set_xlim(-1.08, 1.08)
axc.set_ylim(-1.24, 1.26)
axc.set_aspect("equal", adjustable="box", anchor="W")
axc.axis("off")
axc.set_title(r"code space $\mathcal C$", fontsize=FS["title"], pad=8)
S.panel_label(axc, "c", x=-0.02, y=1.0)
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(VMIN, VMAX))
cb = fig.colorbar(sm, cax=cax)
cb.set_ticks([2.4, 2.6, 2.8, 3.0])
cb.set_label(r"photon number $\bar n$", fontsize=FS["label"])
cb.ax.tick_params(labelsize=FS["tick"])
axc.apply_aspect()
pc = axc.get_position()
cax.set_position([pc.x1 + 0.10 / W, pc.y0 + 0.17 * pc.height, 0.20 / W, 0.66 * pc.height])

# ================================================================= legend (the cases A, B, C of the note)
hs = [Line2D([], [], **GEO), Line2D([], [], **PH),
      tuple(Line2D([], [], marker=mk, ms=ms, **MK_ON) for mk, ms in MARK.values()),
      Line2D([], [], mfc=S.INK, **STAR),
      Line2D([], [], lw=1.6, **DIAM),
      Line2D([], [], **THIN)]
names = ["time optimal", "photon optimal", r"A, keeps the qubit encoded", r"B, transfer of $|0_L\rangle$, not a gate",
         r"C, exact flip leaving $\mathcal C$", r"single inputs under $\omega Y_L$"]
axl.legend(hs, names, loc="center left", bbox_to_anchor=(0.0, 0.5), ncol=1, fontsize=FS["legend"], labelspacing=0.85,
           handlelength=2.6, handletextpad=0.7, borderaxespad=0.0, frameon=False,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)})
fig.savefig(HERE / "S28_note.png", dpi=300)
plt.close(fig)

# ================================================================= caption (one line), every number from num_facts.json
A80 = val("A", "D80")
avg_dev = val("B", "nbar_octa_avg_maxdev")
n_all = SUM["n_flips"]
n_shown = len(on)
n_frame = sum(1 for f in FL if f.get("family") == "S1frame")
assert n_frame == 6 and all(abs(f["T_over_Tstar"] - 1) < 1e-12 for f in FL if f.get("family") == "S1frame")
lin = val("C", "summary")["max_abs_rel_dev_E"]
WORDS = ["zero", "one", "two", "three", "four", "five", "six"]
cap = (r"\caption{Numerical check at $\Delta=0.4$ (Fock cutoff 80), with the cases A, B and C of Sec.~\ref{sec:cases}. "
       r"(a)~Photon number along the flip brachistochrone $\omega Y_L$ for the six cardinal states of the orthonormal "
       r"basis (thin lines) and their mean (crimson dashed), which equals the average over the code space and stays "
       rf"within $10^{{{math.ceil(math.log10(avg_dev))}}}$ of $\bar n_{{\mathcal C}}={nC:.3f}$ (blue band), whereas "
       rf"the single inputs vary between {A80['nbar_min_on_C']:.3f} and {A80['nbar_max_on_C']:.3f}. "
       r"(b)~Exposure $\int\bar n\,dt$ averaged over the code space against the duration $T$. "
       rf"All {n_all} flips of case A that we propagated lie on the line $\bar n_{{\mathcal C}}T$ within a relative "
       rf"$10^{{{math.ceil(math.log10(lin))}}}$, and none is faster than $T^*={TS:.0f}$~ns. The plotted range holds "
       rf"{n_shown} of them, slower brachistochrones (circles), random smooth evolutions within the code space "
       rf"(triangles) and two-leg detours (squares), with those within {100 * (TCUT / TS - 1):.0f}\% of $T^*$ in the "
       r"inset. The blue disk with the crimson dashed ring marks the brachistochrone, the common minimum of the "
       rf"duration and of the averaged exposure, and {WORDS[n_frame]} flips that differ from it by a frame rotation. "
       r"The tilted flips of case C (diamonds at $\lambda=0.1$ to 0.45, on a line from "
       rf"$\lambda=0$) fall below the line, and at $\lambda=0.3$ ({t03['T_ns']:.0f}~ns) their averaged exposure is "
       rf"{100 * t03['saving_vs_brach']:.0f}\% below that of the brachistochrone. The generators of the "
       rf"photon-optimal transfer of case B (stars, {VAC['T_ns']:.0f}~ns) give $\ket{{0_L}}$ alone an exposure of "
       rf"{VAC['J_0L_ns']:.0f}~photon~ns (open star) and {VAC['avg_over_code_space_ns']:.0f}~photon~ns averaged over "
       rf"the code space (filled star), but they map $\ket{{1_L}}$ to $\ket{{0_L}}$ with fidelity "
       rf"{VAC['F_1to0']:.2f} only and are not a gate. "
       r"(c)~Bloch sphere of the code space shaded by $\bar n=\bar n_{\mathcal C}+\bm b\cdot\bm r$. The brachistochrone "
       r"carries $\ket{0_L}$ along a great circle (blue band) through the state with the fewest photons "
       r"($\bar n_{\min}$) and through $\ket{+_L}$ at $T^*/2$, and the photon-optimal path of $\ket{0_L}$ alone within "
       r"the code space, found by a numerical search (crimson dashed), coincides with it. Another input $\ket\psi$ "
       r"moves on a small circle about the $Y_L$ axis, which passes through $\ket{-i}$.}" + "\n"
       + r"\label{fig:check}" + "\n")
body = re.sub(r"\\(ref|label)\{[^}]*\}", "", cap.replace(r"\caption{", ""))
body_nomath = "".join(body.split("$")[0::2])                    # prose only
for bad in (":", ";", "i.e.", "e.g.", "code word", "codeword", "code-averaged", "code-preserving", "code state"):
    assert bad not in body_nomath, bad
(HERE / "caption_note.tex").write_text(cap)
print(cap)
print("wrote", HERE / "S28_note.png")
print("points on the line in range:", n_shown, "of", n_all, "| inset", len(near), "| main", len(main),
      "| tilt 0.3 saving", t03["saving_vs_brach"], "| vacuum", VAC["T_ns"], VAC["J_0L_ns"], VAC["avg_over_code_space_ns"])
