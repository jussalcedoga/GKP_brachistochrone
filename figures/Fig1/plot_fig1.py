"""Fig. 1: the time-optimal and photon-optimal paths in state space, and the states along the brachistochrone.
(a, b) one common sphere (Nov 9): the unit sphere of the real subspace span{|0>, |0_L>, |1_L>} (fig1_sphere.json, written
by compute_sphere.py), seen from the same direction in both panels, so they share the shading and contours (<n> of every
state of the sphere) and the positions of the three states; on it the brachistochrone, the photon-optimal path and
(flip) the trade-off path (normalized projections, smallest weight in the subspace 0.973); the five snapshot times of
(c, d) are marked on the brachistochrone, numbered, filled with growing opacity 1 -> 5.
(c, d) Wigner functions of the brachistochrone state at t = 0, T*/4, T*/2, 3T*/4, T*, with the target fidelity
F_T = |<T|p(t)>|^2 = cos^2(theta_0 - omega t).
Reads manuscript/fig1_brachistochrone/results (fig1.json; of fig1_wigner.npz only the grid and the snapshot fractions) and
the path durations of fig2.json/fig3.json; writes only into this folder.  The Wigner functions of (c, d) are recomputed
here for the path states of simulations/paths/fig1_brachistochrone.py, exact on the shown window (mfcommon.wigner_grid;
the stored arrays drop the |1_L> peaks at q = +-5.32, review Nov 1, L1-F1), and saved to fig1_wigner.npz in this folder.
python figures/Fig1/plot_fig1.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))
import mfcommon as M  # noqa: E402,I001

M.FS.update({k_: round(1.25 * v_) for k_, v_ in M.FS.items()})   # Nov 4 (user): 1.25x fonts (drawn at 14 in, printed 3.4 in)
S = M.setup()
import matplotlib  # noqa: E402
from matplotlib import patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

FS = M.FS
SRC = M.MS / "fig1_brachistochrone" / "results"
R = json.loads((SRC / "fig1.json").read_text())
RS = json.loads((HERE / "fig1_sphere.json").read_text())   # (a, b) on one sphere, span{|0>, |0_L>, |1_L>} (compute_sphere.py)


def wigner_snapshots():
    """W(q, p) of the brachistochrone states at the stored fractions t/T* (states as in compute.py, cutoff 80), exact on
    the stored grid |q|, |p| <= 5 (161 points); saved to fig1_wigner.npz next to this script"""
    from common import brachistochrone as B, states
    from common.params import GKP_DELTA
    Z0 = np.load(SRC / "fig1_wigner.npz")                  # grid and fractions only
    D = 80
    vac, g0, g1 = states.vacuum(D), states.gkp(GKP_DELTA, 0, D), states.gkp(GKP_DELTA, 1, D)
    Wg = M.wigner_grid(D, len(Z0["x"]), float(Z0["x"][-1]))
    assert np.allclose(Wg.x, Z0["x"])
    out = dict(x=Wg.x, fr=np.array(Z0["fr"]))
    for name, (S0, T0) in (("gkp", (vac, g0)), ("flip", (g0, g1))):
        Ps = B.path_states(S0, T0, out["fr"] * R["transfers"][name]["theta0"])
        out[name] = np.array([Wg.of_vectors(p[None]) for p in Ps])
    np.savez_compressed(HERE / "fig1_wigner.npz", **out)
    return out


Wz = wigner_snapshots()
TR = R["transfers"]
OM = R["params"]["omega"]
NMAX = 8.0
LEVELS = [1, 2, 3, 4, 5, 6]
_F2 = json.loads((M.MS / "fig2_photon_optimal" / "results" / "fig2.json").read_text())
_F3 = json.loads((M.MS / "fig3_time_vs_photon" / "results" / "fig3.json").read_text())
T_TRUE = {("sphere_prep", "geodesic"): _F2["prep"]["gkp"]["geodesic"]["T_ns"],
          ("sphere_prep", "photon"): _F2["prep"]["gkp"]["photon"]["T_ns"],
          ("sphere_flip", "geodesic"): _F2["flip"]["geodesic"]["T_ns"],
          ("sphere_flip", "photon"): _F2["flip"]["photon"]["T_ns"],
          ("sphere_flip", "tradeoff"): 1e3 * _F3["tradeoff"]["flip"]["eps"]["Yale"]["at_w_star"]["L"] / OM}
STY = {"geodesic": dict(color=S.ROYALBLUE, lw=6.0, ls="-"), "photon": dict(color=S.CRIMSON, lw=5.4, ls=(0, (3.5, 1.4))),
       "tradeoff": dict(color=S.TRADEOFF, lw=5.0, ls=(0, (5, 1.6, 1.4, 1.6)))}
LAB = {"geodesic": "time optimal (brachistochrone)", "photon": "photon optimal", "tradeoff": r"trade-off ($w^*$)"}
CONTOUR = "#8E2C48"                 # photon-number contours: dark rose of the crimson (photon) family, no gray
KET = dict(boxstyle="round,pad=0.25", fc="white", ec=S.INK, lw=1.3, alpha=0.95)
ALPHAS = (0.15, 0.35, 0.55, 0.78, 1.0)
NUM = ("1", "2", "3", "4", "5")


def along(xyz, key, ref):
    q = np.asarray(xyz)
    ang = np.arccos(np.clip(np.sum(q[:-1] * q[1:], 1), -1, 1))
    tt = 1e3 * np.r_[0, np.cumsum(ang)] / OM
    return tt * T_TRUE[(key, ref)] / tt[-1]


def at_times(xyz, times, key, ref):
    q = np.asarray(xyz)
    tt = along(q, key, ref)
    pts = np.array([np.interp(times, tt, q[:, i]) for i in range(3)]).T
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def badge(ax, x, y, j, transform, size):
    a = ALPHAS[j]
    ax.text(x, y, NUM[j], transform=transform, ha="center", va="center", fontsize=size, zorder=10,
            color="white" if a >= 0.5 else S.INK, clip_on=False,
            bbox=dict(boxstyle="circle,pad=0.24", fc=(*matplotlib.colors.to_rgb(S.ROYALBLUE), a), ec=S.ROYALBLUE, lw=1.8))


def sphere(ax, key, labels, title, letter, offsets, num_off, alpha_view=0.30, frame=None, extra=()):
    """one panel on the common sphere of fig1_sphere.json (span{|0>, |0_L>, |1_L>}, Nov 9): `frame` (ex, ey, view) fixes
    the orientation, so both panels share it; `extra` lists further labelled states (xyz, label), drawn as dots"""
    d = RS[key]
    N3 = np.array(d["N3"])
    P = {k: np.array(v["xyz"]) for k, v in d["paths"].items()}
    end = P["geodesic"][-1]
    if frame is None:                        # look at the middle of the brachistochrone, tilted toward the departure axis
        mid = P["geodesic"][0] + end
        mid /= np.linalg.norm(mid)
        v = np.cos(alpha_view) * mid + np.sin(alpha_view) * np.array([0, 0, 1.0])
        v /= np.linalg.norm(v)
        ey = np.array([0, 0, 1.0]) - v[2] * v
        ey /= np.linalg.norm(ey)
        ex = np.cross(ey, v)
    else:
        ex, ey, v = (np.asarray(w, float) for w in frame)
    frame_used = (ex.copy(), ey.copy(), v.copy())          # returned (v is reused below for contour vertices)
    n = 701
    X, Y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    R2 = X ** 2 + Y ** 2
    Z = np.sqrt(np.clip(1 - R2, 0, 1))
    p = X[..., None] * ex + Y[..., None] * ey + Z[..., None] * v
    nb = np.einsum("...i,ij,...j->...", p, N3, p)
    rgba = S.NCMAP(np.clip(nb / NMAX, 0, 1))
    rgba[..., :3] *= (0.86 + 0.14 * Z)[..., None]
    rgba[..., 3] = (R2 <= 1).astype(float)
    ax.imshow(rgba, origin="lower", extent=[-1, 1, -1, 1], interpolation="bilinear", zorder=0)
    cs = ax.contour(X, Y, np.where(R2 <= 0.995, nb, np.nan), levels=LEVELS, colors=[CONTOUR], linewidths=1.7, alpha=0.75,
                    zorder=1)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=S.INK, lw=3.0, zorder=2)

    def proj(q):
        q = np.atleast_2d(q)
        return q @ ex, q @ ey

    for k in ("tradeoff", "photon", "geodesic"):
        if k in P:
            xs, ys = proj(P[k])
            ax.plot(xs, ys, solid_capstyle="round", zorder=4, **STY[k])
    pts = [(P["geodesic"][0], labels[0]), (end, labels[1])]
    if labels[2]:
        pts.append((np.array(d["vacuum"]["xyz"]), labels[2]))
    pts += [(np.array(q, float), lab) for q, lab in extra]
    for (q, lab), (off, ha) in zip(pts, offsets):
        x0, y0 = proj(q)
        ax.annotate(lab, (x0[0], y0[0]), xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=FS["label"] + 2,
                    zorder=9, bbox=KET)
    # states that are no end point of this panel's paths: the vacuum in (b), |1_L> in (a)
    dots = ([np.array(d["vacuum"]["xyz"])] if labels[2] else []) + [np.array(q, float) for q, _ in extra]
    for q in dots:
        x0, y0 = proj(q)
        ax.plot(x0, y0, "o", ms=14, color=S.INK, mec="white", mew=1.8, zorder=6)
    Tn = T_TRUE[(key, "geodesic")]
    snaps = at_times(P["geodesic"], np.array(Wz["fr"]) * Tn, key, "geodesic")    # the snapshot times of (c, d)
    sx, sy = proj(snaps)
    for j in range(5):
        ax.plot(sx[j], sy[j], "o", ms=17, mfc=(*matplotlib.colors.to_rgb(S.ROYALBLUE), ALPHAS[j]), mec=S.ROYALBLUE,
                mew=2.6, zorder=7)
        ax.annotate(NUM[j], (sx[j], sy[j]), xytext=num_off[j], textcoords="offset points", ha="center", va="center",
                    fontsize=FS["small"] + 3, color=S.ROYALBLUE, zorder=8,
                    path_effects=[patheffects.withStroke(linewidth=5, foreground="white")])   # legible over contours
    # contour labels placed clear of the paths, the snapshot markers and numbers, the ket boxes and the rim: for each
    # level, the contour point farthest (beyond its clearance radius) from every item; a level with no clear point is
    # left unlabelled
    pos = ax.get_position()
    ppu = min(pos.width * ax.figure.get_figwidth(), pos.height * ax.figure.get_figheight()) * 72 / 2.06   # pt per unit
    avoid = []
    for k in P:
        xs, ys = proj(P[k])
        avoid += [(x, y, 0.10) for x, y in zip(xs, ys)]
    for j in range(5):
        avoid += [(sx[j], sy[j], 0.10), (sx[j] + num_off[j][0] / ppu, sy[j] + num_off[j][1] / ppu, 0.09)]
    for (q, _), (off, ha) in zip(pts, offsets):
        x0, y0 = proj(q)
        dx = {"right": -0.16, "left": 0.16, "center": 0.0}[ha]
        avoid.append((x0[0] + off[0] / ppu + dx, y0[0] + off[1] / ppu, 0.24))
    for q in dots:
        x0, y0 = proj(q)
        avoid.append((x0[0], y0[0], 0.08))
    av = np.array(avoid)
    manual = []
    for path in cs.get_paths():
        v = path.vertices
        if len(v) == 0:
            continue
        v = v[np.sum(v ** 2, 1) < 0.80]                    # inside r < 0.89: the label is not cut by the rim
        if len(v) == 0:
            continue
        a = np.array(av.tolist() + [(mx, my, 0.16) for mx, my in manual]) if manual else av
        dist = np.min(np.hypot(v[:, None, 0] - a[None, :, 0], v[:, None, 1] - a[None, :, 1]) - a[None, :, 2], 1)
        i = int(np.argmax(dist))
        if dist[i] > 0.02:
            manual.append((float(v[i, 0]), float(v[i, 1])))
    ax.clabel(cs, fmt=r"$%d$", fontsize=FS["small"] - 2, inline=True, inline_spacing=5, colors=[CONTOUR], manual=manual)
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.apply_aspect()                                        # the square sphere box, for the title extent below
    tt = ax.text(0.5, 1.01, title, transform=ax.transAxes, fontsize=FS["title"] - 1, ha="center", va="bottom")
    bb = tt.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    x_left = ax.transAxes.inverted().transform((bb.x0, bb.y0))[0]
    ax.text(x_left - 0.02, 1.0, f"({letter})", transform=ax.transAxes, fontsize=FS["panel"], ha="right", va="bottom")
    return frame_used


def wigner_rows(fig, gs_rows):
    fr, xg = Wz["fr"], Wz["x"]
    ext = [xg[0], xg[-1], xg[0], xg[-1]]
    for row, (name, letter, title) in enumerate((("gkp", "c", r"brachistochrone of the preparation $|0\rangle\to|0_L\rangle$"),
                                                 ("flip", "d", r"brachistochrone of the logical flip $|0_L\rangle\to|1_L\rangle$"))):
        sub = gs_rows[row].subgridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 0.05, 0.07], wspace=0.06)
        Tn = TR[name]["T_star_ns"]
        for k in range(5):
            ax = fig.add_subplot(sub[k])
            im = ax.imshow(Wz[name][k], origin="lower", extent=ext, cmap=S.RBCMAP, vmin=-S.W_LIM, vmax=S.W_LIM)
            badge(ax, 0.12, 1.12, k, ax.transAxes, FS["small"])
            ax.text(0.26, 1.12, rf"$t={fr[k] * Tn:.0f}$ ns", transform=ax.transAxes, fontsize=FS["ann"], ha="left",
                    va="center")
            ax.text(0.05, 0.05, rf"$\mathcal{{F}}_T={TR[name]['F_target_snap'][k]:.2f}$", transform=ax.transAxes, fontsize=FS["small"] + 2,
                    va="bottom", ha="left", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.4))
            ax.set_xticks([-4, 0, 4])
            ax.set_yticks([-4, 0, 4])
            ax.tick_params(direction="out", length=8)
            if row == 1:                                   # q on the bottom row only (same axes), clear of the title of (d)
                ax.set_xlabel(r"$q$", labelpad=2, fontsize=FS["label"] + 3)
            if k == 0:
                ax.set_ylabel(r"$p$", labelpad=2, fontsize=FS["label"] + 3)
                S.panel_label(ax, letter, x=-0.40, y=1.29)
                ax.text(-0.06, 1.29, title, transform=ax.transAxes, fontsize=FS["ann"] + 5, ha="left", va="bottom")
            else:
                ax.set_yticklabels([])
            for sp in ax.spines.values():
                sp.set_visible(True)
        cax = fig.add_subplot(sub[6])
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([-S.W_LIM, 0, S.W_LIM])
        cb.set_ticklabels([r"$-1/\pi$", r"$0$", r"$1/\pi$"])
        cb.ax.tick_params(labelsize=FS["tick"])
        cb.set_label(r"$W(q,p)$", fontsize=FS["label"] + 3)


fig = plt.figure(figsize=(14.0, 15.4))
g = fig.add_gridspec(3, 1, height_ratios=[1.78, 1, 1], hspace=0.52, top=0.905, bottom=0.04, left=0.075, right=0.955)
top = g[0].subgridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.05)
tp = [T_TRUE[("sphere_prep", k)] for k in ("geodesic", "photon")]
tf = [T_TRUE[("sphere_flip", k)] for k in ("geodesic", "tradeoff", "photon")]
ax_a, ax_b = fig.add_subplot(top[0]), fig.add_subplot(top[1])
frame_b = sphere(ax_b, "sphere_flip", (r"$|0_L\rangle$", r"$|1_L\rangle$", r"$|0\rangle$"),
                 rf"$|0_L\rangle\to|1_L\rangle$: ${tf[0]:.0f}$, ${tf[1]:.0f}$, ${tf[2]:.0f}$ ns", "b",
                 [((0, -68), "center"), ((0, -68), "center"), ((0, 30), "center")],
                 num_off=[(-4, -28), (0, -28), (0, -28), (0, -28), (4, -28)])
# Nov 9 (author, advisor): (a) on the same sphere as (b), seen from the same direction, so both panels share the contours
# and the positions of |0>, |0_L> and |1_L>
sphere(ax_a, "sphere_prep", (r"$|0\rangle$", r"$|0_L\rangle$", None),
       rf"preparation: ${tp[0]:.0f}$ vs ${tp[1]:.0f}$ ns", "a",
       [((0, 30), "center"), ((0, -68), "center"), ((0, -68), "center")],
       num_off=[(-30, -8), (-30, 0), (-30, 0), (-30, 0), (-30, 0)],        # on the side away from the photon-optimal path
       frame=frame_b, extra=[(RS["states"]["1L"], r"$|1_L\rangle$")])
sm = plt.cm.ScalarMappable(cmap=S.NCMAP, norm=plt.Normalize(0, NMAX))
cb = fig.colorbar(sm, cax=fig.add_subplot(top[2]))
cb.set_label(r"photon number $\bar n$", fontsize=FS["label"] + 3)
hs = [Line2D([], [], **STY[k]) for k in ("geodesic", "photon", "tradeoff")]
fig.legend(hs, [LAB[k] for k in ("geodesic", "photon", "tradeoff")], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
           fontsize=FS["legend"] + 1, columnspacing=1.0, handlelength=2.0, frameon=False)
wigner_rows(fig, [g[1], g[2]])
S.save(fig, HERE / "fig1")
print("wrote", HERE / "fig1.png", "| durations (ns):", dict(prep=tp, flip=tf),
      "| F_T snaps:", {n: TR[n]["F_target_snap"] for n in ("gkp", "flip")})
