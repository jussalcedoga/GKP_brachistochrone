"""SI figures S12, S14, S16, S21: one transfer followed along its three optimal paths -- the time-optimal path
(brachistochrone), the trade-off path (w* = kappa/mu = 0.79, Yale) and the photon-optimal path -- each run at the same
multiple m of its own minimal time L/omega (so in absolute time the photon-optimal path is the slowest), with the gate
sets that were run on it (CS on every path; CD and SNAP on every path for the flip, on the brachistochrone only for
preparation), all under the budget of Fig. 4 (|Omega| <= 10 g, generator rate <= g, N slices, the tracking protocol,
the same random starts, Lindblad evolution with the Yale rates).
(a) the state in the plane of the start and the brachistochrone direction (z horizontal), with the runs of every gate
    set at t/T = 1/4, 1/2, 3/4 (dots of growing opacity; CS joined to the path state); (b) closeness to the path;
(c) photon number of the runs and of the path; numbers: final fidelity F_T (mean +- s.d. over the usable starts) and the
    complete first-order error eps_1 = mu T + kappa int<n>dt + kappa_phi int Var(n) dt of the path itself;
(d) Wigner functions of the path state and of the run of each gate set at the numbered times, exact on the shown window
    |q|,|p| <= 5 (mfcommon.wigner_grid, review Nov 1, L1-F1), saved to <out>_wigner.npz.
Bands: mean +- s.d. over all starts; thick lines and Wigner panels: the start closest to the mean (mfcommon); every
run is kept (mfcommon.usable returns all runs, trivial SNAP runs included).  The photon-optimal path is drawn dotted in
a dark rose of the crimson family, not in crimson itself, which is the colour of the CS runs in these figures.  Reads manuscript/ (read only); caches the reference-path states in
supplementary_information/_cache.   Used by S12/S14/S16/S21 plot.py:  main(target, k, out_path)
--landscape (S16, S22; one-column SI page): (a)-(c) on the left; on the right (d) as one grid, rows = the times t_k of the
paths (i)-(iii), columns = path, CS, CD, SNAP; legend and colour bar below.  Same data and numbers; the default layout
is unchanged.   python _three_paths.py --target flip --k 2 --out <stem> [--landscape]"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))
import mfcommon as M  # noqa: E402,I001

S = M.setup()
from common import params as P, tracking  # noqa: E402
import compute as CP  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402
import numpy as np  # noqa: E402

FS = M.FS
REFS = ("geodesic", "tradeoff", "photon")
PL = {"geodesic": "time optimal", "tradeoff": r"trade-off ($w^*$)", "photon": "photon optimal"}
# photon-optimal path: dark rose of the crimson (photon) family, distinct from the crimson CS runs; ink would give gray
# tints for the badges and snapshot dots of growing opacity
PC = {"geodesic": S.ROYALBLUE, "tradeoff": S.TRADEOFF, "photon": "#8E2C48"}
PLS = {"geodesic": (0, (5, 2)), "tradeoff": (0, (5, 2)), "photon": (0, (1.2, 1.4))}
PLW = {"geodesic": 5.4, "tradeoff": 5.4, "photon": 3.6}
GATES = ("cs", "cd", "snap")
SNAP = (0.25, 0.5, 0.75)
NUM = ("1", "2", "3")
ALPHAS = (0.25, 0.6, 1.0)
KAPPA, MU = P.kappa_mu("Yale")
KPHI = 2.0 / P.NOISE["Yale"]["Tphic"]
CACHE = M.MF / "supplementary_information" / "_cache"
TITLE = {"flip": r"state transfer $|0_L\rangle\to|1_L\rangle$", "gkp": r"GKP preparation $|0\rangle\to|0_L\rangle$",
         "cat": r"cat preparation $|0\rangle\to|C_2^+\rangle$"}


def path_states(name, ref):
    """dense reference-path states (cutoff 80), cached; the trade-off and photon paths are numerical optima"""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"path_{name}_{ref}.npz"
    if not f.exists():
        np.savez_compressed(f, X=CP.reference_dense(name, ref, 80))
    return np.load(f)["X"]


def path_moments(X):
    """arclength averages of <n> and Var(n) along the path states X (rows)"""
    nv = np.arange(X.shape[1])
    n1 = np.real(np.einsum("ij,j,ij->i", X.conj(), nv, X))
    n2 = np.real(np.einsum("ij,j,ij->i", X.conj(), nv ** 2, X))
    ds = np.arccos(np.clip(np.abs(np.sum(X[:-1].conj() * X[1:], 1)), -1, 1))
    w = np.r_[ds, 0] + np.r_[0, ds]
    return float(np.sum(w * n1) / np.sum(w)), float(np.sum(w * (n2 - n1 ** 2)) / np.sum(w)), float(ds.sum())


def at(r, key, f):
    t = np.array(r["tL_ns"])
    return float(np.interp(f * t[-1], t, np.array(r[key])))


def badge(ax, x, y, j, color, transform=None, pad=0.24):
    a = ALPHAS[j]
    kw = dict(transform=transform) if transform is not None else {}
    ax.text(x, y, NUM[j], ha="center", va="center", fontsize=FS["small"], zorder=9, clip_on=False,
            color="white" if a >= 0.5 else S.INK,
            bbox=dict(boxstyle=f"circle,pad={pad}", fc=(*matplotlib.colors.to_rgb(color), a), ec=color, lw=1.6), **kw)


def ticks(T):
    for step in (100, 200, 250, 500, 1000):
        if T / step <= 3.2:
            return list(np.arange(0, 0.97 * T, step))
    return [0]


def main(name, k, out, landscape=False):
    out = pathlib.Path(out)
    LAND = landscape
    G = M.groups()
    D_ = {(ref, g): G.get((name, ref, g, k), []) for ref in REFS for g in GATES}
    USE = {key: M.usable(v) for key, v in D_.items()}
    REP = {key: M.representative(v) if v else None for key, v in D_.items()}
    gates = [g for g in GATES if any(USE[(ref, g)] for ref in REFS)]
    cols = [ref for ref in REFS if any(USE[(ref, g)] for g in gates)]
    first = {ref: next(USE[(ref, g)][0] for g in gates if USE[(ref, g)]) for ref in cols}
    Tcol = {ref: float(np.mean([r["T_ns"] for g in gates for r in USE[(ref, g)]])) for ref in cols}
    TMAX = max(Tcol.values())
    FLIP = min(first[cols[0]]["z_path"]) < -0.5
    th0 = float(np.arccos(np.sqrt(min(1.0, first["geodesic"]["F_T_path"][0])))) if "geodesic" in cols else None
    m_pace = Tcol["geodesic"] / (1e3 * th0 / P.OMEGA_REF) if th0 else float("nan")
    NMAX = 1.12 * max([max(first[ref]["nbar_path"]) for ref in cols]
                      + [float(np.max(M.curve_stat(v, "nbarL")[1] + M.curve_stat(v, "nbarL")[2])) for v in USE.values() if v])
    # layout (inches)
    if not LAND:
        W_IN, LEFT, RIGHT, CG = 14.0, 1.25, 0.25, 0.32
        COLW = (W_IN - LEFT - RIGHT - (len(cols) - 1) * CG) / len(cols)
        AH = 2.15 if FLIP else 2.75
        TOP = dict(top=0.62, a=AH, ab=0.92, b=2.0, bc=0.32, c=2.25, cd=2.05, leg=1.15)
        nrow = 1 + len(gates)
        WB = COLW - 0.30                         # Wigner block of a column; the last one leaves room for the colour bar
        PW = (WB - 2 * 0.05) / 3
        WIG_H = nrow * PW + (nrow - 1) * 0.06
        H_IN = TOP["top"] + TOP["a"] + TOP["ab"] + TOP["b"] + TOP["bc"] + TOP["c"] + TOP["cd"] + WIG_H + TOP["leg"]
    else:                                        # landscape: (a)-(c) | (d) as one grid (rows t_k of (i)-(iii))
        W_IN, LEFT, RIGHT, CG, GAP_LR = 14.4, 1.10, 0.05, 0.26, 0.22
        PW, GW, GG, RL, RLG = 0.82, 0.05, 0.20, 1.44, 0.44   # Wigner side, gap, gap between the paths; row labels
        ncw = 1 + len(gates)                     # grid columns: path, then the gate sets
        RB_W = RL + ncw * PW + (ncw - 1) * GW
        XR = W_IN - RIGHT - RB_W                 # left edge of the right block (path labels, badges, times)
        XG = XR + RL                             # left edge of the Wigner grid
        COLW = (XR - GAP_LR - LEFT - (len(cols) - 1) * CG) / len(cols)
        AH = COLW * (1.22 / 2.24) if FLIP else COLW * (1.22 / 1.20)
        Y_G0 = 0.52                              # top of the grid (column names above it)
        GRP_H = 3 * PW + 2 * GW
        GRID_H = len(cols) * GRP_H + (len(cols) - 1) * GG
        TOP = dict(top=1.0, a=AH, ab=0.90, bc=0.55, cbot=1.0, leg=1.42)   # top: two-line column titles
        REST = Y_G0 + GRID_H - TOP["top"] - AH - TOP["ab"] - TOP["bc"] - TOP["cbot"]   # (b) + (c)
        TOP.update(b=0.40 * REST, c=0.60 * REST)
        H_IN = Y_G0 + GRID_H + TOP["leg"]
    fig = plt.figure(figsize=(W_IN, H_IN))

    def ax_at(x0, y_top, w, h):
        return fig.add_axes([x0 / W_IN, 1 - (y_top + h) / H_IN, w / W_IN, h / H_IN])

    y_a = TOP["top"]
    y_b = y_a + TOP["a"] + TOP["ab"]
    y_c = y_b + TOP["b"] + TOP["bc"]
    y_d = y_c + TOP["c"] + TOP["cd"] if not LAND else None
    # snapshot dots in (a): marker sizes / edge widths (gate runs; path states), CS link, number offset
    DOT = dict(ms=12, mew=1.6, msp=13, mewp=2.4, lwl=1.8, off=13) if not LAND else \
        dict(ms=10, mew=1.4, msp=11, mewp=2.1, lwl=1.5, off=11)

    def plabel(ax, letter, y, xin=-0.25):
        """landscape: panel label at the common figure x xin (inches), height y in axes fraction of ax"""
        p = ax.get_position()
        S.panel_label(ax, letter, x=(xin / W_IN - p.x0) / p.width, y=y)

    facts = dict(target=name, k=k, m_pace=m_pace, columns={})
    WSAVE = {}                                   # the plotted Wigner arrays of (d)
    for ci, ref in enumerate(cols):
        x0 = LEFT + ci * (COLW + CG)
        r0, T = first[ref], Tcol[ref]
        X = path_states(name, ref)
        mn, mv, Lp = path_moments(X)
        eps = (T * 1e-3) * (MU + KAPPA * mn + KPHI * mv)
        cf = dict(T_ns=T, L_path=Lp, eps1=eps, eps1_two_term=(T * 1e-3) * (MU + KAPPA * mn))
        # (a)
        aw = AH * (2.24 / 1.22) if FLIP else AH * (1.20 / 1.22)
        ax = ax_at(x0 + 0.5 * (COLW - aw), y_a, aw, AH)
        th = np.linspace(0, np.pi, 300)
        ax.plot(np.cos(th), np.sin(th), color=S.INK, lw=1.3, ls=(0, (2, 2)))      # pure states of the plane (no gray)
        xp, zp = np.array(r0["x_path"]), np.array(r0["z_path"])
        ax.plot(zp, xp, color=PC[ref], lw=PLW[ref], ls=PLS[ref], zorder=5)
        for g in gates:
            for r in USE[(ref, g)]:
                ax.plot(r["zL"], r["xL"], color=M.GATE_COLOR[g], lw=M.LW_THIN, alpha=0.20, zorder=2)
            if REP[(ref, g)] is not None:
                ax.plot(REP[(ref, g)]["zL"], REP[(ref, g)]["xL"], color=M.GATE_COLOR[g], lw=M.LW - 0.4, zorder=4)
        for j, f in enumerate(SNAP):
            ip = int(round(f * (len(xp) - 1)))
            for g in gates:
                rr = REP[(ref, g)]
                if rr is None:
                    continue
                xs, zs = at(rr, "xL", f), at(rr, "zL", f)
                if g == "cs":
                    ax.plot([zp[ip], zs], [xp[ip], xs], color=M.GATE_COLOR[g], lw=DOT["lwl"], zorder=6)
                ax.plot(zs, xs, "o", ms=DOT["ms"], mfc=(*matplotlib.colors.to_rgb(M.GATE_COLOR[g]), ALPHAS[j]),
                        mec=M.GATE_COLOR[g], mew=DOT["mew"], zorder=7)
            ax.plot(zp[ip], xp[ip], "o", ms=DOT["msp"], mfc=(*matplotlib.colors.to_rgb(PC[ref]), ALPHAS[j]), mec=PC[ref],
                    mew=DOT["mewp"], zorder=8)
            ax.annotate(NUM[j], (zp[ip], xp[ip]), xytext=(0, DOT["off"]), textcoords="offset points", ha="center",
                        fontsize=FS["small"], color=S.INK, zorder=9)
        ax.set_xlim(1.12, -1.12 if FLIP else -0.08)
        ax.set_ylim(-0.08, 1.14)
        ax.set_aspect("equal")
        ax.set_xticks([1, 0, -1] if FLIP else [1, 0.5, 0])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(r"$z$", labelpad=2)
        if ci == 0:
            ax.set_ylabel(r"$x$")
            if not LAND:
                S.panel_label(ax, "a", x=-0.30 if FLIP else -0.40, y=1.0)
            else:                                # level with the first line of the titles and with "(d)"
                plabel(ax, "a", y=1.0 + (y_a - 0.43) / AH)
        else:
            ax.set_yticklabels([])
        if not LAND:
            ax.set_title(rf"({'i' * (ci + 1)})\enspace " + PL[ref] + rf", ${T:.0f}$ ns", fontsize=FS["small"],
                         pad=8, color=PC[ref])
        else:                                    # two lines (narrow column), clear of the snapshot numbers
            ax.set_title(rf"({'i' * (ci + 1)})\enspace " + PL[ref] + "\n" + rf"${T:.0f}$ ns", fontsize=FS["small"],
                         pad=20, color=PC[ref], linespacing=1.15)
        # (b)
        axb = ax_at(x0, y_b, COLW, TOP["b"])
        for g in gates:
            if USE[(ref, g)]:
                t, m, s, _ = M.curve_stat(USE[(ref, g)], "FL")
                axb.fill_between(t, m - s, m + s, color=M.GATE_COLOR[g], alpha=0.2, lw=0)
                rr = REP[(ref, g)]
                axb.plot(np.array(rr["tL_ns"]), np.array(rr["FL"]), color=M.GATE_COLOR[g], lw=M.LW - 0.4)
        axb.set_xlim(0, 1.03 * TMAX)
        axb.set_ylim(0, 1.03)
        axb.set_xticks(ticks(TMAX))
        axb.set_xticklabels([])
        axb.set_yticks([0.5, 1])
        # (c)
        axc = ax_at(x0, y_c, COLW, TOP["c"])
        axc.plot(np.array(r0["t_ns"]), r0["nbar_path"], color=PC[ref], lw=PLW[ref] - 0.8, ls=PLS[ref], zorder=5)
        lines = []
        for g in gates:
            if USE[(ref, g)]:
                t, m, s, _ = M.curve_stat(USE[(ref, g)], "nbarL")
                axc.fill_between(t, m - s, m + s, color=M.GATE_COLOR[g], alpha=0.2, lw=0)
                rr = REP[(ref, g)]
                axc.plot(np.array(rr["tL_ns"]), np.array(rr["nbarL"]), color=M.GATE_COLOR[g], lw=M.LW - 0.4)
                mF, sF, n = M.stat(USE[(ref, g)], "F_T_final_noisy")
                mC, sC, _ = M.stat(USE[(ref, g)], "C_noisy")
                lines.append((rf"{M.GATE_SHORT[g]}: $F_T={mF:.3f}\pm{sF:.3f}$" if not LAND else
                              (rf"{M.GATE_SHORT[g]}: ${mF:.4f}{{\pm}}{sF:.4f}$" if sF < 5e-4 else
                               rf"{M.GATE_SHORT[g]}: ${mF:.3f}{{\pm}}{sF:.3f}$"), M.GATE_COLOR[g]))   # landscape: F_T in caption;
                # an s.d. below 5e-4 is printed with four decimals (never as +-0.000); {\pm} without binary spacing keeps
                # the four-decimal line inside the narrow panel
                cf[g] = dict(n_usable=n, n_trivial=M.n_trivial(D_[(ref, g)]), F_T_mean=mF, F_T_sd=sF, C_mean=mC, C_sd=sC,
                             expo_L_ns=M.stat(USE[(ref, g)], "expo_L_ns")[0], rep_seed=rr["seed"])
        lines.append((rf"path: $\epsilon_1={100 * eps:.2f}\%$", S.INK))
        step = 0.125 if not LAND else 0.27 / TOP["c"]   # landscape: line spacing 0.27 in
        for i, (txt, c) in enumerate(lines):
            axc.text(0.03 if not LAND else 0.022, 0.96 - step * i, txt, transform=axc.transAxes, ha="left", va="top",
                     fontsize=FS["small"] - (1 if not LAND else 2), color=c,
                     bbox=dict(fc="white", ec="none", alpha=1.0, pad=0.5))   # opaque: no t_k line through the numbers
        axc.set_xlim(0, 1.03 * TMAX)
        if not LAND:
            axc.set_ylim(0, NMAX * (1.0 + 0.135 * len(lines) + 0.15))
        else:                                    # the narrow panel: every curve and band stays below the text block
            ytop = NMAX / (0.96 - step * (len(lines) - 1) - 0.27 / TOP["c"] - 0.02)
            axc.set_ylim(0, ytop)
            axc.set_yticks([v for v in (0, 2.5, 5.0) if v <= NMAX])   # no tick behind the text block
        axc.set_xticks(ticks(TMAX))
        axc.set_xlabel(r"$t$ (ns)")
        for a_ in (axb, axc):
            for f in SNAP:
                a_.axvline(f * T, color=S.INK, lw=1.2, ls=(0, (1, 2)), zorder=1)
            a_.axvline(T, color=PC[ref], lw=1.8, ls=(0, (4, 2)), zorder=1)
        if ci == 0:
            axb.set_ylabel(r"$F$")
            axc.set_ylabel(r"$\bar n$")
            if not LAND:
                S.panel_label(axb, "b", x=-0.30, y=1.0)
                S.panel_label(axc, "c", x=-0.30, y=1.0)
            else:
                plabel(axb, "b", y=0.92)
                plabel(axc, "c", y=0.94)
        else:
            axb.set_yticklabels([])
            axc.set_yticklabels([])
        if LAND:                                 # snapshot badges between (b) and (c), on the dotted lines at t_k
            for j, f in enumerate(SNAP):
                badge(axc, f * T / axc.get_xlim()[1], 1 + 0.5 * TOP["bc"] / TOP["c"], j, PC[ref],
                      transform=axc.transAxes, pad=0.10)
        # (d) Wigner block of this column, exact on the shown window |q|,|p| <= 5 (mfcommon.wigner_grid; review Nov 1,
        # L1-F1): the path row is recomputed for the path states of wigner_snapshots.py (reference path X, cutoff 90,
        # the stored fractions), whose stored W_path is not used; the plotted arrays go to <out>_wigner.npz
        Zc = np.load(M.MS / "fig4_cs_path_following" / "results" / "wigner" / f"{name}_{ref}_cs_k{k}.npz")
        fr = Zc["fracs"]
        ii = [int(np.argmin(np.abs(fr - f))) for f in SNAP]
        WG = M.wigner_grid(90, len(Zc["x"]), float(Zc["x"][-1]))
        assert np.allclose(WG.x, Zc["x"])
        rows = [(None, PC[ref], [WG.of_vectors(gv[None]) for gv in tracking.Path(CP.pad(X, 90)).at(fr[ii])[0]], None)]
        WSAVE.update({"x": WG.x, "fracs": fr[ii], f"{ref}_path": np.array(rows[0][2])})
        Gp, _ = tracking.Path(CP.pad(X, 90)).at(np.array(SNAP))
        for g in gates:
            rr = REP[(ref, g)]
            if rr is None:
                rows.append((g, M.GATE_COLOR[g], None, None))
                continue
            xg, Ws, _ = M.wigner_of_run(rr, SNAP)
            Fr = M.closeness_of_run(rr, Gp, SNAP)
            dF = float(np.max(np.abs(np.array(Fr) - np.array([at(rr, "FL", f) for f in SNAP]))))
            assert dF < 2e-3, (ref, g, dF)           # the panels show the state of the plotted run
            cf.setdefault("wigner_check", {})[g] = dF
            rows.append((g, M.GATE_COLOR[g], Ws, Fr))
            WSAVE[f"{ref}_{g}"] = np.array(Ws)
            WSAVE[f"{ref}_{g}_seed"] = rr["seed"]
        xw = WG.x
        ext = [xw[0], xw[-1], xw[0], xw[-1]]
        for ri, (g, c, Ws, Fs) in enumerate(rows):
            for j in range(3):
                if not LAND:
                    axw = ax_at(x0 + j * (PW + 0.05), y_d + ri * (PW + 0.06), PW, PW)
                else:                            # grid column ri (path, gates), row j of the group of this path
                    yw = Y_G0 + ci * (GRP_H + GG) + j * (PW + GW)
                    axw = ax_at(XG + ri * (PW + GW), yw, PW, PW)
                    if ri == 0:                  # row label: badge j over the time t_j; the path's number left of it
                        xl = -(RL - RLG) / 2 / PW
                        badge(axw, xl, 0.70, j, PC[ref], transform=axw.transAxes, pad=0.10)
                        axw.text(xl, 0.21, rf"${SNAP[j] * T:.0f}$ ns", transform=axw.transAxes, ha="center",
                                 va="center", fontsize=FS["small"] - 2)
                        if j == 1:
                            axw.text(-(RL - 0.5 * RLG) / PW, 0.5, f"({'i' * (ci + 1)})", transform=axw.transAxes,
                                     ha="center", va="center", fontsize=FS["small"], color=PC[ref])
                    if ci == 0 and j == 0:       # column names above the grid
                        axw.text(0.5, 1.0 + 0.07 / PW, "path" if g is None else M.GATE_SHORT[g],
                                 transform=axw.transAxes, ha="center", va="bottom", fontsize=FS["ann"] + 1,
                                 color=S.INK if g is None else c)
                        if ri == 0:
                            S.panel_label(axw, "d", x=-RL / PW, y=1.0 + (Y_G0 - 0.43) / PW)
                axw.set_xticks([])
                axw.set_yticks([])
                if Ws is None:
                    axw.axis("off")
                    continue
                im = axw.imshow(Ws[j], origin="lower", extent=ext, cmap=S.RBCMAP, vmin=-S.W_LIM, vmax=S.W_LIM)
                for sp in axw.spines.values():
                    sp.set_visible(True)
                    sp.set_color(c)
                    sp.set_linewidth(3.0)
                if Fs is not None:
                    axw.text(0.05, 0.05, rf"${Fs[j]:.2f}$", transform=axw.transAxes, fontsize=FS["small"] - 2, zorder=2,
                             bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))   # under the frame (spines: 2.5)
                if ri == 0 and not LAND:
                    axw.text(0.5, 1.10, rf"${SNAP[j] * T:.0f}$ ns", transform=axw.transAxes, ha="center", va="center",
                             fontsize=FS["small"])
                    xa = SNAP[j] * T / axc.get_xlim()[1]
                    yb = -1.05 / TOP["c"]
                    badge(axc, xa, yb, j, PC[ref], transform=axc.transAxes)
                    fig.add_artist(ConnectionPatch(xyA=(xa, yb), coordsA=axc.transAxes, xyB=(0.5, 1.20),
                                                   coordsB=axw.transAxes, arrowstyle="-|>", mutation_scale=18, lw=1.7,
                                                   color=PC[ref], shrinkA=12, shrinkB=2, zorder=0))
                if ci == 0 and j == 0 and not LAND:
                    lab = "path" if g is None else M.GATE_SHORT[g]
                    axw.text(-0.16, 0.5, lab, transform=axw.transAxes, rotation=90, ha="center", va="center",
                             fontsize=FS["ann"], color=c)
                    if ri == 0:
                        S.panel_label(axw, "d", x=-0.62, y=1.22)
        facts["columns"][ref] = cf
    hs = [Line2D([], [], color=PC[r], lw=PLW[r] - 0.8, ls=PLS[r]) for r in cols]
    hs += [Line2D([], [], color=M.GATE_COLOR[g], lw=M.LW) for g in gates]
    if not LAND:
        cax = ax_at(LEFT + (len(cols) - 1) * (COLW + CG) + WB + 0.14, y_d, 0.18, WIG_H)
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([-S.W_LIM, 0, S.W_LIM])
        cb.set_ticklabels([r"$-1/\pi$", r"$0$", r"$1/\pi$"])
        cb.ax.tick_params(labelsize=FS["tick"])
        cb.set_label(r"$W(q,p)$", fontsize=FS["label"])
        fig.legend(hs, [PL[r] + " path" for r in cols] + [M.GATE_NAME[g] for g in gates], loc="lower center",
                   ncol=3, fontsize=FS["legend"] - 1, bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=2.4,
                   frameon=False)
    else:                                        # colour bar under the grid (W(q,p) below it); legend under (a)-(c)
        GW_ALL = ncw * PW + (ncw - 1) * GW
        cax = ax_at(XG + 0.30, Y_G0 + GRID_H + 0.26, GW_ALL - 0.60, 0.16)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([-S.W_LIM, 0, S.W_LIM])
        cb.set_ticklabels([r"$-1/\pi$", r"$0$", r"$1/\pi$"])
        cb.ax.tick_params(labelsize=FS["tick"], pad=4)
        cb.set_label(r"$W(q,p)$", fontsize=FS["label"], labelpad=4)
        fig.legend(hs, [PL[r] + " path" for r in cols] + [M.GATE_NAME[g] for g in gates], loc="lower left",
                   ncol=2, fontsize=FS["legend"] - 1, bbox_to_anchor=(-0.25 / W_IN, 0.0), columnspacing=1.2,
                   handlelength=2.4, frameon=False)
    S.save(fig, out)
    np.savez_compressed(out.with_name(out.name + "_wigner.npz"), **WSAVE)
    out.with_name(out.name + "_facts.json").write_text(json.dumps(facts, indent=1))
    print("wrote", out.with_suffix(".png"), f"| m = {m_pace:.2f}")
    for ref, cf in facts["columns"].items():
        print(f"  {ref}: T={cf['T_ns']:.0f} ns eps1 {100 * cf['eps1']:.2f}% (two-term {100 * cf['eps1_two_term']:.2f}%) " +
              "  ".join(f"{g}: n={cf[g]['n_usable']}(+{cf[g]['n_trivial']}) F_T={cf[g]['F_T_mean']:.4f}+-{cf[g]['F_T_sd']:.4f} "
                        f"C={cf[g]['C_mean']:.3f}" for g in GATES if g in cf))
    return facts


if __name__ == "__main__":
    a = sys.argv
    main(a[a.index("--target") + 1], int(a[a.index("--k") + 1]), a[a.index("--out") + 1], landscape="--landscape" in a)
