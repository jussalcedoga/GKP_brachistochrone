"""Fig. 4 (and SI Figs. S8-S11): following the brachistochrone of a preparation with three gate sets under one budget.

Every curve of a column has the same duration T, the same N slices, the same bounds |Omega| <= 10 g and generator rate
<= g = 7.5/us, the same tracking protocol, the same random starting pulses and the same Lindblad noise (Yale rates):
conditional squeezing (CS), conditional displacements (ECD generator) and SNAP-displacement control, all along the
brachistochrone.  Bands: mean +- s.d. over all starts; thick lines and Wigner panels: one run, the start closest to the
mean (mfcommon.representative); thin lines: every start.  Trivial runs (the optimizer stopped at zero cavity drive, SNAP
only; mfcommon.trivial) are KEPT in every statistic (mfcommon.usable returns all runs) and counted in the caption.
(a) oscillator state in the plane of |0> and the path direction |chi>; in the snapshot column each gate's run is joined,
    in its colour, to the path state at the numbered times t_k; (b) closeness F(t) = <p(t)|rho(t)|p(t)>, with the errors
    at T: first-order error eps_1 of the path itself and 1 - F_T of each gate set (mean over the usable starts);
(c) target fidelity F_T(t); (d) Wigner functions of the path state and of the run of each gate set at the numbered times.
Reads manuscript/fig4_cs_path_following/results (traces80, rho80; of wigner/ only the grid and fractions); writes only
into this folder (or --out).  Wigner panels exact on the shown window (mfcommon.wigner_grid, review Nov 1, L1-F1), saved
to <stem>_wigner.npz.
--landscape (SI Figs. S09-S11, one-column page): (a)-(c) on the left; on the right (d) with times as rows and path/CS/
CD/SNAP as columns, and below it the errors at T (path eps_1, 1 - F_T of each gate set) as a table.  Same data and numbers;
the default layout (main-text Fig. 4) is unchanged.
python figures/Fig4/plot_fig4.py [--target gkp|fock2|cat|binomial] [--snap-warm] [--landscape] [--out <stem>]"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))

import mfcommon as M  # noqa: E402,I001
if "--landscape" not in sys.argv and "--wide" not in sys.argv:   # Nov 4 (user): 1.25x fonts in the one-column Letter
    # figure (drawn at 14 in, printed at 3.4 in); the two-column one (--wide) prints at half its drawn size
    M.FS.update({k_: round(1.25 * v_) for k_, v_ in M.FS.items()})

S = M.setup()
from common import params as P, tracking  # noqa: E402
import compute as CP  # noqa: E402  (simulations/tracking/compute.py: transfers and reference paths)
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402
import numpy as np  # noqa: E402

FS = M.FS
NAME = sys.argv[sys.argv.index("--target") + 1] if "--target" in sys.argv else "gkp"
OUT = (HERE if "--out" not in sys.argv else pathlib.Path(sys.argv[sys.argv.index("--out") + 1]).resolve().parent)
STEM = ("fig4" if "--out" not in sys.argv else pathlib.Path(sys.argv[sys.argv.index("--out") + 1]).name)
TITLE = {"gkp": r"GKP $|0\rangle\to|0_L\rangle$, $\Delta=0.4$", "fock2": r"Fock $|0\rangle\to|2\rangle$",
         "cat": r"cat $|0\rangle\to|C_2^+\rangle$, $\gamma=2$", "binomial": r"$|0\rangle\to(|0\rangle+|4\rangle)/\sqrt2$"}
GATES = ("cs", "cd", "snap")
KS = (1, 2, 3)                                   # 412, 824, 2133 ns on the brachistochrone
KW = 2                                           # the column with the Wigner snapshots
SNAP = (0.2, 0.4, 0.6, 0.8, 1.0)
NUM = ("1", "2", "3", "4", "5")
ALPHAS = (0.15, 0.35, 0.55, 0.78, 1.0)          # opacity of the snapshot dots 1 -> 5 in (a-ii)
ROMAN = ("i", "ii", "iii")
LW, LWR = M.LW, 5.6
PATHC = S.ROYALBLUE
WIDE = "--wide" in sys.argv                      # Letter, two columns: (a)-(c) left (a as wide as b, c), (d) right
DV = "--dvert" in sys.argv                       # --wide: (d) with the numbered times as rows (default: as columns)
LAND = "--landscape" in sys.argv or WIDE         # SI layout (wide): (a)-(c) left, (d) and the errors at T right
KAPPA, MU = P.kappa_mu("Yale")
G = M.groups()
if "--snap-warm" in sys.argv:        # SNAP rows from the warm-started runs of sims/snap_warm.py (no zero-drive trap)
    WARM = {}
    for p in sorted((M.MF / "sims" / "results" / "snap_warm" / "traces").glob(f"warm_{NAME}_geodesic_snap_k*_s*.json")):
        r = json.loads(p.read_text())
        r["seed"] = r["job"]["seed"]
        WARM.setdefault((NAME, "geodesic", "snap", r["job"]["k"]), []).append(r)
    G.update(WARM)
D_ = {(g, k): G.get((NAME, "geodesic", g, k), []) for g in GATES for k in KS}
USE = {key: M.usable(v) for key, v in D_.items()}
REP = {key: (M.representative(v) if v else None) for key, v in D_.items()}
r0 = next(v[0] for v in D_.values() if v)
SEMI = min(r0["z_path"]) < -0.5                  # half arc (Fock 2, cat): z drawn horizontally


def PT(x, z):
    return (z, x) if SEMI else (x, z)


KPHI = 2.0 / P.NOISE["Yale"]["Tphic"]          # cavity dephasing rate (1/us), D[n] at kappa_phi = 2/T_phi,c
_PS = CP.reference_dense(NAME, "geodesic", 80)  # 801 path states, uniform in arclength
_nv = np.arange(_PS.shape[1])
_N1 = np.real(np.einsum("ij,j,ij->i", _PS.conj(), _nv, _PS))
_N2 = np.real(np.einsum("ij,j,ij->i", _PS.conj(), _nv ** 2, _PS))
PATH_MEAN_N, PATH_MEAN_VAR = float(np.mean(_N1)), float(np.mean(_N2 - _N1 ** 2))


def eps1(r):
    """complete first-order error of the reference path itself traversed in the duration T of the trace r:
    mu T + kappa int<n>dt + kappa_phi int Var(n) dt (ancilla errors, photon loss, cavity dephasing; Yale rates)"""
    T = np.array(r["t_ns"])[-1] * 1e-3
    return T * (MU + KAPPA * PATH_MEAN_N + KPHI * PATH_MEAN_VAR)


def ticks(T):
    for step in (100, 200, 250, 500, 1000):
        if T / step <= 3.2:
            return list(np.arange(0, 0.97 * T, step))
    return [0]


def badge(ax, x, y, j, transform=None, size=None, pad=0.24):
    """numbered snapshot badge j (0..4): royal-blue circle (the path colour; ink tints would read as gray) whose fill
    opacity grows from 1 to 5, as the path dots of (a-ii) and the badges of Fig. 1"""
    a = ALPHAS[j]
    kw = dict(transform=transform) if transform is not None else {}
    ax.text(x, y, NUM[j], ha="center", va="center", fontsize=size or FS["small"], zorder=9,
            color="white" if a >= 0.5 else S.INK, clip_on=False,
            bbox=dict(boxstyle=f"circle,pad={pad}", fc=(*matplotlib.colors.to_rgb(PATHC), a), ec=PATHC, lw=1.8), **kw)


def at(r, key, f):
    """value of the Lindblad trace `key` of run r at the fraction f of its duration"""
    t = np.array(r["tL_ns"])
    return float(np.interp(f * t[-1], t, np.array(r[key])))


# ---------------------------------------------------------------- layout (inches; drawn at twice the printed size)
W_IN = 14.0
if not LAND:
    LEFT, RIGHT, CG = 1.45, 1.15, 0.30           # inches: y labels / colour bar of (d), ticks and W label / column gap
    COLW = (W_IN - LEFT - RIGHT - 2 * CG) / 3    # every panel of a column, (a) included, has this axis width
    GAP_W = 0.07
    BLOCK_W = 3 * COLW + 2 * CG                  # (d) spans exactly the three columns
    PANEL_W = (BLOCK_W - 4 * GAP_W) / 5          # Wigner panel side
    LAB_W, CB_GAP, CB_W = 0.62, 0.14, 0.22
    TOPH = dict(a=COLW * (1.22 / 2.24) if SEMI else COLW * (1.17 / 1.22), b=2.45, c=1.85)   # (a) at equal aspect
    GAPS = dict(top=0.65, ab=1.06, bc=0.34, cd=2.85, leg=1.50)     # cd: steep badge arrows that clear every badge
    BADGE_IN = 1.22                              # inches below the (c) axis: the row of snapshot badges
    NW = 4
    WIG_H = NW * PANEL_W + (NW - 1) * GAP_W
    H_IN = GAPS["top"] + TOPH["a"] + GAPS["ab"] + TOPH["b"] + GAPS["bc"] + TOPH["c"] + GAPS["cd"] + WIG_H + GAPS["leg"]
elif WIDE:                                       # Letter, two columns (printed at half size): (a)-(c) | (d) + errors at T
    LEFT, CG, GAP_LR, RIGHT = 1.05, 0.25, 0.45, 0.05
    if DV:                                       # (d): times as rows; table rows (header, (i)-(iii)); RL: row labels
        W_IN, NR, NC, PANEL_W, GAP_W, RL, NTAB, TAB_GAP, TAB_ROW, TAB_DROP = 14.2, 5, 4, 0.90, 0.05, 0.92, 4, 0.22, 0.36, 0.0
    else:                                        # (d): times as columns, large panels; colour bar and key under it
        W_IN, NR, NC, PANEL_W, GAP_W, RL, NTAB, TAB_GAP, TAB_ROW, TAB_DROP = 18.0, 4, 5, 1.47, 0.07, 0.50, 0, 0.0, 0.0, 0.0
    CBV_GAP, CBV_W, CBV_LAB = 0.18, 0.16, 0.72   # (not DV) vertical colour bar right of the grid, its ticks and label
    RB_W = RL + NC * PANEL_W + (NC - 1) * GAP_W + (0.0 if DV else CBV_GAP + CBV_W + CBV_LAB)   # (not DV: no table)
    XR = W_IN - RIGHT - RB_W                     # left edge of the right block
    COLW = (XR - GAP_LR - LEFT - 2 * CG) / 3     # (a) as wide as (b) and (c)
    GRID_H = NR * PANEL_W + (NR - 1) * GAP_W
    CB_GAP, CB_H, CB_TICK = 0.26, 0.16, 0.44
    if DV:                                       # (c) ends level with the table; legend under the whole figure
        RIGHT_H = GRID_H + CB_GAP + CB_H + CB_TICK + TAB_GAP + NTAB * TAB_ROW
        A_H = COLW * (1.22 / 2.24) if SEMI else COLW * (1.17 / 1.22)   # (a) at equal aspect, full column width
        GAPS = dict(top=0.95, ab=0.86, bc=0.56, cbot=0.98, leg=1.08)
        REST = RIGHT_H - A_H - GAPS["ab"] - GAPS["bc"]
        H_IN = GAPS["top"] + RIGHT_H + GAPS["cbot"] + GAPS["leg"]
        TOPH = dict(a=A_H, b=0.53 * REST, c=0.47 * REST)
    else:                                        # much wider than tall: (c) ends level with the Wigner grid, whose
        RIGHT_H = GRID_H                                               # colour bar and key share the band of the
        GAPS = dict(top=0.68, ab=0.80, bc=0.46, cbot=0.84)             # tick labels of (c); (a) as wide as (b), (c)
        A_H = COLW * (1.22 / 2.24) if SEMI else COLW * (1.17 / 1.22)   # and at equal aspect (a true arc)
        REST = RIGHT_H - A_H - GAPS["ab"] - GAPS["bc"]
        TOPH = dict(a=A_H, b=0.53 * REST, c=0.47 * REST)
        H_IN = GAPS["top"] + RIGHT_H + max(GAPS["cbot"], 1.04) + 0.05   # the legend under the colour bar's "-1/pi"
else:                                            # landscape: (a)-(c) | (d) + table of the errors at T
    LEFT, CG, GAP_LR, RIGHT = 1.15, 0.30, 0.24, 0.05
    NR, NC = 5, 4
    PANEL_W, GAP_W, RL = 0.95, 0.05, 0.92        # Wigner side and gap; RL: row labels of (d) (badge over the time)
    RB_W = RL + 4 * PANEL_W + 3 * GAP_W          # right block: row labels + path, CS, CD, SNAP
    XR = W_IN - RIGHT - RB_W                     # left edge of the right block
    COLW = (XR - GAP_LR - LEFT - 2 * CG) / 3
    GRID_H = 5 * PANEL_W + 4 * GAP_W
    CB_GAP, CB_H, CB_TICK = 0.24, 0.16, 0.44     # horizontal colour bar under (d) and its tick labels
    TAB_GAP, TAB_ROW = 0.22, 0.36                # table: header row and rows (i)-(iii)
    RIGHT_H = GRID_H + CB_GAP + CB_H + CB_TICK + TAB_GAP + 4 * TAB_ROW
    GAPS = dict(top=0.74, ab=0.86, bc=0.56, cbot=0.98, leg=1.08)   # bc: the row of snapshot badges of column (ii)
    A_H = COLW * (1.22 / 2.24) if SEMI else 1.72                    # (a) at equal aspect (quarter arc: centred)
    REST = RIGHT_H - A_H - GAPS["ab"] - GAPS["bc"] - GAPS["cbot"]    # (b) + (c): left and right blocks end together
    TOPH = dict(a=A_H, b=0.53 * REST, c=0.47 * REST)
    H_IN = GAPS["top"] + RIGHT_H + GAPS["leg"]
fig = plt.figure(figsize=(W_IN, H_IN))


def ax_at(x0, y_top, w, h):
    """axes at x0 (in from the left) with its top edge y_top (in from the top)"""
    return fig.add_axes([x0 / W_IN, 1 - (y_top + h) / H_IN, w / W_IN, h / H_IN])


LAB_X = -0.30                                    # inches from the left edge: (a)-(d) share this x, left of every y label


def panel_label(ax, letter, y):
    """panel label at the common x LAB_X, at the height y in axes fraction of ax"""
    p = ax.get_position()
    S.panel_label(ax, letter, x=(LAB_X / W_IN - p.x0) / p.width, y=y)


y_a = GAPS["top"]
y_b = y_a + TOPH["a"] + GAPS["ab"]
y_c = y_b + TOPH["b"] + GAPS["bc"]
y_d = y_c + TOPH["c"] + (GAPS["cd"] if not LAND else 0.0)
QUAD = not SEMI
B_TOP = 1.57 if (QUAD and not LAND) else 1.03   # (b): the band above F = 1 holds the errors at T (landscape: table)
# snapshot dots of (a-ii): marker size, edge width (gate runs; path states) and the CS link (smaller in the landscape)
DOT = dict(ms=14, mew=1.8, msp=13, mewp=2.6, lwl=2.0, fs=FS["small"] + 2, off=14) if not LAND else \
    dict(ms=10.5, mew=1.5, msp=10, mewp=2.2, lwl=1.6, fs=FS["small"], off=11)
facts = {"target": NAME, "columns": {}}
axc_sel, T_sel = None, None
table = []                                       # landscape: errors at T of each column, (text, colour) per cell
ctable = []                                      # Letter two-column: minimum closeness C of each column
for col, k in enumerate(KS):
    lists = {g: USE[(g, k)] for g in GATES}
    rr = {g: REP[(g, k)] for g in GATES}
    ref = next(v[0] for v in (D_[(g, k)] for g in GATES) if v)
    Tg = ref["T_ns"]
    x0 = LEFT + col * (COLW + CG)
    # (a) the plane of |0> and chi, as wide as (b) and (c) (landscape quarter arc: narrower, centred)
    aw = COLW if (WIDE or not (LAND and QUAD)) else TOPH["a"] * (1.22 / 1.17)
    ax = ax_a = ax_at(x0 + 0.5 * (COLW - aw), y_a, aw, TOPH["a"])
    th = np.linspace(0, np.pi, 300)
    ax.plot(*PT(np.sin(th), np.cos(th)), color=S.INK, lw=1.4, ls=(0, (2, 2)))     # pure states of the plane (no gray)
    xp, zp = np.array(ref["x_path"]), np.array(ref["z_path"])
    ax.plot(*PT(xp, zp), color=PATHC, lw=LWR, ls=(0, (5, 2)), zorder=3)
    for g in GATES:
        for r in lists[g]:
            ax.plot(*PT(r["xL"], r["zL"]), color=M.GATE_COLOR[g], lw=M.LW_THIN, alpha=0.20, zorder=2)
        if rr[g] is not None:
            ax.plot(*PT(rr[g]["xL"], rr[g]["zL"]), color=M.GATE_COLOR[g], lw=LW, zorder=4 + GATES.index(g) * 0.1)
        if rr[g] is not None and M.trivial(rr[g]):  # the run shown never leaves |0>: also the mean over all starts
            _, mx, _, _ = M.curve_stat(lists[g], "xL")
            _, mz, _, _ = M.curve_stat(lists[g], "zL")
            ax.plot(*PT(mx, mz), color=M.GATE_COLOR[g], lw=LW, zorder=8)   # solid, on top of CS/CD
    idx = [int(round(f * (len(xp) - 1))) for f in SNAP]
    if k == KW:                                    # every gate's run at the same times t_k: dots of growing opacity 1 -> 5
        for j, f in enumerate(SNAP):
            for g in GATES:
                if rr[g] is None:
                    continue
                xs, zs = at(rr[g], "xL", f), at(rr[g], "zL", f)
                if g == "cs":                      # the closest run: joined to the path state at the same time
                    ax.plot(*PT([xp[idx[j]], xs], [zp[idx[j]], zs]), color=M.GATE_COLOR[g], lw=DOT["lwl"], zorder=5)
                ax.plot(*PT(xs, zs), "o", ms=DOT["ms"], mfc=(*matplotlib.colors.to_rgb(M.GATE_COLOR[g]), ALPHAS[j]),
                        mec=M.GATE_COLOR[g], mew=DOT["mew"], zorder=6 + j * 0.01)
            end = LAND and SEMI and j == len(SNAP) - 1  # landscape half arc: t_5 (end point) labelled to its right
            ax.annotate(NUM[j], PT(xp[idx[j]], zp[idx[j]]),
                        xytext=((9, 9) if not LAND else (7, 7)) if not SEMI else ((9, 1) if end else (0, DOT["off"])),
                        textcoords="offset points", fontsize=DOT["fs"], color=S.INK, zorder=9,
                        ha="left" if (not SEMI or end) else "center",
                        **(dict(va="center") if end else {}))
    for j in range(5 if k == KW else 0):           # path states at t_k (snapshot column only), filled 1 -> 5
        ax.plot(*PT(xp[idx[j]], zp[idx[j]]), "o", ms=DOT["msp"], mfc=(*matplotlib.colors.to_rgb(PATHC), ALPHAS[j]),
                mec=PATHC, mew=DOT["mewp"], zorder=8)
    if SEMI:
        ax.set_xlim(1.12, -1.12)
        ax.set_ylim(-0.08, 1.14)
        ax.set_xticks([1, 0, -1])
    else:
        ax.set_xlim(-0.1, 1.12)
        ax.set_ylim(-0.05, 1.12)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        if LAND:                                   # narrow quarter-arc panel: short tick labels
            ax.set_xticklabels(["0", "0.5", "1"])
            ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_aspect("equal")
    ALAB = FS["label"] + (3 if (WIDE and not DV) else 8 if LAND else 5)
    ax.set_xlabel(r"$z$" if SEMI else r"$x$", labelpad=0 if (WIDE and not DV) else 2, fontsize=ALAB)
    if col == 0:
        ax.set_ylabel(r"$x$" if SEMI else r"$z$", fontsize=ALAB)
    else:
        ax.set_yticklabels([])
    ax.set_title(rf"({ROMAN[col]})\enspace$T={Tg:.0f}$ ns", fontsize=FS["title"] + (2 if not LAND else -3 if WIDE else 0),
                 pad=10 if not LAND else 24)       # landscape: clear of the snapshot numbers above the arc
    if col == 0:
        panel_label(ax, "a", y=1.0 if not LAND else 1.0 + 0.33 / TOPH["a"])   # landscape: level with the titles
    # (b) closeness to the path at equal times
    ax = ax_b = ax_at(x0, y_b, COLW, TOPH["b"])
    for g in GATES:
        if lists[g]:
            t, m, s, _ = M.curve_stat(lists[g], "FL")
            ax.fill_between(t, m - s, m + s, color=M.GATE_COLOR[g], alpha=0.20, lw=0)
            ax.plot(np.array(rr[g]["tL_ns"]), np.array(rr[g]["FL"]), color=M.GATE_COLOR[g], lw=LW)
    Tmax = Tg
    ax.set_xlim(0, Tmax)
    ax.set_ylim(0.0, B_TOP)
    ax.set_yticks([0.5, 1])                        # no 0 label: it would meet the 1.0 of (c) below
    ax.set_xticks(ticks(Tmax))
    ax.set_xticklabels([])
    if col == 0:
        if WIDE and not DV:                        # Letter two-column: the symbol (caption), same size as F_T and z
            ax.set_ylabel(r"$F$", fontsize=ALAB)
        else:
            ax.set_ylabel(r"closeness $F$", fontsize=FS["label"] + (-3 if (WIDE or not LAND) else -1))   # within (b)
        panel_label(ax, "b", y=0.96 if not LAND else 1.07)
    else:
        ax.set_yticklabels([])
    # errors at T (percent): first-order error of the path itself and 1 - F_T of every gate set (mean over usable runs)
    errs = [(rf"$\epsilon_1$: {100 * eps1(ref):.2f}\%" if QUAD and not LAND else
             rf"path $\epsilon_1$: {100 * eps1(ref):.2f}\%", S.INK)]   # Letter: eps_1 of the path (caption), 2 x 2 fits
    cells = [(f"{100 * eps1(ref):.2f}", S.INK)]   # the same numbers, for the table of the landscape layout
    ccells = []                                   # Letter two-column table: minimum closeness C of each gate set
    col_facts = dict(T_ns=Tg, eps1_path=eps1(ref))
    for g in GATES:
        if lists[g]:
            mF, sF, n = M.stat(lists[g], "F_T_final_noisy")
            mC, sC, _ = M.stat(lists[g], "C_noisy")
            errs.append((rf"{M.GATE_SHORT[g]}: {100 * (1 - mF):.2f}\%" if 1 - mF < 0.1 else
                         rf"{M.GATE_SHORT[g]}: {100 * (1 - mF):.1f}\%", M.GATE_COLOR[g]))
            cells.append((f"{100 * (1 - mF):.2f}" if 1 - mF < 0.1 else f"{100 * (1 - mF):.1f}", M.GATE_COLOR[g]))
            ccells.append((f"{mC:.2f}", M.GATE_COLOR[g]))   # minimum closeness C under noise, mean over the starts
            col_facts[g] = dict(n_usable=n, n_trivial=M.n_trivial(D_[(g, k)]), F_T_mean=mF, F_T_sd=sF, C_mean=mC, C_sd=sC,
                                F_T_min=float(min(r["F_T_final_noisy"] for r in lists[g])),
                                F_T_max=float(max(r["F_T_final_noisy"] for r in lists[g])),
                                rep_seed=rr[g]["seed"], rep_F_T=rr[g]["F_T_final_noisy"], rep_C=rr[g]["C_noisy"])
        else:
            cells.append(None)
            ccells.append(None)
    table.append(cells)
    ctable.append(ccells)
    facts["columns"][str(k)] = col_facts
    # (c) target fidelity
    ax = ax_at(x0, y_c, COLW, TOPH["c"])
    ax.plot(np.array(ref["t_ns"]), ref["F_T_path"], color=PATHC, lw=LWR - 0.6, ls=(0, (5, 2)))
    for g in GATES:
        if lists[g]:
            t, m, s, _ = M.curve_stat(lists[g], "FTL")
            ax.fill_between(t, m - s, m + s, color=M.GATE_COLOR[g], alpha=0.20, lw=0)
            ax.plot(np.array(rr[g]["tL_ns"]), np.array(rr[g]["FTL"]), color=M.GATE_COLOR[g], lw=LW)
    ax.set_xlim(0, Tmax)
    ax.set_ylim(0.0, 1.03)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks(ticks(Tmax))
    ax.set_xlabel(r"$t$ (ns)", fontsize=FS["label"] + 3)
    if not LAND or WIDE:                           # larger tick labels: the "0" of t clear of the "0.0" of F_T
        ax.tick_params(axis="x", pad=9)
        ax.tick_params(axis="y", pad=7)
    if col == 0:
        ax.set_ylabel(r"$F_T$", fontsize=ALAB if (WIDE and not DV) else FS["label"] + (5 if LAND else 3))
        panel_label(ax, "c", y=0.97)
    else:
        ax.set_yticklabels([])
    if LAND:                                       # errors at T: table under (d); snapshot badges between (b-ii), (c-ii)
        if k == KW:
            for j, f in enumerate(SNAP):
                badge(ax, f * Tg / ax.get_xlim()[1], 1 + 0.5 * GAPS["bc"] / TOPH["c"], j, transform=ax.transAxes,
                      pad=0.10)
    elif QUAD:                                     # 2 x 2 block above F = 1 in (b)
        for i, (txt, c) in enumerate(errs):
            ax_b.text(0.01 + 0.47 * (i % 2), 0.975 - 0.16 * (i // 2), txt, transform=ax_b.transAxes, ha="left", va="top",
                      fontsize=FS["small"] - 2, color=c)
    else:                                          # half arc: F_T starts near 0, the upper left of (c) is empty
        for i, (txt, c) in enumerate(errs):
            ax.text(0.03, 0.96 - 0.135 * i, txt, transform=ax.transAxes, ha="left", va="top", fontsize=FS["small"],
                    color=c, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    for axx in (ax_b, ax):
        if k == KW:
            for f in SNAP:
                axx.axvline(f * Tg, color=S.INK, lw=1.4, ls=(0, (1, 2)), zorder=1,
                            ymax=(1.03 / B_TOP) if (axx is ax_b and QUAD) else 1)
    if k == KW:
        axc_sel, T_sel = ax, Tg

# (d) Wigner functions at the numbered times of the snapshot column: path state and the run of each gate set, exact on
# the shown window |q|,|p| <= 5 (mfcommon.wigner_grid; review Nov 1, L1-F1).  The path row is recomputed for the path
# states of manuscript/.../wigner_snapshots.py (same reference path, cutoff 90, same stored fractions), whose stored
# W_path is not used; the plotted arrays are saved to <stem>_wigner.npz next to the figure.
Zc = np.load(M.MS / "fig4_cs_path_following" / "results" / "wigner" / f"{NAME}_geodesic_cs_k{KW}.npz")
fr = Zc["fracs"]
ii = [int(np.argmin(np.abs(fr - f))) for f in SNAP]
S9, T9 = CP.transfer(NAME, 90)
path9 = tracking.Path(CP.pad(CP.reference_dense(NAME, "geodesic", 80), 90))
WG = M.wigner_grid(90, len(Zc["x"]), float(Zc["x"][-1]))
assert np.allclose(WG.x, Zc["x"])
rows = [("brachistochrone", PATHC, [WG.of_vectors(gv[None]) for gv in path9.at(fr[ii])[0]], None)]
WSAVE = dict(x=WG.x, fracs=fr[ii], path=np.array(rows[0][2]))
Gp, _ = path9.at(np.array(SNAP))
check = {}
for g in GATES:
    r = REP[(g, KW)]
    if r is None:
        continue
    x, Ws, used = M.wigner_of_run(r, SNAP)
    Fr = M.closeness_of_run(r, Gp, SNAP)
    Ftr = [at(r, "FL", f) for f in SNAP]
    check[g] = float(np.max(np.abs(np.array(Fr) - np.array(Ftr))))
    assert check[g] < 2e-3, (g, Fr, Ftr)          # the Wigner panels show the state of the plotted run
    rows.append((M.GATE_SHORT[g], M.GATE_COLOR[g], Ws, Fr))
    WSAVE[g] = np.array(Ws)
    WSAVE[f"{g}_seed"] = r["seed"]
facts["wigner_check_max_dF"] = check
xg = WG.x
ext = [xg[0], xg[-1], xg[0], xg[-1]]
if not LAND:
    for i, (lab, c, Ws, Fs) in enumerate(rows):   # (d) from the left edge of column (i) to the right edge of (iii)
        yt = y_d + i * (PANEL_W + GAP_W)
        axl = ax_at(LEFT - LAB_W, yt, LAB_W, PANEL_W)
        axl.axis("off")
        axl.text(0.62, 0.5, "path" if i == 0 else lab, rotation=90, ha="center", va="center", fontsize=FS["ann"],
                 color=c, transform=axl.transAxes)   # "path" as in the landscape layout (the legend names it)
        for j in range(5):
            ax = ax_at(LEFT + j * (PANEL_W + GAP_W), yt, PANEL_W, PANEL_W)
            im = ax.imshow(Ws[j], origin="lower", extent=ext, cmap=S.RBCMAP, vmin=-S.W_LIM, vmax=S.W_LIM)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(c)
                sp.set_linewidth(3.4)
            if i == 0:                             # the time of the snapshot, centred above the column
                ax.text(0.5, 1.105, rf"$t={SNAP[j] * T_sel:.0f}$ ns", transform=ax.transAxes, ha="center",
                        va="center", fontsize=FS["ann"] + 1)
                if axc_sel is not None:            # badge j under the time axis of (c-ii), at t_k, with an arrow
                    xa = SNAP[j] * T_sel / axc_sel.get_xlim()[1]
                    yb = -BADGE_IN / TOPH["c"]
                    badge(axc_sel, xa, yb, j, transform=axc_sel.transAxes)
                    fig.add_artist(ConnectionPatch(xyA=(xa, yb), coordsA=axc_sel.transAxes, xyB=(0.5, 1.24),
                                                   coordsB=ax.transAxes, arrowstyle="-|>", mutation_scale=22, lw=2.0,
                                                   color=S.INK, shrinkA=13, shrinkB=3, zorder=0))
            if Fs is not None:
                ax.text(0.04, 0.04, rf"$F={Fs[j]:.2f}$", transform=ax.transAxes, fontsize=FS["small"], zorder=2,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))   # under the frame (spines: 2.5)
            if i == 0 and j == 0:
                panel_label(ax, "d", y=1.13)
    cax = ax_at(LEFT + BLOCK_W + CB_GAP, y_d, CB_W, WIG_H)
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-S.W_LIM, 0, S.W_LIM])
    cb.set_ticklabels([r"$-1/\pi$", r"$0$", r"$1/\pi$"])
    cb.ax.tick_params(labelsize=FS["tick"] + 2, pad=4)
    fig.canvas.draw()                            # W(q,p) just right of the narrow middle tick label "0", not beyond "-1/pi"
    zero = next(t for t in cb.ax.get_yticklabels() if "0" in t.get_text() and "pi" not in t.get_text())
    bb = zero.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    pc = cax.get_position()
    fig.text(bb.x1 + 0.10 / W_IN, pc.y0 + pc.height / 2, r"$W(q,p)$", rotation=90, ha="left", va="center",
             fontsize=FS["label"] + 2)
elif WIDE and not DV:
    # (d), Letter two-column: columns t_1 ... t_5 of the snapshot column (ii) (badge over the time), rows path, CS, CD,
    # SNAP; the errors at T are tabulated under the colour bar
    XG = XR + RL
    for i, (lab, c, Ws, Fs) in enumerate(rows):
        yt = y_a + i * (PANEL_W + GAP_W)
        fig.text((XG - 0.5 * RL) / W_IN, 1 - (yt + 0.5 * PANEL_W) / H_IN, "path" if i == 0 else lab, rotation=90,
                 ha="center", va="center", fontsize=FS["ann"] + 2, color=c)
        for j in range(5):
            ax = ax_at(XG + j * (PANEL_W + GAP_W), yt, PANEL_W, PANEL_W)
            im = ax.imshow(Ws[j], origin="lower", extent=ext, cmap=S.RBCMAP, vmin=-S.W_LIM, vmax=S.W_LIM)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(c)
                sp.set_linewidth(3.0)
            if Fs is not None:                     # closeness of the run to the path state at t_j
                ax.text(0.05, 0.05, rf"$F={Fs[j]:.2f}$", transform=ax.transAxes, fontsize=FS["small"] - 1, zorder=2,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
            if i == 0:                             # column header: badge j over the time t_j
                badge(ax, 0.5, 1.0 + 0.50 / PANEL_W, j, transform=ax.transAxes, pad=0.10)
                ax.text(0.5, 1.0 + 0.16 / PANEL_W, rf"${SNAP[j] * T_sel:.0f}$ ns", transform=ax.transAxes,
                        ha="center", va="center", fontsize=FS["small"])
                if j == 0:                         # level with "(a)"
                    S.panel_label(ax, "d", x=-RL / PANEL_W - 0.12, y=1.0 + 0.33 / PANEL_W)
else:
    # (d), landscape: rows t_1 ... t_5 of the snapshot column (ii) (badge over the time), columns path, CS, CD, SNAP
    XG = XR + RL                                 # left edge of the Wigner grid
    XC = XT = [XG + i * (PANEL_W + GAP_W) + 0.5 * PANEL_W for i in range(4)]
    for i, (lab, c, Ws, Fs) in enumerate(rows):
        fig.text(XC[i] / W_IN, 1 - (y_a - 0.07) / H_IN, "path" if i == 0 else lab, ha="center", va="bottom",
                 fontsize=FS["ann"] + 2, color=c)
        for j in range(5):
            ax = ax_at(XG + i * (PANEL_W + GAP_W), y_a + j * (PANEL_W + GAP_W), PANEL_W, PANEL_W)
            im = ax.imshow(Ws[j], origin="lower", extent=ext, cmap=S.RBCMAP, vmin=-S.W_LIM, vmax=S.W_LIM)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(c)
                sp.set_linewidth(3.0)
            if Fs is not None:                     # closeness of the run to the path state at t_j
                ax.text(0.05, 0.05, rf"${Fs[j]:.2f}$", transform=ax.transAxes, fontsize=FS["small"] - 2, zorder=2,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))   # under the frame (spines: 2.5)
            if i == 0:                             # row label: badge j over the time t_j
                xl = -0.5 * RL / PANEL_W
                badge(ax, xl, 0.70, j, transform=ax.transAxes, pad=0.10)
                ax.text(xl, 0.21, rf"${SNAP[j] * T_sel:.0f}$ ns", transform=ax.transAxes, ha="center", va="center",
                        fontsize=FS["small"])
                if j == 0:                         # level with "(a)"
                    S.panel_label(ax, "d", x=-RL / PANEL_W, y=1.0 + 0.33 / PANEL_W)
if LAND:
    GW = NC * PANEL_W + (NC - 1) * GAP_W         # width of the Wigner grid
    TH = WIDE and not DV                         # Letter two-column (d): vertical colour bar right of the grid, legend
    if TH:                                       # under it; the errors at T and the closeness: a table of the Letter
        cax = ax_at(XG + GW + CBV_GAP, y_a, CBV_W, GRID_H)
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([-S.W_LIM, 0, S.W_LIM])
        cb.set_ticklabels([r"$-1/\pi$", r"$0$", r"$1/\pi$"])
        cb.ax.tick_params(labelsize=FS["tick"] + 2, pad=4)
        fig.canvas.draw()                        # W(q,p) just right of the narrow middle tick label "0"
        zero = next(t for t in cb.ax.get_yticklabels() if "0" in t.get_text() and "pi" not in t.get_text())
        bb = zero.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        pc = cax.get_position()
        fig.text(bb.x1 + 0.10 / W_IN, pc.y0 + pc.height / 2, r"$W(q,p)$", rotation=90, ha="left", va="center",
                 fontsize=FS["label"] + 1)
    else:
        ycb = y_a + GRID_H + CB_GAP              # colour bar under the grid, W(q,p) to its left
        xcb0, xcb1 = XG + 0.92, XG + GW - 0.30
        cax = ax_at(xcb0, ycb, xcb1 - xcb0, CB_H)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([-S.W_LIM, 0, S.W_LIM])
        cb.set_ticklabels([r"$-1/\pi$", r"$0$", r"$1/\pi$"])
        cb.ax.tick_params(labelsize=FS["tick"] + 2, pad=4)
        fig.text((xcb0 - 0.42) / W_IN, 1 - (ycb + 0.5 * CB_H) / H_IN, r"$W(q,p)$", ha="right", va="center",
                 fontsize=FS["label"] + 1)
        yt0 = ycb + CB_H + CB_TICK + TAB_GAP

        # errors at T (%) of the columns (i)-(iii): eps_1 of the path, 1 - F_T of each gate set (mean over the starts)
        def tab(x, y, s, c, ha="center"):
            fig.text(x / W_IN, 1 - y / H_IN, s, ha=ha, va="center", fontsize=FS["ann"], color=c)

        tab(XG - 0.12, yt0 + 0.5 * TAB_ROW, r"(\%)", S.INK, ha="right")
        tab(XT[0], yt0 + 0.5 * TAB_ROW, r"$\epsilon_1$", S.INK)
        tab(0.5 * (XT[1] + XT[3]), yt0 + 0.5 * TAB_ROW, r"$1-F_T$", S.INK)
        for r_, cells in enumerate(table):
            y = yt0 + (r_ + 1.5) * TAB_ROW
            tab(XG - 0.12, y, f"({ROMAN[r_]})", S.INK, ha="right")
            for i, cell in enumerate(cells):
                if cell is not None:
                    tab(XT[i], y, cell[0], cell[1])
# legend: path and the three gate sets
hs = [Line2D([], [], color=PATHC, lw=LWR, ls=(0, (5, 2)))] + [Line2D([], [], color=M.GATE_COLOR[g], lw=LW) for g in GATES]
labs = ["brachistochrone"] + [M.GATE_NAME[g] for g in GATES]
if WIDE and not DV:                              # legend with the full names under the grid and its colour bar
    fig.legend(hs, labs, loc="upper center", ncol=2, fontsize=FS["legend"] - 3,
               bbox_to_anchor=((XG + 0.5 * (GW + CBV_GAP + CBV_W + CBV_LAB)) / W_IN, 1 - (y_a + GRID_H + 0.34) / H_IN),
               handlelength=1.8, labelspacing=0.25, columnspacing=1.4, borderaxespad=0.0, borderpad=0.0, frameon=False)
else:
    fig.legend(hs, labs, loc="lower center", ncol=2, fontsize=FS["legend"] + (3 if LAND else 0),   # Letter: under (d),
               bbox_to_anchor=(0.5 if LAND else (LEFT + 0.5 * BLOCK_W) / W_IN, 0.0),          # clear of its colour bar
               columnspacing=2.4 if LAND else 1.0, handlelength=2.6 if LAND else 1.8, frameon=False)
S.save(fig, OUT / STEM)
np.savez_compressed(OUT / f"{STEM}_wigner.npz", **WSAVE)
(OUT / f"{STEM}_facts.json").write_text(json.dumps(facts, indent=1))
print("wrote", OUT / f"{STEM}.png", "| wigner closeness check (max |dF|):", check)
for k, cf in facts["columns"].items():
    print(f"  k={k} T={cf['T_ns']:.0f} ns: eps1 path {100 * cf['eps1_path']:.2f}%  " + "  ".join(
        f"{g}: n={cf[g]['n_usable']} (+{cf[g]['n_trivial']} trivial) F_T={cf[g]['F_T_mean']:.4f}+-{cf[g]['F_T_sd']:.4f} "
        f"C={cf[g]['C_mean']:.3f}+-{cf[g]['C_sd']:.3f}" for g in GATES if g in cf))
