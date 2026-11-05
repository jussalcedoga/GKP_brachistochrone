"""Shared data loading and drawing of Supplementary Figs. S25-S27, the pulses (Nov 1).

Reads only the files of results/pulses/ (pulses_extract.py, pulses_stats.py, pulses_closed_target.py); no simulation.
Representation, the same in every figure: one column per gate set, rows
  (a) qubit drive Omega/10g = (Omega_x + i Omega_y)/10g,
  (b) sigma_z-conditional part u_- = (u_g - u_e)/2 of the branch amplitudes u_j = xi_j (CS) or varepsilon_j (CD),
  (c) common part u_+ = (u_g + u_e)/2, the cavity drive varepsilon for SNAP,
      since sum_j |j><j| (x) G(u_j) = 1 (x) G(u_+) + sigma_z (x) G(u_-) for a generator G linear in u,
  (d) oscillator purity Tr rho^2 (closed evolution, cutoff 90, 8 points per slice; = qubit purity, the joint state is
      pure), the shown run and the mean +- s.d. over the starts (band clipped to the physical range [1/2, 1]).
Layout 'cols' (one column per gate set) is used for S25-S27: each time axis is about 50% wider than in the 'rows'
candidate (one row per gate set, S25/_candidates/S25_rows.png), and the conditional part, common part and purity of one
run share a vertically aligned time axis.  Every drawn run is checked independently by sims/pulses_fig_check.py.
Controls are piecewise constant (steps), real part solid, imaginary part dashed, both in the gate colour, the modulus
+-|.| shaded, normalized to the bounds |Omega| <= 10 g and |u_j| <= g (so |u_+-| <= g), bounds dotted at +-1.
For SNAP the conditional row holds no control: H_SNAP = -chi_d |e><e| (x) n = -(chi_d/2) n + (chi_d/2) sigma_z (x) n, a
fixed conditional rotation.
"""
from __future__ import annotations
import layout as _L  # repository layout: data/ and output/ (src/layout.py)

import json
import pathlib
import sys

MF = _L.vfile(__file__).parents[1]
sys.path.insert(0, str(MF))
import mfcommon as M  # noqa: E402,I001

S = M.setup()
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

FS = M.FS
R = MF / "sims" / "results" / "pulses"
INK = S.INK
PATHC = S.ROYALBLUE
DOT = (0, (1, 2.2))
ALPHAS = (0.15, 0.35, 0.55, 0.78, 1.0)       # badge opacity 1 -> 5, as plot_fig4.badge
LW_STEP = 3.0                                # piecewise-constant controls (50 slices per panel)
LW_PUR = M.LW                                # smooth purity of the shown run


def _pm(u):
    return 0.5 * (u[0] + u[1]), 0.5 * (u[0] - u[1])


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, int(len(v))


# ----------------------------------------------------------------------------------------------- data
def tracking_column(gate, T):
    """representative Fig. 4 run (mfcommon.representative, stored flag) of GKP preparation along the brachistochrone"""
    z = np.load(R / f"tracking_gkp_{gate}_T{T}.npz")
    zc = np.load(R / f"closed_tracking_gkp_{gate}_T{T}.npz")
    i = int(np.flatnonzero(z["representative"])[0])
    assert int(zc["seed"][i]) == int(z["seed"][i]) and bool(zc["representative"][i])
    col = dict(gate=gate, color=M.GATE_COLOR[gate], title=M.GATE_SHORT[gate], T_ns=float(z["T_ns"]), N=int(z["N"]),
               te=np.asarray(z["t_edges_ns"], float), Om=z["Omega_over_bound"][i], seed=int(z["seed"][i]),
               chi=1.0 if gate == "snap" else None, pur_t=np.asarray(zc["t_ns"], float),
               pur_rep=[(np.asarray(zc["purity"][i, 0], float), "-")], pur_all=np.asarray(zc["purity"][:, 0], float))
    if gate == "snap":
        col["eps"] = z["eps"][i]
    else:
        col["up"], col["um"] = _pm((z["u_g"][i], z["u_e"][i]))
    F = np.asarray(z["F_T_noisy"], float)
    col["facts"] = dict(seed=col["seed"], F_rep=float(F[i]), F_stat=ms(F), n=len(F), n_trivial=int(z["trivial"].sum()),
                        trivial_seeds=[int(s) for s in z["seed"][z["trivial"]]],
                        purity_min_rep=float(col["pur_rep"][0][0][::8].min()),
                        purity_min_stat=ms(col["pur_all"][:, ::8].min(axis=1)),
                        purity_final_stat=ms(col["pur_all"][:, -1]))
    return col


TARGET_TITLE = {"cs": "CS", "cd": "CD", "snap": r"SNAP, $\chi_d=g$", "snap4": r"SNAP, $\chi_d=4g$"}


def target_column(gk):
    """target-only pulse at 412 ns (Letter Table I), start whose Lindblad fidelity is closest to the mean"""
    z = np.load(R / f"closed_target_{gk}_T412.npz")
    i = int(np.flatnonzero(z["representative"])[0])
    gate = "snap" if gk.startswith("snap") else gk
    C = z["C"][i]                                       # units of g: Omega_x, Omega_y, Re, Im of u_g, u_e
    T, N = float(z["T_ns"]), int(z["N"])
    col = dict(gate=gate, color=M.GATE_COLOR[gate], title=TARGET_TITLE[gk], T_ns=T, N=N,
               te=T * np.arange(N + 1) / N, Om=(C[:, 0] + 1j * C[:, 1]) / 10.0, seed=int(z["seed"][i]),
               chi=float(z["chi_d_over_g"]) if gate == "snap" else None, pur_t=np.asarray(z["t_ns"], float),
               pur_rep=[(np.asarray(z["purity"][i], float), "-")], pur_all=np.asarray(z["purity"], float))
    if gate == "snap":
        col["eps"] = C[:, 2] + 1j * C[:, 3]
    else:
        col["up"], col["um"] = _pm((C[:, 2] + 1j * C[:, 3], C[:, 4] + 1j * C[:, 5]))
    F = np.asarray(z["F90_noisy"], float)
    col["facts"] = dict(seed=col["seed"], F_rep=float(F[i]), F_stat=ms(F), n=len(F),
                        purity_min_rep=float(col["pur_rep"][0][0][::8].min()),
                        purity_min_stat=ms(col["pur_all"][:, ::8].min(axis=1)),
                        purity_final_stat=ms(col["pur_all"][:, -1]))
    return col


def flip_column(gate, T=824, seed=None, title=None):
    """gate-only flip pulse (sims/flip_gate.py, mode target): the representative (score closest to the mean) or `seed`;
    purity row: the smallest purity over the inputs |0_L>, |1_L>, |+_L> at each time (band only for the representative)"""
    z = np.load(R / f"flip_target_{gate}_T{T}.npz")
    zc = np.load(R / f"closed_flip_target_{gate}_T{T}.npz")
    i = int(np.flatnonzero(z["representative"])[0]) if seed is None else int(np.flatnonzero(z["seed"] == seed)[0])
    assert int(zc["seed"][i]) == int(z["seed"][i]) and list(zc["inputs"]) == ["0", "1", "+"]
    pmin = np.asarray(zc["purity"], float).min(axis=1)                    # (starts, t): smallest over the inputs
    col = dict(gate=gate, color=M.GATE_COLOR[gate], title=title or M.GATE_SHORT[gate], T_ns=float(z["T_ns"]),
               N=int(z["N"]), te=np.asarray(z["t_edges_ns"], float), Om=z["Omega_over_bound"][i], seed=int(z["seed"][i]),
               chi=1.0 if gate == "snap" else None, pur_t=np.asarray(zc["t_ns"], float), pur_rep=[(pmin[i], "-")],
               pur_all=pmin, band=seed is None, representative=seed is None)
    if gate == "snap":
        col["eps"] = z["eps"][i]
    else:
        col["up"], col["um"] = _pm((z["u_g"][i], z["u_e"][i]))
    F = np.asarray(z["avg6_opt_lind"], float)
    pm = np.asarray(zc["purity"], float)[:, :, ::8].min(axis=2)          # (starts, inputs) minimum at the slice ends
    col["facts"] = dict(seed=col["seed"], F_rep=float(F[i]), F_stat=ms(F), F_best=float(F.max()), n=len(F),
                        purity_min_rep=[float(v) for v in pm[i]], purity_min_over_inputs_stat=ms(pm.min(axis=1)),
                        purity_final_min_over_inputs_rep=float(np.asarray(zc["purity"])[i, :, -1].min()),
                        nbar_max_rep=float(np.asarray(zc["nbar"])[i, :, ::8].max()),
                        nbar_max_stat=ms(np.asarray(zc["nbar"])[:, :, ::8].max(axis=(1, 2))))
    return col


def shape_facts(col):
    """numbers of the drawn pulse itself (definitions of sims/pulses_stats.control_stats): conditional fraction
    int|u_-|^2 / int(|u_+|^2 + |u_-|^2), slices with |Omega| at its bound (>= 0.999), time-averaged |Omega|/10g;
    SNAP: largest |eps|/g and slices with |eps| at its bound"""
    r = np.abs(col["Om"])
    f = dict(drive_at_bound=float(np.mean(r >= 0.999)), drive_mean=float(r.mean()))
    if col["gate"] == "snap":
        a = np.abs(col["eps"])
        f.update(osc_max=float(a.max()), osc_at_bound=float(np.mean(a >= 0.999)))
    else:
        um2, up2 = np.abs(col["um"]) ** 2, np.abs(col["up"]) ** 2
        f.update(cond_frac=float(um2.sum() / (um2 + up2).sum()), um_mean=float(np.sqrt(um2).mean()),
                 up_mean=float(np.sqrt(up2).mean()))
    col["facts"]["shape"] = f
    return f


_ROWS_A = []


def check_against_A(col, fname):
    """the drawn run against data-A's independent per-start numbers (pulses_stats_A_rows.json): conditional fraction,
    drive at bound, and the minimum purity at the slice ends; returns the largest deviation"""
    if not _ROWS_A:
        _ROWS_A.extend(json.loads((R / "pulses_stats_A_rows.json").read_text()))
    r = next(r for r in _ROWS_A if r["file"] == fname and int(r["seed"]) == col["seed"])
    f = shape_facts(col)
    dev = [abs(r["controls"]["drive_at_bound"] - f["drive_at_bound"]), abs(r["controls"]["drive_mean"] - f["drive_mean"])]
    if "cond_frac" in f:
        dev.append(abs(r["controls"]["cond_frac"] - f["cond_frac"]))
    pm = col["facts"]["purity_min_rep"]
    pm = pm if isinstance(pm, list) else [pm]
    dev += [abs(r["closed"][nm]["purity_min"] - v) for nm, v in zip(("0", "1", "+"), pm)]
    d = float(max(dev))
    assert d < 1e-6, (fname, col["seed"], d)
    col["facts"]["max_dev_vs_data_A"] = d
    return d


def fmt_pm(m, s):
    """mean +- s.d. in the convention of the SI text: the s.d. to one significant digit, two if that digit is 1, and the
    mean to the same decimal place"""
    lead, e = f"{s:.1e}".split("e")                  # '1.9e-04': leading digit and decade after rounding
    d = max(-int(e) + (1 if lead[0] == "1" else 0), 0)
    return f"{m:.{d}f}" + r"\pm" + f"{s:.{d}f}"


def table_pm(entry, m, s):
    """'0.797(15)' of a table, checked against the data (mean and s.d. rounded to the decimal place of the entry), as
    '0.797\\pm0.015'"""
    import re
    q, e = re.fullmatch(r"([0-9.]+)\(([0-9]+)\)", entry).groups()
    d = len(q.split(".")[1])
    sd = f"{int(e) * 10.0 ** -d:.{d}f}"
    assert f"{m:.{d}f}" == q and f"{s:.{d}f}" == sd, (entry, m, s)
    return q + r"\pm" + sd


# ----------------------------------------------------------------------------------------------- drawing
def _steps(ax, te, v, color, ls="-", lw=LW_STEP, z=3):
    ax.plot(te, np.r_[v, v[-1]], drawstyle="steps-post", color=color, ls=ls, lw=lw, zorder=z,
            solid_joinstyle="miter", dash_joinstyle="miter")


def _control(ax, col, u):
    te, c = col["te"], col["color"]
    a = np.abs(u)
    ax.fill_between(te, -np.r_[a, a[-1]], np.r_[a, a[-1]], step="post", color=c, alpha=0.16, lw=0, zorder=1)
    _steps(ax, te, u.real, c, "-", z=4)
    _steps(ax, te, u.imag, c, (0, (2.2, 1.3)), z=3)
    for y in (-1, 1):
        ax.axhline(y, color=INK, lw=1.5, ls=DOT, zorder=2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1, 0, 1])


def _empty_snap(ax, col, fs=None):
    """conditional row of SNAP: u_- = 0 (no control), a fixed conditional rotation from the dispersive term (chi_d in
    the title or the caption)"""
    for y in (-1, 1):
        ax.axhline(y, color=INK, lw=1.5, ls=DOT, zorder=2)
    ax.axhline(0, color=col["color"], lw=LW_STEP, zorder=3)
    box = dict(fc="white", ec="none", pad=1.5)
    ax.text(0.5, 0.71, "no conditional drive", transform=ax.transAxes, ha="center", va="center",
            fontsize=fs or FS["ann"], color=INK, bbox=box, zorder=5)
    ax.text(0.5, 0.29, r"fixed term $\tfrac12\chi_d\,\sigma_z\otimes\hat n$", transform=ax.transAxes,
            ha="center", va="center", fontsize=fs or FS["ann"], color=INK, bbox=box, zorder=5)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1, 0, 1])


def _purity(ax, col, band=True):
    """purity row; the band mean +- s.d. is clipped to the physical range [1/2, 1] of the purity of one factor of a
    pure qubit-oscillator state (the clipped amount is recorded in facts['band_clip'])"""
    c = col["color"]
    t = col["pur_t"]
    if band and col.get("band", True):
        m, s = col["pur_all"].mean(axis=0), col["pur_all"].std(axis=0, ddof=1)
        col["facts"]["band_clip"] = dict(min_mean_minus_sd=float((m - s).min()), max_mean_plus_sd=float((m + s).max()))
        ax.fill_between(t, np.clip(m + s, 0.5, 1.0), np.clip(m - s, 0.5, 1.0), color=c, alpha=0.20, lw=0, zorder=1)
    for p, ls in col["pur_rep"]:
        ax.plot(t, p, color=c, lw=LW_PUR if ls == "-" else LW_PUR - 0.8, ls=ls, zorder=3,
                solid_capstyle="butt", dash_capstyle="butt")
    ax.axhline(0.5, color=INK, lw=1.5, ls=DOT, zorder=2)
    ax.set_ylim(0.44, 1.04)
    ax.set_yticks([0.5, 1])
    ax.set_yticklabels(["0.5", "1"])


def ticks(T):
    for step in (100, 200, 250, 500):
        if T / step <= 4.5:
            return list(np.arange(0, T + 1e-6, step))
    return [0]


def badge(ax, x, y, j, transform, size=None, pad=0.22):
    """numbered snapshot badge j (0..4) of Fig. 4: royal-blue circle, fill opacity growing from 1 to 5"""
    a = ALPHAS[j]
    ax.text(x, y, str(j + 1), ha="center", va="center", fontsize=size or FS["small"], zorder=9, transform=transform,
            color="white" if a >= 0.5 else INK, clip_on=False,
            bbox=dict(boxstyle=f"circle,pad={pad}", fc=(*matplotlib.colors.to_rgb(PATHC), a), ec=PATHC, lw=1.8))


ROWS = ("drive", "cond", "common", "purity")
YLAB = {"drive": r"$\Omega/10g$", "cond": r"$u_-/g$", "common": r"$u_+/g$", "purity": r"$\mathrm{Tr}\,\rho^2$"}
QTITLE = {"drive": r"drive $\Omega/10g$", "cond": r"conditional $u_-/g$", "common": r"common $u_+/g$",
          "purity": r"purity $\mathrm{Tr}\,\rho^2$"}           # column titles of the 'rows' candidate


def draw(cols, out, layout="cols", snapshots=False, band=True, pur_label=None, W_IN=15.6, row_h=1.72,
         row_gap=0.2, left=1.62, right=0.30, col_gap=0.52):
    """layout 'cols': one column per gate set, rows (a)-(d) as in the module docstring;
    layout 'rows': one row per gate set, columns (a)-(d) (S25 candidate)"""
    nG, nQ = len(cols), len(ROWS)
    T = cols[0]["T_ns"]
    top = 0.62 + (0.62 if snapshots else 0.0)                       # gate titles (+ badge row)
    bottom = 0.98 + 0.62 + 0.42                                     # ticks and x label + two-row legend
    if layout == "cols":
        nr, nc = nQ, nG
        cw = (W_IN - left - right - (nc - 1) * col_gap) / nc
    else:
        nr, nc = nG, nQ
        left = 1.25
        col_gap = 1.02
        cw = (W_IN - left - right - (nc - 1) * col_gap) / nc
        top = 0.62 + (0.62 if snapshots else 0.0)
    H_IN = top + nr * row_h + (nr - 1) * row_gap + bottom
    fig = plt.figure(figsize=(W_IN, H_IN))

    def ax_at(x0, y_top, w, h):
        return fig.add_axes([x0 / W_IN, 1 - (y_top + h) / H_IN, w / W_IN, h / H_IN])

    LAB_X = 0.10                                                    # inches: shared x of the panel labels
    axes = {}
    for gi, col in enumerate(cols):
        for qi, q in enumerate(ROWS):
            r, c = (qi, gi) if layout == "cols" else (gi, qi)
            ax = ax_at(left + c * (cw + col_gap), top + r * (row_h + row_gap), cw, row_h)
            axes[(gi, qi)] = ax
            if q == "drive":
                _control(ax, col, col["Om"])
            elif q == "cond":
                _control(ax, col, col["um"]) if col["gate"] != "snap" else _empty_snap(
                    ax, col, fs=FS["ann"] if cw >= 2.9 else FS["small"] - 1)
            elif q == "common":
                _control(ax, col, col["eps"] if col["gate"] == "snap" else col["up"])
            else:
                _purity(ax, col, band=band)
            ax.set_xlim(0, col["T_ns"])
            ax.set_xticks(ticks(col["T_ns"]))
            if r < nr - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$t$ (ns)")
            if layout == "cols":
                if c == 0:
                    long = q == "purity" and pur_label
                    ax.set_ylabel(pur_label if long else YLAB[q], fontsize=FS["label"] + (0 if long else 2),
                                  labelpad=4, y=0.38 if long else 0.5)     # long label clear of the panel label
                    p = ax.get_position()
                    S.panel_label(ax, "abcd"[qi], x=(LAB_X / W_IN - p.x0) / p.width, y=0.80)
                else:
                    ax.set_yticklabels([])
            elif r == 0:                                            # label on the title line, clear of the badges
                yt = 1 + ((0.62 + 0.30) if snapshots else 0.30) / row_h
                ax.text(-0.30, yt, f"({'abcd'[qi]})", transform=ax.transAxes, fontsize=FS["panel"], va="center",
                        ha="left")
            if snapshots:
                for k in range(1, 6):
                    ax.axvline(k * col["T_ns"] / 5, color=INK, lw=1.4, ls=DOT, zorder=0)
            ax.tick_params(axis="both", which="major", pad=6)
    # titles and badges
    if layout == "cols":
        for gi, col in enumerate(cols):
            ax = axes[(gi, 0)]
            yt = 1 + ((0.62 + 0.30) if snapshots else 0.30) / row_h
            ax.text(0.5, yt, col["title"], transform=ax.transAxes, ha="center", va="center", fontsize=FS["label"] + 3,
                    color=col["color"])
            if snapshots:
                for j in range(5):
                    badge(ax, (j + 1) / 5, 1 + 0.34 / row_h, j, ax.transAxes)
    else:
        for qi, q in enumerate(ROWS):
            ax = axes[(0, qi)]
            yt = 1 + ((0.62 + 0.30) if snapshots else 0.30) / row_h
            ax.text(0.5, yt, QTITLE[q], transform=ax.transAxes, ha="center", va="center", fontsize=FS["label"],
                    color=INK)
            if snapshots:
                for j in range(5):
                    badge(ax, (j + 1) / 5, 1 + 0.34 / row_h, j, ax.transAxes)
        for gi, col in enumerate(cols):
            ax = axes[(gi, 0)]
            ax.text(-0.42, 0.5, col["title"], transform=ax.transAxes, ha="center", va="center", rotation=90,
                    fontsize=FS["label"] + 3, color=col["color"])
    if layout == "cols":                                            # one x for the y labels of the first column
        fig.align_ylabels([axes[(0, qi)] for qi in range(nQ)])
    # legend, two rows: line styles (ink) | the two shadings, keyed in the gate colours of the figure (an ink tint
    # would read as gray), since the modulus of (a)-(c) and the band of (d) are both light shadings
    gc = list(dict.fromkeys(col["color"] for col in cols))
    mod = tuple(Patch(fc=c, alpha=0.16, lw=0) for c in gc)
    bnd = tuple((Patch(fc=c, alpha=0.20, lw=0), Line2D([], [], color=c, lw=LW_PUR - 1.2)) for c in gc)
    hs = [Line2D([], [], color=INK, lw=LW_STEP), mod, Line2D([], [], color=INK, lw=LW_STEP, ls=(0, (2.2, 1.3))), bnd]
    labs = [r"real part ($\Omega_x$, $\mathrm{Re}\,u$)", r"modulus $\pm|\Omega|$, $\pm|u|$ (a--c)",
            r"imaginary part ($\Omega_y$, $\mathrm{Im}\,u$)",
            ("shown run and mean $\\pm$ s.d. over the starts (d)" if band else "shown run (d)")]
    fig.legend(hs, labs, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=FS["legend"], frameon=False,
               handlelength=3.0, handleheight=0.9, columnspacing=2.4, labelspacing=0.45,
               handler_map={mod: HandlerTuple(ndivide=None, pad=0.0), bnd: HandlerTuple(ndivide=None, pad=0.0)})
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)
    return out
