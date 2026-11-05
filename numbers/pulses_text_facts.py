"""Every number of the new SI section on the optimized pulses (Sec. X, Nov 2 2026), from the data.

No simulation.  Reads (read only)
  sims/results/pulses/tracking_gkp_<gate>_T<ns>.npz, flip_target_<gate>_T824.npz      (data-A: slice variables X)
  sims/results/pulses/closed_<case>.npz, closed_target_<gk>_T412.npz                  (closed traces, 8 points/slice)
  sims/results/pulses/capability_pulses.jsonl                                         (data-B: target-only X)
  sims/results/pulses/pulses_smooth.jsonl                                             (data-B: smoothing test rows)
  sims/results/pulses/pulses_facts_A.json, pulses_facts_B.json, pulses_fig_check.json, closed_target_check.json
  supplementary_information/S2{5,6,7}_*/S2x_facts.json                               (numbers of the captions)
and writes sims/results/pulses/pulses_text_facts.json with
  table  : the rows of the statistics table (mean, s.d., n, and the formatted LaTeX cell) for every column,
  text   : every number quoted in the prose of the section (value and formatted string),
  checks : recomputations from the raw arrays against the stored facts (all must hold, else the script stops).
Statistics are mean and sample s.d. (ddof = 1) over all starts, every start included (SNAP runs that stay at the
vacuum too).  Closed-evolution purity and photon number at the slice ends (every 8th point), as pulses_stats.py.
The cap of Eq. (purity-cap): for a pure qubit-oscillator state the oscillator state has the eigenvalues lambda and
1 - lambda, so lambda_max = [1 + (2 Tr rho^2 - 1)^(1/2)]/2 >= F (closeness), and C <= min_t lambda_max(t).
  python numbers/pulses_text_facts.py
Audit of Nov 2: the shares |u_-| >= 0.99 g and 'drive and conditional part both at least
half their bound' are recomputed here from X with an explicit tolerance; added the like-for-like flip/preparation numbers
at 412 ns, the independent Lindblad check of pulses_check_B.json, and checks of the comparative wording.
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import math
import pathlib
from collections import defaultdict

import numpy as np

HERE = _L.vfile(__file__).parent
MF = HERE.parent
P = HERE / "results" / "pulses"
SI = MF / "supplementary_information"
GATES = ("cs", "cd", "snap")
SUB = 8


def ms(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else 0.0, n=int(v.size),
                min=float(v.min()), max=float(v.max()))


def sd_round(sd):
    """s.d. rounded to one significant digit, or two when the leading digit is 1; returns (value, decimals)"""
    if sd <= 0:
        return 0.0, None
    e = math.floor(math.log10(sd))
    s2 = round(sd, -e + 1)
    if f"{s2:.10e}"[0] == "1":
        dec = -e + 1
    else:
        dec = -e
        s1 = round(sd, dec)
        if f"{s1:.10e}"[0] == "1":       # rounding up to a leading 1 (0.096 -> 0.10)
            dec = -math.floor(math.log10(s1)) + 1
    return round(sd, dec), dec


def paren(mean, sd, maxdec=3, mindec=None):
    """mean(s.d.) in the parenthesis notation of Table I, s.d. digits in units of the last digit"""
    s, dec = sd_round(sd)
    if dec is None or dec > maxdec:
        dec = maxdec if mindec is None else max(mindec, maxdec)
        s = round(sd, dec)
    if mindec is not None and dec < mindec:
        dec = mindec
    m = f"{mean:.{max(dec, 0)}f}"
    if m.startswith("-") and float(m) == 0:
        m = m[1:]
    digits = int(round(s * 10 ** dec))
    if digits == 0:
        return m
    return f"{m}({digits})"


def pm(mean, sd, maxdec=4):
    """mean \\pm s.d. with the s.d. rounding of the SI text"""
    s, dec = sd_round(sd)
    if dec is None:
        return f"{mean:.{maxdec}f}"
    dec = min(dec, maxdec)
    return f"{mean:.{max(dec, 0)}f}\\pm{s:.{max(dec, 0)}f}"


def frac_stats_from_X(X, gate):
    """per start: fraction of oscillator amplitudes at the bound (a >= 0.999, both branches for CS and ECD), conditional
    fraction, time-averaged qubit drive |Omega|/10g, from the slice variables (r, f_Omega, a_g, f_g, a_e, f_e)"""
    r, ag, fg, ae, fe = X[..., 0], X[..., 2], X[..., 3], X[..., 4], X[..., 5]
    drive_mean = r.mean(axis=1)
    if gate.startswith("snap"):
        atb = (ag >= 0.999).mean(axis=1)
        cond = np.full(len(X), np.nan)
    else:
        atb = np.concatenate([ag >= 0.999, ae >= 0.999], axis=1).mean(axis=1)
        ug, ue = ag * np.exp(1j * fg), ae * np.exp(1j * fe)
        up, um = (ug + ue) / 2, (ug - ue) / 2
        cond = (np.abs(um) ** 2).sum(axis=1) / (np.abs(up) ** 2 + np.abs(um) ** 2).sum(axis=1)
    return dict(osc_at_bound=atb, cond_frac=cond, drive_mean=drive_mean)


def closed_stats(path):
    """per start: smallest purity and largest photon number over the inputs at the slice ends; smallest purity on
    the fine grid (for the cap)"""
    z = np.load(path)
    pur, nb = z["purity"], z["nbar"]
    if pur.ndim == 2:                       # closed_target: (starts, points)
        pur, nb = pur[:, None, :], nb[:, None, :]
    ends = slice(0, None, SUB)
    return dict(purity_min=pur[:, :, ends].min(axis=(1, 2)), nbar_max=nb[:, :, ends].max(axis=(1, 2)),
                purity_min_fine=pur.min(axis=(1, 2)), purity_final_min=pur[:, :, -1].min(axis=1),
                seed=z["seed"])


def parse_pm(s):
    """'0.833 +- 0.033 (n = 20)' -> (0.833, 0.033, 20)"""
    a, b = s.split("+-")
    return float(a), float(b.split("(")[0]), int(b.split("=")[1].strip(" )"))


A = json.load(open(P / "pulses_facts_A.json"))
B = json.load(open(P / "pulses_facts_B.json"))
FC = json.load(open(P / "pulses_fig_check.json"))
CT = json.load(open(P / "closed_target_check.json"))
S25 = json.load(open(SI / "S25_pulses_tracking" / "S25_facts.json"))
S26 = json.load(open(SI / "S26_pulses_target" / "S26_facts.json"))
S27 = json.load(open(SI / "S27_pulses_flip" / "S27_facts.json"))
checks, text, table = {}, {}, []


def check(name, ok, info=None):
    checks[name] = dict(ok=bool(ok), info=info)
    if not ok:
        raise SystemExit(f"CHECK FAILED: {name} {info}")


# ------------------------------------------------------------------ smoothing: recompute Delta from the rows
rows = [json.loads(l) for l in open(P / "pulses_smooth.jsonl")]


def metric(r):
    return r["eval"]["lind"]["avg6_opt"] if r["set"] == "flip" else r["F_T_noisy"]


base, deltas = {}, defaultdict(list)
for r in rows:
    key = (r["set"], r["gate"], r["seed"], round(r["T_ns"]))
    if r["sigma_ns"] == 0:
        base[key] = metric(r)
for r in rows:
    if r["sigma_ns"] == 0 or r["edge"] != "zero":
        continue
    key = (r["set"], r["gate"], r["seed"], round(r["T_ns"]))
    deltas[(r["set"], r["gate"], round(r["T_ns"]), r["sigma_ns"])].append(metric(r) - base[key])


def smooth_cell(setname, gate, T, sig):
    d = np.array(deltas[(setname, gate, T, sig)]) * 1e3
    st = ms(d)
    bkey = {"track": f"track_{gate}_{T}", "target": f"target_{gate}_{T}", "flip": f"flip_{gate}_{T}"}[setname]
    q = "d_avg6_opt_noisy" if setname == "flip" else "d_F_T_noisy"
    bst = B["smoothing"]["groups"][bkey]["sigma"][str(int(sig))][q]
    check(f"smooth_{bkey}_{sig}", abs(bst["mean"] * 1e3 - st["mean"]) < 1e-9 and abs(bst["sd"] * 1e3 - st["sd"]) < 1e-9
          and bst["n"] == st["n"], (st["mean"], bst["mean"] * 1e3))
    return st


# ------------------------------------------------------------------ table rows
def add_row(group, label, T, gate, ctrl, closed, smooth, n, footnote=""):
    row = dict(group=group, label=label, T=T, gate=gate, n=n, footnote=footnote, cells={})
    row["cells"]["osc_at_bound"] = ctrl["osc_at_bound"]
    row["cells"]["cond_frac"] = ctrl["cond_frac"]
    row["cells"]["drive_mean"] = ctrl["drive_mean"]
    row["cells"]["purity_min"] = closed["purity_min"]
    row["cells"]["nbar_max"] = closed["nbar_max"]
    for s in (1.0, 2.0, 4.0):
        row["cells"][f"dF_{int(s)}"] = smooth[s] if smooth else None
    fmt = {}
    for k, v in row["cells"].items():
        if v is None or (isinstance(v, dict) and (v.get("mean") is None or np.isnan(v["mean"]))):
            fmt[k] = "--"
        elif k.startswith("dF_"):
            fmt[k] = "$" + paren(v["mean"], v["sd"], maxdec=2) + "$"
        elif k == "nbar_max":
            fmt[k] = paren(v["mean"], v["sd"], maxdec=2)
        else:
            fmt[k] = paren(v["mean"], v["sd"], maxdec=3)
    row["fmt"] = fmt
    table.append(row)
    return row


LAB = {"cs": "CS", "cd": "ECD", "snap": "SNAP"}

# preparation along the brachistochrone (Fig. 4)
cap_rows = {}
for gate in GATES:
    for T, k in ((412, 1), (824, 2), (2133, 3)):
        case = f"tracking_gkp_{gate}_T{T}"
        z = np.load(P / f"{case}.npz")
        X = z["X"]
        c = frac_stats_from_X(X, gate)
        st = A["stats"][case]["controls"]
        check(f"ctrl_{case}", abs(c["osc_at_bound"].mean() - st["osc_at_bound"]["mean"]) < 1e-12
              and abs(c["drive_mean"].mean() - st["drive_mean"]["mean"]) < 1e-12
              and (gate == "snap" or abs(c["cond_frac"].mean() - st["cond_frac"]["mean"]) < 1e-12))
        cl = closed_stats(P / f"closed_{case}.npz")
        sc = A["stats"][case]["closed"]["0"]
        check(f"closed_{case}", abs(cl["purity_min"].mean() - sc["purity_min"]["mean"]) < 1e-6
              and abs(cl["nbar_max"].mean() - sc["nbar_max"]["mean"]) < 1e-6
              and np.array_equal(cl["seed"], z["seed"]))
        ctrl = {kk: (ms(v) if not np.all(np.isnan(v)) else None) for kk, v in c.items()}
        closed = {kk: ms(cl[kk]) for kk in ("purity_min", "nbar_max")}
        smooth = {s: smooth_cell("track", gate, T, s) for s in (1.0, 2.0, 4.0)} if T < 2000 else None
        foot = ""
        if gate == "snap" and int(z["trivial"].sum()) > 0:
            foot = "trivial"
        add_row("track", LAB[gate], T, gate, ctrl, closed, smooth, len(X), foot)
        # cap of Eq. (purity-cap) against the closed closeness, per start
        lam = 0.5 * (1 + np.sqrt(np.clip(2 * cl["purity_min_fine"] - 1, 0, None)))
        Cc = z["C_closed"]
        cap_rows[case] = dict(cap=ms(lam), C_closed=ms(Cc), min_gap=float((lam - Cc).min()),
                              holds=bool(np.all(Cc <= lam + 1e-6)), trivial=z["trivial"].tolist())
        check(f"cap_{case}", cap_rows[case]["holds"], cap_rows[case]["min_gap"])

# target only (Table I), 412 ns
cap_jsonl = [json.loads(l) for l in open(P / "capability_pulses.jsonl")]
for gk, lab in (("cs", "CS"), ("cd", "ECD"), ("snap", r"SNAP, $\chi_d=g$"), ("snap4", r"SNAP, $\chi_d=4g$")):
    zz = np.load(P / f"closed_target_{gk}_T412.npz")
    X = zz["X"]
    rowsj = sorted([r for r in cap_jsonl if r["gate"] == gk and r["T_ns"] == 412.0], key=lambda r: r["seed"])
    check(f"target_rows_{gk}", len(rowsj) == 8 and [r["seed"] for r in rowsj] == list(range(1000, 1008)),
          sorted({(r["gate"], r.get("model"), r.get("chi_d_over_g")) for r in cap_jsonl}))
    if gk.startswith("snap"):
        check(f"target_chi_{gk}", all(abs((r.get("chi_d_over_g") or 0) - (4.0 if gk == "snap4" else 1.0)) < 1e-12
                                      for r in rowsj), [r.get("chi_d_over_g") for r in rowsj])
    Xj = np.array([r["X"] for r in rowsj])
    check(f"target_X_{gk}", Xj.shape == X.shape and float(np.abs(Xj - X).max()) == 0.0, Xj.shape)
    c = frac_stats_from_X(X, gk)
    ps = B["pulse_shapes"][f"target_{gk}_412"]
    check(f"ctrl_target_{gk}", abs(c["osc_at_bound"].mean() - ps["osc_at_bound"]["mean"]) < 1e-12
          and abs(c["drive_mean"].mean() - ps["drive_mean"]["mean"]) < 1e-12
          and (gk.startswith("snap") or abs(c["cond_frac"].mean() - ps["cond_frac"]["mean"]) < 1e-12))
    cl = closed_stats(P / f"closed_target_{gk}_T412.npz")
    ct = CT["cases"][gk]
    check(f"closed_target_{gk}", abs(cl["purity_min"].mean() - ct["purity_min_slice_ends"]["mean"]) < 1e-9
          and abs(cl["nbar_max"].mean() - ct["nbar_max"]["mean"]) < 1e-6)
    ctrl = {kk: (ms(v) if not np.all(np.isnan(v)) else None) for kk, v in c.items()}
    closed = {kk: ms(cl[kk]) for kk in ("purity_min", "nbar_max")}
    smooth = {s: smooth_cell("target", gk, 412, s) for s in (1.0, 2.0, 4.0)}
    add_row("target", lab, 412, gk, ctrl, closed, smooth, len(X))

# logical flip as a gate, gate objective alone, 824 ns
for gate in GATES:
    case = f"flip_target_{gate}_T824"
    z = np.load(P / f"{case}.npz")
    X = z["X"]
    c = frac_stats_from_X(X, gate)
    st = A["stats"][case]["controls"]
    check(f"ctrl_{case}", abs(c["osc_at_bound"].mean() - st["osc_at_bound"]["mean"]) < 1e-12
          and abs(c["drive_mean"].mean() - st["drive_mean"]["mean"]) < 1e-12
          and (gate == "snap" or abs(c["cond_frac"].mean() - st["cond_frac"]["mean"]) < 1e-12))
    cl = closed_stats(P / f"closed_{case}.npz")
    sm = A["stats"][case]["closed_min_over_inputs"]["purity_min"]
    check(f"closed_{case}", abs(cl["purity_min"].mean() - sm["mean"]) < 1e-6)
    gvt = parse_pm(A["checks"]["gate_vs_tracking_824ns"]["per_gate"][gate]["target"]["nbar_max"])
    check(f"nbar_{case}", abs(cl["nbar_max"].mean() - gvt[0]) < 0.006)
    ctrl = {kk: (ms(v) if not np.all(np.isnan(v)) else None) for kk, v in c.items()}
    closed = {kk: ms(cl[kk]) for kk in ("purity_min", "nbar_max")}
    smooth = {s: smooth_cell("flip", gate, 824, s) for s in (1.0, 2.0, 4.0)}
    add_row("flip", LAB[gate], 824, gate, ctrl, closed, smooth, len(X))


def row(group, gate, T):
    return next(r for r in table if r["group"] == group and r["gate"] == gate and r["T"] == T)


def put(key, value, s):
    text[key] = dict(value=value, s=s)


# ------------------------------------------------------------------ numbers of the prose
# purity row: final purity is at least F_T^2 (closed)
put("min_gap_final_purity_FT2", A["checks"]["entanglement"]["caveat"], "")

# saturation at 2133 ns, CS and ECD
for gate in ("cs", "cd"):
    for T in (412, 2133):
        v = row("track", gate, T)["cells"]["osc_at_bound"]
        put(f"atb_{gate}_{T}", v, f"{v['mean']:.2f}")
# conditional fraction of the tracking pulses
for gate in ("cs", "cd"):
    for T in (412, 824, 2133):
        v = row("track", gate, T)["cells"]["cond_frac"]
        put(f"cond_{gate}_{T}", v, f"{v['mean']:.2f}")
mincf = min(row("track", g_, T_)["cells"]["cond_frac"]["min"] for g_ in ("cs", "cd") for T_ in (412, 824, 2133))
put("cond_min_track_cs_cd", mincf, f"{mincf:.2f}")
check("cond_share_above_half_every_start", mincf > 0.5, mincf)
# ECD generator within 1% of full amplitude (|u_-| >= 0.99 g) in tracking pulses, recomputed from X and checked
# against data-A (audit, Nov 2: the text now states the rule |u_-| >= 0.99 g instead of |u_-| = g)
TIE = 1e-12                                     # explicit tolerance, so that |u_-| = g/2 exactly (one branch at the bound,
                                                # the other at zero) counts as 'at least half' independently of rounding


def um_abs(X):
    return np.abs(0.5 * (X[..., 2] * np.exp(1j * X[..., 3]) - X[..., 4] * np.exp(1j * X[..., 5])))


pc = A["checks"]["pure_conditional_at_bound"]["per_case"]
for gate in ("cs", "cd"):
    for T in (412, 824):
        Xg = np.load(P / f"tracking_gkp_{gate}_T{T}.npz")["X"]
        v = ms((um_abs(Xg) >= 0.99).mean(axis=1))
        m, s, n = parse_pm(pc[f"tracking_gkp_{gate}_T{T}"])
        check(f"pureum_{gate}_{T}_vs_A", abs(v["mean"] - m) < 6e-4 and n == v["n"], (v["mean"], m))
        put(f"pureum_{gate}_{T}", v, f"{v['mean']:.2f}")
# simultaneous qubit drive and conditional part (CS tracking), both at least half their bound, recomputed from X with
# the explicit tolerance TIE (audit, Nov 2: data-A's rule |u_-| >= 0.5 decided the exact ties by rounding)
for T in (412, 824, 2133):
    Xg = np.load(P / f"tracking_gkp_cs_T{T}.npz")["X"]
    on_r, on_c = Xg[..., 0] >= 0.5 - TIE, um_abs(Xg) >= 0.5 - TIE
    both, donly = ms((on_r & on_c).mean(axis=1)), ms((on_r & ~on_c).mean(axis=1))
    put(f"both_on_cs_{T}", both, f"{both['mean']:.2f}")
    put(f"drive_only_cs_{T}", donly, f"{donly['mean']:.2f}")
check("cs_2133_drive_alone_more_often", text["drive_only_cs_2133"]["value"]["mean"] > text["both_on_cs_2133"]["value"]["mean"])
check("cs_412_824_mostly_together", all(text[f"both_on_cs_{T}"]["value"]["mean"] > 0.5 for T in (412, 824)))
# qubit drive strength
qd = A["checks"]["qubit_drive_strength"]["per_case"]
for gate in GATES:
    for T in (412, 824, 2133):
        e = qd[f"tracking_gkp_{gate}_T{T}"]
        put(f"Omega_g_{gate}_{T}", e["mean_Omega_over_g"], f"{e['mean_Omega_over_g']:.1f}")
        m, s, n = parse_pm(e["frac_Omega_le_chi_d"])
        put(f"le_chi_{gate}_{T}", dict(mean=m, sd=s, n=n), f"{m:.2f}")
# SNAP starts that leave the vacuum drive the qubit at about twice chi_d = g (audit, Nov 2: replaces the number-selective
# reading, which the data do not single out, since ECD keeps |Omega| <= chi_d in a similar share of its slices)
for T in (412, 824, 2133):
    zs = np.load(P / f"tracking_gkp_snap_T{T}.npz")
    nt = ~zs["trivial"].astype(bool)
    om = float((10 * zs["X"][..., 0][nt].mean(axis=1)).mean())          # |Omega| = 10 g r, starts that leave the vacuum
    check(f"snap_Omega_nontrivial_{T}", abs(om - qd[f"tracking_gkp_snap_T{T}"]["mean_Omega_over_g"]) < 1e-9
          and 1.5 < om < 2.5, (om, qd[f"tracking_gkp_snap_T{T}"]["mean_Omega_over_g"]))
# minimum purity of the tracking pulses (SNAP: also the starts that leave the vacuum)
en = A["checks"]["entanglement"]["per_case"]
for gate in GATES:
    for T in (412, 824, 2133):
        v = row("track", gate, T)["cells"]["purity_min"]
        put(f"pmin_{gate}_{T}", v, f"{v['mean']:.2f}")
for T in (412, 824):
    m, s, n = parse_pm(en[f"tracking_gkp_snap_T{T}"]["nontrivial_purity_min"])
    put(f"pmin_snap_nontriv_{T}", dict(mean=m, sd=s, n=n), f"{m:.2f}")
    put(f"n_snap_nontriv_{T}", n, f"{n}")
check("cs_entangles_less_than_cd_every_T", all(text[f"pmin_cs_{T}"]["value"]["mean"] > text[f"pmin_cd_{T}"]["value"]["mean"]
                                               for T in (412, 824, 2133)))
check("snap_nontriv_vs_cd_824_about_equal", abs(text["pmin_snap_nontriv_824"]["value"]["mean"]
                                                - text["pmin_cd_824"]["value"]["mean"]) < 0.03)
check("snap_nontriv_412_less_entangled_than_cs", text["pmin_snap_nontriv_412"]["value"]["mean"]
      > text["pmin_cs_412"]["value"]["mean"])
check("target_cs_deeper_than_track_cs_412", row("target", "cs", 412)["cells"]["purity_min"]["mean"]
      < row("track", "cs", 412)["cells"]["purity_min"]["mean"])
check("target_cd_deeper_than_target_cs", row("target", "cd", 412)["cells"]["purity_min"]["mean"]
      < row("target", "cs", 412)["cells"]["purity_min"]["mean"])
check("cs_drives_hardest", all(row("track", "cs", T)["cells"]["drive_mean"]["mean"]
                               > max(row("track", g_, T)["cells"]["drive_mean"]["mean"] for g_ in ("cd", "snap"))
                               for T in (412, 824, 2133)))
check("flip_cs_every_start_close_to_maxent", row("flip", "cs", 824)["cells"]["purity_min"]["max"] < 0.55)
sz = A["checks"]["snap_zero_drive"]["per_T"]["412"]
put("snap_nontriv_FT_412", sz["nontrivial_F_T_noisy"], pm(sz["nontrivial_F_T_noisy"]["mean"], sz["nontrivial_F_T_noisy"]["sd"]))
vmax = max(sz["trivial_max_cavity_drive_over_bound"],
           A["checks"]["snap_zero_drive"]["per_T"]["824"]["trivial_max_cavity_drive_over_bound"])
e_ = math.floor(math.log10(vmax))
m_ = math.ceil(vmax / 10 ** e_ * 10 - 1e-9) / 10           # upper bound with two significant digits
put("snap_trivial_max_drive", vmax, f"{m_:.1f}\\times10^{{{e_}}}")
check("snap_trivial_max_drive_str", m_ * 10 ** e_ >= vmax, (vmax, m_, e_))
# the trivial runs are exactly those with a cavity drive this small (recomputed from X)
for T in (412, 824):
    z = np.load(P / f"tracking_gkp_snap_T{T}.npz")
    amax = z["X"][..., 2].max(axis=1)
    check(f"snap_trivial_flags_{T}", np.array_equal(amax <= vmax + 1e-15, z["trivial"].astype(bool)), amax.tolist())
# gate tracking: every ECD start entangles nearly maximally at 824 ns
for W in (10, 70):
    clW = closed_stats(P / f"closed_flip_track_cd_T824_W{W}.npz")
    put(f"ecd_track_W{W}_pmin_max", float(clW["purity_min"].max()), f"{clW['purity_min'].max():.3f}")
    check(f"ecd_track_W{W}_every_start_maxent", clW["purity_min"].max() < 0.52, clW["purity_min"].tolist())
put("snap_trivial_FT", sz["trivial_F_T_noisy"]["mean"], f"{sz['trivial_F_T_noisy']['mean']:.3f}")
put("n_trivial_412", A["checks"]["snap_zero_drive"]["per_T"]["412"]["n_trivial"], "13")
put("n_trivial_824", A["checks"]["snap_zero_drive"]["per_T"]["824"]["n_trivial"], "2")
check("n_trivial", text["n_trivial_412"]["value"] == 13 and text["n_trivial_824"]["value"] == 2)
# cap of the closeness by the purity, ECD and CS at 824 ns (means over the starts)
for gate in GATES:
    for T in (412, 824, 2133):
        cr = cap_rows[f"tracking_gkp_{gate}_T{T}"]
        put(f"cap_{gate}_{T}", cr["cap"], f"{cr['cap']['mean']:.2f}")
        put(f"Cclosed_{gate}_{T}", cr["C_closed"], f"{cr['C_closed']['mean']:.2f}")
# target only against tracking, CS at 412 ns
v = row("target", "cs", 412)["cells"]["purity_min"]
put("pmin_target_cs", v, f"{v['mean']:.2f}")
for gk in ("snap", "snap4", "cd"):
    v = row("target", gk, 412)["cells"]["purity_min"]
    put(f"pmin_target_{gk}", v, f"{v['mean']:.2f}")
    v = row("target", gk, 412)["cells"]["drive_mean"]
    put(f"drive_target_{gk}", v, f"{v['mean']:.2f}")
# flip
fl = A["checks"]["gate_vs_tracking_824ns"]["per_gate"]
v = row("flip", "cs", 824)["cells"]
put("flip_cs_pmin", v["purity_min"], pm(v["purity_min"]["mean"], v["purity_min"]["sd"], 3))
put("flip_cs_nbar", v["nbar_max"], pm(v["nbar_max"]["mean"], v["nbar_max"]["sd"], 2))
v = row("track", "cs", 824)["cells"]
put("track_cs_824_pmin", v["purity_min"], pm(v["purity_min"]["mean"], v["purity_min"]["sd"], 3))
put("track_cs_824_nbar", v["nbar_max"], pm(v["nbar_max"]["mean"], v["nbar_max"]["sd"], 2))
stall = A["checks"]["ecd_flip_stall"]["per_T"]["824"]
sc, pu, cf = np.array(stall["score"]), np.array(stall["purity_min"]), np.array(stall["cond_frac"])
seeds = np.array(np.load(P / "flip_target_cd_T824.npz")["seed"])
stalled = sc < 0.8
near_prod = stalled & (pu > 0.9)
maxent = stalled & (pu < 0.51)
put("ecd_n_stalled", int(stalled.sum()), f"{int(stalled.sum())}")
put("ecd_n_stalled_product", int(near_prod.sum()), f"{int(near_prod.sum())}")
put("ecd_n_stalled_maxent", int(maxent.sum()), f"{int(maxent.sum())}")
check("ecd_stall_split", stalled.sum() == near_prod.sum() + maxent.sum() == 7, (stalled.sum(), near_prod.sum(), maxent.sum()))
put("ecd_stalled_product_pmin_range", [float(pu[near_prod].min()), float(pu[near_prod].max())],
    f"{pu[near_prod].min():.3f}--{pu[near_prod].max():.3f}")     # three decimals, so that the range contains the 0.985
check("ecd_maxent_starts_maximal", float(pu[maxent].max()) < 0.502, pu[maxent].tolist())   # of the representative
check("ecd_rep_in_product_range", S27["cd"]["seed"] in seeds[near_prod].tolist())
put("ecd_stalled_product_cond_range", [float(cf[near_prod].min()), float(cf[near_prod].max())],
    f"{cf[near_prod].min():.2f}--{cf[near_prod].max():.2f}")
best = int(np.argmax(sc))
put("ecd_best_seed", int(seeds[best]), f"{int(seeds[best])}")
put("ecd_best_score", float(sc[best]), f"{sc[best]:.3f}")
put("ecd_best_pmin", float(pu[best]), f"{pu[best]:.3f}")
put("ecd_best_cond", float(cf[best]), f"{cf[best]:.2f}")
check("ecd_best_vs_S27", int(seeds[best]) == S27["cd_best_seed"] and abs(sc[best] - S27["cd_best"]["F_rep"]) < 1e-12)
rep = S27["cd"]
put("ecd_rep_pmin", min(rep["purity_min_rep"]), f"{min(rep['purity_min_rep']):.3f}")
put("ecd_rep_cond", rep["shape"]["cond_frac"], f"{rep['shape']['cond_frac']:.2f}")
for W in ("W10", "W70"):
    m, s, n = parse_pm(fl["cd"][f"track_{W}"]["purity_min"])
    put(f"ecd_track_{W}_pmin", dict(mean=m, sd=s, n=n), f"{m:.3f}")
    m, s, n = parse_pm(fl["cd"][f"track_{W}"]["cond_frac"])
    put(f"ecd_track_{W}_cond", dict(mean=m, sd=s, n=n), f"{m:.2f}")
m, s, n = parse_pm(fl["cd"]["target"]["cond_frac"])
put("ecd_target_cond", dict(mean=m, sd=s, n=n), pm(m, s, 2))
# like-for-like check of 'the flip entangles far more deeply than preparation' (audit, Nov 2): at 412 ns the CS flip
# optimized for the gate alone against the CS preparation optimized for the final fidelity alone (no tracking term in
# either), smallest purity and largest photon number at the slice ends, the flip over the inputs 0_L, 1_L, +_L
cl412 = closed_stats(P / "closed_flip_target_cs_T412.npz")
f412 = {k: ms(cl412[k]) for k in ("purity_min", "nbar_max")}
check("flip412_cs_seeds", sorted(cl412["seed"].tolist()) == list(range(3000, 3008)), cl412["seed"].tolist())
tcs = row("target", "cs", 412)["cells"]
put("flip412_cs_pmin", f412["purity_min"], pm(f412["purity_min"]["mean"], f412["purity_min"]["sd"], 3))
put("flip412_cs_nbar", f412["nbar_max"], pm(f412["nbar_max"]["mean"], f412["nbar_max"]["sd"], 2))
put("target_cs_pmin_pm", tcs["purity_min"], pm(tcs["purity_min"]["mean"], tcs["purity_min"]["sd"], 3))
put("target_cs_nbar_pm", tcs["nbar_max"], pm(tcs["nbar_max"]["mean"], tcs["nbar_max"]["sd"], 2))
check("flip412_deeper_than_target_prep", f412["purity_min"]["max"] < tcs["purity_min"]["min"],
      (f412["purity_min"], tcs["purity_min"]))
check("flip412_nbar_more_than_twice_target_prep", f412["nbar_max"]["mean"] > 2 * tcs["nbar_max"]["mean"],
      (f412["nbar_max"]["mean"], tcs["nbar_max"]["mean"]))
check("flip412_nbar_max_is_SecIX_7.1", f"{f412['nbar_max']['max']:.1f}" == "7.1", f412["nbar_max"]["max"])
# SNAP at chi_d = g, target only, entangles less than the other gate sets of Table I
check("target_snap_least_entangled", row("target", "snap", 412)["cells"]["purity_min"]["mean"]
      > max(row("target", g_, 412)["cells"]["purity_min"]["mean"] for g_ in ("cs", "cd", "snap4")))
check("target_snap4_drives_harder", row("target", "snap4", 412)["cells"]["drive_mean"]["mean"]
      > row("target", "snap", 412)["cells"]["drive_mean"]["mean"])
v = A["stats"]["flip_target_snap_T824"]["closed_min_over_inputs"]["purity_final_min"]
put("flip_snap_final_purity", v, pm(v["mean"], v["sd"], 2))
v = A["stats"]["flip_target_cs_T824"]["closed_min_over_inputs"]["purity_final_min"]
put("flip_cs_final_purity", v, f"{v['mean']:.3f}")
# smoothing
bw = B["bandwidth"]
put("bw_1", bw["1"]["f_half_power_MHz"], f"{bw['1']['f_half_power_MHz']:.0f}")
put("bw_4", bw["4"]["f_half_power_MHz"], f"{bw['4']['f_half_power_MHz']:.0f}")
put("sig_dt_1_412", bw["1"]["sigma_over_dt_412"], f"{bw['1']['sigma_over_dt_412']:.2f}")
allmean1 = [abs(c["mean"]) for (st_, g_, T_, s_), _ in deltas.items() if s_ == 1.0
            for c in [ms(np.array(deltas[(st_, g_, T_, s_)]) * 1e3)]]
put("max_abs_mean_d1", max(allmean1), f"{max(allmean1):.2f}")
check("max_abs_mean_d1_below_1e-3", max(allmean1) < 1.0)
worst1 = min(min(v) for (st_, g_, T_, s_), v in deltas.items() if s_ == 1.0) * 1e3
put("worst_start_d1", worst1, f"{worst1:.1f}")
check("worst_start_d1_vs_B", abs(worst1 - B["derived"]["worst_start"]["1"]["d"] * 1e3) < 1e-9)
check("worst_start_d1_below_3e-3", worst1 > -3.0)
mean4 = {(st_, g_, T_): ms(np.array(v) * 1e3)["mean"] for (st_, g_, T_, s_), v in deltas.items() if s_ == 4.0}
wk = min(mean4, key=mean4.get)
put("worst_mean_d4", dict(case=list(wk), value=mean4[wk]), f"{-mean4[wk] / 1e3:.2f}")
put("cs_track412_d4", mean4[("track", "cs", 412)], f"{-mean4[('track', 'cs', 412)] / 1e3:.3f}")
check("worst_mean_d4_is_snap4", wk == ("target", "snap4", 412))
put("same_order_all_sigma", all(B["derived"]["ranking"][k]["same_order_all_sigma"] for k in B["derived"]["ranking"]), "")
check("same_order", text["same_order_all_sigma"]["value"])
es = B["derived"]["end_share"]
vals1 = [v["1"] for k, v in es.items() if v["1"] is not None]
put("end_share_1_range", [min(vals1), max(vals1)], f"{min(vals1):.1f}--{max(vals1):.1f}")
check("end_share_range", 0.55 <= min(vals1) and max(vals1) <= 1.05, vals1)
# relative L2 change of the controls at sigma = 1 ns (tracking and target): CS against ECD
rel = {}
for k in ("track_cs_412", "track_cs_824", "target_cs_412", "track_cd_412", "track_cd_824", "target_cd_412"):
    g1 = B["smoothing"]["groups"][k]["sigma"]["1"]
    rel[k] = (g1["rel_dC_drive"]["mean"], g1["rel_dC_osc"]["mean"])
put("relL2", rel, "")
put("relL2_cs_osc_range", [min(rel[k][1] for k in rel if "_cs_" in k), max(rel[k][1] for k in rel if "_cs_" in k)],
    f"{min(rel[k][1] for k in rel if '_cs_' in k):.2f}--{max(rel[k][1] for k in rel if '_cs_' in k):.2f}")
put("relL2_cd_osc_range", [min(rel[k][1] for k in rel if "_cd_" in k), max(rel[k][1] for k in rel if "_cd_" in k)],
    f"{min(rel[k][1] for k in rel if '_cd_' in k):.2f}--{max(rel[k][1] for k in rel if '_cd_' in k):.2f}")
put("n_smooth_rows", len(rows), f"{len(rows)}")
put("bound_smoothed", [B["smoothing"]["bounds"]["max_Omega_over_10g_smoothed"], B["smoothing"]["bounds"]["max_osc_over_g_smoothed"]], "")
# reproduction
rp = A["reproduction"]
put("repro_n", rp["n_pulses"], f"{rp['n_pulses']}")
put("repro_FT_C", rp["max_dev_tracking_FT_C"], "")
put("repro_flip", rp["max_dev_flip_metric"], "")
tr_traces = max(v.get("max_dev_traces", 0) for v in rp["per_case"].values())
put("repro_traces", tr_traces, "")
check("repro_bounds", rp["n_errors"] == 0 and rp["max_dev_tracking_FT_C"] < 1e-13 and tr_traces < 3e-13
      and rp["max_dev_flip_metric"] < 1e-15)
br = B["reproduction"]
dmax = max(max(abs(p["d_F60"]), abs(p["d_F90_closed"]), abs(p["d_F90_noisy"])) for p in br["per_seed"])
put("target_repro_max", dmax, "0")
check("target_repro_exact", dmax == 0.0 and len(br["per_seed"]) == 32)
fcmax = max(v.get("d_FT_closed", 0.0) for v in FC["cases"].values())
put("figcheck_FT_closed", fcmax, "")
check("figcheck", fcmax < 3e-13, fcmax)
check("figcheck_prep_cases", sorted(k for k, v in FC["cases"].items() if "d_FT_closed" in v) ==
      sorted(k for k in FC["cases"] if k.startswith(("S25", "S26"))) and len(FC["cases"]) == 11)
fcp = max(v["d_purity"] for v in FC["cases"].values())
put("figcheck_purity", fcp, "")
n_track = sum(A["cases"][k]["n_starts"] for k in A["cases"] if k.startswith("tracking"))
n_flip = sum(A["cases"][k]["n_starts"] for k in A["cases"] if k.startswith("flip"))
put("repro_n_track", n_track, f"{n_track}")
put("repro_n_flip", n_flip, f"{n_flip}")
check("repro_split", n_track == 180 and n_flip == 144 and n_track + n_flip == rp["n_pulses"])
# share of the mean shortfall 1 - C that the purity cap explains, CS and ECD at every duration
shares = {}
for gate in ("cs", "cd"):
    for T in (412, 824, 2133):
        cr = cap_rows[f"tracking_gkp_{gate}_T{T}"]
        shares[f"{gate}_{T}"] = (1 - cr["cap"]["mean"]) / (1 - cr["C_closed"]["mean"])
put("capshare", shares, "")
put("capshare_range", [min(shares.values()), max(shares.values())],
    f"{100 * min(shares.values()):.0f}--{100 * max(shares.values()):.0f}\\%")
# scores quoted in the prose (Table I means of the target-only pulses, Supplementary Table 11 entries)
put("tabI_cs", S26["cs"]["F_stat"][0], f"{S26['cs']['F_stat'][0]:.3f}")
for gk in ("cd", "snap", "snap4"):
    put(f"tabI_{gk}", S26[gk]["F_stat"][0], f"{S26[gk]['F_stat'][0]:.2f}")
put("flip_cd_W10_score", A["cases"]["flip_track_cd_T824_W10"]["score_mean"],
    f"{A['cases']['flip_track_cd_T824_W10']['score_mean']:.3f}")
check("flip_cd_W10_score_tab", text["flip_cd_W10_score"]["s"] == "0.778")
flip_cs_pmin_max = float(row("flip", "cs", 824)["cells"]["purity_min"]["max"])
put("flip_cs_pmin_max", flip_cs_pmin_max, f"{flip_cs_pmin_max:.2f}")


def ceil1(x):
    """upper bound with one significant digit, as LaTeX"""
    e = math.floor(math.log10(x))
    m = math.ceil(x / 10 ** e - 1e-12)
    if m == 10:
        m, e = 1, e + 1
    return f"{m}\\times10^{{{e}}}"


# independent Lindblad integrator (data-B, pulses_check_B.py: exact exponentials of review_oct28/lindblad_crosscheck.py,
# controls rebuilt from Eq. (7) with Gauss-Legendre averaging of the exact convolution), smoothed and unsmoothed pulses
CB = json.load(open(P / "pulses_check_B.json"))["spot"]
sig = sorted({float(c["key"].split("|")[4]) for c in CB["checks"]})
sets = sorted({c["key"].split("|")[0] for c in CB["checks"]})
check("indep_lindblad_coverage", len(CB["checks"]) == 17 and 0.0 in sig and max(sig) > 0
      and sets == ["flip", "target", "track"], (len(CB["checks"]), sig, sets))
dprep = max(abs(c["d_F_T_noisy"]) for c in CB["checks"] if "d_F_T_noisy" in c)
dflip = max(abs(c["d_avg6_opt_noisy"]) for c in CB["checks"] if "d_avg6_opt_noisy" in c)
check("indep_lindblad_summary", dprep == CB["summary"]["max_abs_d_F_T_noisy"]
      and dflip == CB["summary"]["max_abs_d_avg6_opt_noisy"])
put("indep_lind_n", len(CB["checks"]), f"{len(CB['checks'])}")
put("indep_lind_prep_s", dprep, ceil1(dprep))
put("indep_lind_flip_s", dflip, ceil1(dflip))
# every table cell carries its s.d. in parentheses (or is a dash), so the table caption needs no rule for bare values
check("table_cells_parenthesized", all(c == "--" or "(" in c for r in table for c in r["fmt"].values()))
put("repro_FT_C_s", rp["max_dev_tracking_FT_C"], ceil1(rp["max_dev_tracking_FT_C"]))
put("repro_traces_s", tr_traces, ceil1(tr_traces))
put("repro_flip_s", rp["max_dev_flip_metric"], ceil1(rp["max_dev_flip_metric"]))
put("figcheck_s", fcmax, ceil1(fcmax))
put("figcheck_purity_s", fcp, ceil1(fcp))
for T in (412, 824):
    v = A["stats"][f"tracking_gkp_snap_T{T}"]["nontrivial_closed"]["purity_min"]
    put(f"pmin_snap_nontriv_full_{T}", v, paren(v["mean"], v["sd"]))
# at sigma = 4 ns CS loses more than ECD in every protocol (track 412, 824, target, flip)
cmp4 = {}
for st_, T in (("track", 412), ("track", 824), ("target", 412), ("flip", 824)):
    cmp4[f"{st_}_{T}"] = (mean4[(st_, "cs", T)], mean4[(st_, "cd", T)])
put("cs_vs_cd_d4", cmp4, "")
check("cs_loses_more_than_cd_at_4ns", all(a < b for a, b in cmp4.values()), cmp4)
put("dt_412", 412 / 50, "8.24")
check("dt", abs(S26["dt_ns"] - 8.24) < 1e-12 and abs(S27["dt_ns"] - 16.48) < 1e-12)
# caption numbers (from the figure facts), re-formatted here as a cross-check of the captions
put("S25_F_rep", {g: S25[g]["F_rep"] for g in GATES}, ", ".join(f"{S25[g]['F_rep']:.3f}" for g in GATES))
put("S26_F_rep", {g: S26[g]["F_rep"] for g in ("cs", "cd", "snap", "snap4")},
    ", ".join(f"{S26[g]['F_rep']:.3f}" for g in ("cs", "cd", "snap", "snap4")))

out = dict(description=__doc__, table=table, text=text, checks=checks, cap=cap_rows)
(_L.out_dir("numbers") / "pulses_text_facts.json").write_text(json.dumps(out, indent=1, default=float))
print("checks passed:", len(checks))
for r in table:
    print(r["group"], r["label"], r["T"], r["n"], r["footnote"], " | ".join(f"{k}={v}" for k, v in r["fmt"].items()))
for k, v in text.items():
    print(k, "->", v["s"], "|", json.dumps(v["value"], default=float)[:160])
for k, v in cap_rows.items():
    print(k, json.dumps(v, default=float)[:300])
