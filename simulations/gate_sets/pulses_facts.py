"""Facts of the pulse study of result set A (Nov 1): every number of the pulse section that comes from the stored
pulses, assembled from pulses_index_A.json (pulses_extract.py), reproduce_A.json (pulses_reproduce.py) and
pulses_stats_A.json, pulses_stats_A_rows.json, closed_*.npz (pulses_stats.py), with the physics checks evaluated by
explicit rules (each check states its rule, its numbers and whether it holds).  Statistics: mean +- sample s.d. (ddof 1)
over all starts of a case, n stated.  Writes manuscript_figures/sims/results/pulses/pulses_facts_A.json.
  python simulations/gate_sets/pulses_facts.py
"""
from __future__ import annotations
import layout as _L  # repository layout: data/ and output/ (src/layout.py)

import json
import math
import pathlib

import numpy as np

HERE = _L.vfile(__file__).parent
OUT = HERE / "results" / "pulses"
G = 7.5                                  # 1/us (params.G_SQ)


def ms(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)),
                min=float(v.min()), max=float(v.max()))


def short(d, p=3):
    return None if d is None else f"{d['mean']:.{p}f} +- {d['sd']:.{p}f} (n = {d['n']})"


def main():
    index = json.loads((OUT / "pulses_index_A.json").read_text())
    rep = json.loads((OUT / "reproduce_A.json").read_text())
    stats = json.loads((OUT / "pulses_stats_A.json").read_text())["cases"]
    rows = json.loads((OUT / "pulses_stats_A_rows.json").read_text())
    byfile = {}
    for r in rows:
        byfile.setdefault(r["file"][:-4], []).append(r)
    for v in byfile.values():
        v.sort(key=lambda r: r["i"])
    idx = {c["file"][:-4]: c for c in index["cases"]}
    case = lambda fam, gate, T, W=None: f"{fam}_{gate}_T{T}" + (f"_W{W}" if W else "")  # noqa: E731
    TR = [("tracking_gkp", g, T) for g in ("cs", "cd", "snap") for T in (412, 824, 2133)]
    FT = [("flip_target", g, T) for g in ("cs", "cd", "snap") for T in (412, 824)]
    FK = [("flip_track", g, T, W) for g in ("cs", "cd", "snap") for T in (412, 824) for W in (10, 70)]
    facts = dict(description=__doc__, units="amplitudes as fractions of their bounds (r = |Omega|/10g, a_j = |u_j|/g); "
                 "u_+-, |u_-| in units of g; rotation angle int|Omega|dt in units of pi; times in ns",
                 g_per_us=G, Omega_max_over_g=10.0, chi_d_over_g=1.0)
    # ------------------------------------------------------------------ data and representative starts
    facts["cases"] = {c: dict(gate=e["gate"], T_ns=e["T_ns"], N=e["N"], n_starts=e["n_starts"], rep_seed=e["rep_seed"],
                              rep_rule=e["rep_rule"], score=e["score"], score_mean=e["score_mean"],
                              score_sd=e["score_sd"], **({"n_trivial": e["n_trivial"],
                                                          "trivial_seeds": e["trivial_seeds"],
                                                          "rep_nontrivial_seed": e["rep_nontrivial_seed"]}
                                                         if "n_trivial" in e else {"W": e["W"],
                                                                                   "tabulated": e["tabulated"],
                                                                                   "score_best": e["score_best"]}))
                      for c, e in idx.items()}
    for c, e in facts["cases"].items():                 # largest possible int|Omega|dt = 10 g T
        e["rotation_angle_bound_over_pi"] = 10 * G * e["T_ns"] * 1e-3 / math.pi
    # ------------------------------------------------------------------ reproduction
    facts["reproduction"] = dict(
        rule="every extracted pulse re-simulated with the routine that produced the stored numbers (traces.run for "
             "tracking, flip_gate.evaluate for the flip), cutoff 90, closed and Lindblad; threshold 1e-5",
        n_pulses=rep["n_rows"], n_errors=rep["n_errors"], max_dev_tracking_FT_C=rep["max_dev_tracking_FT_C"],
        max_dev_flip_metric=rep["max_dev_flip_metric"],
        per_case={c: {k: v for k, v in e.items() if k in ("n", "max_dev_FT_C", "max_dev_traces", "max_dev_metric",
                                                         "max_dev_grape_closed")} for c, e in rep["cases"].items()},
        closed_propagator_check=dict(
            rule="closed propagation of pulses_stats.py (GRAPE models, exact slice unitaries) against stored values",
            tracking_max_dev_F_T_closed=max(stats[case(*t)]["max_F_T_closed_check"] for t in TR),
            flip_max_dev_F_0to1_closed=max(stats[case(*t)]["max_dev_F_0to1_closed_vs_stored"] for t in FT + FK),
            flip_max_dev_F_1to0_closed=max(stats[case(*t)]["max_dev_F_1to0_closed_vs_stored"] for t in FT + FK)),
        holds=bool(rep["n_errors"] == 0 and rep["max_dev_tracking_FT_C"] < 1e-5 and rep["max_dev_flip_metric"] < 1e-5))
    # ------------------------------------------------------------------ per-case statistics (compact)
    keys_c = ("osc_at_bound", "osc_at_zero", "osc_mean", "drive_at_bound", "drive_at_zero", "drive_mean",
              "rotation_angle_over_pi", "drive_le_chi", "jumps_Omega", "cond_frac", "um_mean", "up_mean", "um_at_bound",
              "up_at_bound", "both_branches_at_bound", "overlap_both", "overlap_drive_only", "overlap_cond_only",
              "overlap_neither", "corr_r_um", "jumps_um", "jumps_up", "osc_max", "jumps_eps")
    keys_q = ("purity_min", "purity_min_fine", "t_purity_min_over_T", "purity_final", "purity_mean", "Pe_max", "Pe_mean",
              "Pe_final", "nbar_max", "nbar_final", "nbar_mean")
    facts["stats"] = {}
    for c, e in stats.items():
        facts["stats"][c] = dict(n_starts=e["n_starts"], rep_seed=e["rep_seed"],
                                 controls={k: e["controls"][k] for k in keys_c if k in e["controls"]},
                                 controls_rep={k: e["controls_rep"][k] for k in keys_c if k in e["controls_rep"]},
                                 closed={nm: {k: d[k] for k in keys_q} for nm, d in e["closed"].items()},
                                 closed_rep=e["closed_rep"])
        for k in ("n_trivial", "trivial_osc_max", "trivial_drive_mean", "nontrivial_controls", "nontrivial_closed",
                  "F_0to1_closed", "F_1to0_closed"):
            if k in e:
                facts["stats"][c][k] = e[k]
        # per start: oscillator amplitude at a bound (upper or lower), and the same for the qubit drive
        R = byfile[c]
        facts["stats"][c]["controls"]["osc_bound_or_zero"] = ms([r["controls"]["osc_at_bound"] + r["controls"]["osc_at_zero"]
                                                               for r in R])
        facts["stats"][c]["controls"]["drive_bound_or_zero"] = ms([r["controls"]["drive_at_bound"]
                                                                 + r["controls"]["drive_at_zero"] for r in R])
        # flip: smallest purity over the three inputs, per start
        if c.startswith("flip"):
            facts["stats"][c]["closed_min_over_inputs"] = dict(
                purity_min=ms([min(r["closed"][nm]["purity_min"] for nm in r["closed"]) for r in R]),
                purity_final_min=ms([min(r["closed"][nm]["purity_final"] for nm in r["closed"]) for r in R]))
    checks = {}
    st = facts["stats"]
    # 1 bang-bang
    bb = {}
    for t in TR + FT + FK:
        c = case(*t)
        cc = st[c]["controls"]
        bb[c] = dict(osc_at_bound=short(cc["osc_at_bound"]), osc_bound_or_zero=short(cc["osc_bound_or_zero"]),
                     drive_at_bound=short(cc["drive_at_bound"]), drive_bound_or_zero=short(cc["drive_bound_or_zero"]),
                     near_bang_bang_osc=bool(cc["osc_bound_or_zero"]["mean"] >= 0.9),
                     near_bang_bang_drive=bool(cc["drive_bound_or_zero"]["mean"] >= 0.9))
    checks["bang_bang"] = dict(
        rule="near bang-bang if, on average over the starts, >= 90% of the (slice, branch) amplitudes sit at a bound "
             "(a >= 0.999 or a <= 0.001); the same for the qubit drive r",
        expectation="amplitude-bounded time-optimal control saturates the bounds (Pontryagin), except on singular "
                    "arcs; the stored pulses are not time optimal (fixed T, tracking or target objective)",
        per_case=bb,
        holds_for=[c for c, v in bb.items() if v["near_bang_bang_osc"]],
        fails_for=[c for c, v in bb.items() if not v["near_bang_bang_osc"]])
    # 2 conditional part (CS, ECD)
    cp = {}
    for t in [x for x in TR + FT + FK if x[1] in ("cs", "cd")]:
        c = case(*t)
        R = byfile[c]
        cf = [r["controls"]["cond_frac"] for r in R]
        cp[c] = dict(cond_frac=short(st[c]["controls"]["cond_frac"]), cond_frac_min=float(min(cf)),
                     um_at_bound=short(st[c]["controls"]["um_at_bound"]),
                     both_branches_at_bound=short(st[c]["controls"]["both_branches_at_bound"]))
    checks["conditional_part"] = dict(
        rule="sum_j |j><j| (x) G(u_j) = 1 (x) G(u_+) + sigma_z (x) G(u_-); without u_- the qubit and the oscillator "
             "never entangle, and by Proposition thm:nogo the oscillator then evolves by a Gaussian unitary; holds if "
             "the conditional fraction int|u_-|^2/int(|u_+|^2 + |u_-|^2) is > 0.1 in every start of the tracking and "
             "CS cases",
        per_case=cp,
        holds=bool(all(v["cond_frac_min"] > 0.1 for c, v in cp.items() if c.startswith("tracking") or "_cs_" in c)))
    # 3 entanglement made and undone (closed evolution, slice ends)
    en = {}
    for t in TR + FT + FK:
        c = case(*t)
        if c.startswith("tracking"):
            q = st[c]["closed"]["0"]
            pm, pf = q["purity_min"], q["purity_final"]
        else:
            q = st[c]["closed_min_over_inputs"]
            pm, pf = q["purity_min"], q["purity_final_min"]
        en[c] = dict(purity_min=short(pm), purity_final=short(pf, 4),
                     made=bool(pm["mean"] < 0.9), undone=bool(pf["mean"] > 0.96))
        if c.startswith("tracking") and "nontrivial_closed" in st[c]:
            en[c]["nontrivial_purity_min"] = short(st[c]["nontrivial_closed"]["purity_min"])
            en[c]["nontrivial_purity_final"] = short(st[c]["nontrivial_closed"]["purity_final"], 4)
    # the final purity is not independent of the fidelity: lambda_max >= F_T, so purity >= F_T^2
    lb = []
    for t in TR:
        z = np.load(OUT / f"{case(*t)}.npz")
        pf = np.array([r["closed"]["0"]["purity_final"] for r in byfile[case(*t)]])
        lb.append(float(np.min(pf - z["F_T_closed"] ** 2)))
    checks["entanglement"] = dict(
        rule="closed evolution, cutoff 90, slice ends; Tr rho_osc^2 = 1 for a product state and 1/2 for maximal "
             "qubit-oscillator entanglement; 'made' if the mean minimum purity < 0.9, 'undone' if the mean final "
             "purity > 0.96 (flip: smallest over the inputs 0_L, 1_L, +_L)",
        per_case=en,
        caveat="the final purity is bounded below by F_T^2 (lambda_max >= F_T), so a high final purity follows from a "
               "high closed F_T and is not independent evidence; min over tracking starts of (purity_final - F_T^2) = "
               f"{min(lb):.2e}",
        made_in=[c for c, v in en.items() if v["made"]], not_made_in=[c for c, v in en.items() if not v["made"]],
        not_undone_in=[c for c, v in en.items() if not v["undone"]])
    # 4 entanglement ranking along the brachistochrone: CS stays closer to product states than ECD and SNAP
    rk = {}
    for T in (412, 824, 2133):
        def pmin(g):
            s = st[case("tracking_gkp", g, T)]
            return s["nontrivial_closed"]["purity_min"] if "nontrivial_closed" in s else s["closed"]["0"]["purity_min"]
        rk[str(T)] = {g: short(pmin(g)) for g in ("cs", "cd", "snap")}
        rk[str(T)]["cs_highest"] = bool(pmin("cs")["mean"] > max(pmin("cd")["mean"], pmin("snap")["mean"]))
    checks["entanglement_ranking_tracking"] = dict(
        rule="mean minimum purity over the starts (SNAP: non-trivial starts only, the trivial ones never leave |0>); "
             "consistent with the tangential capacity, by which CS can push along the path without entangling and "
             "ECD and SNAP cannot (v = 0), but not a proof of it",
        per_T=rk, holds=bool(all(v["cs_highest"] for v in rk.values())))
    # 5 simultaneous rather than interleaved
    si = {}
    for t in [x for x in TR + FT + FK if x[1] in ("cs", "cd")]:
        c = case(*t)
        cc = st[c]["controls"]
        si[c] = dict(both_on=short(cc["overlap_both"]), drive_only=short(cc["overlap_drive_only"]),
                     cond_only=short(cc["overlap_cond_only"]), neither=short(cc["overlap_neither"]),
                     corr_r_um=short(cc["corr_r_um"], 2) if cc.get("corr_r_um") else None,
                     simultaneous=bool(cc["overlap_both"]["mean"] > cc["overlap_drive_only"]["mean"]
                                       + cc["overlap_cond_only"]["mean"]))
    checks["rotations_vs_conditional_part"] = dict(
        rule="slices with r >= 0.5 ('drive on') and |u_-| >= 0.5 g ('conditional on'); a literal interleaving of "
             "rotations and conditional gates (SI Sec. V) would put the slices in 'drive only' or 'conditional only' "
             "and give a negative correlation of r and |u_-|; 'simultaneous' if both-on exceeds the two alone together",
        per_case=si, simultaneous_in=[c for c, v in si.items() if v["simultaneous"]],
        not_simultaneous_in=[c for c, v in si.items() if not v["simultaneous"]])
    # 6 sigma_z-conditional generator at the bound (u_e = -u_g, |u_-| = g)
    checks["pure_conditional_at_bound"] = dict(
        rule="fraction of slices with |u_-| >= 0.99 g, that is u_e = -u_g with both branches at the bound "
             "(the ECD gate generator for displacements, the sigma_z-conditional squeeze for CS)",
        per_case={case(*t): short(st[case(*t)]["controls"]["um_at_bound"]) for t in TR + FT if t[1] in ("cs", "cd")})
    # 7 ECD flip stall: unentangled, unconditional pulses
    stall = {}
    for T in (412, 824):
        c = case("flip_target", "cd", T)
        z = np.load(OUT / f"{c}.npz")
        R = byfile[c]
        sc = z["avg6_opt_lind"]
        pm = np.array([min(r["closed"][nm]["purity_min"] for nm in r["closed"]) for r in R])
        cf = np.array([r["controls"]["cond_frac"] for r in R])
        lo = sc < 0.8
        stall[str(T)] = dict(n_stalled=int(lo.sum()), n=int(len(sc)), score=[float(x) for x in sc],
                             purity_min=[float(x) for x in pm], cond_frac=[float(x) for x in cf],
                             stalled_purity_min=(ms(pm[lo]) if lo.any() else None),
                             others_purity_min=(ms(pm[~lo]) if (~lo).any() else None),
                             stalled_cond_frac=(ms(cf[lo]) if lo.any() else None),
                             others_cond_frac=(ms(cf[~lo]) if (~lo).any() else None))
    checks["ecd_flip_stall"] = dict(
        rule="ECD gate-only flip pulses with score < 0.8 (the stall near the best plain displacement): smallest purity "
             "over the three inputs and conditional fraction, against the other starts",
        per_T=stall)
    # 8 SNAP zero-drive runs and drive strength
    sn = {}
    for T in (412, 824):
        c = case("tracking_gkp", "snap", T)
        s = st[c]
        z = np.load(OUT / f"{c}.npz")
        nt = ~z["trivial"]
        sn[str(T)] = dict(n_trivial=s["n_trivial"], trivial_max_cavity_drive_over_bound=s["trivial_osc_max"],
                          trivial_qubit_drive_mean=short(s["trivial_drive_mean"]),
                          trivial_F_T_noisy=ms(z["F_T_noisy"][~nt]), nontrivial_F_T_noisy=ms(z["F_T_noisy"][nt]),
                          nontrivial_C_noisy=ms(z["C_noisy"][nt]))
    checks["snap_zero_drive"] = dict(
        rule="largest cavity drive |eps|/g over the slices of every trivial run (mfcommon.trivial), and their qubit "
             "drive; with the oscillator in |0> the qubit drive and the dispersive term do not act on it",
        per_T=sn)
    dr = {}
    for t in TR:
        c = case(*t)
        s = st[c]
        dd = s["nontrivial_controls"]["drive_mean"] if "nontrivial_controls" in s else s["controls"]["drive_mean"]
        le = s["nontrivial_controls"]["drive_le_chi"] if "nontrivial_controls" in s else s["controls"]["drive_le_chi"]
        dr[c] = dict(drive_mean_over_bound=short(dd), mean_Omega_over_g=10 * dd["mean"], frac_Omega_le_chi_d=short(le))
    checks["qubit_drive_strength"] = dict(
        rule="time-averaged |Omega| (fraction of 10 g) and fraction of slices with |Omega| <= chi_d = g (r <= 0.1); "
             "SNAP non-trivial starts only; number-selective qubit rotations need |Omega| <~ chi_d",
        per_case=dr)
    # 9 gate versus tracking pulses (flip, 824 ns), and preparation versus flip
    gv = {}
    for g in ("cs", "cd", "snap"):
        row = {}
        for lab, c in (("target", case("flip_target", g, 824)), ("track_W10", case("flip_track", g, 824, 10)),
                       ("track_W70", case("flip_track", g, 824, 70)), ("prep_tracking", case("tracking_gkp", g, 824))):
            cc, q = st[c]["controls"], st[c]
            pm = q["closed_min_over_inputs"]["purity_min"] if c.startswith("flip") else q["closed"]["0"]["purity_min"]
            nm = (ms([max(r["closed"][nm_]["nbar_max"] for nm_ in r["closed"]) for r in byfile[c]])
                  if c.startswith("flip") else q["closed"]["0"]["nbar_max"])
            row[lab] = dict(osc_at_bound=short(cc["osc_at_bound"]), osc_mean=short(cc["osc_mean"]),
                            drive_mean=short(cc["drive_mean"]), rotation_over_pi=short(cc["rotation_angle_over_pi"], 1),
                            cond_frac=(short(cc["cond_frac"]) if "cond_frac" in cc else None),
                            purity_min=short(pm), nbar_max=short(nm, 2))
        gv[g] = row
    checks["gate_vs_tracking_824ns"] = dict(rule="same duration (824 ns), all starts; flip: purity minimum and photon "
                                                 "maximum over the inputs", per_gate=gv)
    # 10 SNAP gate tracking at 412 ns: a quarter turn through the dispersive term (observation)
    obs = {}
    for W in (10, 70):
        c = case("flip_track", "snap", 412, W)
        zc = np.load(OUT / f"closed_{c}.npz")
        z = np.load(OUT / f"{c}.npz")
        t = zc["t_ns"] * 1e-3
        Pe0 = zc["Pe"][:, 0, :]                          # input 0_L
        phase = G * np.trapezoid(Pe0, t, axis=1) if hasattr(np, "trapezoid") else G * np.trapz(Pe0, t, axis=1)
        obs[f"W{W}"] = dict(max_cavity_drive_over_g=ms(np.max(np.abs(z["eps"]), axis=1)),
                            chi_d_int_Pe_dt_over_pi_2=ms(phase / (math.pi / 2)), score=ms(z["avg6_opt_lind"]),
                            F_0to1_closed=st[c]["F_0to1_closed"])
    checks["snap_flip_tracking_412ns"] = dict(
        rule="observation, not a claim: chi_d int P_e dt is the phase-space rotation angle that the dispersive term "
             "imparts on the |e> branch; a quarter turn maps |0_L> to |+_L> (F(0_L -> 1_L) = 1/2) and scores 2/3 "
             "against X_L",
        per_W=obs)
    facts["checks"] = checks
    (OUT / "pulses_facts_A.json").write_text(json.dumps(facts, indent=1))
    print("wrote", OUT / "pulses_facts_A.json")
    print("reproduction holds:", facts["reproduction"]["holds"], rep["max_dev_tracking_FT_C"], rep["max_dev_flip_metric"])
    for k, v in checks.items():
        print(k, {kk: vv for kk, vv in v.items() if kk in ("holds", "holds_for", "fails_for", "made_in", "not_made_in",
                                                          "not_undone_in", "simultaneous_in", "not_simultaneous_in")})


if __name__ == "__main__":
    main()
