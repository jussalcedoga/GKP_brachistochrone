"""Every number of data-B (target-only pulses of Letter Table I and the smoothing test of the pulses), from the data.

Reads   sims/results/pulses/capability_pulses.jsonl        (capability_pulses.py: the 412 ns target-only pulses)
        sims/results/capability_noisy.jsonl                 (the stored Table I runs, for the reproduction check)
        sims/results/pulses/pulses_smooth.jsonl            (pulses_smooth.py: smoothed pulses, every start and sigma)
        sims/results/pulses/pulses_smooth_smoke.json       (convolution check, grid convergence, timing)
        sims/results/pulses/capability_pulses_smoke.jsonl  (timing of one target-only run)
        manuscript_figures/fig4/fig4_facts.json             (representative starts and trivial counts of Fig. 4)
        main.tex of the paper                        (the quoted Table I entries, read only)
        sims/results/pulses/pulses_check_B.json            (pulses_check_B.py: independent checks, spot checks)
Writes  sims/results/pulses/pulses_facts_B.json and sims/results/pulses/README_B.md (numbers filled in from the json).
Statistics are mean and sample s.d. (ddof = 1) over all starts (none excluded).  Delta = value at sigma minus the value
of the same start at sigma = 0 on the same sub-slice grid (flip: the native evaluation, exact per slice).
  python simulations/gate_sets/pulses_facts_B.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import math
import pathlib
import re
from collections import defaultdict

import sys

import numpy as np

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "manuscript"))
from common.params import G_SQ  # noqa: E402
MF = HERE.parent
ROOT = MF.parent
P = MF / "sims" / "results" / "pulses"
GATES = {"track": ("cs", "cd", "snap"), "target": ("cs", "cd", "snap", "snap4"), "flip": ("cs", "cd", "snap")}
SIG = (1.0, 2.0, 4.0)


def ms(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                min=float(v.min()), max=float(v.max()), max_abs=float(np.abs(v).max()))


def jl(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


# ----------------------------------------------------------------------------- 1. reproduction of Table I (412 ns)
def table_quoted():
    """Table I entries of the Letter at 412 ns: mean, its decimals, s.d. digits, best"""
    tex = _L.paper_text("supplement.tex")   # Supplementary Table 15 (Nov 5)
    block = tex[tex.index(r"\label{tab:capability}"):]
    block = block[:block.index(r"\end{tabular}")]
    rows = {}
    for lab, gk in (("CS &", "cs"), ("CD &", "cd"), (r"SNAP, $\chi_d=g$ &", "snap"), (r"SNAP, $\chi_d=4g$ &", "snap4")):
        line = next(l for l in block.splitlines() if l.strip().startswith(lab))
        m = re.search(r"&\s*([0-9.]+)\((\d+)\)\s*\[([0-9.]+)\]", line)
        rows[gk] = dict(text=m.group(0)[1:].strip(), mean=m.group(1), sd_digits=m.group(2), best=m.group(3))
    return rows


def reproduction():
    new = {(r["gate"], r["seed"]): r for r in jl(P / "capability_pulses.jsonl")}
    old = {(r["gate"], r["seed"]): r for r in jl(MF / "sims" / "results" / "capability_noisy.jsonl")
           if abs(r["T_ns"] - 412) < 1}
    out = dict(n_new=len(new), n_old_412=len(old), per_gate={}, per_seed=[])
    keys = ("F60", "F90_closed", "F90_noisy")
    allmax = {k: 0.0 for k in keys}
    for (gk, s), r in sorted(new.items()):
        o = old.get((gk, s))
        if o is None:
            continue
        d = {k: float(r[k] - o[k]) for k in keys}
        out["per_seed"].append(dict(gate=gk, seed=s, **{f"d_{k}": d[k] for k in keys},
                                    wall_s=r.get("wall_s")))
        for k in keys:
            allmax[k] = max(allmax[k], abs(d[k]))
    out["max_abs_diff"] = allmax
    out["reproduced_to_1e-6"] = bool(max(allmax.values()) <= 1e-6) if out["per_seed"] else None
    q = table_quoted()
    for gk in GATES["target"]:
        v = np.array([r["F90_noisy"] for (g, s), r in new.items() if g == gk])
        vo = np.array([o["F90_noisy"] for (g, s), o in old.items() if g == gk])
        if v.size == 0:
            continue
        dec = len(q[gk]["mean"].split(".")[1])
        sd_q = int(q[gk]["sd_digits"])
        st = dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)), best=float(v.max()),
                  stored_mean=float(vo.mean()), stored_sd=float(vo.std(ddof=1)), stored_best=float(vo.max()),
                  quoted=q[gk]["text"])
        st["matches_quoted"] = dict(
            mean=f"{st['mean']:.{dec}f}" == q[gk]["mean"],
            sd=int(round(st["sd"] * 10 ** dec)) == sd_q,
            best=f"{st['best']:.3f}" == q[gk]["best"])
        st["formatted"] = f"{st['mean']:.{dec}f}({int(round(st['sd'] * 10 ** dec))}) [{st['best']:.3f}]"
        out["per_gate"][gk] = st
    sm = jl(P / "capability_pulses_smoke.jsonl")
    if sm:
        out["smoke"] = dict(gate=sm[0]["gate"], seed=sm[0]["seed"], wall_s=sm[0]["wall_s"],
                            d_F90_noisy=float(sm[0]["F90_noisy"] - old[(sm[0]["gate"], sm[0]["seed"])]["F90_noisy"]))
    walls = [r["wall_s"] for r in new.values() if "wall_s" in r]
    if walls:
        out["wall_s"] = ms(walls)
    return out


# ----------------------------------------------------------------------------- 2. smoothing
def metrics(r):
    if r["set"] == "flip":
        L, Cc = r["eval"]["lind"], r["eval"]["closed"]
        return dict(avg6_opt_noisy=L["avg6_opt"], avg6_opt_closed=Cc["avg6_opt"], avg6_noisy=L["avg6"],
                    F_0to1_noisy=L["F_0to1"], nbar_max=r["eval"]["nbar_max"])
    m = dict(F_T_noisy=r["F_T_noisy"], F_T_closed=r["F_T_closed"])
    if r["set"] == "track":
        m.update(C_noisy=r["C_noisy"], C_closed=r["C_closed"])
    return m


def stored_metrics(r):
    s = r["stored"]
    if r["set"] == "flip":
        return dict(avg6_opt_noisy=s["lind"]["avg6_opt"], avg6_opt_closed=s["closed"]["avg6_opt"],
                    avg6_noisy=s["lind"]["avg6"], F_0to1_noisy=s["lind"]["F_0to1"])
    return {k: v for k, v in s.items()}


def smoothing():
    rows = jl(P / "pulses_smooth.jsonl")
    f4 = json.loads((MF / "fig4" / "fig4_facts.json").read_text())
    rep = {(g, int(k)): f4["columns"][str(k)][g]["rep_seed"] for k in (1, 2) for g in ("cs", "cd", "snap")}
    ntriv_f4 = {(g, int(k)): f4["columns"][str(k)][g]["n_trivial"] for k in (1, 2) for g in ("cs", "cd", "snap")}
    by = defaultdict(dict)
    for r in rows:
        by[(r["set"], r["gate"], round(r["T_ns"]))][(r["seed"], r["sigma_ns"], r.get("edge", "zero"))] = r
    out = dict(n_rows=len(rows), groups={}, bounds={}, baseline_vs_stored={})
    bmax = {"sigma0": [0.0, 0.0], "smoothed": [0.0, 0.0]}
    wmin, rowsum = 0.0, 0.0
    for (st, gk, T), d in sorted(by.items()):
        for edge in ("zero", "hold"):
            if not any(e == edge for (_, sg, e) in d if sg > 0):
                continue
            grp, bm, wm, rs = _group(st, gk, T, d, edge, rep, ntriv_f4, out)
            out["groups"][f"{st}_{gk}_{T}" + ("" if edge == "zero" else "_hold")] = grp
            for i in (0, 1):
                bmax["smoothed"][i] = max(bmax["smoothed"][i], bm["smoothed"][i])
                bmax["sigma0"][i] = max(bmax["sigma0"][i], bm["sigma0"][i])
            wmin, rowsum = min(wmin, wm), max(rowsum, rs)
    out["bounds"] = dict(max_Omega_over_10g_sigma0=bmax["sigma0"][0], max_osc_over_g_sigma0=bmax["sigma0"][1],
                         max_Omega_over_10g_smoothed=bmax["smoothed"][0], max_osc_over_g_smoothed=bmax["smoothed"][1],
                         kernel_weight_min=wmin, kernel_rowsum_max=rowsum)
    return out


def _group(st, gk, T, d, edge, rep, ntriv_f4, out):
    bmax = {"sigma0": [0.0, 0.0], "smoothed": [0.0, 0.0]}
    wmin, rowsum = 0.0, 0.0
    if True:
        seeds = sorted({s for s, _, _ in d})
        grp = dict(n_starts=len(seeds), seeds=seeds, edge=edge, M=sorted({r["M"] for r in d.values()}),
                   h_ns=sorted({round(r["h_ns"], 6) for r in d.values()}), sigma={})
        if st == "track":
            k = 1 if T < 600 else 2
            grp["k"] = k
            grp["rep_seed_fig4"] = rep[(gk, k)]
            triv = sorted({s for (s, sg, e), r in d.items() if r.get("trivial")})
            grp["trivial_seeds"] = triv
            grp["n_trivial"] = len(triv)
            grp["n_trivial_fig4"] = ntriv_f4[(gk, k)]
        base = {s: d[(s, 0.0, "zero")] for s in seeds if (s, 0.0, "zero") in d}
        # sigma = 0 on the sub-slice grid against the stored values
        dev = defaultdict(float)
        for s, r in base.items():
            mm, sm_ = metrics(r), stored_metrics(r)
            for key in sm_:
                if key in mm:
                    dev[key] = max(dev[key], abs(mm[key] - sm_[key]))
        if edge == "zero":
            out["baseline_vs_stored"][f"{st}_{gk}_{T}"] = dict(dev)
        grp["sigma"]["0"] = {key: ms([metrics(base[s])[key] for s in base]) for key in metrics(next(iter(base.values())))}
        for sg in SIG:
            have = [s for s in seeds if (s, sg, edge) in d and s in base]
            if not have:
                continue
            m0 = {s: metrics(base[s]) for s in have}
            m1 = {s: metrics(d[(s, sg, edge)]) for s in have}
            e = dict(n=len(have))
            for key in m0[have[0]]:
                e[key] = ms([m1[s][key] for s in have])
                e["d_" + key] = ms([m1[s][key] - m0[s][key] for s in have])
                if st == "track":
                    nt = [s for s in have if s not in grp["trivial_seeds"]]
                    if nt and len(nt) < len(have):
                        e["d_" + key + "_nontrivial"] = ms([m1[s][key] - m0[s][key] for s in nt])
                    if grp["rep_seed_fig4"] in have:
                        e["rep_" + key] = m1[grp["rep_seed_fig4"]][key]
                        e["rep_d_" + key] = m1[grp["rep_seed_fig4"]][key] - m0[grp["rep_seed_fig4"]][key]
            ch = [d[(s, sg, edge)]["checks"] for s in have]
            e["rel_dC_drive"] = ms([c["rel_dC"]["drive"] for c in ch])
            e["rel_dC_osc"] = ms([c["rel_dC"]["osc"] for c in ch])
            e["area_ratio_drive"] = ms([c["area_ratio_drive"] for c in ch])
            e["bound_Omega_max"] = float(max(c["bound_Omega"] for c in ch))
            e["bound_osc_max"] = float(max(c["bound_osc"] for c in ch))
            bmax["smoothed"][0] = max(bmax["smoothed"][0], e["bound_Omega_max"])
            bmax["smoothed"][1] = max(bmax["smoothed"][1], e["bound_osc_max"])
            wmin = min(wmin, min(c["w_min"] for c in ch))
            rowsum = max(rowsum, max(c["w_rowsum_max"] for c in ch))
            grp["sigma"][f"{sg:g}"] = e
        for s, r in base.items():
            bmax["sigma0"][0] = max(bmax["sigma0"][0], r["checks"]["bound_Omega"])
            bmax["sigma0"][1] = max(bmax["sigma0"][1], r["checks"]["bound_osc"])
    return grp, bmax, wmin, rowsum


def smoke():
    p = P / "pulses_smooth_smoke.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    conv = defaultdict(dict)
    for r in s["convergence"]:
        tag = f"{r['set']}_{r['gate']}_{round(r['T_ns'])}_s{r['seed']}_sigma{r['sigma_ns']:g}"
        mk = "native" if str(r.get("grid", "")).startswith("native") else str(r["M"])
        conv[tag][mk] = dict(metrics(r), h_ns=r["h_ns"], wall_s=r["wall_s"], stored=stored_metrics(r))
    # differences between successive grids
    for tag, d in conv.items():
        Ms = sorted(int(m) for m in d if m.isdigit())
        pairs = list(zip(Ms[:-1], Ms[1:])) + ([(Ms[0], "native")] if "native" in d and Ms else [])
        if pairs:
            diffs = {}
            for a, b in pairs:
                diffs[f"{a}->{b}"] = {k: abs(d[str(b)][k] - d[str(a)][k]) for k in d[str(a)]
                                      if isinstance(d[str(a)][k], float) and k not in ("h_ns", "wall_s")}
            d["grid_diffs"] = diffs
    return dict(convolution_check=s["convolution_check_cs_k1_s11"], convergence=conv)


# control statistics with the definitions of data-A (copy of pulses_stats.control_stats, thresholds BOUND, ZERO), so
# that the target-only pulses are described with the same numbers as the tracking and flip pulses; checked against
# data-A's per-start rows on the flip gate pulses (pulse_shapes()['check_vs_A'])
BOUND, ZERO = 0.999, 0.001


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def _jumps(z, thr=0.1):
    ok = (np.abs(z[1:]) > thr) & (np.abs(z[:-1]) > thr)
    d = np.abs(_wrap(np.angle(z[1:]) - np.angle(z[:-1])))
    return int(np.sum(ok & (d > np.pi / 2)))


def control_stats(gate, X, dt_us):
    X = np.asarray(X, float)
    r, fO = X[:, 0], X[:, 1]
    Om = r * np.exp(1j * fO)
    s = dict(drive_at_bound=float(np.mean(r >= BOUND)), drive_at_zero=float(np.mean(r <= ZERO)), drive_mean=float(r.mean()),
             rotation_angle_over_pi=float(np.sum(10 * G_SQ * r * dt_us) / math.pi), jumps_Omega=_jumps(Om))
    if gate == "snap":
        a = X[:, 2]
        eps = a * np.exp(1j * X[:, 3])
        s.update(osc_at_bound=float(np.mean(a >= BOUND)), osc_at_zero=float(np.mean(a <= ZERO)), osc_mean=float(a.mean()),
                 osc_max=float(a.max()), jumps_eps=_jumps(eps), drive_le_chi=float(np.mean(r <= 0.1)))
        return s
    ag, ae = X[:, 2], X[:, 4]
    ug, ue = ag * np.exp(1j * X[:, 3]), ae * np.exp(1j * X[:, 5])
    up, um = 0.5 * (ug + ue), 0.5 * (ug - ue)
    A = np.r_[ag, ae]
    on_r, on_c = r >= 0.5, np.abs(um) >= 0.5
    rc = np.corrcoef(r, np.abs(um))[0, 1] if (r.std() > 0 and np.abs(um).std() > 0) else float("nan")
    s.update(osc_at_bound=float(np.mean(A >= BOUND)), osc_at_zero=float(np.mean(A <= ZERO)), osc_mean=float(A.mean()),
             osc_g_at_bound=float(np.mean(ag >= BOUND)), osc_e_at_bound=float(np.mean(ae >= BOUND)),
             both_branches_at_bound=float(np.mean((ag >= BOUND) & (ae >= BOUND))),
             cond_frac=float(np.sum(np.abs(um) ** 2) / np.sum(np.abs(up) ** 2 + np.abs(um) ** 2)),
             um_mean=float(np.abs(um).mean()), up_mean=float(np.abs(up).mean()),
             um_at_bound=float(np.mean(np.abs(um) >= 0.99)), up_at_bound=float(np.mean(np.abs(up) >= 0.99)),
             jumps_um=_jumps(um), jumps_up=_jumps(up),
             overlap_both=float(np.mean(on_r & on_c)), overlap_drive_only=float(np.mean(on_r & ~on_c)),
             overlap_cond_only=float(np.mean(~on_r & on_c)), overlap_neither=float(np.mean(~on_r & ~on_c)),
             corr_r_um=float(rc), drive_le_chi=float(np.mean(r <= 0.1)))
    return s


def pulse_shapes():
    """control statistics of the target-only pulses of Table I (412 ns), mean +- s.d. over the 8 starts, with the
    definitions of data-A; check of the copied function against data-A's rows on the flip gate pulses"""
    groups = defaultdict(list)
    for r in jl(P / "capability_pulses.jsonl"):
        g = r["gate"] if r["gate"] != "snap4" else "snap"
        groups[f"target_{r['gate']}_{round(r['T_ns'])}"].append(control_stats(g, r["X"], 1e-3 * r["T_ns"] / len(r["X"])))
    out = {g: {k: ms([d[k] for d in lst]) for k in lst[0]} for g, lst in sorted(groups.items())}
    rowsA = P / "pulses_stats_A_rows.json"
    if rowsA.exists():
        A = {(x["file"], x["seed"]): x["controls"] for x in json.loads(rowsA.read_text())}
        dev, n = 0.0, 0
        for r in jl(MF / "sims" / "results" / "flip_gate.jsonl"):
            if r["mode"] != "target":
                continue
            key = (f"flip_target_{r['gate']}_T{round(r['T_ns'])}.npz", r["seed"])
            if key in A:
                mine = control_stats(r["gate"], r["X"], 1e-3 * r["T_ns"] / len(r["X"]))
                for k, v in mine.items():
                    if isinstance(v, float) and math.isnan(v):
                        continue
                    dev = max(dev, abs(v - A[key][k]))
                n += 1
        out["check_vs_A"] = dict(n_pulses=n, max_abs_dev=dev)
    return out


def checks():
    """summary of pulses_check_B.json (independent checks of completeness, bounds, statistics, reproduction and the
    spot checks with the independent smoothing and integrator)"""
    p = P / "pulses_check_B.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    rw, rp = c["rows"], c["reproduction"]
    out = dict(n_rows=rw["n_rows"], n_unique_keys=rw["n_unique_keys"],
               n_starts={k: v["n_starts"] for k, v in rw["cases"].items()},
               every_variant_has_every_start=all(v["every_variant_has_every_start"] for v in rw["cases"].values()),
               starts_equal_dataA=all(v["seeds_equal"] for v in rw["vs_dataA"]["cases"].values()),
               trivial_equal_dataA=all(v.get("trivial_equal", True) for v in rw["vs_dataA"]["cases"].values()),
               X_vs_dataA_max_abs=rw["vs_dataA"]["max_abs_dX"], target_seeds_ok=rw["target_seeds_ok"],
               bounds=rw["bounds"], stats_vs_facts=rw["stats_vs_facts"],
               reproduction=dict(n_bitwise_equal=rp["n_bitwise_equal"], n=rp["n_new"], max_abs_diff=rp["max_abs_diff"],
                                 C_vs_formula_max_abs=rp["C_vs_formula_max_abs"],
                                 table_equal={k: v["equal"] for k, v in rp["table"].items()},
                                 table_line={k: v["line"] for k, v in rp["table"].items()}))
    if "spot" in c:
        out["spot"] = dict(c["spot"]["summary"], n=len(c["spot"]["checks"]),
                           cases=[s["key"] for s in c["spot"]["checks"]])
    return out


def derived(Sm):
    """ranking of the gate sets at every sigma (means), the worst single start, the growth of the mean loss with sigma
    (exponents log2 of successive ratios) and the share of the loss due to the lower drive at the ends (zero vs hold)"""
    G = Sm["groups"]
    out = dict(ranking={}, worst_start={}, growth={}, end_share={})
    for st, T, gates, q in (("track", 412, ("cs", "cd", "snap"), "F_T_noisy"), ("track", 824, ("cs", "cd", "snap"), "F_T_noisy"),
                            ("target", 412, ("cs", "cd", "snap", "snap4"), "F_T_noisy"),
                            ("flip", 824, ("cs", "cd", "snap"), "avg6_opt_noisy")):
        rk = {}
        for sg in ("0", "1", "2", "4"):
            mean = {g: G[f"{st}_{g}_{T}"]["sigma"][sg][q]["mean"] for g in gates}
            rk[sg] = dict(means=mean, order=sorted(gates, key=lambda g: -mean[g]))
        rk["same_order_all_sigma"] = len({tuple(v["order"]) for v in rk.values()}) == 1
        out["ranking"][f"{st}_{T}"] = rk
    for sg in ("1", "2", "4"):
        worst, wmean = None, None
        for name, gp in G.items():
            if name.endswith("_hold"):
                continue
            q = "avg6_opt_noisy" if name.startswith("flip") else "F_T_noisy"
            e = gp["sigma"][sg]["d_" + q]
            if worst is None or e["min"] < worst[1]:
                worst = (name, e["min"])
            if wmean is None or e["mean"] < wmean[1]:
                wmean = (name, e["mean"])
        out["worst_start"][sg] = dict(case=worst[0], d=worst[1], worst_mean_case=wmean[0], worst_mean_d=wmean[1])
    for name, gp in G.items():
        q = "avg6_opt_noisy" if name.startswith("flip") else "F_T_noisy"
        d = {sg: gp["sigma"][sg]["d_" + q]["mean"] for sg in ("1", "2", "4")}
        ex = {}
        for a, b in (("1", "2"), ("2", "4")):
            if d[a] < -1e-5 and d[b] < -1e-5:
                ex[f"{a}->{b}"] = math.log2(d[b] / d[a])
        out["growth"][name] = dict(d_mean=d, exponent=ex)
        if not name.endswith("_hold") and name + "_hold" in G:
            h = {sg: G[name + "_hold"]["sigma"][sg]["d_" + q]["mean"] for sg in ("1", "2", "4")}
            out["end_share"][name] = {sg: (1 - h[sg] / d[sg]) if d[sg] < -1e-4 else None for sg in ("1", "2", "4")}
    return out


def bandwidths():
    """half-power frequency of the Gaussian filter, |H(f)|^2 = exp(-(2 pi f sigma)^2) = 1/2, and sigma / Delta t"""
    return {f"{s:g}": dict(f_half_power_MHz=1e3 * math.sqrt(math.log(2)) / (2 * math.pi * s),
                           sigma_over_dt_412=s / (412.0 / 50), sigma_over_dt_824=s / (824.0 / 50)) for s in SIG}


# ----------------------------------------------------------------------------- README (numbers from the facts)
NAMES = {"cs": "CS", "cd": "ECD", "snap": "SNAP", "snap4": "SNAP (chi_d = 4g)"}


def f3(x):
    return f"{1e3 * x:+.2f}"


def readme(F):
    R, Sm, K = F["reproduction"], F["smoothing"], F["smoke"]
    L = ["# data-B: target-only pulses of Table I and the smoothing test of the pulses", "",
         "Written by `simulations/gate_sets/pulses_facts_B.py` from the data files (every number below is in "
         "`pulses_facts_B.json`). Statistics are mean and sample s.d. over all starts (none excluded).", ""]
    L += ["## 1. Target-only GKP preparation at 412 ns (Letter Table I), with the pulses", "",
          "`sims/capability_pulses.py` repeats `sims/capability_noisy.py` (imported, numerics unchanged) for CS, ECD, "
          "SNAP (chi_d = g, 4g), seeds 1000-1007, and stores the slice variables X and the Cartesian controls in "
          "`capability_pulses.jsonl`.", ""]
    if R.get("per_seed"):
        md = R["max_abs_diff"]
        L += [f"Per-start reproduction against `capability_noisy.jsonl` ({len(R['per_seed'])} starts): max |dF60| = "
              f"{md['F60']:.1e}, max |dF90_closed| = {md['F90_closed']:.1e}, max |dF90_noisy| = {md['F90_noisy']:.1e} "
              f"(reproduced to 1e-6: {R['reproduced_to_1e-6']}).", "",
              "| gate set | reproduced mean (s.d.) [best] | quoted in Table I | mean, s.d., best match |",
              "|---|---|---|---|"]
        for gk, st in R["per_gate"].items():
            mq = st["matches_quoted"]
            L.append(f"| {NAMES[gk]} | {st['formatted']} | {st['quoted']} | {mq['mean']}, {mq['sd']}, {mq['best']} |")
        if "wall_s" in R:
            L += ["", f"Wall time per start {R['wall_s']['mean']:.0f} +- {R['wall_s']['sd']:.0f} s (one core)."]
    L += ["", "## 2. Smoothing test", "",
          "Each Cartesian control is convolved with a normalized Gaussian of s.d. sigma, with zero drive before t = 0 "
          "and after T (duration kept, so the integrated drive near both ends is lower). Closed form (error "
          "functions), averaged over sub-slices of h = 0.2575 ns (M = 32 per slice at 412 ns, 64 at 824 ns), "
          "evaluated at cutoff 90 with `common/lindblad.evolve` and `CSModel.propagate` (preparation) or "
          "`flip_gate.evaluate` (flip gate). Delta = smoothed minus the same start at sigma = 0 on the same grid. "
          "Diagnostic `hold`: first and last slice values continued outside [0, T] (bandwidth only, no end loss).", ""]
    b = Sm["bounds"]
    L += [f"Bounds over all {Sm['n_rows']} rows: max |Omega|/10g - 1 = {b['max_Omega_over_10g_smoothed'] - 1:.1e} and "
          f"max |branch amplitude|/g - 1 = {b['max_osc_over_g_smoothed'] - 1:.1e} for the smoothed pulses (sigma = 0: "
          f"{b['max_Omega_over_10g_sigma0'] - 1:.1e}, {b['max_osc_over_g_sigma0'] - 1:.1e}), so every smoothed control "
          f"stays in its bound disk up to rounding; kernel weights >= {b['kernel_weight_min']:.1e}, row sums - 1 <= "
          f"{b['kernel_rowsum_max'] - 1:.1e}. For SNAP the branch column is the cavity drive.", ""]
    bw = F["bandwidth"]
    L += ["Half-power bandwidth of the filter: " + ", ".join(
        f"sigma = {s} ns -> {v['f_half_power_MHz']:.0f} MHz (sigma/Delta t = {v['sigma_over_dt_412']:.2f} at 412 ns)"
        for s, v in bw.items()) + ".", ""]
    if K:
        L += ["Convolution check (closed form against a brute-force quadrature, CS 412 ns start 11): max deviation "
              + ", ".join(f"{v['max_abs_diff_over_bound']:.1e} g (sigma = {s} ns)" for s, v in K["convolution_check"].items())
              + ".", ""]
        L += ["Grid convergence (sigma = 1 ns, |change| of F_T or gate metric when h is halved):", ""]
        for tag, d in K["convergence"].items():
            if "grid_diffs" in d and "sigma1" in tag:
                L.append(f"- {tag}: " + "; ".join(
                    f"M {k}: " + ", ".join(f"{kk} {vv:.1e}" for kk, vv in v.items() if kk.startswith(("F_T", "avg6_opt")))
                    for k, v in d["grid_diffs"].items()))
        L.append("")
    L += ["Baseline (sigma = 0 on the sub-slice grid) against the stored values, max |difference|:", ""]
    for g, dv in Sm["baseline_vs_stored"].items():
        L.append(f"- {g}: " + ", ".join(f"{k} {v:.1e}" for k, v in dv.items()))
    L += ["", "Change of the figures of merit (units of 1e-3, mean +- s.d. over starts [min, max]):", "",
          "| case | n | value at sigma = 0 | quantity | sigma = 1 ns | sigma = 2 ns | sigma = 4 ns | hold, 2 ns | hold, 4 ns |",
          "|---|---|---|---|---|---|---|---|---|"]
    for gname, gp in Sm["groups"].items():
        if gname.endswith("_hold"):
            continue
        hold = Sm["groups"].get(gname + "_hold", {}).get("sigma", {})
        s0 = gp["sigma"]["0"]
        for q in (("F_T_noisy", "F_T_closed", "C_noisy") if not gname.startswith("flip") else
                  ("avg6_opt_noisy", "avg6_opt_closed")):
            if q not in s0:
                continue
            cells = []
            for sg in ("1", "2", "4"):
                e = gp["sigma"].get(sg, {}).get("d_" + q)
                cells.append(f"{f3(e['mean'])} +- {1e3 * e['sd']:.2f} [{f3(e['min'])}, {f3(e['max'])}]" if e else "")
            for sg in ("2", "4"):
                e = hold.get(sg, {}).get("d_" + q)
                cells.append(f"{f3(e['mean'])} +- {1e3 * e['sd']:.2f}" if e else "")
            L.append(f"| {gname} | {gp['n_starts']} | {s0[q]['mean']:.4f} +- {s0[q]['sd']:.4f} | {q} | " + " | ".join(cells) + " |")
    if F.get("pulse_shapes"):
        ps = F["pulse_shapes"]
        L += ["", "## 3. Control statistics of the target-only pulses (412 ns, mean +- s.d. over 8 starts)", "",
              "Definitions of data-A (`pulses_stats.control_stats`, copied): fractions of slices with the oscillator "
              "amplitude at its bound (a >= 0.999, both branches pooled for CS and ECD) or at zero, the same for the "
              "qubit drive, conditional fraction int|u_-|^2 / int(|u_+|^2 + |u_-|^2) with u_+- = (u_g +- u_e)/2, qubit "
              "rotation int|Omega|dt / pi (bound 10 g T / pi = 9.84 at 412 ns). For SNAP the oscillator amplitude is the "
              "cavity drive."]
        if "check_vs_A" in ps:
            L.append(f"Copy checked against data-A's rows on {ps['check_vs_A']['n_pulses']} flip gate pulses: max "
                     f"|difference| {ps['check_vs_A']['max_abs_dev']:.1e}.")
        L += ["", "| pulses | osc at bound | osc at zero | drive at bound | mean drive / 10g | conditional fraction | "
              "qubit rotation / pi |", "|---|---|---|---|---|---|---|"]

        def c(d, k):
            return f"{d[k]['mean']:.2f} +- {d[k]['sd']:.2f}" if k in d else ""
        for g, d in ps.items():
            if g == "check_vs_A":
                continue
            L.append(f"| {g} | {c(d, 'osc_at_bound')} | {c(d, 'osc_at_zero')} | {c(d, 'drive_at_bound')} | "
                     f"{c(d, 'drive_mean')} | {c(d, 'cond_frac')} | {c(d, 'rotation_angle_over_pi')} |")
    L += ["", "Relative L2 change of the controls (mean over starts): " + "; ".join(
        f"{g} sigma {sg}: drive {gp['sigma'][sg]['rel_dC_drive']['mean']:.3f}, osc {gp['sigma'][sg]['rel_dC_osc']['mean']:.3f}"
        for g, gp in Sm["groups"].items() if not g.endswith("_hold") for sg in ("1", "4") if sg in gp["sigma"]) + ".", ""]
    Dv = F.get("derived")
    if Dv:
        L += ["## 4. Derived summaries (from the table above)", "",
              "Mean figure of merit of each gate set at sigma = 0, 1, 2, 4 ns (zero drive outside [0, T]) and the order "
              "of the gate sets:", ""]
        for case, rk in Dv["ranking"].items():
            L.append(f"- {case}: " + "; ".join(
                f"sigma {sg}: " + ", ".join(f"{NAMES[g]} {rk[sg]['means'][g]:.4f}" for g in rk[sg]["order"])
                for sg in ("0", "1", "2", "4")) + f" (same order at every sigma: {rk['same_order_all_sigma']})")
        L += ["", "Largest loss of a single start and largest mean loss (units of 1e-3): " + "; ".join(
            f"sigma {sg} ns: start {f3(w['d'])} ({w['case']}), mean {f3(w['worst_mean_d'])} ({w['worst_mean_case']})"
            for sg, w in Dv["worst_start"].items()) + ".", "",
            "Growth of the mean loss with sigma, exponent p of |Delta| ~ sigma^p between successive sigma "
            "(log2 of the ratio; only where both losses exceed 1e-5):", ""]
        for name, gr in Dv["growth"].items():
            if gr["exponent"]:
                L.append(f"- {name}: " + ", ".join(f"{k} p = {v:.1f}" for k, v in gr["exponent"].items()))
        L += ["", "Share of the mean loss due to the lower drive near the ends, 1 - Delta(hold)/Delta(zero):", ""]
        for name, es in Dv["end_share"].items():
            L.append(f"- {name}: " + ", ".join(f"sigma {sg}: {v:.2f}" for sg, v in es.items() if v is not None))
        L.append("")
    Ck = F.get("checks")
    if Ck:
        rp = Ck["reproduction"]
        L += ["## 5. Independent checks (`sims/pulses_check_B.py` -> `pulses_check_B.json`)", "",
              f"- Completeness: {Ck['n_rows']} rows, {Ck['n_unique_keys']} distinct (pulse, sigma, edge); starts per "
              "case " + ", ".join(f"{k} {v}" for k, v in Ck["n_starts"].items()) + "; every sigma and edge has every "
              f"start: {Ck['every_variant_has_every_start']}. Starts equal data-A's independent extraction "
              f"(mfcommon.groups, flip_gate.jsonl): {Ck['starts_equal_dataA']}, trivial SNAP starts equal: "
              f"{Ck['trivial_equal_dataA']}, max |X - X_A| = {Ck['X_vs_dataA_max_abs']:.1e}; target seeds 1000-1007 "
              f"of every gate set: {Ck['target_seeds_ok']}.",
              f"- Bounds over every row: max |Omega|/10g - 1 = {Ck['bounds']['Omega_minus_1']:.1e}, max |branch|/g - 1"
              f" = {Ck['bounds']['osc_minus_1']:.1e}, polar round trip {Ck['bounds']['polar_roundtrip_max']:.1e}.",
              f"- Statistics recomputed from the rows: {Ck['stats_vs_facts']['n_compared']} means and s.d. of Delta, "
              f"max |difference| {Ck['stats_vs_facts']['max_abs_dev']:.1e}.",
              f"- Table I: {rp['n_bitwise_equal']} of {rp['n']} starts bitwise equal to capability_noisy.jsonl (F60, "
              f"F90_closed, F90_noisy); stored C equals the formulas of Eq. (7) to {rp['C_vs_formula_max_abs']:.1e}; "
              "the entries re-formatted from the 8 starts equal the current main.tex: "
              + ", ".join(f"{NAMES[k]} {v}" for k, v in rp["table_equal"].items()) + "."]
        if "spot" in Ck:
            s = Ck["spot"]
            L += [f"- Spot checks ({s['n']} pulses: " + ", ".join(s["cases"]) + "): smoothed pulses rebuilt "
                  "independently (pointwise exact convolution with the normal CDF, 8-point Gauss-Legendre average per "
                  "sub-slice, controls from the formulas of Eq. (7)) and evaluated with the independent integrator of "
                  "analytics_audit/review_oct28/lindblad_crosscheck.py (exact expm_multiply per sub-slice; flip: "
                  "flip_gate.evaluate). Max |difference| from the rows: F_T noisy "
                  f"{s['max_abs_d_F_T_noisy']:.1e}, F_T closed {s['max_abs_d_F_T_closed']:.1e}, C noisy "
                  f"{s['max_abs_d_C_noisy']:.1e}, C closed {s['max_abs_d_C_closed']:.1e}, flip gate metric noisy "
                  f"{s['max_abs_d_avg6_opt_noisy']:.1e}, closed {s['max_abs_d_avg6_opt_closed']:.1e}; bounds of the "
                  f"rebuilt pulses - 1 <= {max(s['max_bound_Omega'], s['max_bound_osc']) - 1:.1e}. The noisy F_T "
                  "difference is the same at sigma = 0 (Lindblad splitting of src/common/lindblad.py against exact "
                  "exponentials), so it cancels in Delta."]
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sm_ = smoothing()
    facts = dict(reproduction=reproduction(), smoothing=sm_, smoke=smoke(), bandwidth=bandwidths(),
                 pulse_shapes=pulse_shapes(), derived=derived(sm_), checks=checks(),
                 conventions=dict(statistics="mean and sample s.d. (ddof=1) over all starts, none excluded",
                                  delta="value at sigma minus value of the same start at sigma=0 on the same grid",
                                  kernel="normalized Gaussian, std sigma, zero drive outside [0, T], duration kept; "
                                         "diagnostic 'hold': end values continued outside [0, T]",
                                  cutoff=90, noise="Yale rates (params.NOISE['Yale'])"))
    (P / "pulses_facts_B.json").write_text(json.dumps(facts, indent=1))
    (P / "README_B.md").write_text(readme(facts))
    print(json.dumps(facts["reproduction"].get("per_gate"), indent=1))
    print("max |diff|", facts["reproduction"].get("max_abs_diff"))
