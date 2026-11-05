"""Fig. 4 / SM data: following the time-optimal, photon-optimal and trade-off paths with the deck's tracking protocol,
conditional squeezing (CS) against conditional displacement (CD, the ECD generator), under the same constraints.

Protocol (src/common/tracking.py; protocol deck Eqs. (16)-(21), (28), (29), Table 5), identical for every job:
  bounds |Omega| <= 10 g, |xi_q| or |eps_q| <= g, g = 7.5 rad/us; durations T_g = g T in {2.58, 3.09, 6.18, 16.0, 30.0}
  (the deck's grid; T = 344, 412, 824, 2133, 4000 ns), N = 50, 50, 50, 100, 100 slices; random start (seed 2); stages
  A-E; tracking weight = largest w of the ladder with F_T >= 0.99 under Lindblad noise (photon loss, cavity dephasing,
  qubit relaxation and dephasing at the rates of the Yale device); cutoff 60, every chosen pulse re-checked at 80.
Jobs:
  main  Fock |2>, cat (alpha = 2), GKP |0_L> (Delta = 0.4) from the vacuum; references: brachistochrone, photon-optimal
        path, trade-off path at w* = kappa/mu (Yale) (for Fock |2> the three coincide, Thm. 1); CS on every path, CD
        (zero tangential capacity for every parity-even target) on the brachistochrone
  si    logical flip |0_L> -> |1_L> (three references), Fock |4> (brachistochrone)
  val   the deck's own settings (g = 15 rad/us, its light loss set) for the cat and Fock |2> at 206 and 1067 ns
Usage: python simulations/tracking/compute.py [main|si|val|all] [--workers 64] [--smoke]
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time
import traceback

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from common import brachistochrone as B, grape, lindblad, paths, photon_core as pc, states, tracking  # noqa: E402
from common.params import G_SQ, GKP_DELTA, OMEGA_REF, kappa_mu  # noqa: E402

JOBS = HERE / "results" / "jobs"
TG = (2.58, 3.09, 6.18, 16.0, 30.0)
NSL = {2.58: 50, 3.09: 50, 6.18: 50, 16.0: 100, 30.0: 100}
D_OPT, D_CHECK = 60, 80                  # optimization cutoff (population at n >= 40 penalized); check cutoff
SMOKE = False


def transfer(name, D):
    if name == "flip":
        return states.gkp(GKP_DELTA, 0, D), states.gkp(GKP_DELTA, 1, D)
    return states.vacuum(D), states.target(name, D, Delta=GKP_DELTA)


def reference_dense(name, ref, D):
    """dense reference path (rows: full Fock vectors) from start to target"""
    S, T = transfer(name, D)
    c0, th, chi = B.geodesic_parts(S, T)
    if ref == "geodesic":
        return B.path_states(S, T, np.linspace(0, th, 801))
    k, m = kappa_mu("Yale")
    if name == "flip":
        L = pc.Logical(GKP_DELTA, D)
        N = np.diag(L.nv)
        U_ph, _ = pc.optimize_path(N, L.u0, L.uT, [pc.resample(L.detour(), 30)], 30)
        geo = L.geo(np.linspace(0, L.th, 32))
        u0, uT = L.u0, L.uT
    else:
        P = paths.Prep(T)
        N = P.N
        U_ph, _ = paths.photon_path_prep(P)
        geo = P.geo(np.linspace(0, P.th, 32))
        u0, uT = P.u0, P.uT
    U = U_ph if ref == "photon" else paths.tradeoff_family(N, u0, uT, geo, U_ph, [k / m])[0]
    X, _ = pc.dense(U, 41)
    full = paths.to_full(X, D)
    return full * np.sign(np.real(np.vdot(S, full[0])))


def job_list(which):
    jobs = []
    if which in ("main", "all"):
        for name, refs in (("fock2", ("geodesic",)), ("cat", ("geodesic", "photon", "tradeoff")),
                           ("gkp", ("geodesic", "photon", "tradeoff"))):
            for ref in refs:
                for gate in (("cs", "cd") if ref == "geodesic" else ("cs",)):
                    for tg in TG:
                        jobs.append(dict(set="main", name=name, ref=ref, gate=gate, Tg=tg, g=G_SQ, noise="Yale"))
    if which in ("si", "all"):
        for name, refs in (("flip", ("geodesic", "photon", "tradeoff")), ("fock4", ("geodesic",))):
            for ref in refs:
                for gate in ("cs", "cd"):
                    for tg in TG:
                        jobs.append(dict(set="si", name=name, ref=ref, gate=gate, Tg=tg, g=G_SQ, noise="Yale"))
    if which in ("val", "all"):
        for name in ("cat", "fock2"):
            for tg in (3.09, 16.0):
                jobs.append(dict(set="val", name=name, ref="geodesic", gate="cs", Tg=tg, g=15.0, noise="deck"))
    for j in jobs:
        j["id"] = f"{j['set']}_{j['name']}_{j['ref']}_{j['gate']}_Tg{j['Tg']}"
    if which == "seeds3":                  # Fig. 4: random starts 11-20 (20 in total) at 412, 824, 2133 ns
        sj = [dict(name="gkp", ref=ref, gate=gate, k=k, seed=seed)
              for ref, gate in (("geodesic", "cs"), ("geodesic", "cd"), ("photon", "cs")) for k in (1, 2, 3)
              for seed in range(11, 21)]
        for j in sj:
            j.update(set="seed", g=G_SQ, noise="Yale", Tg=TG[j["k"]], pace="path",
                     id=f"seed_{j['name']}_{j['ref']}_{j['gate']}_k{j['k']}_s{j['seed']}")
        jobs += sj
    if which in ("snap", "snap_poc"):      # v2 figures: SNAP-displacement control, same protocol, bounds and starts as CD
        if which == "snap_poc":            # feasibility: one start per configuration (part of the production set)
            sj = [dict(name=n, ref=r, gate="snap", k=k, seed=1) for n, r, k in
                  (("gkp", "geodesic", 1), ("gkp", "geodesic", 2), ("fock2", "geodesic", 2), ("cat", "geodesic", 2),
                   ("flip", "geodesic", 2), ("flip", "photon", 2))]
        else:
            sj = [dict(name="gkp", ref="geodesic", gate="snap", k=k, seed=s) for k in range(4)
                  for s in range(1, 11 if k == 0 else 21)]
            sj += [dict(name=n, ref="geodesic", gate="snap", k=k, seed=s) for n in ("fock2", "cat", "binomial")
                   for k in range(4) for s in range(1, 6)]
            sj += [dict(name="flip", ref=r, gate="snap", k=k, seed=s) for r in ("geodesic", "tradeoff", "photon")
                   for k in range(4) for s in range(1, 21 if k == 2 else 4)]
        for j in sj:
            j.update(set="seed", g=G_SQ, noise="Yale", Tg=TG[j["k"]], pace="path",
                     id=f"seed_{j['name']}_{j['ref']}_{j['gate']}_k{j['k']}_s{j['seed']}")
        jobs += sj
    if which == "seeds4":                  # Fig. 5: the logical flip at m = 2.81 (k = 2), 20 random starts per curve
        sj = [dict(name="flip", ref=ref, gate=gate, k=2, seed=seed)
              for ref in ("geodesic", "tradeoff", "photon") for gate, s0 in (("cs", 6), ("cd", 4)) for seed in range(s0, 21)]
        for j in sj:
            j.update(set="seed", g=G_SQ, noise="Yale", Tg=TG[j["k"]], pace="path",
                     id=f"seed_{j['name']}_{j['ref']}_{j['gate']}_k{j['k']}_s{j['seed']}")
        jobs += sj
    if which == "seeds2":                  # Fig. 4: random starts 6-10 for GKP preparation (10 starts in total)
        sj = [dict(name="gkp", ref="geodesic", gate=gate, k=k, seed=seed) for gate in ("cs", "cd") for k in range(4)
              for seed in range(6, 11)]
        sj += [dict(name="gkp", ref="photon", gate="cs", k=k, seed=seed) for k in range(4) for seed in range(4, 11)]
        for j in sj:
            j.update(set="seed", g=G_SQ, noise="Yale", Tg=TG[j["k"]], pace="path",
                     id=f"seed_{j['name']}_{j['ref']}_{j['gate']}_k{j['k']}_s{j['seed']}")
        jobs += sj
    if which in ("seeds",):
        # Repeated random starts for confidence intervals, durations k = 0..3 of the deck grid (344, 412, 824, 2133 ns on
        # the brachistochrone).  Path-paced durations: every reference path is run at the same multiple m_k of its own
        # minimal time L_path/omega, m_k = g T_k / (g T*_geo) (so the brachistochrone keeps the grid); in absolute time
        # the photon-optimal path is then the slowest.  Seed 2 of the grid jobs already exists (sets main/si).
        sj = []
        for name in ("gkp", "fock2", "cat", "binomial"):           # Fig. 4 and SM: brachistochrone, CS vs CD
            for gate in ("cs", "cd"):
                for k in range(4):
                    for seed in ((1, 3, 4, 5) if name != "binomial" else (1, 2, 3, 4, 5)):
                        sj.append(dict(name=name, ref="geodesic", gate=gate, k=k, seed=seed))
        for gate, seeds_all in (("cs", (1, 2, 3, 4, 5)), ("cd", (1, 2, 3))):   # Fig. 5 and SM: the logical flip
            for ref in ("geodesic", "tradeoff", "photon"):
                for k in range(4):
                    for seed in seeds_all:
                        if ref == "geodesic" and seed == 2:
                            continue
                        sj.append(dict(name="flip", ref=ref, gate=gate, k=k, seed=seed))
        for name in ("gkp", "cat"):                                 # SM: the time-photon compromise for preparation
            for ref in ("tradeoff", "photon"):
                for k in range(4):
                    for seed in (1, 2, 3):
                        sj.append(dict(name=name, ref=ref, gate="cs", k=k, seed=seed))
        for j in sj:
            j.update(set="seed", g=G_SQ, noise="Yale", Tg=TG[j["k"]], pace="path",
                     id=f"seed_{j['name']}_{j['ref']}_{j['gate']}_k{j['k']}_s{j['seed']}")
        jobs += sj
    return jobs


def pad(X, D):
    out = np.zeros((X.shape[0], D), complex)
    out[:, :X.shape[1]] = X
    return out


def run(job):
    out = JOBS / f"{job['id']}.json"
    if out.exists():
        return job["id"], "skip"
    t0 = time.time()
    g, name = job["g"], job["name"]
    T = job["Tg"] / g
    N = NSL[job["Tg"]]
    noise = lindblad.rates(job["noise"])
    M = tracking.make_model(job["gate"], D_OPT, g, 10 * g)
    S, Tg = transfer(name, D_OPT)
    dense = reference_dense(name, job["ref"], D_OPT)
    path = tracking.Path(dense)
    if job.get("pace") == "path":                       # same multiple of each path's own minimal time L/omega
        c0, th, chi = B.geodesic_parts(S, Tg)
        m = job["Tg"] / (g * th / OMEGA_REF)
        T = m * path.L / OMEGA_REF
        job["m"], job["T_path_ns"] = m, 1e3 * path.L / OMEGA_REF
    psi0 = grape.joint(S)
    X = grape.random_start(N, np.random.default_rng(job.get("seed", 2)))
    sched = tracking.SCHEDULE if not SMOKE else tracking.SCHEDULE[:1] + tracking.SCHEDULE[-2:]
    log = []
    for (nm, frac, thr, w, wf, thf, it, free) in sched:
        X, _, inf = tracking.stage(M, X, N, frac * T, psi0, path, thr, w, wf, thf, 3 if SMOKE else it, free)
        log.append(dict(stage=nm, **inf))
    pol = tracking.POLISH
    rungs, Xs = [], {}
    Xprev = X
    for w in (tracking.LADDER if not SMOKE else tracking.LADDER[:2]):
        Xw, _, inf = tracking.stage(M, Xprev, N, T, psi0, path, pol["thr"], w, pol["w_fin"], pol["thr_fin"],
                                    3 if SMOKE else pol["iters"])
        Fl = tracking.lindblad_F(M, Xw, T, N, psi0, Tg, noise)
        rungs.append(dict(w=w, F_T=inf["F_T"], C_ends=inf["C"], F_loss=Fl))
        Xs[w] = Xw
        Xprev = Xw
    ok = [r for r in rungs if r["F_loss"] >= tracking.F_LOSS_MIN]
    if ok:                                                # two bisections above the largest passing rung
        wp = max(r["w"] for r in ok)
        above = [r["w"] for r in rungs if r["w"] > wp]
        if above:
            wf_ = min(above)
            for _ in range(2):
                wb = float(np.sqrt(wp * wf_))
                Xb, _, inf = tracking.stage(M, Xs[wp], N, T, psi0, path, pol["thr"], wb, pol["w_fin"], pol["thr_fin"],
                                            3 if SMOKE else pol["iters"])
                Fl = tracking.lindblad_F(M, Xb, T, N, psi0, Tg, noise)
                rungs.append(dict(w=wb, F_T=inf["F_T"], C_ends=inf["C"], F_loss=Fl, bisection=True))
                Xs[wb] = Xb
                if Fl >= tracking.F_LOSS_MIN:
                    wp = wb
                else:
                    wf_ = wb
        w_star, rule = wp, "largest w with F_loss >= 0.99"
    else:
        w_star, rule = max(rungs, key=lambda r: r["F_loss"])["w"], "none reaches 0.99: best F_loss"
    Xc = Xs[w_star]
    diag = tracking.diagnostics(M, Xc, T, N, psi0, path, Tg)
    F_loss = next(r["F_loss"] for r in rungs if r["w"] == w_star)
    # cutoff check at D = 90 (unitary closeness and fidelity, and the noisy fidelity)
    M9 = tracking.make_model(job["gate"], D_CHECK, g, 10 * g)
    S9, T9 = transfer(name, D_CHECK)
    path9 = tracking.Path(pad(dense, D_CHECK))
    d9 = tracking.diagnostics(M9, Xc, T, N, grape.joint(S9), path9, T9)
    F_loss9 = tracking.lindblad_F(M9, Xc, T, N, grape.joint(S9), T9, noise)
    res = dict(job=job, T_us=T, N=N, D=D_OPT, path_L=path.L, w_star=w_star, rule=rule, F_loss=F_loss,
               F_loss_D90=F_loss9, check_D90=dict(C=d9["C"], F_T=d9["F_T"]), stages=log, ladder=rungs,
               X=Xc.tolist(), path_F_loss=tracking.path_lindblad_F(path, D_OPT, T, Tg, noise), **diag,
               wall_s=time.time() - t0)
    out.write_text(json.dumps(res))
    return job["id"], f"{time.time() - t0:.0f}s w*={w_star:g} C={diag['C']:.3f} F_loss={F_loss:.4f}"


JOBS80 = HERE / "results" / "jobs80"
D_L3, D_L3_CHECK, L3_ITERS = 80, 90, 100
L3_TOP = 2 / 3                                # guard threshold n >= L3_TOP * 80; --guard 0.55 for pulses that leak upward


def level3(jid):
    """deck level 3 at a higher cutoff: polish the chosen pulse at cutoff 80 (100 iterations at w*), then up to three lower
    rungs of the ladder; keep the first with F_T >= 0.99 under noise, else the best; check at cutoff 90"""
    out = JOBS80 / f"{jid}.json"
    if out.exists():
        return jid, "skip"
    t0 = time.time()
    r = json.loads((JOBS / f"{jid}.json").read_text())
    job = r["job"]
    g, name, T, N = job["g"], job["name"], r["T_us"], r["N"]
    noise = lindblad.rates(job["noise"])
    M = tracking.make_model(job["gate"], D_L3, g, 10 * g, L3_TOP)
    S, Tg = transfer(name, D_L3)
    dense = reference_dense(name, job["ref"], D_L3)
    path = tracking.Path(dense)
    psi0 = grape.joint(S)
    pol = tracking.POLISH
    Xprev = np.array(r["X"])
    w0 = r["w_star"]
    tried = []
    for w in [w0] + sorted([w for w in tracking.LADDER if w < w0 - 1e-9], reverse=True)[:3]:
        Xw, _, inf = tracking.stage(M, Xprev, N, T, psi0, path, pol["thr"], w, pol["w_fin"], pol["thr_fin"], L3_ITERS)
        Fl = tracking.lindblad_F(M, Xw, T, N, psi0, Tg, noise)
        tried.append(dict(w=w, F_T=inf["F_T"], C_ends=inf["C"], F_loss=Fl, X=Xw))
        Xprev = Xw
        if Fl >= tracking.F_LOSS_MIN:
            break
    best = next((t for t in tried if t["F_loss"] >= tracking.F_LOSS_MIN), None) or max(tried, key=lambda t: t["F_loss"])
    rule = ("level 3 at cutoff 80: first rung with F_loss >= 0.99" if best["F_loss"] >= tracking.F_LOSS_MIN
            else "level 3 at cutoff 80: no rung reaches 0.99, best F_loss")
    Xc = best["X"]
    diag = tracking.diagnostics(M, Xc, T, N, psi0, path, Tg)
    M9 = tracking.make_model(job["gate"], D_L3_CHECK, g, 10 * g)
    S9, T9 = transfer(name, D_L3_CHECK)
    path9 = tracking.Path(pad(dense, D_L3_CHECK))
    d9 = tracking.diagnostics(M9, Xc, T, N, grape.joint(S9), path9, T9)
    F_loss9 = tracking.lindblad_F(M9, Xc, T, N, grape.joint(S9), T9, noise)
    res = dict(job=job, T_us=T, N=N, D=D_L3, guard_n=int(np.ceil(L3_TOP * D_L3)), path_L=path.L, w_star=best["w"],
               rule=rule, F_loss=best["F_loss"],
               F_loss_D90=F_loss9, check_D90=dict(C=d9["C"], F_T=d9["F_T"]),
               level3=[{k: v for k, v in t.items() if k != "X"} for t in tried],
               level2=dict(w_star=w0, F_loss_D60=r["F_loss"], F_T_D60=r["F_T"], C_D60=r["C"]),
               stages=r["stages"], ladder=r["ladder"], X=Xc.tolist(),
               path_F_loss=tracking.path_lindblad_F(path, D_L3, T, Tg, noise), **diag, wall_s=time.time() - t0)
    out.write_text(json.dumps(res))
    return jid, f"{time.time() - t0:.0f}s w*={best['w']:g} C={diag['C']:.3f} F_loss={best['F_loss']:.4f} (D90 {F_loss9:.4f})"


def safe_level3(jid):
    try:
        return level3(jid)
    except Exception:                                      # noqa: BLE001
        (JOBS80 / f"{jid}.err").write_text(traceback.format_exc())
        return jid, "ERROR (see .err)"


def safe_run(job):
    try:
        return run(job)
    except Exception:                                      # noqa: BLE001
        (JOBS / f"{job['id']}.err").write_text(traceback.format_exc())
        return job["id"], "ERROR (see .err)"


def main():
    global JOBS, SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--smoke", action="store_true", help="3 iterations per stage, 3 stages, 2 rungs, into results/smoke")
    ap.add_argument("--guard", type=float, default=2 / 3, help="level 3: guard threshold as a fraction of the cutoff")
    ap.add_argument("--only", default="", help="level 3: comma-separated job ids")
    a = ap.parse_args()
    jobs = job_list(a.which)
    if a.smoke:
        SMOKE, JOBS = True, HERE / "results" / "smoke"
        JOBS.mkdir(parents=True, exist_ok=True)
        pick = {}
        for j in jobs:                                     # one job per (name, ref, gate) at the shortest duration
            pick.setdefault((j["name"], j["ref"], j["gate"], j["set"]), j)
        for j in pick.values():
            (JOBS / f"{j['id']}.json").unlink(missing_ok=True)
            print(run(j), flush=True)
        print("smoke OK", flush=True)
        return
    if a.which == "level3":                              # after 'all': re-polish every chosen pulse at cutoff 80
        global L3_TOP
        L3_TOP = a.guard
        JOBS80.mkdir(parents=True, exist_ok=True)
        ids = a.only.split(",") if a.only else sorted(p.stem for p in JOBS.glob("*.json"))
        if a.only == "seeds":
            ids = sorted(p.stem for p in JOBS.glob("seed_*.json") if not (JOBS80 / p.name).exists())
        ids.sort(key=lambda i: -(float(i.split("_Tg")[1]) if "_Tg" in i else (2.58, 3.09, 6.18, 16.0)[int(i.split("_k")[1][0])]))
        print(f"level 3: {len(ids)} pulses, {a.workers} workers", flush=True)
        t0 = time.time()
        with mp.Pool(min(a.workers, len(ids))) as pool:
            for i, (jid, st) in enumerate(pool.imap_unordered(safe_level3, ids)):
                print(f"[{(time.time() - t0) / 60:6.1f} min] {i + 1}/{len(ids)} {jid} {st}", flush=True)
        return
    JOBS.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (JOBS / f"{j['id']}.json").exists()]
    todo.sort(key=lambda j: -NSL[j["Tg"]] * j["Tg"] * (1.4 if j["name"] == "flip" else 1.0)
              * (1.3 if j["ref"] == "photon" else 1.0))                 # long, many-slice jobs first
    print(f"{len(todo)} jobs, {a.workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for i, (jid, st) in enumerate(pool.imap_unordered(safe_run, todo)):
            print(f"[{(time.time() - t0) / 60:6.1f} min] {i + 1}/{len(todo)} {jid} {st}", flush=True)


if __name__ == "__main__":
    main()
