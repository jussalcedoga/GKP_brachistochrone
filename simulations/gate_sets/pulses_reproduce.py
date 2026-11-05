"""Reproduction of the stored pulses (result set A, Nov 1): every pulse extracted by pulses_extract.py is re-simulated
with the repository's own evaluation routines and compared with the stored numbers.

  tracking  every Fig. 4 GKP run (3 gate sets x 412, 824, 2133 ns x 20 starts): manuscript/fig4_cs_path_following/
            traces.run (the routine that wrote traces80; cutoff 90, closed evolution 8 points per slice, Lindblad
            evolution of src/common/lindblad.py with the Yale rates, 8 points per slice, sub-steps <= 2 ns), redirected to
            a scratch folder, then compared field by field with results/traces80/<id>.json: F_T_final, C (closed),
            F_T_final_noisy, C_noisy (Lindblad), F_T_final_noisy_CS, the exposures and every stored time trace.
  flip      every pulse of sims/results/flip_gate.jsonl (mode target, and the rungs W = 10 and 70 of mode track):
            flip_gate.evaluate
            (the routine that wrote the stored eval; cutoff 90, independent integrator of analytics_audit/review_oct28,
            closed and Lindblad gate metrics), compared key by key; in addition the closed gate metric is recomputed
            with a third propagator, the GRAPE models of common/ (eigen-decomposition per slice), as a convention check.
Writes manuscript_figures/sims/results/pulses/reproduce_A.jsonl (one line per pulse) and reproduce_A.json (maxima per
case).  Retraced files go to <scratch>/retrace (not part of the results).
  OMP_NUM_THREADS=1 python simulations/gate_sets/pulses_reproduce.py --workers 28
  [--scratch DIR] [--only tracking|flip] [--summary]
"""
from __future__ import annotations
import layout as _L  # repository layout: data/ and output/ (src/layout.py)

import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

HERE = _L.vfile(__file__).parent
MF = HERE.parent
sys.path.insert(0, str(MF))
sys.path.insert(0, str(HERE))
import mfcommon as M  # noqa: E402,I001

import numpy as np  # noqa: E402

OUT = HERE / "results" / "pulses"
LINES = OUT / "reproduce_A.jsonl"
SUMMARY = OUT / "reproduce_A.json"
JOBS80 = M.MS / "fig4_cs_path_following" / "results" / "jobs80"
TRACES80 = M.MS / "fig4_cs_path_following" / "results" / "traces80"
FLIP = HERE / "results" / "flip_gate.jsonl"
SCRATCH = pathlib.Path(sys.argv[sys.argv.index("--scratch") + 1]) if "--scratch" in sys.argv else OUT / "_retrace"
SCALARS = ("F_T_final", "C", "F_T_final_noisy", "C_noisy", "F_T_final_noisy_CS", "expo_L_ns", "expo_ns")
ARRAYS = ("F", "F_T", "x", "z", "nbar", "FL", "FTL", "xL", "zL", "nbarL", "wL")
EVKEYS = ("avg6", "avg6_opt", "F_0to1", "F_1to0")


def tracking_tasks():
    G = M.groups()
    out = []
    for gate in ("cs", "cd", "snap"):
        for k in (1, 2, 3):
            for r in G[("gkp", "geodesic", gate, k)]:
                out.append(("tracking", r["job"]["id"], gate, k, r["seed"], r["T_ns"]))
    return out


def flip_tasks():
    out = []
    for i, line in enumerate(open(FLIP)):
        r = json.loads(line)
        if r["mode"] == "target":
            out.append(("flip", i, r["gate"], "target", r["seed"], r["T_ns"], None))
        else:
            for j, rg in enumerate(r["rungs"]):
                if float(rg["w"]) in (10.0, 70.0):        # the rungs extracted by pulses_extract.py
                    out.append(("flip", i, r["gate"], "track", r["seed"], r["T_ns"], j))
    return out


def run_tracking(task):
    _, jid, gate, k, seed, T_ns = task
    import traces as TR                                  # simulations/tracking/traces.py
    TR.JOBS, TR.OUT, TR.RHO, TR.FORCE = JOBS80, SCRATCH / "traces80", SCRATCH / "rho80", True
    TR.OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    TR.run(jid)
    new = json.loads((TR.OUT / f"{jid}.json").read_text())
    old = json.loads((TRACES80 / f"{jid}.json").read_text())
    dev = {}
    for key in SCALARS:
        if key in old:
            dev[key] = abs(float(new[key]) - float(old[key]))
    for key in ARRAYS:
        a, b = np.asarray(new[key], float), np.asarray(old[key], float)
        dev[key] = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
    return dict(kind="tracking", id=jid, gate=gate, k=k, seed=seed, T_ns=T_ns, dev=dev,
                stored=dict(F_T_closed=old["F_T_final"], C_closed=old["C"], F_T_noisy=old["F_T_final_noisy"],
                            C_noisy=old["C_noisy"]),
                new=dict(F_T_closed=new["F_T_final"], C_closed=new["C"], F_T_noisy=new["F_T_final_noisy"],
                         C_noisy=new["C_noisy"]), wall_s=time.time() - t0)


def closed_metric_grape(X, gate, T):
    """closed gate metric with the GRAPE model of common/ (cutoff 90) and the code words of the stored evaluation"""
    import flip_gate as FG
    from common import tracking
    from common.params import G_SQ
    RB = FG.RB
    B, _ = RB.code()
    model = tracking.make_model(gate, RB.D, G_SQ, 10 * G_SQ)
    C, _ = model.controls(np.asarray(X, float))
    D = RB.D
    A = np.zeros((2, 2, 2), complex)
    for i in range(2):
        traj, _, _ = model.propagate(C, T / len(X), RB.joint(B[i]))
        psi = traj[-1]
        for q in range(2):
            for kk in range(2):
                A[q, kk, i] = np.vdot(B[kk], psi[q * D:(q + 1) * D])
    return RB.gate_metrics(np.einsum("qki,qlj->ijkl", A, A.conj()))


def run_flip(task):
    _, i, gate, mode, seed, T_ns, j = task
    import flip_gate as FG                               # simulations/gate_sets/flip_gate.py (evaluate only)
    t0 = time.time()
    with open(FLIP) as f:
        for n, line in enumerate(f):
            if n == i:
                r = json.loads(line)
                break
    src = r if mode == "target" else r["rungs"][j]
    X, T = np.array(src["X"]), r["T_ns"] / 1e3
    ev_old = src["eval"]
    ev = FG.evaluate(X, gate, T)
    dev = {}
    for part in ("closed", "lind"):
        for key in EVKEYS:
            dev[f"{part}.{key}"] = abs(ev[part][key] - ev_old[part][key])
        d = ev[part]["phi_opt"] - ev_old[part]["phi_opt"]
        dev[f"{part}.phi_opt"] = abs((d + math.pi) % (2 * math.pi) - math.pi)
    dev["nbar_max"] = abs(ev["nbar_max"] - ev_old["nbar_max"])
    gm = closed_metric_grape(X, gate, T)
    dev["grape_closed.avg6_opt"] = abs(gm["avg6_opt"] - ev_old["closed"]["avg6_opt"])
    dev["grape_closed.avg6"] = abs(gm["avg6"] - ev_old["closed"]["avg6"])
    return dict(kind="flip", line=i, gate=gate, mode=mode, seed=seed, T_ns=T_ns,
                W=(None if mode == "target" else float(src["w"])), dev=dev,
                stored=dict(avg6_opt_lind=ev_old["lind"]["avg6_opt"], avg6_opt_closed=ev_old["closed"]["avg6_opt"]),
                new=dict(avg6_opt_lind=ev["lind"]["avg6_opt"], avg6_opt_closed=ev["closed"]["avg6_opt"],
                         grape_avg6_opt_closed=gm["avg6_opt"]), wall_s=time.time() - t0)


def worker(task):
    try:
        return run_tracking(task) if task[0] == "tracking" else run_flip(task)
    except Exception as e:                                # noqa: BLE001
        import traceback
        return dict(kind=task[0], task=list(task), error=repr(e), tb=traceback.format_exc())


def key_of(task):
    return json.dumps(list(task))


def summarize():
    rows = [json.loads(line) for line in open(LINES)]
    errs = [r for r in rows if "error" in r]
    cases = {}
    for r in rows:
        if "error" in r:
            continue
        if r["kind"] == "tracking":
            c = f"tracking_gkp_{r['gate']}_T{round(r['T_ns'])}"
        else:
            c = f"flip_{r['mode']}_{r['gate']}_T{round(r['T_ns'])}" + (f"_W{r['W']:g}" if r["W"] is not None else "")
        e = cases.setdefault(c, dict(n=0, max_dev={}))
        e["n"] += 1
        for k, v in r["dev"].items():
            e["max_dev"][k] = max(e["max_dev"].get(k, 0.0), v)
    head = {}
    for c, e in cases.items():
        md = e["max_dev"]
        if c.startswith("tracking"):
            e["max_dev_FT_C"] = max(md[k] for k in ("F_T_final", "C", "F_T_final_noisy", "C_noisy"))
            e["max_dev_traces"] = max(md[k] for k in ARRAYS)
        else:
            e["max_dev_metric"] = max(v for k, v in md.items() if k.split(".")[-1] in EVKEYS
                                      and not k.startswith("grape"))
            e["max_dev_grape_closed"] = max(md["grape_closed.avg6_opt"], md["grape_closed.avg6"])
        head[c] = e
    allmax_tr = max((e["max_dev_FT_C"] for c, e in head.items() if c.startswith("tracking")), default=None)
    allmax_fl = max((e["max_dev_metric"] for c, e in head.items() if c.startswith("flip")), default=None)
    SUMMARY.write_text(json.dumps(dict(n_rows=len(rows), n_errors=len(errs), errors=errs[:5],
                                       max_dev_tracking_FT_C=allmax_tr, max_dev_flip_metric=allmax_fl,
                                       threshold=1e-5, cases=head), indent=1))
    print(f"{len(rows)} rows, {len(errs)} errors; max |dev| tracking F_T/C {allmax_tr}, flip metric {allmax_fl}")


def main():
    if "--summary" in sys.argv:
        summarize()
        return
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 28
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else ""
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tasks = ([] if only == "flip" else tracking_tasks()) + ([] if only == "tracking" else flip_tasks())
    done = set()
    if LINES.exists():
        for line in open(LINES):
            r = json.loads(line)
            if "error" not in r:
                done.add(r.get("_key"))
    tasks = [t for t in tasks if key_of(t) not in done]
    # longest first: flip at 824 ns (Lindblad on the vectorized generator), then flip 412, then tracking 2133 ...
    tasks.sort(key=lambda t: (t[0] != "flip", -t[5]))
    print(f"{len(tasks)} tasks, {W} workers", flush=True)
    t0 = time.time()
    with mp.Pool(W) as pool, open(LINES, "a") as f:
        for n, (t, res) in enumerate(zip(tasks, pool.imap(worker, tasks))):
            res["_key"] = key_of(t)
            f.write(json.dumps(res) + "\n")
            f.flush()
            dv = res.get("dev", {})
            print(f"[{(time.time() - t0) / 60:6.1f} min] {n + 1}/{len(tasks)} {t[:6]} "
                  f"max dev {max(dv.values()) if dv else res.get('error')}", flush=True)
    summarize()


if __name__ == "__main__":
    main()
