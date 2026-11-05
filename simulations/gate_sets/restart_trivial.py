"""Restart the trivial tracking runs (the pulse never moved the oscillator; SNAP only, mfcommon.trivial) with the same
protocol and the same random start, preceded by three target-only stages.

Why: for SNAP-displacement control the pulse with zero cavity drive is a local minimum of the tracking objective. A
displacement takes |0> to |1>, outside the span of the path and the target (both parity even), so every first-order
change of the drive leaves the closeness F_k and the target fidelity unchanged and the second-order change lowers them.
Random starts that fall into this basin end at the trivial pulse. Three target-only stages (tracking weight w = 0; final
weights 200, 2e4 and 1e5; 800, 800 and 1200 L-BFGS-B iterations, the schedule of manuscript/tools/capability.py) placed
before stage A move the pulse out of the basin (a single stage at final weight 1e4 does not); every later step is the
production protocol of simulations/tracking/compute.py unchanged: stages A-E, the weight ladder with the
Lindblad selection (largest w with F_T >= 0.99, else the best), the level-3 polish at cutoff 80, and the Lindblad traces
and saved states at cutoff 90.  Outputs go to manuscript_figures/sims/results/{jobs,jobs80,traces80,rho80}; nothing in
manuscript/ is written.
python simulations/gate_sets/restart_trivial.py --test | --all [--workers 34]"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))
import mfcommon as M  # noqa: E402,I001

from common import tracking  # noqa: E402
import compute as CP  # noqa: E402
import traces as TR  # noqa: E402

RES = HERE / "results"
CP.JOBS, CP.JOBS80 = RES / "jobs", RES / "jobs80"
TR.JOBS, TR.OUT, TR.RHO = RES / "jobs80", RES / "traces80", RES / "rho80"
# target-only pre-stages (tracking weight 0): the schedule of manuscript/tools/capability.py, which leaves the trivial
# basin (one stage at w_fin = 1e4 does not: tested on binomial k=2 seed 1, Oct 27)
WARM = [("W1", 1.0, 0.0, 0.0, 200.0, 0.99999, 800, False), ("W2", 1.0, 0.0, 0.0, 2e4, 0.99999, 800, False),
        ("W3", 1.0, 0.0, 0.0, 1e5, 0.99999, 1200, False)]       # (name, T_m/T, thr, w, w_fin, thr_fin, iterations, free)
tracking.SCHEDULE = WARM + list(tracking.SCHEDULE)


def todo():
    G = M.groups()
    jobs = []
    for (name, ref, gate, k), lst in sorted(G.items()):
        for r in lst:
            if M.trivial(r):
                jb = {kk: vv for kk, vv in r["job"].items() if kk not in ("m", "T_path_ns")}
                jb.update(set="seed", k=k, seed=r["seed"], restart_of=jb["id"])
                jb["id"] = f"restart_{name}_{ref}_{gate}_k{k}_s{r['seed']}"
                jobs.append(jb)
    return jobs


def one(job):
    t0 = time.time()
    a = CP.safe_run(job)
    b = CP.safe_level3(job["id"])
    c = TR.run(job["id"])
    return job["id"], f"{(time.time() - t0) / 60:.1f} min | {a[1]} | {b[1]} | {c[1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--name", default="", help="restart only the trivial runs of this transfer (e.g. binomial)")
    ap.add_argument("--workers", type=int, default=34)
    a = ap.parse_args()
    for d in (CP.JOBS, CP.JOBS80, TR.OUT, TR.RHO):
        d.mkdir(parents=True, exist_ok=True)
    jobs = todo()
    (RES / "restart_jobs.json").write_text(json.dumps(jobs, indent=1))
    if a.name:
        jobs = [j for j in jobs if j["name"] == a.name]
    print(f"{len(jobs)} trivial runs to restart", flush=True)
    if a.test:
        jobs = [j for j in jobs if j["name"] == "binomial" and j["k"] in (1, 2) and j["seed"] == 1]
        for j in jobs:                                   # a fresh test: remove the outputs of the earlier single-stage test
            for d in (CP.JOBS, CP.JOBS80, TR.OUT):
                (d / f"{j['id']}.json").unlink(missing_ok=True)
    t0 = time.time()
    with mp.Pool(min(a.workers, len(jobs))) as pool:
        for i, (jid, st) in enumerate(pool.imap_unordered(one, jobs)):
            print(f"[{(time.time() - t0) / 60:6.1f} min] {i + 1}/{len(jobs)} {jid} {st}", flush=True)
    print("end", flush=True)


if __name__ == "__main__":
    main()
