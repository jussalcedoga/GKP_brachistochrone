"""Statistics and physics checks of the stored pulses (result set A, Nov 1), from the files of pulses_extract.py.

Per start (every start included, no exclusion) and then mean +- sample s.d. (ddof = 1) over the starts of each case:
  controls  fraction of slices with the oscillator amplitude at its bound (a >= 0.999) or at zero (a <= 0.001), the same
            for the qubit drive (r = |Omega|/10g), time averages <a> and <r> (fractions of the bounds); for CS and ECD the
            common and sigma_z-conditional parts u_+- = (u_g +- u_e)/2 of the branch amplitudes (units of g), since
            sum_j |j><j| (x) G(u_j) = 1 (x) G(u_+) + sigma_z (x) G(u_-) for a generator G linear in u, the conditional
            fraction int|u_-|^2 dt / int(|u_+|^2 + |u_-|^2) dt, the fractions of slices with |u_-| >= 0.99 g (the
            sigma_z-conditional gate xi_e = -xi_g at the bound) and |u_+| >= 0.99 g; the total qubit rotation angle
            int |Omega| dt; phase jumps |Delta f| > pi/2 between consecutive slices in which both amplitudes exceed 0.1 of
            the bound; the overlap of qubit drive and conditional part (slices with r >= 0.5 and |u_-| >= 0.5 g, either
            alone, neither) and the Pearson correlation of r and |u_-| over the slices; SNAP: fraction of slices with
            |Omega| <= chi_d (r <= 0.1) and the largest cavity drive.
  closed    the closed evolution at cutoff 90 (GRAPE models of common/, exact slice unitaries, 8 points per slice) from
            |0>|g> (tracking) or from |0_L>|g>, |1_L>|g>, |+_L>|g> (flip; the code words of sims/flip_gate.code): the
            oscillator purity Tr rho_osc^2 (= qubit purity, the joint state is pure), the qubit excited population P_e and
            <n>, at the slice ends (the reported numbers) and on the fine grid (check of the minimum purity); final
            closed F_T (tracking) and F(0_L -> 1_L) (flip) as a check of the propagation against the stored values.
Writes manuscript_figures/sims/results/pulses/{pulses_stats_A.json, pulses_stats_A_rows.json, closed_<case>.npz};
pulses_facts.py then assembles pulses_facts_A.json.
  OMP_NUM_THREADS=1 python simulations/gate_sets/pulses_stats.py --workers 28
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

from common import tracking  # noqa: E402
from common.params import G_SQ  # noqa: E402

OUT = HERE / "results" / "pulses"
D = 90
SUB = 8
BOUND, ZERO = 0.999, 0.001


def ms(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)),
                min=float(v.min()), max=float(v.max()))


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def jumps(z, thr=0.1):
    """phase jumps |Delta arg z| > pi/2 between consecutive slices with |z| > thr on both"""
    ok = (np.abs(z[1:]) > thr) & (np.abs(z[:-1]) > thr)
    d = np.abs(wrap(np.angle(z[1:]) - np.angle(z[:-1])))
    return int(np.sum(ok & (d > np.pi / 2)))


def control_stats(gate, X, dt_us):
    r, fO = X[:, 0], X[:, 1]
    Om = r * np.exp(1j * fO)                             # units of the bound 10 g
    s = dict(drive_at_bound=float(np.mean(r >= BOUND)), drive_at_zero=float(np.mean(r <= ZERO)), drive_mean=float(r.mean()),
             rotation_angle_over_pi=float(np.sum(10 * G_SQ * r * dt_us) / math.pi), jumps_Omega=jumps(Om))
    if gate == "snap":
        a = X[:, 2]
        eps = a * np.exp(1j * X[:, 3])
        s.update(osc_at_bound=float(np.mean(a >= BOUND)), osc_at_zero=float(np.mean(a <= ZERO)), osc_mean=float(a.mean()),
                 osc_max=float(a.max()), jumps_eps=jumps(eps), drive_le_chi=float(np.mean(r <= 0.1)))
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
             jumps_um=jumps(um), jumps_up=jumps(up),
             overlap_both=float(np.mean(on_r & on_c)), overlap_drive_only=float(np.mean(on_r & ~on_c)),
             overlap_cond_only=float(np.mean(~on_r & on_c)), overlap_neither=float(np.mean(~on_r & ~on_c)),
             corr_r_um=float(rc), drive_le_chi=float(np.mean(r <= 0.1)))
    return s


def closed_traces(gate, X, T_us, psi0s):
    """joint states at t = j T/(SUB N), j = 0..SUB N, for the columns of psi0s (2D, m)"""
    model = tracking.make_model(gate, D, G_SQ, 10 * G_SQ)
    C, _ = model.controls(np.asarray(X, float))
    H = model.hamiltonians(C)
    lam, V = np.linalg.eigh(H)
    h = T_us / len(X) / SUB
    P = psi0s.copy()
    out = [P.copy()]
    for k in range(len(X)):
        U = (V[k] * np.exp(-1j * lam[k] * h)) @ V[k].conj().T
        for _ in range(SUB):
            P = U @ P
            out.append(P.copy())
    return np.array(out)                                 # (SUB N + 1, 2D, m)


def diag(traj):
    """purity, P_e, nbar along traj (t, 2D, m) -> arrays (m, t)"""
    V = traj.reshape(traj.shape[0], 2, D, traj.shape[2])
    rq = np.einsum("tinm,tjnm->tmij", V, V.conj())       # reduced qubit state = Gram matrix of the branches
    pur = np.sum(np.abs(rq) ** 2, axis=(2, 3)).T
    pe = np.real(rq[:, :, 1, 1]).T
    nb = np.einsum("tinm,n->mt", np.abs(V) ** 2, np.arange(D, dtype=float))
    return pur, pe, nb


def osc_fid(psi, target):
    v = psi.reshape(2, D)
    return float(np.sum(np.abs(v.conj() @ target) ** 2))


def run(task):
    fname, i = task
    z = np.load(OUT / fname)
    gate = fname.split("_")[2]
    X = z["X"][i]
    N, T_ns = int(z["N"]), float(z["T_ns"])
    T_us = T_ns / 1e3
    res = dict(file=fname, i=i, seed=int(z["seed"][i]), rep=bool(z["representative"][i]),
               controls=control_stats(gate, X, T_us / N))
    if fname.startswith("tracking"):
        import compute as CP
        S, Tg = CP.transfer("gkp", D)
        from common import grape
        psi0 = grape.joint(S)[:, None]
        traj = closed_traces(gate, X, T_us, psi0)
        res["F_T_closed_check"] = abs(osc_fid(traj[-1, :, 0], Tg) - float(z["F_T_closed"][i]))
        res["trivial"] = bool(z["trivial"][i])
        inputs = ("0",)
    else:
        import flip_gate as FG
        k0, k1, e1, c, theta, inp = FG.code(D)
        from common import grape
        inputs = ("0", "1", "+")
        psi0 = np.stack([grape.joint(inp[nm].astype(complex)) for nm in inputs], axis=1)
        traj = closed_traces(gate, X, T_us, psi0)
        res["F_0to1_closed"] = osc_fid(traj[-1, :, 0], k1.astype(complex))
        res["F_1to0_closed"] = osc_fid(traj[-1, :, 1], k0.astype(complex))
    pur, pe, nb = diag(traj)
    ends = slice(0, None, SUB)
    res["closed"] = {}
    for m, nm in enumerate(inputs):
        p_end = pur[m, ends]
        res["closed"][nm] = dict(purity_min=float(p_end.min()), purity_min_fine=float(pur[m].min()),
                                 t_purity_min_over_T=float(np.argmin(p_end) / N), purity_final=float(p_end[-1]),
                                 purity_mean=float(p_end.mean()), Pe_max=float(pe[m, ends].max()),
                                 Pe_mean=float(pe[m, ends].mean()), Pe_final=float(pe[m, -1]),
                                 nbar_max=float(nb[m, ends].max()), nbar_final=float(nb[m, -1]),
                                 nbar_mean=float(nb[m, ends].mean()))
    res["_traces"] = dict(purity=pur.astype(np.float32), Pe=pe.astype(np.float32), nbar=nb.astype(np.float32))
    return res


def aggregate(rows):
    """per case: mean +- s.d. over all starts of every per-start number, and the representative's values"""
    cases = {}
    for r in rows:
        cases.setdefault(r["file"], []).append(r)
    out = {}
    for fname, lst in cases.items():
        lst.sort(key=lambda r: r["i"])
        z = np.load(OUT / fname)
        e = dict(file=fname, T_ns=float(z["T_ns"]), N=int(z["N"]), n_starts=len(lst),
                 seeds=[r["seed"] for r in lst], rep_seed=next(r["seed"] for r in lst if r["rep"]))
        e["controls"] = {k: ms([r["controls"][k] for r in lst]) for k in lst[0]["controls"]
                         if not (isinstance(lst[0]["controls"][k], float) and math.isnan(lst[0]["controls"][k]))}
        if "corr_r_um" in lst[0]["controls"]:
            cv = [r["controls"]["corr_r_um"] for r in lst if not math.isnan(r["controls"]["corr_r_um"])]
            e["controls"]["corr_r_um"] = ms(cv) if cv else None
        e["controls_rep"] = next(r["controls"] for r in lst if r["rep"])
        e["closed"] = {nm: {k: ms([r["closed"][nm][k] for r in lst]) for k in lst[0]["closed"][nm]}
                       for nm in lst[0]["closed"]}
        e["closed_rep"] = next(r["closed"] for r in lst if r["rep"])
        if fname.startswith("tracking"):
            e["max_F_T_closed_check"] = float(max(r["F_T_closed_check"] for r in lst))
            triv = [r for r in lst if r["trivial"]]
            e["n_trivial"] = len(triv)
            if triv:
                e["trivial_osc_max"] = float(max(r["controls"]["osc_max"] for r in triv))
                e["trivial_drive_mean"] = ms([r["controls"]["drive_mean"] for r in triv])
                nt = [r for r in lst if not r["trivial"]]
                e["nontrivial_controls"] = {k: ms([r["controls"][k] for r in nt]) for k in
                                            ("osc_at_bound", "osc_mean", "drive_at_bound", "drive_mean", "drive_le_chi",
                                             "rotation_angle_over_pi")}
                e["nontrivial_closed"] = {k: ms([r["closed"]["0"][k] for r in nt]) for k in
                                          ("purity_min", "purity_final", "nbar_max", "Pe_max")}
        else:
            gate = fname.split("_")[2]
            e["F_0to1_closed"] = ms([r["F_0to1_closed"] for r in lst])
            e["F_1to0_closed"] = ms([r["F_1to0_closed"] for r in lst])
            for key, lab in (("F_0to1", "F_0to1_closed"), ("F_1to0", "F_1to0_closed")):
                st = _stored_closed(gate, z, key)
                e[f"max_dev_{lab}_vs_stored"] = float(np.max(np.abs(np.array([r[lab] for r in lst]) - st)))
        # traces for the figures: (starts, inputs, SUB N + 1)
        tr = {k: np.stack([r["_traces"][k] for r in lst]) for k in ("purity", "Pe", "nbar")}
        np.savez_compressed(OUT / f"closed_{fname}", t_ns=float(z["T_ns"]) * np.arange(SUB * int(z["N"]) + 1) /
                            (SUB * int(z["N"])), sub=SUB, inputs=np.array(list(lst[0]["closed"])), seed=np.array(
                                e["seeds"]), representative=np.array([r["rep"] for r in lst]), **tr)
        out[fname[:-4]] = e
    return out


_STORED = {}


def _stored_closed(gate, z, key):
    """stored closed F_0to1 / F_1to0 of the evaluation in flip_gate.jsonl for the starts of the extracted file z"""
    W = None if np.isnan(float(z["W"])) else float(z["W"])
    if not _STORED:
        for line in open(HERE / "results" / "flip_gate.jsonl"):
            r = json.loads(line)
            if r["mode"] == "target":
                _STORED[(r["gate"], round(r["T_ns"]), None, r["seed"])] = r["eval"]["closed"]
            else:
                for rg in r["rungs"]:
                    _STORED[(r["gate"], round(r["T_ns"]), float(rg["w"]), r["seed"])] = rg["eval"]["closed"]
    return np.array([_STORED[(gate, round(float(z["T_ns"])), W, int(s))][key] for s in z["seed"]])


def main():
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 28
    index = json.loads((OUT / "pulses_index_A.json").read_text())
    tasks = []
    for c in index["cases"]:
        for i in range(c["n_starts"]):
            tasks.append((c["file"], i))
    tasks.sort(key=lambda t: (not t[0].startswith("flip"), -int(t[0].split("_T")[1].split("_")[0].split(".")[0])))
    print(f"{len(tasks)} pulses, {W} workers", flush=True)
    t0 = time.time()
    rows = []
    with mp.Pool(W) as pool:
        for n, r in enumerate(pool.imap_unordered(run, tasks)):
            rows.append(r)
            if (n + 1) % 20 == 0:
                print(f"[{(time.time() - t0) / 60:5.1f} min] {n + 1}/{len(tasks)}", flush=True)
    out = {}
    by_file = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append(r)
    for fname, lst in by_file.items():
        out.update(aggregate(lst))
    for r in rows:
        r.pop("_traces", None)
    (OUT / "pulses_stats_A.json").write_text(json.dumps(dict(description=__doc__, cases=out), indent=1))
    (OUT / "pulses_stats_A_rows.json").write_text(json.dumps(rows))
    print("wrote", OUT / "pulses_stats_A.json", flush=True)


if __name__ == "__main__":
    main()
