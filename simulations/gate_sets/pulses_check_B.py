"""Independent checks of the data-B results (target-only pulses of Table I and the smoothing test) (result set B, Nov 1).

1  completeness   every row of pulses_smooth.jsonl: one row per pulse, sigma and edge, no duplicates; the starts of each
                  case equal data-A's independent extraction (tracking_gkp_*.npz and flip_target_*.npz, built from
                  mfcommon.groups and flip_gate.jsonl) and the slice variables X used here equal data-A's X; the target
                  starts are seeds 1000-1007 of every gate set
2  bounds         over every row: max |Omega|/10g and max |branch amplitude|/g (SNAP: the cavity drive), the smallest
                  kernel weight, the largest kernel row sum, the polar round trip
3  statistics     the mean and sample s.d. of every change Delta (smoothed minus sigma = 0, same start) recomputed from the
                  rows and compared with pulses_facts_B.json
4  reproduction   capability_pulses.jsonl against capability_noisy.jsonl per start, and the Table I entries of the
                  CURRENT main.tex of the paper re-formatted from the 8 starts
5  spot checks    smoothed pulses rebuilt independently: Cartesian controls from the formulas of Letter Eq. (7)
                  (Omega = 10 g r e^{i f_Omega}, branch amplitude g a_j e^{i f_j}), the exact pointwise convolution of the
                  step pulse with the Gaussian (normal CDF at each point, zero drive outside [0, T] or end values held),
                  averaged over each sub-slice by 8-point Gauss-Legendre quadrature (pulses_smooth.py uses the
                  integrated CDF in closed form instead), and evaluated with the independent integrator of
                  analytics_audit/review_oct28/lindblad_crosscheck.py (Hamiltonians rebuilt from the formulas, exact
                  expm_multiply of the vectorized Lindblad generator per sub-slice; SNAP at chi_d = 4 g by adding
                  -3 g n |e><e|), closeness on its own geodesic; flip pulses with flip_gate.evaluate on the rebuilt
                  pulse.  Compared with the rows of pulses_smooth.jsonl.
Writes sims/results/pulses/pulses_check_B.json
  OMP_NUM_THREADS=1 python simulations/gate_sets/pulses_check_B.py --workers 16
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import math
import multiprocessing as mp
import pathlib
import re
import sys
import time
from collections import defaultdict

HERE = _L.vfile(__file__).parent
MF = HERE.parent
ROOT = MF.parent
MS = ROOT / "manuscript"
for p in (MF / "analytics_audit" / "review_oct28", MS / "fig4_cs_path_following", MS, HERE):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.special import ndtr  # noqa: E402
from scipy.sparse.linalg import expm_multiply  # noqa: E402

import lindblad_crosscheck as LC  # noqa: E402

P = MF / "sims" / "results" / "pulses"
RES = MF / "sims" / "results"
JOBS80 = MS / "fig4_cs_path_following" / "results" / "jobs80"
G = LC.G
D = LC.D
SUBL = 8


def jl(p):
    return [json.loads(l) for l in open(p)]


def ms(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else 0.0)


# ----------------------------------------------------------------------------- 1-3 over every row
def rows_checks():
    rows = jl(P / "pulses_smooth.jsonl")
    out = dict(n_rows=len(rows))
    keys = [(r["set"], r["gate"], round(r["T_ns"]), r["seed"], r["sigma_ns"], r["edge"]) for r in rows]
    out["n_unique_keys"] = len(set(keys))
    per = defaultdict(lambda: defaultdict(set))
    for (st, g, T, s, sg, e) in keys:
        per[f"{st}_{g}_{T}"][f"{sg:g}_{e}"].add(s)
    out["cases"] = {}
    for c, d in sorted(per.items()):
        seeds = sorted(set().union(*d.values()))
        out["cases"][c] = dict(n_starts=len(seeds), variants=sorted(d),
                               every_variant_has_every_start=all(v == set(seeds) for v in d.values()))
    # starts and X against data-A's extraction
    cmpA = {}
    xdev = 0.0
    for st, fn in (("track", "tracking_gkp_{g}_T{T}.npz"), ("flip", "flip_target_{g}_T{T}.npz")):
        for g in ("cs", "cd", "snap"):
            for T in ((412, 824) if st == "track" else (824,)):
                z = np.load(P / fn.format(g=g, T=T), allow_pickle=True)
                seedsA = sorted(int(s) for s in z["seed"])
                mine = {r["seed"]: r for r in rows if r["set"] == st and r["gate"] == g and round(r["T_ns"]) == T}
                ok = sorted(mine) == seedsA
                for i, s in enumerate(z["seed"]):
                    r = mine.get(int(s))
                    if r is None:
                        continue
                    X = load_X(r)[0]
                    xdev = max(xdev, float(np.abs(X - z["X"][i]).max()))
                extra = {}
                if st == "track":
                    trivA = sorted(int(s) for s, t in zip(z["seed"], z["trivial"]) if t)
                    trivB = sorted({r["seed"] for r in mine.values() if r.get("trivial")})
                    extra = dict(trivial_A=trivA, trivial_B=trivB, trivial_equal=trivA == trivB)
                cmpA[f"{st}_{g}_{T}"] = dict(n_A=len(seedsA), n_B=len(mine), seeds_equal=ok, **extra)
    out["vs_dataA"] = dict(cases=cmpA, max_abs_dX=xdev)
    tg = sorted({(r["gate"], r["seed"]) for r in rows if r["set"] == "target"})
    out["target_seeds_ok"] = tg == sorted((g, s) for g in ("cs", "cd", "snap", "snap4") for s in range(1000, 1008))
    # bounds
    b = dict(Omega=max(r["checks"]["bound_Omega"] for r in rows), osc=max(r["checks"]["bound_osc"] for r in rows),
             w_min=min(r["checks"]["w_min"] for r in rows), rowsum_max=max(r["checks"]["w_rowsum_max"] for r in rows),
             polar_roundtrip_max=max(r["checks"]["polar_roundtrip"] for r in rows))
    b["Omega_minus_1"], b["osc_minus_1"] = b["Omega"] - 1.0, b["osc"] - 1.0
    out["bounds"] = b
    # statistics against the facts
    F = json.loads((P / "pulses_facts_B.json").read_text())["smoothing"]["groups"]
    by = defaultdict(dict)
    for r in rows:
        by[(r["set"], r["gate"], round(r["T_ns"]))][(r["seed"], r["sigma_ns"], r["edge"])] = r

    def met(r):
        if r["set"] == "flip":
            return dict(avg6_opt_noisy=r["eval"]["lind"]["avg6_opt"], avg6_opt_closed=r["eval"]["closed"]["avg6_opt"])
        m = dict(F_T_noisy=r["F_T_noisy"], F_T_closed=r["F_T_closed"])
        if r["set"] == "track":
            m.update(C_noisy=r["C_noisy"], C_closed=r["C_closed"])
        return m
    dev, n = 0.0, 0
    table = {}
    for (st, g, T), d in by.items():
        seeds = sorted({s for s, _, _ in d})
        for e in ("zero", "hold"):
            name = f"{st}_{g}_{T}" + ("" if e == "zero" else "_hold")
            for sg in (1.0, 2.0, 4.0):
                m0 = {s: met(d[(s, 0.0, "zero")]) for s in seeds}
                m1 = {s: met(d[(s, sg, e)]) for s in seeds}
                for k in m0[seeds[0]]:
                    v = ms([m1[s][k] - m0[s][k] for s in seeds])
                    f = F[name]["sigma"][f"{sg:g}"]["d_" + k]
                    dev = max(dev, abs(v["mean"] - f["mean"]), abs(v["sd"] - f["sd"]))
                    n += 1
                    table[f"{name}|{sg:g}|{k}"] = v
    out["stats_vs_facts"] = dict(n_compared=n, max_abs_dev=dev)
    out["delta_table"] = table
    return out


# ----------------------------------------------------------------------------- 4 reproduction of Table I
def reproduction():
    new = {(r["gate"], r["seed"]): r for r in jl(P / "capability_pulses.jsonl")}
    old = {(r["gate"], r["seed"]): r for r in jl(RES / "capability_noisy.jsonl") if abs(r["T_ns"] - 412) < 1}
    out = dict(n_new=len(new), n_old_412=len(old), same_keys=sorted(new) == sorted(old))
    keys = ("F60", "F90_closed", "F90_noisy")
    out["max_abs_diff"] = {k: max(abs(new[c][k] - old[c][k]) for c in new) for k in keys}
    out["n_bitwise_equal"] = int(sum(all(new[c][k] == old[c][k] for k in keys) for c in new))
    # C stored in the jsonl against the formulas of Eq. (7)
    cdev = 0.0
    for r in new.values():
        cdev = max(cdev, float(np.abs(cart(np.array(r["X"])) - np.array(r["C"])).max()))
    out["C_vs_formula_max_abs"] = cdev
    tex = _L.paper_text("supplement.tex")   # Supplementary Table 15 (Nov 5)
    blk = tex[tex.index(r"\label{tab:capability}"):]
    blk = blk[:blk.index(r"\end{tabular}")]
    out["table"] = {}
    for lab, gk in (("CS &", "cs"), ("CD &", "cd"), (r"SNAP, $\chi_d=g$ &", "snap"), (r"SNAP, $\chi_d=4g$ &", "snap4")):
        line = next(l for l in blk.splitlines() if l.strip().startswith(lab))
        m = re.search(r"&\s*([0-9.]+)\((\d+)\)\s*\[([0-9.]+)\]", line)
        v = np.array([r["F90_noisy"] for (g, s), r in new.items() if g == gk])
        dec = len(m.group(1).split(".")[1])
        fm = f"{v.mean():.{dec}f}({int(round(v.std(ddof=1) * 10 ** dec))}) [{v.max():.3f}]"
        q = f"{m.group(1)}({m.group(2)}) [{m.group(3)}]"
        out["table"][gk] = dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)), best=float(v.max()),
                                formatted=fm, quoted=q, equal=fm == q, line=line.strip())
    return out


# ----------------------------------------------------------------------------- 5 spot checks
def cart(X):
    r, fO, ag, fg, ae, fe = np.asarray(X, float).T
    return np.stack([10 * G * r * np.cos(fO), 10 * G * r * np.sin(fO), G * ag * np.cos(fg), G * ag * np.sin(fg),
                     G * ae * np.cos(fe), G * ae * np.sin(fe)], 1)


def smooth_indep(C, T, M, sig_us, edge):
    """cell averages over N M sub-slices of the Gaussian-smoothed step pulse, pointwise exact (normal CDF), 8-point
    Gauss-Legendre per sub-slice"""
    N = len(C)
    h = T / (N * M)
    xg, wg = np.polynomial.legendre.leggauss(8)
    t = (np.arange(N * M)[:, None] + 0.5 * (xg[None, :] + 1)) * h          # (NM, 8)
    tk = np.arange(N + 1) * T / N
    if sig_us == 0:
        idx = np.minimum((t / (T / N)).astype(int), N - 1)
        vals = C[idx]
    else:
        Phi = ndtr((t[..., None] - tk[None, None, :]) / sig_us)            # (NM, 8, N+1)
        Wt = Phi[..., :-1] - Phi[..., 1:]                                   # weight of slice k at time t
        if edge == "hold":
            Wt[..., 0] += ndtr(-t / sig_us)
            Wt[..., -1] += ndtr((t - T) / sig_us)
        vals = np.einsum("aqk,kj->aqj", Wt, C)
    return 0.5 * np.einsum("q,aqj->aj", wg, vals)


def polar(Cf):
    X = np.zeros((len(Cf), 6))
    for j, b in ((0, 10 * G), (2, G), (4, G)):
        z = Cf[:, j] + 1j * Cf[:, j + 1]
        X[:, j], X[:, j + 1] = np.abs(z) / b, np.angle(z)
    return X


def ham(gate, x, chi_over_g):
    H = LC.hamiltonian(gate, x)
    if gate == "snap" and chi_over_g != 1.0:
        H = (H - (chi_over_g - 1.0) * G * sp.kron(LC.csr(LC.P_Q[1]), LC.csr(LC.N_OSC))).tocsr()
    return H


def load_X(r):
    if r["set"] == "track":
        j = json.loads((JOBS80 / f"{r['jid']}.json").read_text())
        return np.array(j["X"]), float(j["T_us"])
    if r["set"] == "target":
        for c in jl(P / "capability_pulses.jsonl"):
            if c["gate"] == r["gate"] and c["seed"] == r["seed"]:
                return np.array(c["X"]), 1e-3 * c["T_ns"]
    for c in jl(RES / "flip_gate.jsonl"):
        if c["mode"] == "target" and c["gate"] == r["gate"] and c["seed"] == r["seed"] and abs(c["T_ns"] - 824) < 1:
            return np.array(c["X"]), 1e-3 * c["T_ns"]
    raise KeyError(r["seed"])


def spot(task):
    key, r = task
    t0 = time.time()
    X, T = load_X(r)
    N, M = len(X), r["M"]
    gate = "snap" if r["gate"] == "snap4" else r["gate"]
    chi = 4.0 if r["gate"] == "snap4" else 1.0
    C = cart(X)
    Cs = smooth_indep(C, T, M, 1e-3 * r["sigma_ns"], r["edge"])
    Xf = polar(Cs)
    om = float(np.abs(Cs[:, 0] + 1j * Cs[:, 1]).max() / (10 * G))
    br = float(np.abs(Cs[:, 2] + 1j * Cs[:, 3]).max() / G)
    if gate in ("cs", "cd"):
        br = max(br, float(np.abs(Cs[:, 4] + 1j * Cs[:, 5]).max() / G))
    out = dict(key=key, bound_Omega=om, bound_osc=br)
    if r["set"] == "flip":
        import flip_gate as FG
        ev = FG.evaluate(Xf, gate, T)
        out.update(avg6_opt_noisy=ev["lind"]["avg6_opt"], avg6_opt_closed=ev["closed"]["avg6_opt"],
                   row_avg6_opt_noisy=r["eval"]["lind"]["avg6_opt"], row_avg6_opt_closed=r["eval"]["closed"]["avg6_opt"])
    else:
        h = T / (N * M)
        S, Tt = LC.transfer("gkp", D)
        psi = np.r_[S, np.zeros(D, complex)]
        Sp, chiv, th = LC.geodesic("gkp")
        rec = M // SUBL
        LDm = LC.vec_dissipator()
        V = np.outer(psi, psi.conj()).ravel()
        Fc, Fn = [], []

        def pstate(j):
            s = j / (N * SUBL)
            return math.cos(s * th) * Sp + math.sin(s * th) * chiv

        def fo(ro, v):
            return float(np.real(np.vdot(v, ro @ v)))
        ro0 = np.outer(S, S.conj())
        Fc.append(fo(ro0, pstate(0)))
        Fn.append(Fc[0])
        for k, x in enumerate(Xf):
            H = ham(gate, x, chi)
            psi = expm_multiply((-1j * h) * H, psi)
            V = expm_multiply(((LC.vec_hamiltonian(H) + LDm) * h).tocsr(), V)
            if (k + 1) % rec == 0:
                j = (k + 1) // rec
                p = pstate(j)
                v2 = psi.reshape(2, D)
                roc = np.outer(v2[0], v2[0].conj()) + np.outer(v2[1], v2[1].conj())
                Fc.append(fo(roc, p))
                Fn.append(fo(LC.osc_of_vec(V), p))
        v2 = psi.reshape(2, D)
        roc = np.outer(v2[0], v2[0].conj()) + np.outer(v2[1], v2[1].conj())
        out.update(F_T_closed=fo(roc, Tt), F_T_noisy=fo(LC.osc_of_vec(V), Tt),
                   row_F_T_closed=r["F_T_closed"], row_F_T_noisy=r["F_T_noisy"],
                   trace_rho=float(np.real(np.trace(V.reshape(2 * D, 2 * D)))))
        if r["set"] == "track":
            out.update(C_closed=float(min(Fc)), C_noisy=float(min(Fn)), row_C_closed=r["C_closed"],
                       row_C_noisy=r["C_noisy"], n_record=len(Fc))
    for k in list(out):
        if k.startswith("row_"):
            out["d_" + k[4:]] = out[k[4:]] - out[k]
    out["wall_s"] = time.time() - t0
    print(key, {k: (round(v, 9) if isinstance(v, float) else v) for k, v in out.items() if k.startswith(("d_", "bound"))},
          f"{out['wall_s']:.0f}s", flush=True)
    return out


SPOTS = [("track", "cs", 412, 11, 0.0, "zero"), ("track", "cs", 412, 11, 4.0, "zero"),
         ("track", "cs", 412, 11, 1.0, "hold"), ("track", "cd", 412, 9, 2.0, "zero"),
         ("track", "snap", 412, 9, 4.0, "zero"), ("track", "snap", 412, 6, 4.0, "zero"),
         ("track", "cs", 824, 16, 4.0, "zero"), ("track", "cd", 824, 18, 2.0, "hold"),
         ("track", "snap", 824, 5, 4.0, "zero"), ("target", "cs", 412, 1000, 4.0, "zero"),
         ("target", "cd", 412, 1004, 1.0, "hold"), ("target", "snap", 412, 1000, 2.0, "zero"),
         ("target", "snap4", 412, 1001, 4.0, "zero"), ("target", "snap4", 412, 1001, 0.0, "zero"),
         ("flip", "cs", 824, 3000, 4.0, "zero"), ("flip", "snap", 824, 3007, 2.0, "zero"),
         ("flip", "cd", 824, 3003, 4.0, "hold")]


def spot_checks(W):
    rows = {(r["set"], r["gate"], round(r["T_ns"]), r["seed"], r["sigma_ns"], r["edge"]): r
            for r in jl(P / "pulses_smooth.jsonl")}
    tasks = [("|".join(map(str, k)), rows[k]) for k in SPOTS]
    with mp.Pool(min(W, len(tasks))) as pool:
        res = pool.map(spot, tasks)
    S, Tt = LC.transfer("gkp", D)
    import compute as CP
    S9, T9 = CP.transfer("gkp", D)
    summ = {}
    for q in ("F_T_noisy", "F_T_closed", "C_noisy", "C_closed", "avg6_opt_noisy", "avg6_opt_closed"):
        v = [abs(r["d_" + q]) for r in res if "d_" + q in r]
        if v:
            summ["max_abs_d_" + q] = float(max(v))
    summ["max_bound_Omega"] = float(max(r["bound_Omega"] for r in res))
    summ["max_bound_osc"] = float(max(r["bound_osc"] for r in res))
    summ["target_overlap_LC_vs_compute"] = float(abs(np.vdot(Tt, T9)) ** 2)
    summ["initial_overlap_LC_vs_compute"] = float(abs(np.vdot(S, S9)) ** 2)
    return dict(summary=summ, checks=res)


if __name__ == "__main__":
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16
    t0 = time.time()
    out = dict(rows=rows_checks(), reproduction=reproduction())
    print(json.dumps({k: v for k, v in out["rows"].items() if k not in ("delta_table",)}, indent=1), flush=True)
    print(json.dumps(out["reproduction"], indent=1), flush=True)
    if "--no-spot" not in sys.argv:
        out["spot"] = spot_checks(W)
        print(json.dumps(out["spot"]["summary"], indent=1), flush=True)
    out["wall_s"] = time.time() - t0
    (P / "pulses_check_B.json").write_text(json.dumps(out, indent=1))
    print("check finished", flush=True)
