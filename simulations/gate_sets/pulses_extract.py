"""Pulses of the paper as Cartesian controls (result set A, Nov 1): the stored slice variables, no new optimization.

Sources (read only):
  * Fig. 4 tracking runs, GKP |0> -> |0_L> along the brachistochrone, CS, ECD, SNAP at 412, 824, 2133 ns (k = 1, 2, 3),
    20 starts each: exactly the runs of figures/Fig4/plot_fig4.py in its default (main-text) mode, that is
    mfcommon.groups() = manuscript/fig4_cs_path_following/load.groups("traces80") for the key
    ("gkp", "geodesic", gate, k).  Slice variables X from manuscript/fig4_cs_path_following/results/jobs80/<id>.json,
    stored F_T and C from results/traces80/<id>.json (cutoff 90; closed: F_T_final, C; Lindblad: F_T_final_noisy, C_noisy).
    The SNAP rows of the main Fig. 4 are these traces80 runs (seeds 1-20), not the warm-started runs of sims/snap_warm.py
    (those enter only Supplementary Fig. S11, binomial word, via --snap-warm).
  * The logical flip as a gate, manuscript_figures/sims/results/flip_gate.jsonl (written by sims/flip_gate.py): mode
    "target" (gate objective alone) and mode "track" (gate tracking; of the nine stored rungs of the weight ladder, the
    rungs W = 10 and 70 of Supplementary Table 11 are extracted), CS, ECD, SNAP, T = 412 and 824 ns, seeds 3000-3007.  Score = six-state average fidelity with X_L after the best frame update, Lindblad,
    cutoff 90 (eval.lind.avg6_opt), the number of Supplementary Table 11.
Representative start:
  * tracking: mfcommon.representative (smallest mean-squared distance of the Lindblad (x, z, F) traces from the averages
    over the starts), checked against rep_seed of manuscript_figures/fig4/fig4_facts.json.  For SNAP at 412 ns that start
    is a zero-drive run; the start chosen by the same rule among the non-trivial starts is stored as rep_nontrivial.
  * flip: no (x, z, F) traces exist for the gate pulses, so the representative is the start whose score is closest to the
    mean score over the starts of its case (ties: lowest seed).
Controls (Letter Eq. (7), SI Sec. VII): Omega = 10 g r e^{i f_Omega}; branch amplitude u_j = g a_j e^{i f_j} (xi_j for CS,
varepsilon_j for ECD); SNAP: cavity drive varepsilon = g a_g e^{i f_g} at chi_d = g, (a_e, f_e) unused.  The map is
model.controls() of src/src/common/grape.py (CS) and tracking.py (CDModel, SNAPModel).
Writes manuscript_figures/sims/results/pulses/{tracking_gkp_<gate>_T<ns>.npz, flip_<mode>_<gate>_T<ns>[_W<w>].npz,
pulses_index_A.json}.
  python simulations/gate_sets/pulses_extract.py
"""
from __future__ import annotations
import layout as _L  # repository layout: data/ and output/ (src/layout.py)

import json
import pathlib
import sys
from collections import defaultdict

HERE = _L.vfile(__file__).parent
MF = HERE.parent
sys.path.insert(0, str(MF))
import mfcommon as M  # noqa: E402,I001

import numpy as np  # noqa: E402

from common import tracking  # noqa: E402
from common.params import G_SQ  # noqa: E402

OUT = HERE / "results" / "pulses"
JOBS80 = M.MS / "fig4_cs_path_following" / "results" / "jobs80"
FLIP = HERE / "results" / "flip_gate.jsonl"
GATES = ("cs", "cd", "snap")
KS = (1, 2, 3)
TAB_RUNGS = (10.0, 70.0)                      # tracking rungs tabulated in Supplementary Table 11


def cartesian(gate, X):
    """polar slice variables (N, 6) -> complex controls: Omega (rad/us), branch amplitudes in units of g"""
    model = tracking.make_model(gate, 4, G_SQ, 10 * G_SQ)      # the map does not depend on the cutoff
    C, _ = model.controls(np.asarray(X, float))
    out = dict(Omega=C[:, 0] + 1j * C[:, 1])
    if gate == "snap":
        out["eps"] = (C[:, 2] + 1j * C[:, 3]) / G_SQ          # (a_e, f_e) multiply zero generators: not stored
    else:
        out["u_g"] = (C[:, 2] + 1j * C[:, 3]) / G_SQ
        out["u_e"] = (C[:, 4] + 1j * C[:, 5]) / G_SQ
    # independent check of the map against the formulas of SI Sec. VII
    X = np.asarray(X, float)
    assert np.allclose(out["Omega"], 10 * G_SQ * X[:, 0] * np.exp(1j * X[:, 1]), atol=1e-12)
    if gate == "snap":
        assert np.allclose(out["eps"], X[:, 2] * np.exp(1j * X[:, 3]), atol=1e-12)
    else:
        assert np.allclose(out["u_g"], X[:, 2] * np.exp(1j * X[:, 3]), atol=1e-12)
        assert np.allclose(out["u_e"], X[:, 4] * np.exp(1j * X[:, 5]), atol=1e-12)
    return out


def pack(runs, gate, N, T_ns):
    """stack the controls of the runs (list of dicts with X) into arrays (starts, slices)"""
    X = np.array([r["X"] for r in runs], float)
    assert X.shape[1:] == (N, 6), X.shape
    cc = [cartesian(gate, x) for x in X]
    d = dict(X=X, N=N, T_ns=T_ns, dt_ns=T_ns / N, t_edges_ns=T_ns * np.arange(N + 1) / N, g_per_us=G_SQ,
             Omega_max_per_us=10 * G_SQ, chi_d_over_g=(1.0 if gate == "snap" else np.nan))
    d["Omega_over_g"] = np.array([c["Omega"] for c in cc]) / G_SQ
    d["Omega_over_bound"] = d["Omega_over_g"] / 10.0
    for key in (("eps",) if gate == "snap" else ("u_g", "u_e")):
        d[key] = np.array([c[key] for c in cc])
    return d


def tracking_cases(index):
    facts = json.loads((MF / "fig4" / "fig4_facts.json").read_text())
    G = M.groups()
    for gate in GATES:
        for k in KS:
            lst = sorted(G[("gkp", "geodesic", gate, k)], key=lambda r: r["seed"])
            cf = facts["columns"][str(k)][gate]
            rep = M.representative(lst)
            assert len(lst) == cf["n_usable"] == 20, (gate, k, len(lst))
            assert rep["seed"] == cf["rep_seed"], (gate, k, rep["seed"], cf["rep_seed"])
            triv = np.array([M.trivial(r) for r in lst])
            assert int(triv.sum()) == cf["n_trivial"], (gate, k)
            nontriv = [r for r, t in zip(lst, triv) if not t]
            rep_nt = M.representative(nontriv) if nontriv else None
            runs = []
            for r in lst:
                jb = json.loads((JOBS80 / f"{r['job']['id']}.json").read_text())
                assert abs(1e3 * jb["T_us"] - r["T_ns"]) < 1e-9 and jb["N"] == r["N"]
                runs.append(dict(X=jb["X"], w_star=jb["w_star"], id=r["job"]["id"]))
            N, T_ns = lst[0]["N"], lst[0]["T_ns"]
            d = pack(runs, gate, N, T_ns)
            d.update(seed=np.array([r["seed"] for r in lst]), job_id=np.array([r["job"]["id"] for r in lst]),
                     w_star=np.array([x["w_star"] for x in runs], float),
                     F_T_closed=np.array([r["F_T_final"] for r in lst]), C_closed=np.array([r["C"] for r in lst]),
                     F_T_noisy=np.array([r["F_T_final_noisy"] for r in lst]),
                     C_noisy=np.array([r["C_noisy"] for r in lst]), trivial=triv,
                     representative=np.array([r is rep for r in lst]),
                     rep_nontrivial=np.array([(rep_nt is not None and r is rep_nt) for r in lst]),
                     # Lindblad traces of traces80 (8 points per slice, t = 0 included), for figures
                     tL_ns=np.array([r["tL_ns"] for r in lst]), FTL=np.array([r["FTL"] for r in lst]),
                     FL=np.array([r["FL"] for r in lst]), nbarL=np.array([r["nbarL"] for r in lst]))
            name = f"tracking_gkp_{gate}_T{round(T_ns)}.npz"
            np.savez_compressed(OUT / name, **d)
            index.append(dict(file=name, family="tracking_gkp", transfer="vacuum -> |0_L>, Delta = 0.4",
                              path="brachistochrone", gate=gate, k=k, T_ns=T_ns, N=N, n_starts=len(lst),
                              seeds=[int(s) for s in d["seed"]], rep_seed=int(rep["seed"]),
                              rep_nontrivial_seed=(int(rep_nt["seed"]) if rep_nt is not None else None),
                              n_trivial=int(triv.sum()), trivial_seeds=[int(r["seed"]) for r, t in zip(lst, triv) if t],
                              rep_rule="mfcommon.representative (Lindblad x, z, F traces), equals fig4_facts rep_seed",
                              score="F_T_noisy (Lindblad, cutoff 90)",
                              score_mean=float(d["F_T_noisy"].mean()), score_sd=float(d["F_T_noisy"].std(ddof=1))))
            print(name, "rep", rep["seed"], "trivial", int(triv.sum()), flush=True)


def flip_cases(index):
    recs = [json.loads(line) for line in open(FLIP)]
    groups = defaultdict(list)
    for r in recs:
        T = round(r["T_ns"])
        if r["mode"] == "target":
            groups[("target", r["gate"], T, None)].append(dict(X=r["X"], seed=r["seed"], N=r["N"], T_ns=r["T_ns"],
                                                                 F_T_D60=r["F_T_D60"], C_D60=r["C_D60"],
                                                                 ev=r["eval"]))
        else:
            for rg in r["rungs"]:
                if float(rg["w"]) not in TAB_RUNGS:    # the rungs of Supplementary Table 11 (W = 10, 70) only
                    continue
                groups[("track", r["gate"], T, float(rg["w"]))].append(
                    dict(X=rg["X"], seed=r["seed"], N=r["N"], T_ns=r["T_ns"], F_T_D60=rg["F_T_D60"],
                         C_D60=rg["C_D60"], ev=rg["eval"]))
    for (mode, gate, T, w), lst in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] or 0)):
        lst = sorted(lst, key=lambda x: x["seed"])
        score = np.array([x["ev"]["lind"]["avg6_opt"] for x in lst])
        dist = np.abs(score - score.mean())
        irep = int(np.argmin(dist))                   # argmin returns the first, i.e. the lowest seed, on ties
        N, T_ns = lst[0]["N"], lst[0]["T_ns"]
        d = pack(lst, gate, N, T_ns)
        ev = lambda part, key: np.array([x["ev"][part][key] for x in lst])  # noqa: E731
        d.update(seed=np.array([x["seed"] for x in lst]), W=(np.nan if w is None else w),
                 avg6_opt_lind=score, avg6_lind=ev("lind", "avg6"), phi_opt_lind=ev("lind", "phi_opt"),
                 F_0to1_lind=ev("lind", "F_0to1"), F_1to0_lind=ev("lind", "F_1to0"),
                 avg6_opt_closed=ev("closed", "avg6_opt"), avg6_closed=ev("closed", "avg6"),
                 nbar_max_closed=np.array([x["ev"]["nbar_max"] for x in lst]),
                 F_T_D60=np.array([x["F_T_D60"] for x in lst]), C_D60=np.array([x["C_D60"] for x in lst]),
                 representative=np.arange(len(lst)) == irep)
        name = f"flip_{mode}_{gate}_T{T}" + (f"_W{w:g}" if w is not None else "") + ".npz"
        np.savez_compressed(OUT / name, **d)
        index.append(dict(file=name, family=f"flip_{mode}", transfer="logical flip X_L as a gate (inputs 0_L, 1_L, +_L)",
                          gate=gate, T_ns=T_ns, N=N, W=w, tabulated=(mode == "target" or (T == 824 and w in TAB_RUNGS)),
                          n_starts=len(lst), seeds=[int(x["seed"]) for x in lst], rep_seed=int(lst[irep]["seed"]),
                          rep_rule="start whose score is closest to the mean score over the starts (ties: lowest seed)",
                          score="avg6_opt_lind (six-state average with X_L, best frame update, Lindblad, cutoff 90)",
                          score_mean=float(score.mean()), score_sd=float(score.std(ddof=1)),
                          score_best=float(score.max()), rep_score=float(score[irep])))
        print(name, "rep", lst[irep]["seed"], f"score {score.mean():.4f} +- {score.std(ddof=1):.4f}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    tracking_cases(index)
    flip_cases(index)
    keys_common = ("X (starts, N, 6): slice variables (r, f_Omega, a_g, f_g, a_e, f_e); N, T_ns, dt_ns, t_edges_ns (N+1); "
                   "Omega_over_g, Omega_over_bound (starts, N) complex, Omega = Omega_x + i Omega_y; u_g, u_e (starts, N) "
                   "complex branch amplitudes xi_j (CS) or varepsilon_j (ECD) in units of g; eps (SNAP) cavity drive in "
                   "units of g at chi_d = g; seed; representative (bool)")
    (OUT / "pulses_index_A.json").write_text(json.dumps(dict(
        description=__doc__, keys_common=keys_common,
        keys_tracking="job_id, w_star, F_T_closed, C_closed, F_T_noisy, C_noisy (stored, cutoff 90), trivial, "
                      "rep_nontrivial, tL_ns, FTL, FL, nbarL (Lindblad traces of traces80, 8 points per slice)",
        keys_flip="W (nan for target), avg6_opt_lind, avg6_lind, phi_opt_lind, F_0to1_lind, F_1to0_lind, "
                  "avg6_opt_closed, avg6_closed, nbar_max_closed (stored eval, cutoff 90), F_T_D60, C_D60 (per input "
                  "0_L, 1_L, +_L, cutoff 60)",
        cases=index), indent=1))
    print("wrote", len(index), "cases to", OUT)


if __name__ == "__main__":
    main()
