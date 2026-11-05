"""Closed-evolution traces of the target-only GKP pulses at 412 ns (Letter Table I) for Supplementary Fig. S26
(Nov 1).  No optimization: the pulses are read from results/pulses/capability_pulses.jsonl.

For every start (CS, ECD, SNAP chi_d = g, SNAP chi_d = 4 g; seeds 1000-1007) the stored slice variables X are mapped to
controls by the model of sims/capability_noisy.py (model(gk, 90), cutoff 90), checked against the stored Cartesian
controls C, and propagated from |0>|g> with the exact slice unitaries at 8 points per slice, as in sims/pulses_stats.py
(closed_traces, diag): oscillator purity Tr rho^2 (equal to the qubit purity, the joint state is pure), excited-qubit
population P_e and <n>.  Check: the final closed F_T against the stored F90_closed.  Representative start of each gate
set: the start whose Lindblad fidelity F90_noisy is closest to the mean over the 8 starts (ties: lowest seed).
Writes results/pulses/closed_target_<gk>_T412.npz (t_ns, seed, F90_noisy, F90_closed, representative, X, C, purity,
Pe, nbar) and results/pulses/closed_target_check.json.
  OMP_NUM_THREADS=1 python simulations/gate_sets/pulses_closed_target.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE))
import capability_noisy as CN  # noqa: E402  (paths of manuscript/, compute as CN.CP, model())
import numpy as np  # noqa: E402

from common import grape  # noqa: E402
from common.params import G_SQ  # noqa: E402

R = HERE / "results" / "pulses"
D, SUB = 90, 8
GK = ("cs", "cd", "snap", "snap4")


def traces(gk, X, T_us):
    """closed evolution from |0>|g> at 8 points per slice: purity, P_e, <n> (t,) and the final joint state"""
    model = CN.model(gk, D)
    C, _ = model.controls(np.asarray(X, float))
    H = model.hamiltonians(C)
    lam, V = np.linalg.eigh(H)
    S, Tt = CN.CP.transfer("gkp", D)
    psi = grape.joint(S)
    h = T_us / len(X) / SUB
    out = [psi.copy()]
    for k in range(len(X)):
        U = (V[k] * np.exp(-1j * lam[k] * h)) @ V[k].conj().T
        for _ in range(SUB):
            psi = U @ psi
            out.append(psi.copy())
    P = np.array(out).reshape(-1, 2, D)                       # (t, qubit, Fock)
    rq = np.einsum("tin,tjn->tij", P, P.conj())             # reduced qubit state
    pur = np.sum(np.abs(rq) ** 2, axis=(1, 2))
    pe = np.real(rq[:, 1, 1])
    nb = np.einsum("tin,n->t", np.abs(P) ** 2, np.arange(D, dtype=float))
    FT = float(np.sum(np.abs(P[-1].conj() @ Tt) ** 2))
    return C, pur, pe, nb, FT


def main():
    rows = [json.loads(l) for l in open(R / "capability_pulses.jsonl")]
    check = {}
    for gk in GK:
        lst = sorted((r for r in rows if r["gate"] == gk), key=lambda r: r["seed"])
        assert [r["seed"] for r in lst] == list(range(1000, 1008)), gk
        F = np.array([r["F90_noisy"] for r in lst])
        dev = np.abs(F - F.mean())
        rep = np.zeros(len(lst), bool)
        rep[int(np.flatnonzero(dev == dev.min())[0])] = True
        T_us = lst[0]["T_ns"] / 1e3
        res = [traces(gk, r["X"], T_us) for r in lst]
        dC = max(float(np.max(np.abs(c[0] - np.array(r["C"])))) for c, r in zip(res, lst))
        dF = max(abs(c[4] - r["F90_closed"]) for c, r in zip(res, lst))
        N = len(lst[0]["X"])
        np.savez_compressed(R / f"closed_target_{gk}_T412.npz", t_ns=lst[0]["T_ns"] * np.arange(SUB * N + 1) / (SUB * N),
                            T_ns=lst[0]["T_ns"], N=N, sub=SUB, chi_d_over_g=np.nan if lst[0]["chi_d_over_g"] is None
                            else float(lst[0]["chi_d_over_g"]), seed=np.array([r["seed"] for r in lst]),
                            F90_noisy=F, F90_closed=np.array([r["F90_closed"] for r in lst]), representative=rep,
                            X=np.array([r["X"] for r in lst]), C=np.array([r["C"] for r in lst]) / G_SQ,
                            purity=np.array([c[1] for c in res]), Pe=np.array([c[2] for c in res]),
                            nbar=np.array([c[3] for c in res]))
        pmin = np.array([c[1][::SUB].min() for c in res])
        check[gk] = dict(max_dev_C_per_us=dC, max_dev_F_T_closed=dF, rep_seed=int(lst[int(np.argmax(rep))]["seed"]),
                         rep_F90_noisy=float(F[rep][0]), F90_noisy_mean=float(F.mean()), F90_noisy_sd=float(F.std(ddof=1)),
                         purity_min_slice_ends=dict(mean=float(pmin.mean()), sd=float(pmin.std(ddof=1)),
                                                    rep=float(pmin[rep][0])),
                         purity_final=dict(mean=float(np.mean([c[1][-1] for c in res])),
                                           sd=float(np.std([c[1][-1] for c in res], ddof=1))),
                         nbar_max=dict(mean=float(np.mean([c[3][::SUB].max() for c in res])),
                                       sd=float(np.std([c[3][::SUB].max() for c in res], ddof=1))))
        print(gk, json.dumps(check[gk]), flush=True)
    (R / "closed_target_check.json").write_text(json.dumps(dict(description=__doc__, cases=check), indent=1))
    print("wrote", R / "closed_target_check.json")


if __name__ == "__main__":
    main()
