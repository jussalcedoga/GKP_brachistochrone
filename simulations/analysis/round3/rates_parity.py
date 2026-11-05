"""Round 3 (Oct 28): two evaluations of the stored benchmark pulses with the independent integrator of
analytics_audit/review_oct28 (cutoff 90, Lindblad Eq. (7)).

 1  Ancilla dephasing.  Sivak et al. (Nature 616, 50 (2023), Table S1) quote T1q = 280 us, T2R = 62 us (Ramsey) and
    T2E = 238 us (echo) for the ancilla; T_phi,q = 414 us of the manuscript follows from 1/T2E = 1/(2 T1q) + 1/T_phi.
    Without echoes the Ramsey value applies: T_phi,q = 1/(1/62 - 1/560) us = 69.7 us.  Every pulse is re-evaluated with
    the Yale rates and T_phi,q = 69.7 us.
 2  Photon-number parity.  For each pulse, the final oscillator state under the Yale rates: P_odd = Tr[(1 - Pi) rho],
    Pi = sum_{n even} |n><n|, the fidelity F_T, and the fidelity conditioned on even parity F_even = <T|rho|T>/P_even
    (every target is parity even), together with the closed-evolution P_odd.  Under conditional squeezing only photon
    loss changes the parity (the Hamiltonian, cavity dephasing and the ancilla channels commute with Pi), so P_odd is
    the probability of an odd number of loss events and a parity measurement after the pulse flags it.
Runs: Fig. 4 (GKP preparation along the brachistochrone; CS, ECD, SNAP; 412, 824, 2133 ns; 20 starts) and the flip
pulses along the brachistochrone at m = 2.81 (|0_L> -> |1_L> transfer; 20 starts).
Writes rates_parity.json next to this file.
  OMP_NUM_THREADS=1 python simulations/analysis/round3/rates_parity.py --workers 12
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
from scipy.sparse.linalg import expm_multiply

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent / "review_oct28"))
import robustness as RB  # noqa: E402

LC, D = RB.LC, RB.D
T1Q, T2R = 280.0, 62.0
TPHI_RAMSEY = 1.0 / (1.0 / T2R - 1.0 / (2 * T1Q))


def rates(tphiq):
    r = dict(LC.RATES)
    r["gphi"] = 1.0 / tphiq
    return r


def task(jid):
    t0 = time.time()
    X, T, jb = RB.load_pulse(jid)
    S, tgt = LC.transfer(jb["name"], D)
    u = RB.joint(S)
    dt = T / len(X)
    psi = u.copy()
    for x in X:
        psi = expm_multiply((-1j * dt * LC.hamiltonian(jb["gate"], x)).tocsr(), psi)
    po = np.abs(psi[:D]) ** 2 + np.abs(psi[D:]) ** 2
    out = dict(id=jid, name=jb["name"], gate=jb["gate"], T_ns=1e3 * T, P_odd_closed=float(po[1::2].sum()),
               F_closed=float(abs(np.vdot(tgt, psi[:D])) ** 2 + abs(np.vdot(tgt, psi[D:])) ** 2))
    V0 = np.outer(u, u.conj()).ravel()[:, None]
    for name, LDm in (("yale", RB.LD()), ("ramsey", RB.LD_rates(rates(TPHI_RAMSEY)))):
        rho = RB.osc_of(RB.lindblad_final(X, jb["gate"], T, V0, LDm)[:, 0])
        F = float(np.real(np.vdot(tgt, rho @ tgt)))
        p_odd = float(np.real(np.sum(np.diag(rho)[1::2])) / np.real(np.trace(rho)))
        out[name] = dict(F=F, P_odd=p_odd, F_even=F / (1 - p_odd))
    out["wall_s"] = time.time() - t0
    return out


def main():
    import mfcommon as mf
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 12
    Gr = mf.groups()
    keys = [("gkp", g, k) for g in RB.GATES for k in (1, 2, 3)] + [("flip", g, 2) for g in RB.GATES]
    ids = {key: [r["job"]["id"] for r in Gr[(key[0], "geodesic", key[1], key[2])]] for key in keys}
    tasks = sorted((i for v in ids.values() for i in v), key=lambda i: -(2.5 if "_k3_" in i else 1))
    print(f"{len(tasks)} pulses; T_phi,q (Ramsey) = {TPHI_RAMSEY:.1f} us", flush=True)
    with mp.Pool(W) as pool:
        res = list(pool.imap_unordered(task, tasks))
    groups = {}
    for key, lst in ids.items():
        rs = [o for o in res if o["id"] in lst]
        g = dict(T_ns=rs[0]["T_ns"], n=len(rs))
        for f in ("F_closed", "P_odd_closed"):
            g[f] = RB.ms([o[f] for o in rs])
        for nm in ("yale", "ramsey"):
            for f in ("F", "P_odd", "F_even"):
                g[f"{nm}_{f}"] = RB.ms([o[nm][f] for o in rs])
        groups[f"{key[0]}/{key[1]}/k{key[2]}"] = g
        print(key, {k: (round(v["mean"], 4), round(v["sd"], 4)) for k, v in g.items() if isinstance(v, dict)}, flush=True)
    (HERE / "rates_parity.json").write_text(json.dumps(dict(T_phi_ramsey_us=TPHI_RAMSEY, groups=groups, runs=res),
                                                       indent=1))
    print("wrote", HERE / "rates_parity.json")


if __name__ == "__main__":
    main()
