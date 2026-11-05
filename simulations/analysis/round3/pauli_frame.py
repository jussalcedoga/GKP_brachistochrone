"""Round 3: six-state average fidelity with X_L when the frame update is restricted to Paulis (phi in {0, pi}, i.e. I or
Z_L) instead of an arbitrary virtual Z rotation, for the gate pulses (sims/results/flip_gate.jsonl: target-only pulses and
every rung of the tracking runs) and for the stored transfer pulses of Table gate (jobs80, flip at m = 2.81).
Closed evolution at cutoff 90 (the frame restriction does not depend on the noise model).  Reports, per group, the mean
of avg6 at the best angle, at the best Pauli frame, and the largest difference.  Writes pauli_frame.json.
  python simulations/analysis/round3/pauli_frame.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import math
import multiprocessing as mp
import pathlib
import sys

import numpy as np
from scipy.sparse.linalg import expm_multiply

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent / "review_oct28"))
import robustness as RB  # noqa: E402
import mfcommon as mf  # noqa: E402

D = RB.D
CARD = RB.CARD


def channel(X, gate, T):
    B, _ = RB.code()
    dt = T / len(X)
    Psi = np.stack([RB.joint(B[0]), RB.joint(B[1])], axis=1)
    for x in X:
        Psi = expm_multiply((-1j * dt * RB.LC.hamiltonian(gate, x)).tocsr(), Psi)
    A = np.array([[[np.vdot(B[k], Psi[q * D:(q + 1) * D, i]) for i in range(2)] for k in range(2)] for q in range(2)])
    return np.einsum("qki,qlj->ijkl", A, A.conj())


def avg(K, phi):
    _, Gm = RB.code()
    out = []
    for a in CARD.values():
        a = np.array(a, complex)
        a = a / math.sqrt(np.real(np.conj(a) @ Gm @ a))
        M = np.einsum("i,j,ijkl->kl", a, a.conj(), K)
        b = np.array([a[1] * np.exp(-0.5j * phi), a[0] * np.exp(0.5j * phi)])
        out.append(float(np.real(np.conj(b) @ M @ b) / np.real(np.conj(b) @ Gm @ b)))
    return float(np.mean(out))


def task(t):
    group, gate, T, X = t
    K = channel(np.array(X), gate, T)
    grid = np.linspace(-math.pi, math.pi, 721)
    vals = [avg(K, p) for p in grid]
    return dict(group=group, best=float(max(vals)), phi_best=float(grid[int(np.argmax(vals))]),
                pauli=float(max(avg(K, 0.0), avg(K, math.pi))))


def main():
    tasks = []
    for line in open(mf.MF / "sims" / "results" / "flip_gate.jsonl"):
        r = json.loads(line)
        T = r["T_ns"] * 1e-3
        if r["mode"] == "target":
            tasks.append((f"gate-target/{r['gate']}/{round(r['T_ns'])}", r["gate"], T, r["X"]))
        else:
            for g in r["rungs"]:
                tasks.append((f"gate-track/{r['gate']}/{round(r['T_ns'])}/W{g['w']:g}", r["gate"], T, g["X"]))
    G = mf.groups()
    for ref in ("geodesic", "tradeoff", "photon"):
        for gate in ("cs", "cd", "snap"):
            for r in G[("flip", ref, gate, 2)]:
                X, T, jb = RB.load_pulse(r["job"]["id"])
                tasks.append((f"transfer/{ref}/{gate}", gate, T, X.tolist()))
    print(len(tasks), "pulses", flush=True)
    with mp.Pool(24) as pool:
        res = pool.map(task, tasks)
    out = {}
    for g in sorted({r["group"] for r in res}):
        rs = [r for r in res if r["group"] == g]
        out[g] = dict(n=len(rs), best=float(np.mean([r["best"] for r in rs])), pauli=float(np.mean([r["pauli"] for r in rs])),
                      max_diff=float(max(r["best"] - r["pauli"] for r in rs)),
                      phi_best=[round(r["phi_best"], 3) for r in rs])
        if not g.startswith("gate-track") or g.endswith("W10") or g.endswith("W2000"):
            print(f"{g:32s} n={len(rs):3d} best-angle {out[g]['best']:.4f} Pauli {out[g]['pauli']:.4f} "
                  f"max diff {out[g]['max_diff']:.4f}", flush=True)
    (HERE / "pauli_frame.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
