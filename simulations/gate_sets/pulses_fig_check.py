"""Independent check of every pulse drawn in Supplementary Figs. S25-S27 (Nov 2).  No optimization.

For each drawn run the slice variables X are read from the ORIGINAL result files (manuscript/.../jobs80/<job_id>.json for
the Fig. 4 tracking runs, sims/results/flip_gate.jsonl for the flip, sims/results/pulses/capability_pulses.jsonl for the
target-only runs of Table I, whose fidelities are compared with sims/results/capability_noisy.jsonl) and compared with
the arrays that the figures draw (results/pulses/*.npz).  The joint Hamiltonian is then rebuilt from Letter Eq. (7) with
operators written here (not the models of common/), Omega = 10 g r e^{i f_Omega}, branch amplitude g a_j e^{i f_j}
(SNAP: cavity drive g a_g e^{i f_g}, dispersive term -chi_d |e><e| (x) n), and propagated in closed evolution with
scipy.linalg.expm at 8 points per slice (cutoff 90).  The oscillator purity Tr rho^2 is computed from the reduced
oscillator density matrix (the figures use the qubit Gram matrix) and compared with the drawn curves, and the final
fidelity with the stored closed value.  Also checked: |u_+-| <= g, the identity
sum_j |j><j| (x) G(u_j) = 1 (x) G(u_+) + sigma_z (x) G(u_-) on the drawn controls, and, for SNAP, u_- = 0.
Writes results/pulses/pulses_fig_check.json.
  OMP_NUM_THREADS=1 python simulations/gate_sets/pulses_fig_check.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import math
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))
import mfcommon as M  # noqa: E402,I001

import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from common.params import G_SQ  # noqa: E402

R = HERE / "results" / "pulses"
JOBS = M.MS / "fig4_cs_path_following" / "results" / "jobs80"
D, SUB = 90, 8
a = np.diag(np.sqrt(np.arange(1, D)), 1).astype(complex)
ad = a.conj().T
n_op = np.diag(np.arange(D)).astype(complex)
I2, ID = np.eye(2), np.eye(D)
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.diag([1.0, -1.0]).astype(complex)              # sigma_z |g> = +|g> (SI Sec. V)
Pg, Pe = np.diag([1.0, 0.0]).astype(complex), np.diag([0.0, 1.0]).astype(complex)


def G_cs(u):                                           # (i/2)(xi^* a^2 - xi a^dag^2), Letter Eq. (7)
    return 0.5j * (np.conj(u) * a @ a - u * ad @ ad)


def G_d(u):                                            # (i/sqrt2)(eps^* a - eps a^dag)
    return (1j / math.sqrt(2)) * (np.conj(u) * a - u * ad)


def hamiltonians(gate, X, chi_over_g=1.0):
    """joint H (qubit (x) oscillator) per slice in units of 1/us, from the slice variables X (N, 6)"""
    X = np.asarray(X, float)
    out = []
    for r, fO, ag, fg, ae, fe in X:
        Om = 10 * G_SQ * r * np.exp(1j * fO)
        H = 0.5 * (Om.real * np.kron(sx, ID) + Om.imag * np.kron(sy, ID))
        if gate == "snap":
            H = H - chi_over_g * G_SQ * np.kron(Pe, n_op) + np.kron(I2, G_d(G_SQ * ag * np.exp(1j * fg)))
        else:
            G = G_cs if gate == "cs" else G_d
            H = H + np.kron(Pg, G(G_SQ * ag * np.exp(1j * fg))) + np.kron(Pe, G(G_SQ * ae * np.exp(1j * fe)))
        out.append(H)
    return out


def purity_traces(Hs, T_us, psi0s):
    """oscillator purity Tr rho_osc^2 at t = j T/(SUB N) from the reduced oscillator density matrix; final states"""
    h = T_us / len(Hs) / SUB
    P = np.array(psi0s, complex).T                       # (2D, m)
    pur = [[] for _ in range(P.shape[1])]

    def rec(P):
        for m in range(P.shape[1]):
            v = P[:, m].reshape(2, D)
            rho = v.T @ v.conj()                         # sum_q |v_q><v_q| on the oscillator
            pur[m].append(float(np.real(np.trace(rho @ rho))))
    rec(P)
    for H in Hs:
        U = expm(-1j * H * h)
        for _ in range(SUB):
            P = U @ P
            rec(P)
    return np.array(pur), P


def joint(osc):
    return np.kron(np.array([1, 0], complex), np.asarray(osc, complex))


def decomposition_check(gate, X):
    """|u_+-| <= g and sum_j |j><j| (x) G(u_j) = 1 (x) G(u_+) + sigma_z (x) G(u_-) on every slice"""
    X = np.asarray(X, float)
    if gate == "snap":
        return dict(max_abs_u_plus=float(np.max(X[:, 2])), u_minus=0.0)
    ug, ue = X[:, 2] * np.exp(1j * X[:, 3]), X[:, 4] * np.exp(1j * X[:, 5])
    up, um = 0.5 * (ug + ue), 0.5 * (ug - ue)
    G = G_cs if gate == "cs" else G_d
    dev = max(float(np.max(np.abs(np.kron(Pg, G(x)) + np.kron(Pe, G(y)) - np.kron(I2, G(p)) - np.kron(sz, G(q)))))
              for x, y, p, q in zip(ug, ue, up, um))
    return dict(max_abs_u_plus=float(np.abs(up).max()), max_abs_u_minus=float(np.abs(um).max()),
                max_abs_u_plus_or_minus_sum=float(np.max(np.abs(up) + np.abs(um))), identity_dev=dev)


def check_drawn(gate, X, z, i):
    """the arrays drawn by sims/pulses_figlib.py against the slice variables"""
    X = np.asarray(X, float)
    d = [float(np.max(np.abs(z["Omega_over_bound"][i] - X[:, 0] * np.exp(1j * X[:, 1]))))]
    if gate == "snap":
        d.append(float(np.max(np.abs(z["eps"][i] - X[:, 2] * np.exp(1j * X[:, 3])))))
    else:
        d.append(float(np.max(np.abs(z["u_g"][i] - X[:, 2] * np.exp(1j * X[:, 3])))))
        d.append(float(np.max(np.abs(z["u_e"][i] - X[:, 4] * np.exp(1j * X[:, 5])))))
    return max(d)


def main():
    import compute as CP
    out = {}
    S0, Tg = CP.transfer("gkp", D)
    # S25: Fig. 4 tracking runs at 824 ns, representative starts
    for gate in ("cs", "cd", "snap"):
        z = np.load(R / f"tracking_gkp_{gate}_T824.npz")
        zc = np.load(R / f"closed_tracking_gkp_{gate}_T824.npz")
        i = int(np.flatnonzero(z["representative"])[0])
        X = json.loads((JOBS / f"{z['job_id'][i]}.json").read_text())["X"]
        T_us = float(z["T_ns"]) / 1e3
        pur, P = purity_traces(hamiltonians(gate, X), T_us, [joint(S0)])
        FT = float(np.sum(np.abs(P[:, 0].reshape(2, D).conj() @ Tg) ** 2))
        out[f"S25_{gate}"] = dict(seed=int(z["seed"][i]), job=str(z["job_id"][i]),
                                  dX_raw=float(np.max(np.abs(np.asarray(X) - z["X"][i]))),
                                  d_drawn=check_drawn(gate, X, z, i),
                                  d_purity=float(np.max(np.abs(pur[0] - zc["purity"][i, 0]))),
                                  d_FT_closed=abs(FT - float(z["F_T_closed"][i])), **decomposition_check(gate, X))
        print(f"S25_{gate}", out[f"S25_{gate}"], flush=True)
    # S27: logical-flip gate pulses at 824 ns (mode target), representative starts and the best ECD start
    import flip_gate as FG
    recs = [json.loads(line) for line in open(HERE / "results" / "flip_gate.jsonl")]
    k0, k1, e1, c, theta, inp = FG.code(D)
    for gate, which in (("cs", "rep"), ("cd", "rep"), ("cd", "best"), ("snap", "rep")):
        z = np.load(R / f"flip_target_{gate}_T824.npz")
        zc = np.load(R / f"closed_flip_target_{gate}_T824.npz")
        sc = np.asarray(z["avg6_opt_lind"], float)
        i = int(np.flatnonzero(z["representative"])[0]) if which == "rep" else int(np.argmax(sc))
        seed = int(z["seed"][i])
        r = next(r for r in recs if r["mode"] == "target" and r["gate"] == gate and abs(r["T_ns"] - 824) < 1
                 and r["seed"] == seed)
        X = r["X"]
        pur, P = purity_traces(hamiltonians(gate, X), float(z["T_ns"]) / 1e3, [joint(inp[nm]) for nm in ("0", "1", "+")])
        out[f"S27_{gate}_{which}"] = dict(seed=seed, dX_raw=float(np.max(np.abs(np.asarray(X) - z["X"][i]))),
                                          d_drawn=check_drawn(gate, X, z, i),
                                          d_purity=float(np.max(np.abs(pur - zc["purity"][i]))),
                                          d_score_vs_jsonl=abs(float(r["eval"]["lind"]["avg6_opt"]) - float(sc[i]))
                                          if "lind" in r.get("eval", {}) else None,
                                          **decomposition_check(gate, X))
        print(f"S27_{gate}_{which}", out[f"S27_{gate}_{which}"], flush=True)
    # S26: target-only pulses at 412 ns (Letter Table I)
    rows = [json.loads(line) for line in open(R / "capability_pulses.jsonl")]
    orig = [json.loads(line) for line in open(HERE / "results" / "capability_noisy.jsonl")]
    for gk in ("cs", "cd", "snap", "snap4"):
        z = np.load(R / f"closed_target_{gk}_T412.npz")
        i = int(np.flatnonzero(z["representative"])[0])
        seed = int(z["seed"][i])
        r = next(r for r in rows if r["gate"] == gk and r["seed"] == seed)
        o = [x for x in orig if x.get("gate") == gk and x.get("seed") == seed and abs(x.get("T_ns", 412) - 412) < 1]
        gate = "snap" if gk.startswith("snap") else gk
        chi = float(r["chi_d_over_g"]) if r.get("chi_d_over_g") is not None else 1.0
        pur, P = purity_traces(hamiltonians(gate, r["X"], chi), float(z["T_ns"]) / 1e3, [joint(S0)])
        FT = float(np.sum(np.abs(P[:, 0].reshape(2, D).conj() @ Tg) ** 2))
        out[f"S26_{gk}"] = dict(seed=seed, chi_d_over_g=chi, dX=float(np.max(np.abs(np.asarray(r["X"]) - z["X"][i]))),
                                d_purity=float(np.max(np.abs(pur[0] - z["purity"][i]))),
                                d_FT_closed=abs(FT - float(r["F90_closed"])),
                                n_orig_rows=len(o), d_F90_noisy_vs_capability_noisy=(
                                    abs(float(o[0]["F90_noisy"]) - float(r["F90_noisy"])) if o else None),
                                **decomposition_check(gate, r["X"]))
        print(f"S26_{gk}", out[f"S26_{gk}"], flush=True)
    worst = dict(dX=max(v.get("dX_raw", v.get("dX", 0.0)) for v in out.values()),
                 d_drawn=max(v.get("d_drawn", 0.0) for v in out.values()),
                 d_purity=max(v["d_purity"] for v in out.values()),
                 d_FT_closed=max(v.get("d_FT_closed", 0.0) for v in out.values()),
                 identity_dev=max(v.get("identity_dev", 0.0) for v in out.values()),
                 max_abs_u=max(max(v.get("max_abs_u_plus", 0.0), v.get("max_abs_u_minus", 0.0)) for v in out.values()))
    print("WORST", worst)
    (R / "pulses_fig_check.json").write_text(json.dumps(dict(description=__doc__, cases=out, worst=worst), indent=1))
    print("wrote", R / "pulses_fig_check.json")


if __name__ == "__main__":
    main()
