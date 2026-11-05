"""Round 5: cutoff dependence of Supplementary Table order (target fidelity at T* with the normal-ordered expansion of
H_opt truncated at total order K).  Everything after the Fock-basis H_opt is done in mpmath (dps 60): the coefficients
h_jk, the matrix elements of sum_{j+k<=K} h_jk a^dag^j a^k assembled directly,
    <n|.|m> = sum_q h_{n-q,m-q} sqrt(n! m!)/q!   over (n-q)+(m-q) <= K,
and the exponential.  All targets are real and parity even, so H_opt = i a with a real antisymmetric, the truncated
operator conserves parity, and exp(-i H_K t) = exp(a_K t) is computed on the even sector.  Validation: K = full order
reproduces F = 1, and D = 40 reproduces the table.
python tables/si_table1_truncation_order.py"""
import multiprocessing as mp_
import time

import mpmath as mp
import numpy as np

from common import states, brachistochrone as BR

mp.mp.dps = 60
KS = [2, 4, 6, 8, 12, 16, 24, 40, 56]


def pair(name, D):
    if name == "flip":
        return states.gkp(0.4, 0, D), states.gkp(0.4, 1, D)
    T = {"fock2": states.fock(2, D), "fock4": states.fock(4, D), "cat2": states.cat(2.0, D),
         "gkp0": states.gkp(0.4, 0, D)}[name]
    return states.vacuum(D), T


def run(task):
    name, D, K = task
    t0 = time.time()
    S, T = pair(name, D)
    A = BR.H_opt(S, T, 1.0)
    assert np.max(np.abs(A.real)) < 1e-12 and np.max(np.abs(S.imag)) < 1e-12 and np.max(np.abs(T.imag)) < 1e-12
    a = A.imag
    _, th, _ = BR.geodesic_parts(S, T)
    fact = [mp.factorial(k) for k in range(D)]
    sq = [mp.sqrt(f) for f in fact]
    eta = {}
    for j in range(D):
        for k in range(j % 2, D, 2):
            acc = mp.mpf(0)
            for p in range(min(j, k) + 1):
                v = a[j - p, k - p]
                if v != 0.0:
                    acc += (-1) ** p * mp.mpf(v) / (fact[p] * sq[j - p] * sq[k - p])
            eta[j, k] = acc
    ev = list(range(0, D, 2))
    N = len(ev)
    Kmax = 10 ** 9 if K is None else K
    M = mp.matrix(N, N)
    for i, n in enumerate(ev):
        for jj, m in enumerate(ev):
            acc = mp.mpf(0)
            for q in range(min(n, m) + 1):
                if (n - q) + (m - q) <= Kmax:
                    acc += eta[n - q, m - q] * sq[n] * sq[m] / fact[q]
            M[i, jj] = acc
    nrm = float(mp.mnorm(M, 1))
    E = mp.expm(M * mp.mpf(th))
    s = mp.matrix([mp.mpf(float(S[n].real)) for n in ev])
    psi = E * s
    ov = mp.fsum(mp.mpf(float(T[n].real)) * psi[i] for i, n in enumerate(ev))
    nrm_psi = mp.sqrt(mp.fsum(psi[i] ** 2 for i in range(N)))
    return name, D, ("full" if K is None else K), float(ov ** 2), nrm, float(nrm_psi), time.time() - t0


def main():
    tasks = [("gkp0", 40, None), ("flip", 40, None)]
    for name in ("fock2", "fock4", "cat2", "gkp0", "flip"):
        for D in (40, 60, 80):
            for K in KS:
                tasks.append((name, D, K))
    tasks.sort(key=lambda t: -(t[1] * (100 if t[2] is None else t[2])))
    with mp_.Pool(32) as pool:
        res = pool.map(run, tasks, chunksize=1)
    res.sort(key=lambda r: (r[0], r[1], 999 if r[2] == "full" else r[2]))
    for r in res:
        print(f"{r[0]:6s} D={r[1]:3d} K={str(r[2]):>4s}  F={r[3]:.4f}  |H_K|_1={r[4]:.2e}  |psi|={r[5]:.12f}  ({r[6]:.0f} s)",
              flush=True)
    print("\nTable (rows: target/cutoff; columns K = " + " ".join(map(str, KS)) + ")")
    for name in ("fock2", "fock4", "cat2", "gkp0", "flip"):
        for D in (40, 60, 80):
            row = {r[2]: r[3] for r in res if r[0] == name and r[1] == D}
            print(f"{name:6s} D={D:3d} " + " ".join(f"{row[K]:.3f}" for K in KS), flush=True)


if __name__ == "__main__":
    main()
