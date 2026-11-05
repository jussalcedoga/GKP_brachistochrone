"""Mean Wigner functions of the tracking pulses against the reference path at selected times (Figs. 4, 5 and SM).

For every configuration (transfer, reference path, gate set, duration index k) with traces, the oscillator states of
all random starts at the fractions t/T in traces.FR (saved by traces.py to results/rho80, cutoff 90, qubit traced out)
are averaged.  The Wigner function is linear in rho, so W of the mean state is the mean Wigner function.  Two versions:
  W_mean_L   Lindblad master equation with the Yale rates (the evolution shown in the figures),
  W_mean     closed evolution (for comparison),
with the closeness F to the reference path and the target fidelity F_T of the mean state, the reference-path state
W_path at the same fraction, and the per-start closeness (for the spread over the starts).
Writes results/wigner/<name>_<ref>_<gate>_k<k>.npz.   python simulations/tracking/wigner_snapshots.py [--traces traces80]
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common import tracking  # noqa: E402
from common.wigner import grid  # noqa: E402
import compute as CP  # noqa: E402
import load as L  # noqa: E402

TRN = sys.argv[sys.argv.index("--traces") + 1] if "--traces" in sys.argv else "traces80"
RHO = HERE / "results" / "rho80"
OUT = HERE / "results" / "wigner"
D = 90                                        # evaluation cutoff of traces.py


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G = L.groups(TRN)
    Wg = grid(D, 5.0, 121)
    for key in sorted(G):
        lst = [r for r in G[key] if (RHO / f"{r['job']['id']}.npz").exists()]
        if not lst:
            print("no saved states", key, flush=True)
            continue
        name, ref, gate, k = key
        dst = OUT / f"{name}_{ref}_{gate}_k{k}.npz"
        if (dst.exists() and "--force" not in sys.argv and int(np.load(dst)["n_starts"]) == len(lst)
                and dst.stat().st_mtime > max((RHO / f"{r['job']['id']}.npz").stat().st_mtime for r in lst)):
            print(f"{key}: up to date ({len(lst)} starts)", flush=True)   # keeps the files behind earlier figures
            continue
        S, Tg = CP.transfer(name, D)
        Z = [np.load(RHO / f"{r['job']['id']}.npz") for r in lst]
        fr = Z[0]["fr"]
        path = tracking.Path(CP.pad(CP.reference_dense(name, ref, 80), D))
        Gp, _ = path.at(fr)
        out = dict(x=Wg.x, fracs=fr, t_ns=np.mean([z["T_ns"] for z in Z]) * fr, n_starts=len(lst),
                   seeds=np.array([r["seed"] for r in lst]), W_path=np.array([Wg.of_vectors(gv[None]) for gv in Gp]),
                   F_T_path=np.array([float(abs(np.vdot(Tg, gv)) ** 2) for gv in Gp]))
        rep = L.representative(lst)                       # the start closest to the mean (a single physical run)
        irep = next(i for i, r in enumerate(lst) if r is rep)
        out["seed_rep"] = rep["seed"]
        for suf, fld in (("", "rho"), ("_L", "rhoL")):
            R = np.array([z[fld] for z in Z]).astype(complex)                 # (starts, fractions, D, D)
            Rm = R.mean(0)
            Fs = np.real(np.einsum("fi,sfij,fj->sf", Gp.conj(), R, Gp))      # per-start closeness at each fraction
            out[f"W_rep{suf}"] = np.array([Wg.of_rho(rr) for rr in R[irep]])
            out[f"F_rep{suf}"] = Fs[irep]
            out[f"W_mean{suf}"] = np.array([Wg.of_rho(rr) for rr in Rm])
            out[f"F_mean{suf}"] = np.real(np.einsum("fi,fij,fj->f", Gp.conj(), Rm, Gp))
            out[f"F_starts{suf}"] = Fs
            out[f"F_T_mean{suf}"] = np.real(np.einsum("i,fij,j->f", Tg.conj(), Rm, Tg))
        np.savez_compressed(dst, **out)
        print(f"{key}: {len(lst)} starts; F_mean_L = " + " ".join(f"{v:.3f}" for v in out["F_mean_L"]), flush=True)


if __name__ == "__main__":
    main()
