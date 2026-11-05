"""Round 3 (Oct 28): is the advantage of conditional squeezing a matter of its larger energy uncertainty?

At equal amplitude bounds the CS pulses of Fig. 4 use 2.3 (412 ns) and 1.9 (824 ns) times omega on average, against 1.2 and
1.6 for conditional displacements (Table VIII).  Halving the squeezing rate, g -> g/2, together with the qubit-drive bound
(|Omega| <= 10 g/2), maps every CS pulse of duration T onto a pulse of duration 2T with the same closed-evolution states at
the rescaled times: H'(t) = H(t/2)/2.  The 412-ns CS pulses of Fig. 4 therefore are, at half the rate, 824-ns pulses whose
mean energy uncertainty is 2.33/2 = 1.17 omega, below the 1.60 omega of the 824-ns ECD pulses.  This script evaluates them
with the Lindblad equation (Yale rates) over 824 ns at cutoff 90: target fidelity and minimum closeness to the GKP
brachistochrone traversed in 824 ns (closed evolution, as in the benchmark), for the 20 starts.
Writes cs_half_rate.json next to this file.
  python simulations/analysis/round3/cs_half_rate.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

import numpy as np

import mfcommon as mf

sys.path.insert(0, str(mf.MS / "fig4_cs_path_following"))
from common import grape, lindblad, tracking  # noqa: E402
from common.params import G_SQ  # noqa: E402
import compute as CP  # noqa: E402

HERE = _L.vfile(__file__).parent
JOBS = mf.MS / "fig4_cs_path_following" / "results" / "jobs80"
D = 90


def main():
    runs = mf.groups()[("gkp", "geodesic", "cs", 1)]
    S, Tt = CP.transfer("gkp", D)
    path = tracking.Path(CP.pad(CP.reference_dense("gkp", "geodesic", 60), D))
    half = grape.CSModel(D, G_SQ / 2, 10 * G_SQ / 2, "two")
    full = grape.CSModel(D, G_SQ, 10 * G_SQ, "two")
    noise = lindblad.rates("Yale")
    out = []
    for r in runs:
        job = json.loads((JOBS / f"{r['job']['id']}.json").read_text())
        X, T, N = np.array(job["X"]), float(job["T_us"]), int(job["N"])
        d_half = tracking.diagnostics(half, X, 2 * T, N, grape.joint(S), path, Tt)
        d_full = tracking.diagnostics(full, X, T, N, grape.joint(S), path, Tt)
        F_half = tracking.lindblad_F(half, X, 2 * T, N, grape.joint(S), Tt, noise)
        out.append(dict(id=r["job"]["id"], T_ns=1e3 * T, C_full=d_full["C"], C_half=d_half["C"], FT_closed_full=d_full["F_T"],
                        FT_closed_half=d_half["F_T"], F_noisy_half=float(F_half)))
        print(out[-1], flush=True)
    s = {k: (float(np.mean([o[k] for o in out])), float(np.std([o[k] for o in out], ddof=1)))
         for k in ("C_full", "C_half", "FT_closed_half", "F_noisy_half")}
    print("mean, s.d.:", s)
    (HERE / "cs_half_rate.json").write_text(json.dumps(dict(summary=s, runs=out), indent=1))


if __name__ == "__main__":
    main()
