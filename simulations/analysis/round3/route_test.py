"""Round 3 (Oct 28): corrected route test (manuscript/si/flip_route_test/compute.py is read only).

Error found in the original: photon_core.apply_route returns times in seconds (t = arclength/OMEGA, OMEGA in s^-1), and
the exposure of the second input was formed as 1e3 * sum(nbar dt), i.e. in photon ms, about 1e-6 of its value in photon
ns.  The 'mean exposure of the two inputs' was therefore half the exposure of |0_L> alone (e.g. 370 = 739.5/2 for the
brachistochrone).  Here e1 = 1e9 * sum(nbar dt).

The rank-two Hamiltonians that carry |0_L> along a path (arc by arc) are applied to |1_L>: fidelity with |0_L> (a flip
must map |1_L> -> |0_L>), peak weight outside the code space of either input, exposure of each input and their mean (the
code-space average for an orthonormal basis; <0_L|1_L> = 0.014).  Paths: brachistochrone, code-space penalty
ell = 100, 30, 10, 3, photon optimal (ell = 0), trade-off w* for the Yale and CS-proposal rates.
Writes route_test.json and route_test.md next to this file.
  python simulations/analysis/round3/route_test.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib

import numpy as np

from common import paths, photon_core as pc
from common.params import OMEGA_REF, kappa_mu

HERE = _L.vfile(__file__).parent
D = 80
L = pc.Logical(0.4, D)
N = np.diag(L.nv)
geo = L.geo(np.linspace(0, L.th, 32))
routes = {"brachistochrone": geo}
prev = geo
for ell in (100.0, 30.0, 10.0, 3.0, 0.0):
    st = [prev] + ([pc.resample(L.detour(), 30)] if ell <= 3 else [])
    U, _ = pc.optimize_path(L.Nl(ell), L.u0, L.uT, st, 30)
    prev = U
    routes[f"ell={ell:g}"] = U
for nk in ("Yale", "CSproposal"):
    k, m = kappa_mu(nk)
    routes[f"trade-off w*={k / m:.3f} ({nk})"] = paths.tradeoff_family(N, L.u0, L.uT, geo, routes["ell=0"], [k / m])[0]

rows, res = [], {}
for name, U in routes.items():
    out, t = pc.apply_route(U, L.uT, L.nv)                    # second input |1_L>; t in seconds
    F10 = float((out[-1] @ L.u0) ** 2)
    leak1 = float(np.max(1 - np.sum(out * (out @ L.Pi), 1)))
    X0, _ = pc.dense(U, 41)
    leak0 = float(np.max(1 - np.sum(X0 * (X0 @ L.Pi), 1)))
    e0 = 1e3 * pc.exposure(U, N) / OMEGA_REF                 # photon ns (OMEGA_REF in rad/us)
    nb1 = np.sum(out ** 2 * L.nv, 1)
    e1 = 1e9 * float(np.sum(0.5 * (nb1[1:] + nb1[:-1]) * np.diff(t)))
    T = 1e3 * pc.length(U) / OMEGA_REF
    assert abs(1e9 * t[-1] - T) < 1e-6 * T, (t[-1], T)
    res[name] = dict(T_ns=T, F_1to0=F10, leak_0=leak0, leak_1=leak1, expo_0_ns=e0, expo_1_ns=e1,
                     mean_expo_ns=0.5 * (e0 + e1), nbar1_max=float(nb1.max()))
    rows.append(f"| {name} | {T:.0f} | {F10:.4f} | {100 * leak0:.1f}% | {100 * leak1:.1f}% | {e0:.0f} | {e1:.0f} | "
                f"{0.5 * (e0 + e1):.0f} |")
    print(rows[-1], flush=True)
hdr = ("| route of |0_L> | T (ns) | |1_L> -> |0_L> | peak leakage |0_L> | peak leakage |1_L> | exposure |0_L> | "
       "exposure |1_L> | mean (photon ns) |\n|---|---|---|---|---|---|---|---|")
(HERE / "route_test.json").write_text(json.dumps(res, indent=1))
(HERE / "route_test.md").write_text(hdr + "\n" + "\n".join(rows) + "\n")
print("wrote", HERE / "route_test.json")
