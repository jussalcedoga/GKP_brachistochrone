"""Round 4: exposure and duration of the closed forms for the transfers of Supplementary Table vacuum.
A photon-optimal transfer through the vacuum is the reversed preparation of the initial state followed by the preparation
of the final state; the closed form (all normal modes, compute_fig2.modal_leg) and the one-mode form
(compute_fig2.closed_leg) are applied on each leg, and exposures and lengths add.
Writes results/transfer_closed.json.
python simulations/analysis/round4/transfer_closed.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import importlib.util
import json
import pathlib

import numpy as np

from common import photon_core as pc, states

HERE = _L.vfile(__file__).parent
MF = HERE.parents[1]
spec = importlib.util.spec_from_file_location("compute_fig2", _L.REPO / "figures" / "Fig2" / "compute_fig2.py")
CF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CF)
D = CF.D
NS = 1e3 / CF.OM


def full(name):
    if name.startswith("gkp"):
        _, mu, De = name.split()
        a, b = (states.gkp(float(De), k, D).astype(complex) for k in (0, 1))
        a, b = (x / np.linalg.norm(x) * np.exp(-1j * np.angle(x[0])) for x in (a, b))
        if mu == "+":
            v = a + (b if np.real(np.vdot(a, b)) >= 0 else -b)
            return v / np.linalg.norm(v)
        return a if mu == "0" else b
    nv, cat = pc.target("Cat", 2.0, D)
    v = np.zeros(D, complex)
    v[nv.astype(int)] = cat
    return v


def leg(name, kind):
    P = CF.paths.Prep(full(name))
    X = CF.modal_leg(P) if kind == "modal" else CF.closed_leg(P)
    return pc.exposure(X, P.N) * NS, pc.length(X) * NS


def main():
    transfers = {"flip, Delta=0.4": ("gkp 0 0.4", "gkp 1 0.4"), "flip, Delta=0.3": ("gkp 0 0.3", "gkp 1 0.3"),
                 "flip, Delta=0.5": ("gkp 0 0.5", "gkp 1 0.5"), "|0_L> -> |+_L>": ("gkp 0 0.4", "gkp + 0.4"),
                 "cat(2) -> |0_L>": ("cat", "gkp 0 0.4")}
    out = {}
    for t, (i, f) in transfers.items():
        row = {}
        for kind in ("modal", "one-mode"):
            (Ji, Ti), (Jf, Tf) = leg(i, kind), leg(f, kind)
            row[kind] = dict(J_ns=Ji + Jf, T_ns=Ti + Tf)
        out[t] = row
        print(f"{t:16s}: closed form J = {row['modal']['J_ns']:.2f} photon ns in {row['modal']['T_ns']:.1f} ns; "
              f"one-mode J = {row['one-mode']['J_ns']:.2f} in {row['one-mode']['T_ns']:.1f} ns", flush=True)
    (HERE / "results" / "transfer_closed.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
