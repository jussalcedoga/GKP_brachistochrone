"""Fig. 4 / SM analysis of the tracking benchmark: one table of every job, the tangential capacities, Wigner snapshots
of the representative pulses.  Writes results/fig4.json and results/fig4_table.md.
python simulations/tracking/analyze.py [--smoke]"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common import brachistochrone as B, grape, tracking  # noqa: E402
from common.params import G_SQ  # noqa: E402
from common.wigner import grid  # noqa: E402
import compute as CP  # noqa: E402

SMOKE = "--smoke" in sys.argv
JOBS = HERE / "results" / ("smoke" if SMOKE else "jobs80")      # level-3 pulses (cutoff 80, checked at 90)
OUT = HERE / "results" / ("smoke" if SMOKE else "") / "fig4.json"
KEYS = ("C", "C_ends", "F_mean", "bound", "F_T", "F_loss", "F_loss_D90", "path_F_loss", "w_star", "expo_ns", "nbar_max",
        "Lambda", "path_L", "T_us", "N")


def capacity(name, D=80):
    """tangential capacities at the start of the brachistochrone (deck Eqs. (6)-(9)) for both gate sets: the largest
    speed along the path direction chi that the conditional generator can give from |S> with the qubit in |0>"""
    S, T = CP.transfer(name, D)
    c0, th, chi = B.geodesic_parts(S, T)
    out = {}
    for gate in ("cs", "cd"):
        M = tracking.make_model(gate, D, 1.0, 10.0)
        HA, HB = M.Hc[2][:D, :D], M.Hc[3][:D, :D]          # generators on the qubit-|0> block, unit rate
        cA = np.real(np.vdot(chi, -1j * (HA @ S)))
        cB = np.real(np.vdot(chi, -1j * (HB @ S)))
        v = float(np.hypot(cA, cB))                         # in units of g
        out[gate] = dict(v_over_g=v, T_tan_ns=(1e3 * th / (v * G_SQ) if v > 1e-9 else None))
    out["theta0"] = float(th)
    out["T_star_ns"] = float(1e3 * th / (G_SQ / np.sqrt(2)))
    return out


def snapshots(job_res, n_snap=8, L=5.0, npts=121):
    """Wigner functions of the pulse (oscillator, qubit traced out) and of the path state at equal times"""
    jb = job_res["job"]
    D = 80
    M = tracking.make_model(jb["gate"], D, jb["g"], 10 * jb["g"])
    S, Tg = CP.transfer(jb["name"], D)
    path = tracking.Path(CP.reference_dense(jb["name"], jb["ref"], D))
    X = np.array(job_res["X"])
    C, _ = M.controls(X)
    N = len(X)
    traj, _, _ = M.propagate(C, job_res["T_us"] / N, grape.joint(S))
    idx = [int(round(f * N)) for f in np.linspace(0, 1, n_snap)]
    G = grid(D, L, npts)
    Gp, _ = path.at(np.array(idx) / N)
    Fk = [float(np.real(sum(abs(np.vdot(Gp[i], traj[k].reshape(2, D)[q])) ** 2 for q in range(2))))
          for i, k in enumerate(idx)]
    return dict(t_ns=[1e3 * job_res["T_us"] * k / N for k in idx], W_pulse=[G.of_joint(traj[k]).tolist() for k in idx],
                W_path=[G.of_vectors(g[None]).tolist() for g in Gp], F=Fk, x=G.x.tolist())


def main():
    rows = []
    for p in sorted(JOBS.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "job" in r:
            rows.append(r)
    table = {}
    for r in rows:
        jb = r["job"]
        e = {k: r.get(k) for k in KEYS}
        e.update(F_T_D90=r["check_D90"]["F_T"], C_D90=r["check_D90"]["C"], T_ns=1e3 * r["T_us"], rule=r["rule"])
        table.setdefault(jb["set"], {}).setdefault(jb["name"], {}).setdefault(jb["ref"], {}).setdefault(jb["gate"], {})[
            str(jb["Tg"])] = e
    res = dict(table=table, capacity={n: capacity(n) for n in ("fock2", "cat", "gkp", "fock4", "flip")},
               Tg=list(CP.TG), g=G_SQ, counts=len(rows))
    # representative pulses: GKP preparation along the brachistochrone, CS, at T_g = 16 (or the longest available)
    rep = {}
    for name in ("gkp", "flip"):
        for gate in ("cs", "cd"):
            cands = [r for r in rows if r["job"]["name"] == name and r["job"]["ref"] == "geodesic" and r["job"]["gate"] == gate
                     and r["job"]["set"] in ("main", "si")]
            if not cands:
                continue
            pick = [r for r in cands if abs(r["job"]["Tg"] - 16.0) < 1e-9] or sorted(cands, key=lambda r: r["job"]["Tg"])[-1:]
            rep[f"{name}_{gate}"] = dict(Tg=pick[0]["job"]["Tg"], C=pick[0]["C"], F_loss=pick[0]["F_loss"],
                                         **snapshots(pick[0]))
    res["snapshots"] = rep
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res))
    lines = ["| set | transfer | path | gate | T_g | T (ns) | C | F_T | F_T Lindblad | path F Lindblad | w* | exposure (photon ns) "
             "| Lambda |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, d1 in table.items():
        for n, d2 in d1.items():
            for ref, d3 in d2.items():
                for gate, d4 in d3.items():
                    for tg, e in sorted(d4.items(), key=lambda kv: float(kv[0])):
                        lines.append(f"| {s} | {n} | {ref} | {gate} | {tg} | {e['T_ns']:.0f} | {e['C']:.3f} | {e['F_T']:.4f} | "
                                     f"{e['F_loss']:.4f} | {e['path_F_loss']:.4f} | {e['w_star']:g} | {e['expo_ns']:.0f} | "
                                     f"{e['Lambda']:.2f} |")
    (OUT.parent / "fig4_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    print("capacity:", json.dumps(res["capacity"]))
    print("wrote", OUT, len(rows), "jobs")


if __name__ == "__main__":
    main()
