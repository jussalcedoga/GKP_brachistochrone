"""Time traces of the tracking pulses for the path-comparison figures (format of the protocol deck, slides 25-33).

For every selected job: the oscillator state rho(t) of the pulse (qubit traced out) on a fine grid (4 points per slice),
re-simulated at cutoff 90 (the level-3 pulses are optimized at 80; at 80 the CS flip pulses overstate F_T by up to 0.5%
through truncation, while 90 agrees with 100 and 110 to <= 6e-4, tools/cutoff_scan.py; the reference path is computed at
80 and padded), and
  x(t) = 2 Re<S|rho|chi>,  z(t) = <S|rho|S> - <chi|rho|chi>   (plane of the start and the brachistochrone direction chi;
                                                              a path state at angle a maps to (sin 2a, cos 2a))
  F_T(t) = <T|rho|T>, closeness F(t) = <p(t)|rho|p(t)> to the reference path at the same fraction of the path,
  w_in(t) = <S|rho|S> + <chi|rho|chi> (weight inside the plane), <n>(t);
the same for the reference path, and the same quantities under the Lindblad master equation with the Yale rates (single-
photon loss, cavity dephasing, qubit decay and dephasing; fields with suffix L, eight points per slice, t = 0 included):
tL_ns, xL, zL, FL, FTL, nbarL, wL.  The oscillator states at the snapshot fractions FR, closed and Lindblad, are saved
to results/rho80/<job id>.npz for the mean Wigner functions.  One pulse is also re-evaluated with self-Kerr
K = 100 s^-1 (the deck's value) to bound that omission.
Writes results/traces80/<job id>.json.   python simulations/tracking/traces.py [--workers 64]
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import argparse
import json
import multiprocessing as mp
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common import brachistochrone as B, grape, lindblad, tracking  # noqa: E402
import compute as CP  # noqa: E402

JOBS = HERE / "results" / "jobs80"          # level-3 pulses (cutoff 80); --src jobs for the cutoff-60 pulses
OUT = HERE / "results" / "traces80"
RHO = HERE / "results" / "rho80"
D = 90                                        # evaluation cutoff (directory names keep "80": the level-3 pulses)
D_PATH = 80
SUB = 8                                       # closed record points per slice
SUBL = 8                                      # Lindblad record points per slice (C converged to <= 5e-4, tools/c_grid_check.py)
FORCE = False
FR = (0.0, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0)   # snapshot fractions (Figs. 4, 5); exact for N = 50, 100


def osc_rho(psi):
    v = psi.reshape(2, D)
    return v[0][:, None] * v[0].conj()[None, :] + v[1][:, None] * v[1].conj()[None, :]


def run(jid):
    out = OUT / f"{jid}.json"
    if out.exists() and not FORCE:
        return jid, "skip"
    r = json.loads((JOBS / f"{jid}.json").read_text())
    jb = r["job"]
    g = jb["g"]
    M = tracking.make_model(jb["gate"], D, g, 10 * g)
    S, Tg = CP.transfer(jb["name"], D)
    c0, th, chi = B.geodesic_parts(S, Tg)
    path = tracking.Path(CP.pad(CP.reference_dense(jb["name"], jb["ref"], D_PATH), D))
    X = np.array(r["X"])
    N = len(X)
    T = r["T_us"]
    C, _ = M.controls(X)
    psi0 = grape.joint(S)
    traj, _, _ = M.propagate(np.repeat(C, SUB, axis=0), T / N / SUB, psi0)
    s = np.arange(N * SUB + 1) / (N * SUB)
    G, _ = path.at(s)
    x, z, F, FT, win, nb = [], [], [], [], [], []
    nv = np.arange(D)
    for k in range(len(traj)):
        rho = osc_rho(traj[k])
        Sr, Cr = rho @ S, rho @ chi
        x.append(float(2 * np.real(np.vdot(S, Cr))))
        z.append(float(np.real(np.vdot(S, Sr) - np.vdot(chi, Cr))))
        win.append(float(np.real(np.vdot(S, Sr) + np.vdot(chi, Cr))))
        F.append(float(np.real(np.vdot(G[k], rho @ G[k]))))
        FT.append(float(np.real(np.vdot(Tg, rho @ Tg))))
        nb.append(float(np.real(np.trace(rho * nv[None, :]))))
    xp = [float(2 * np.real(np.vdot(S, gk) * np.vdot(gk, chi))) for gk in G]
    zp = [float(abs(np.vdot(S, gk)) ** 2 - abs(np.vdot(chi, gk)) ** 2) for gk in G]
    FTp = [float(abs(np.vdot(Tg, gk)) ** 2) for gk in G]
    nbp = [float(np.real(np.vdot(gk, nv * gk))) for gk in G]
    noise = lindblad.rates("Yale" if jb["noise"] == "Yale" else jb["noise"])
    rho_T, rec = lindblad.evolve(M, np.repeat(C, SUBL, axis=0), T / N / SUBL, psi0, noise, dt_max=2e-3, record_every=1)
    recL = [osc_rho(psi0)] + rec                                  # oscillator states at t = j T / (SUBL N), j = 0..SUBL N
    sL = np.arange(N * SUBL + 1) / (N * SUBL)
    GL, _ = path.at(sL)
    L = dict(tL_ns=(1e3 * T * sL).tolist(), xL=[], zL=[], FL=[], FTL=[], nbarL=[], wL=[])
    for k, ro in enumerate(recL):
        Sr, Cr = ro @ S, ro @ chi
        L["xL"].append(float(2 * np.real(np.vdot(S, Cr))))
        L["zL"].append(float(np.real(np.vdot(S, Sr) - np.vdot(chi, Cr))))
        L["wL"].append(float(np.real(np.vdot(S, Sr) + np.vdot(chi, Cr))))
        L["FL"].append(float(np.real(np.vdot(GL[k], ro @ GL[k]))))
        L["FTL"].append(float(np.real(np.vdot(Tg, ro @ Tg))))
        L["nbarL"].append(float(np.real(np.trace(ro * nv[None, :]))))
    F_noisy = L["FL"][SUBL::SUBL]                                 # slice ends (kept for the older scripts)
    FT_noisy = L["FTL"][SUBL::SUBL]
    RHO.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RHO / f"{jid}.npz", fr=np.array(FR), T_ns=1e3 * T,
                        rho=np.array([osc_rho(traj[int(round(f * N * SUB))]) for f in FR], np.complex64),
                        rhoL=np.array([recL[int(round(f * N * SUBL))] for f in FR], np.complex64))
    res = dict(job=jb, T_ns=1e3 * T, N=N, D=D, t_ns=(1e3 * T * s).tolist(), x=x, z=z, F=F, F_T=FT, w_in=win, nbar=nb,
               x_path=xp, z_path=zp, F_T_path=FTp, nbar_path=nbp, C=float(min(F)), F_T_final=FT[-1],
               t_slices_ns=(1e3 * T * np.arange(1, N + 1) / N).tolist(), F_noisy=F_noisy, F_T_noisy=FT_noisy,
               F_T_final_noisy=FT_noisy[-1], C_noisy=float(min(L["FL"])), w_star=r["w_star"], **L,
               expo_L_ns=float(np.sum(0.5 * (np.array(L["nbarL"][1:]) + np.array(L["nbarL"][:-1])) * np.diff(L["tL_ns"]))),
               F_T_final_noisy_CS=lindblad.osc_fidelity(lindblad.evolve(M, C, T / N, psi0, lindblad.rates("CSproposal"),
                                                                         dt_max=2e-3), Tg),
               expo_ns=float(np.sum(0.5 * (np.array(nb[1:]) + np.array(nb[:-1])) * np.diff(1e3 * T * s))))
    if jb["name"] == "gkp" and jb["gate"] == "cs" and jb["ref"] == "geodesic" and jb["Tg"] == 30.0:
        nk = dict(noise)
        nk["kerr"] = 100.0 * 1e-6                          # K = 100 s^-1 in rad/us
        res["F_T_noisy_with_kerr"] = lindblad.osc_fidelity(lindblad.evolve(M, C, T / N, psi0, nk, dt_max=2e-3), Tg)
    out.write_text(json.dumps(res))
    return jid, f"C={res['C']:.3f} F_T={res['F_T_final']:.4f} noisy={res['F_T_final_noisy']:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--src", default="jobs80")
    ap.add_argument("--force", action="store_true", help="recompute existing traces")
    a = ap.parse_args()
    global JOBS, OUT, FORCE
    FORCE = a.force
    JOBS = HERE / "results" / a.src
    OUT = HERE / "results" / ("traces80" if a.src == "jobs80" else "traces")
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in JOBS.glob("*.json") if not p.stem.startswith("val_") and (FORCE or not (OUT / p.name).exists()))
    ids.sort(key=lambda i: -(float(i.split("_Tg")[1]) if "_Tg" in i else (2.58, 3.09, 6.18, 16.0)[int(i.split("_k")[1][0])]))
    if not ids:
        return
    with mp.Pool(min(a.workers, len(ids))) as pool:
        for i, (jid, st) in enumerate(pool.imap_unordered(run, ids)):
            print(f"{i + 1}/{len(ids)} {jid} {st}", flush=True)


if __name__ == "__main__":
    main()
