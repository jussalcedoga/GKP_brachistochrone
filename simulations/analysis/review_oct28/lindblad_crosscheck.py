#!/usr/bin/env python3
"""Independent cross-check of the Lindblad numerics behind the tracking benchmark (Figs. 4-5), review of Oct 28.

Independent of manuscript/common (nothing from its propagation, GRAPE or Lindblad code is imported):
  * Hamiltonians rebuilt from the formulas of the Supplement (Secs. "Gate sets with equal resources" and "Following a
    path"): qubit drive (1/2)(Omega_x s_x + Omega_y s_y), Omega = 10 g r e^{i phi};
      CS   sum_q |q><q| (x) (i/2)(xi_q^* a^2 - xi_q a^dag^2),        xi_q  = g a_q e^{i f_q}
      ECD  sum_q |q><q| (x) (i/sqrt2)(eps_q^* a - eps_q a^dag),       eps_q = g a_q e^{i f_q}
      SNAP -chi n |e><e| + 1 (x) (i/sqrt2)(eps^* a - eps a^dag),     eps = g a_0 e^{i f_0}, chi = g, (a_1, f_1) unused
    joint ordering qubit (x) oscillator, |g> = index 0, sigma_- = |g><e|, g = 7.5 / us, Fock cutoff 90.
  * Lindblad equation [main text Eq. (7)]: kappa D[a] + kappa_phi D[n] + gamma_1 D[sigma_-] + (gamma_phi/2) D[sigma_z],
    kappa = 1/606, kappa_phi = 2/5100, gamma_1 = 1/280, gamma_phi = 1/414 (1/us).
  * GKP code words and the brachistochrone reference rebuilt from their definitions (coherent states by recursion).
Methods:
  A  (main) per slice the exact propagator of the vectorized generator, vec(rho(t)) = exp(L_k t) vec(rho), with
     scipy.sparse.linalg.expm_multiply (a-priori truncation control at double precision; no splitting, no Kraus cut-off);
     the closed evolution with expm_multiply of -iH on the state vector;
  A' (convergence) the same with every record interval split into 2 chained expm_multiply calls;
  B  (convergence, different method) the matrix-form master equation, adaptive DOP853 at two tolerances.
  Detection power: method A rerun with one deliberately wrong dissipator convention at a time (VARIANTS), to show
  that the observed agreement excludes such slips.
Read from the manuscript (data only): run selection (mfcommon.groups / representative / trivial), the pulses
results/jobs80/<id>.json, the stored traces results/traces80/<id>.json and states results/rho80/<id>.npz.
Writes only lindblad_crosscheck.json next to this file (nothing with --smoke).

Usage (repository root):
  python simulations/analysis/review_oct28/lindblad_crosscheck.py [--workers 12]
  ... --smoke 3        first 3 slices of the CS 412 ns pulse only, printed, nothing written
"""
from __future__ import annotations
import layout as _L  # repository layout: data/ and output/ (src/layout.py)

import argparse
import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply

HERE = _L.vfile(__file__).parent
REPO = HERE.parents[2]
RES = REPO / "manuscript" / "fig4_cs_path_following" / "results"
OUT_JSON = HERE / "lindblad_crosscheck.json"

G = 7.5                                   # 1/us
D = 90                                    # evaluation cutoff of every reported number
D_PATH = 80                               # the reference path is built at 80 and zero-padded (as in the stored traces)
SUBL = 8                                  # record points per slice (stored traces: t = j T/(8N), j = 0..8N)
DELTA = 0.4
RATES = dict(kappa=1 / 606.0, kphi=2 / 5100.0, g1=1 / 280.0, gphi=1 / 414.0)


# ----------------------------------------------------------------------------- states and reference path
def coherent(gam, n_max):
    """Fock amplitudes of |gam> by the recursion c_n = c_{n-1} gam / sqrt(n), c_0 = exp(-|gam|^2/2)"""
    c = np.zeros(n_max, complex)
    c[0] = math.exp(-abs(gam) ** 2 / 2)
    for n in range(1, n_max):
        c[n] = c[n - 1] * gam / math.sqrt(n)
    return c


def gkp(mu, dim, Delta=DELTA, M=10):
    """|mu_L> ~ exp(-Delta^2 n) sum_{m = mu mod 2, l; |m|,|l| <= M} exp(-i pi m l/2) D(m alpha + i l alpha)|0>,
    alpha = sqrt(pi/2); truncated to dim, normalized, first non-negligible amplitude real positive"""
    al = math.sqrt(math.pi / 2)
    psi = np.zeros(dim, complex)
    for m in range(-M, M + 1):
        if (m - mu) % 2:
            continue
        for l in range(-M, M + 1):
            psi += np.exp(-1j * math.pi * m * l / 2) * coherent(al * (m + 1j * l), dim)
    psi *= np.exp(-Delta ** 2 * np.arange(dim))
    psi /= np.linalg.norm(psi)
    k = int(np.argmax(np.abs(psi) > 1e-12))
    return psi * np.exp(-1j * np.angle(psi[k]))


def vac(dim):
    v = np.zeros(dim, complex)
    v[0] = 1
    return v


def transfer(name, dim):
    return (gkp(0, dim), gkp(1, dim)) if name == "flip" else (vac(dim), gkp(0, dim))


def geodesic(name):
    """(S, chi, theta0) of the brachistochrone at cutoff 80, padded to 90: p(s) = cos(s theta0) S + sin(s theta0) chi"""
    S, T = transfer(name, D_PATH)
    ov = np.vdot(S, T)
    Tg = T * np.exp(-1j * np.angle(ov))
    c0 = abs(ov)
    chi = Tg - c0 * S
    chi /= np.linalg.norm(chi)
    pad = lambda v: np.r_[v, np.zeros(D - D_PATH)]        # noqa: E731
    return pad(S), pad(chi), math.acos(min(c0, 1.0))


# ----------------------------------------------------------------------------- operators, Hamiltonian, generator
def csr(M):
    return sp.csr_matrix(np.asarray(M, complex))


A_OSC = np.diag(np.sqrt(np.arange(1, D)), 1).astype(complex)
AD_OSC = A_OSC.conj().T
N_OSC = np.diag(np.arange(D)).astype(complex)
I_D, I_2 = np.eye(D), np.eye(2)
P_Q = [np.diag([1.0, 0.0]).astype(complex), np.diag([0.0, 1.0]).astype(complex)]     # |g><g|, |e><e|
SM = np.array([[0, 1], [0, 0]], complex)                                               # sigma_- = |g><e|
SZ = np.diag([1.0, -1.0]).astype(complex)


def hamiltonian(gate, x):
    """sparse joint Hamiltonian (1/us) of one slice with variables x = (r, phi, a0, f0, a1, f1)"""
    r, phi, a0, f0, a1, f1 = x
    Om = 10 * G * r * np.exp(1j * phi)
    Hq = 0.5 * np.array([[0, np.conj(Om)], [Om, 0]])      # (1/2)(Om_x s_x + Om_y s_y), |g> = index 0
    H = sp.kron(csr(Hq), csr(I_D))
    if gate == "cs":
        for q, (aq, fq) in enumerate(((a0, f0), (a1, f1))):
            xi = G * aq * np.exp(1j * fq)
            H = H + sp.kron(csr(P_Q[q]), csr(0.5j * (np.conj(xi) * A_OSC @ A_OSC - xi * AD_OSC @ AD_OSC)))
    elif gate == "cd":
        for q, (aq, fq) in enumerate(((a0, f0), (a1, f1))):
            eps = G * aq * np.exp(1j * fq)
            H = H + sp.kron(csr(P_Q[q]), csr((1j / math.sqrt(2)) * (np.conj(eps) * A_OSC - eps * AD_OSC)))
    elif gate == "snap":
        eps = G * a0 * np.exp(1j * f0)
        H = H + sp.kron(csr(I_2), csr((1j / math.sqrt(2)) * (np.conj(eps) * A_OSC - eps * AD_OSC)))
        H = H - G * sp.kron(csr(P_Q[1]), csr(N_OSC))        # chi = g
    else:
        raise ValueError(gate)
    H = H.tocsr()
    assert abs(H - H.conj().T).max() < 1e-12
    return H


VARIANTS = {   # deliberately wrong conventions (detection-power test only; None = Eq. (7))
    "gphi_Dsz_full": "gamma_phi D[sigma_z] instead of (gamma_phi/2) D[sigma_z] (qubit coherence decays at 2/T_phi,q)",
    "kphi_1_over_Tphic": "kappa_phi = 1/T_phi,c instead of 2/T_phi,c",
    "sigma_plus": "D[sigma_+] instead of D[sigma_-] (qubit basis flipped in the decay)",
}


def jump_ops(variant=None):
    """(operator, rate) of Eq. (7), joint space; `variant` switches on one deliberately wrong convention"""
    sm = SM.T if variant == "sigma_plus" else SM
    kphi = RATES["kphi"] / 2 if variant == "kphi_1_over_Tphic" else RATES["kphi"]
    gz = RATES["gphi"] if variant == "gphi_Dsz_full" else RATES["gphi"] / 2
    return [(sp.kron(csr(I_2), csr(A_OSC)).tocsr(), RATES["kappa"]),
            (sp.kron(csr(I_2), csr(N_OSC)).tocsr(), kphi),
            (sp.kron(csr(sm), csr(I_D)).tocsr(), RATES["g1"]),
            (sp.kron(csr(SZ), csr(I_D)).tocsr(), gz)]


def vec_dissipator(variant=None):
    """row-major vectorization, vec(A rho B) = (A (x) B^T) vec(rho):
    D[L] -> L (x) L^* - (1/2) L^dag L (x) 1 - (1/2) 1 (x) (L^dag L)^T"""
    d = 2 * D
    Id = sp.identity(d, complex, format="csr")
    LD = sp.csr_matrix((d * d, d * d), dtype=complex)
    for L, rate in jump_ops(variant):
        LdL = (L.conj().T @ L).tocsr()
        LD = LD + rate * (sp.kron(L, L.conj()) - 0.5 * sp.kron(LdL, Id) - 0.5 * sp.kron(Id, LdL.T))
    return LD.tocsr()


def vec_hamiltonian(H):
    d = H.shape[0]
    Id = sp.identity(d, complex, format="csr")
    return (-1j * (sp.kron(H, Id) - sp.kron(Id, H.T))).tocsr()


class MatrixRHS:
    """d rho/dt of Eq. (7) in matrix form (method B): -i[H, rho] + sum rate (L rho L^dag - {L^dag L, rho}/2)"""

    def __init__(self, H):
        self.d = H.shape[0]
        self.H, self.HT = H.tocsr(), H.T.tocsr()
        self.ops = []
        for L, rate in jump_ops():
            LdL = (L.conj().T @ L).tocsr()
            self.ops.append((L, L.conj().tocsr(), LdL, LdL.T.tocsr(), rate))

    def __call__(self, t, y):
        R = y.reshape(self.d, self.d)
        out = -1j * (self.H @ R - (self.HT @ R.T).T)
        for L, Lc, LdL, LdLT, rate in self.ops:
            LR = L @ R
            out = out + rate * ((Lc @ LR.T).T - 0.5 * (LdL @ R) - 0.5 * (LdLT @ R.T).T)
        return np.asarray(out).reshape(-1)


# ----------------------------------------------------------------------------- one pulse
class Pulse:
    def __init__(self, jid, nslices=None):
        r = json.loads((RES / "jobs80" / f"{jid}.json").read_text())
        jb = r["job"]
        self.id, self.gate, self.name = jid, jb["gate"], jb["name"]
        assert jb["ref"] == "geodesic" and abs(jb["g"] - G) < 1e-12 and jb["noise"] == "Yale"
        self.X = np.array(r["X"])
        self.N_full = len(self.X)
        self.T = float(r["T_us"])
        self.dt = self.T / self.N_full
        self.N = self.N_full if nslices is None else nslices
        S90, T90 = transfer(self.name, D)
        self.psi0 = np.r_[S90, np.zeros(D, complex)]        # |S> (x) |g>
        self.target = T90
        self.Sp, self.chi, self.theta = geodesic(self.name)
        self.nv = np.arange(D)
        self.Hs = [hamiltonian(self.gate, x) for x in self.X[:self.N]]
        self.snap_idx = {}

    def path_state(self, j):
        s = j / (self.N_full * SUBL)
        return math.cos(s * self.theta) * self.Sp + math.sin(s * self.theta) * self.chi

    def measure(self, j, ro, rec):
        p = self.path_state(j)
        rec["F"].append(float(np.real(np.vdot(p, ro @ p))))
        rec["FT"].append(float(np.real(np.vdot(self.target, ro @ self.target))))
        rec["nbar"].append(float(np.real(np.sum(self.nv * np.diag(ro)))))
        if j in self.snap_idx:
            rec["snaps"][self.snap_idx[j]] = ro.copy()


def osc_of_vec(v):
    R = v.reshape(2 * D, 2 * D)
    return R[:D, :D] + R[D:, D:]


def new_rec(n_snap):
    return dict(F=[], FT=[], nbar=[], snaps=[None] * n_snap)


def run_lindblad_expm(P, split=1, variant=None):
    """method A (split = 1) and A' (each record interval as `split` chained calls)"""
    LD = vec_dissipator(variant)
    rec = new_rec(len(P.snap_idx))
    v = np.outer(P.psi0, P.psi0.conj()).reshape(-1)
    P.measure(0, osc_of_vec(v), rec)
    for k in range(P.N):
        Lk = (vec_hamiltonian(P.Hs[k]) + LD).tocsr()
        if split == 1:
            Y = expm_multiply(Lk, v, start=0.0, stop=P.dt, num=SUBL + 1, endpoint=True)
            for j in range(1, SUBL + 1):
                P.measure(k * SUBL + j, osc_of_vec(Y[j]), rec)
            v = Y[-1]
        else:
            Lh = (Lk * (P.dt / (SUBL * split))).tocsr()
            for j in range(1, SUBL * split + 1):
                v = expm_multiply(Lh, v)
                if j % split == 0:
                    P.measure(k * SUBL + j // split, osc_of_vec(v), rec)
    R = v.reshape(2 * D, 2 * D)
    Rh = 0.5 * (R + R.conj().T)
    rec["trace_minus_1"] = float(abs(np.trace(R) - 1))
    rec["min_eig_joint"] = float(np.linalg.eigvalsh(Rh)[0])
    rec["herm_err"] = float(np.max(np.abs(R - R.conj().T)))
    return rec


def run_lindblad_dop(P, rtol, atol):
    """method B: adaptive DOP853 on the matrix-form master equation, restarted at every slice"""
    rec = new_rec(len(P.snap_idx))
    y = np.outer(P.psi0, P.psi0.conj()).reshape(-1)
    P.measure(0, osc_of_vec(y), rec)
    nfev = 0
    for k in range(P.N):
        f = MatrixRHS(P.Hs[k])
        sol = solve_ivp(f, (0.0, P.dt), y, method="DOP853", t_eval=np.linspace(0.0, P.dt, SUBL + 1), rtol=rtol,
                        atol=atol)
        assert sol.success, sol.message
        nfev += sol.nfev
        for j in range(1, SUBL + 1):
            P.measure(k * SUBL + j, osc_of_vec(sol.y[:, j]), rec)
        y = sol.y[:, -1]
    rec["nfev"] = int(nfev)
    return rec


def run_closed(P):
    rec = new_rec(len(P.snap_idx))
    psi = P.psi0.copy()

    def osc(ps):
        v = ps.reshape(2, D)
        return np.outer(v[0], v[0].conj()) + np.outer(v[1], v[1].conj())
    P.measure(0, osc(psi), rec)
    for k in range(P.N):
        Y = expm_multiply((-1j * P.Hs[k]).tocsr(), psi, start=0.0, stop=P.dt, num=SUBL + 1, endpoint=True)
        for j in range(1, SUBL + 1):
            P.measure(k * SUBL + j, osc(Y[j]), rec)
        psi = Y[-1]
    rec["norm_minus_1"] = float(abs(np.linalg.norm(psi) - 1))
    return rec


def worker(task):
    t0 = time.time()
    P = Pulse(task["id"], task.get("nslices"))
    fr = np.load(RES / "rho80" / f"{task['id']}.npz")["fr"]
    P.snap_idx = {int(round(f * P.N_full * SUBL)): i for i, f in enumerate(fr)}
    out = dict(task=task)
    if task["mode"] == "main":
        out["lind"] = run_lindblad_expm(P, 1)
        out["closed"] = run_closed(P)
    elif task["mode"] == "fine":
        out["lind"] = run_lindblad_expm(P, 2)
    elif task["mode"] == "dop":
        out["lind"] = run_lindblad_dop(P, task["rtol"], task["atol"])
    elif task["mode"] == "sens":
        out["lind"] = run_lindblad_expm(P, 1, task["variant"])
    out["wall_s"] = time.time() - t0
    return out


# ----------------------------------------------------------------------------- self-tests (conventions, generator)
def self_tests():
    rng = np.random.default_rng(0)
    out = {}
    d = 2 * D
    LD = vec_dissipator()
    for gate in ("cs", "cd", "snap"):
        x = np.r_[rng.uniform(0, 1), rng.uniform(-3, 3), rng.uniform(0, 1), rng.uniform(-3, 3), rng.uniform(0, 1),
                  rng.uniform(-3, 3)]
        H = hamiltonian(gate, x)
        Z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        R = Z @ Z.conj().T
        R /= np.trace(R)
        a = (vec_hamiltonian(H) + LD) @ R.reshape(-1)
        b = MatrixRHS(H)(0.0, R.reshape(-1))
        Hd = H.toarray()                                     # dense reference of the generator, plain formula
        c = -1j * (Hd @ R - R @ Hd)
        for L, rate in jump_ops():
            Ld = L.toarray()
            c = c + rate * (Ld @ R @ Ld.conj().T - 0.5 * (Ld.conj().T @ Ld @ R + R @ Ld.conj().T @ Ld))
        out[f"generator_{gate}_vec_vs_dense"] = float(np.max(np.abs(a - c.reshape(-1))))
        out[f"generator_{gate}_matrixRHS_vs_dense"] = float(np.max(np.abs(b - c.reshape(-1))))
        out[f"generator_{gate}_trace_of_drho"] = float(abs(np.trace(c)))
    # targets against the manuscript's own construction (states.py only; a check of conventions, not of integration)
    try:
        sys.path.insert(0, str(REPO / "manuscript"))
        from common import states as ST                     # noqa: E402
        out["gkp0_D90_vs_manuscript"] = float(np.max(np.abs(gkp(0, 90) - ST.gkp(DELTA, 0, 90))))
        out["gkp1_D90_vs_manuscript"] = float(np.max(np.abs(gkp(1, 90) - ST.gkp(DELTA, 1, 90))))
        out["gkp0_D80_vs_manuscript"] = float(np.max(np.abs(gkp(0, 80) - ST.gkp(DELTA, 0, 80))))
    except Exception as e:                                   # noqa: BLE001
        out["targets_vs_manuscript_error"] = repr(e)
    return out


# ----------------------------------------------------------------------------- selection and comparison
def select_runs():
    import mfcommon as mf
    Gr = mf.groups()
    sel = []

    def add(label, key, pool=None):
        lst = Gr[key] if pool is None else pool
        r = mf.representative(lst)
        sel.append(dict(label=label, id=r["job"]["id"], group=list(key), n_starts=len(Gr[key]),
                        n_trivial_in_group=mf.n_trivial(Gr[key]), trivial=mf.trivial(r), m=r["job"].get("m"),
                        rule="representative of the group" if pool is None else
                        "representative among the non-trivial starts of the group"))
    add("CS, GKP prep, 412 ns", ("gkp", "geodesic", "cs", 1))
    add("ECD, GKP prep, 412 ns", ("gkp", "geodesic", "cd", 1))
    add("SNAP, GKP prep, 412 ns (Fig. 4 rep.)", ("gkp", "geodesic", "snap", 1))
    key = ("gkp", "geodesic", "snap", 1)
    add("SNAP, GKP prep, 412 ns (non-trivial)", key, [r for r in Gr[key] if not mf.trivial(r)])
    add("CS, GKP prep, 2133 ns", ("gkp", "geodesic", "cs", 3))
    add("CS, flip |0_L>->|1_L>, m = 2.81", ("flip", "geodesic", "cs", 2))
    assert abs(sel[-1]["m"] - 2.81) < 0.01, sel[-1]["m"]
    assert sel[2]["trivial"] and not sel[3]["trivial"]
    return sel


def maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def tdist(A, B):
    M = A - B
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(0.5 * (M + M.conj().T)))))


def h_sub(T, N, per_slice):
    """sub-step of the stored (split-step) evaluation: interval T/(N per_slice) cut into steps <= 2 ns"""
    dt = T / N / per_slice
    return dt / max(1, math.ceil(dt / 2e-3 - 1e-9))


def compare(sel_item, main, fine, dops, sens=()):
    jid = sel_item["id"]
    tr = json.loads((RES / "traces80" / f"{jid}.json").read_text())
    jb = json.loads((RES / "jobs80" / f"{jid}.json").read_text())
    z = np.load(RES / "rho80" / f"{jid}.npz")
    L, C = main["lind"], main["closed"]
    n = len(L["F"])
    assert n == len(tr["FL"]), (n, len(tr["FL"]))
    T, N = jb["T_us"], jb["N"]
    h1, h2 = h_sub(T, N, SUBL), h_sub(T, N, 1)
    F1, F2 = tr["F_T_final_noisy"], jb["F_loss_D90"]
    rich = F1 - (F2 - F1) * h1 / (h2 - h1)
    snapsL = [tdist(L["snaps"][i], z["rhoL"][i].astype(complex)) for i in range(len(z["fr"]))]
    snapsC = [tdist(C["snaps"][i], z["rho"][i].astype(complex)) for i in range(len(z["fr"]))]
    floor = max(tdist(s, s.astype(np.complex64).astype(complex)) for s in L["snaps"])
    q = dict(
        F_T_noisy=dict(stored=tr["F_T_final_noisy"], independent=L["FT"][-1],
                       absdiff=abs(tr["F_T_final_noisy"] - L["FT"][-1]),
                       stored_jobs80_F_loss_D90=F2, absdiff_jobs80=abs(F2 - L["FT"][-1]),
                       stored_substep_ns=1e3 * h1, jobs80_substep_ns=1e3 * h2,
                       richardson_first_order=rich, absdiff_richardson=abs(rich - L["FT"][-1])),
        C_noisy=dict(stored=tr["C_noisy"], independent=min(L["F"]), absdiff=abs(tr["C_noisy"] - min(L["F"])),
                     argmin_stored_ns=tr["tL_ns"][int(np.argmin(tr["FL"]))],
                     argmin_independent_ns=tr["tL_ns"][int(np.argmin(L["F"]))]),
        FL_trace=dict(n_points=n, max_absdiff=maxabs(tr["FL"], L["F"])),
        FTL_trace=dict(max_absdiff=maxabs(tr["FTL"], L["FT"])),
        nbarL_trace=dict(max_absdiff=maxabs(tr["nbarL"], L["nbar"])),
        F_T_closed=dict(stored=tr["F_T_final"], independent=C["FT"][-1], absdiff=abs(tr["F_T_final"] - C["FT"][-1]),
                        stored_jobs80_check_D90=jb["check_D90"]["F_T"],
                        absdiff_jobs80=abs(jb["check_D90"]["F_T"] - C["FT"][-1])),
        C_closed=dict(stored=tr["C"], independent=min(C["F"]), absdiff=abs(tr["C"] - min(C["F"]))),
        F_closed_trace=dict(max_absdiff=maxabs(tr["F"], C["F"])),
        F_T_closed_trace=dict(max_absdiff=maxabs(tr["F_T"], C["FT"])),
        rho80_lindblad_trace_distance=dict(fr=z["fr"].tolist(), per_snapshot=snapsL, at_T=snapsL[-1], max=max(snapsL)),
        rho80_closed_trace_distance=dict(per_snapshot=snapsC, at_T=snapsC[-1], max=max(snapsC)),
        complex64_storage_floor=floor,
        start_checks=dict(FTL0_stored=tr["FTL"][0], FTL0_independent=L["FT"][0], FL0_stored=tr["FL"][0],
                          FL0_independent=L["F"][0]),
        integrity=dict(trace_minus_1=L["trace_minus_1"], min_eig_joint=L["min_eig_joint"], herm_err=L["herm_err"],
                       closed_norm_minus_1=C["norm_minus_1"]),
        wall_s_main=main["wall_s"],
        T_ns=1e3 * T, N=N, gate=jb["job"]["gate"], name=jb["job"]["name"], w_star=jb["w_star"])
    conv = {}
    if fine is not None:
        Fl = fine["lind"]
        conv["expm_split2_vs_main"] = dict(F_T=abs(Fl["FT"][-1] - L["FT"][-1]), FL_trace=maxabs(Fl["F"], L["F"]),
                                           C=abs(min(Fl["F"]) - min(L["F"])),
                                           rho_T=tdist(Fl["snaps"][-1], L["snaps"][-1]), wall_s=fine["wall_s"])
    for dp in dops:
        Dl = dp["lind"]
        conv[f"dop853_rtol{dp['task']['rtol']:g}_vs_main"] = dict(
            F_T=abs(Dl["FT"][-1] - L["FT"][-1]), FL_trace=maxabs(Dl["F"], L["F"]), C=abs(min(Dl["F"]) - min(L["F"])),
            rho_T=tdist(Dl["snaps"][-1], L["snaps"][-1]), nfev=Dl["nfev"], wall_s=dp["wall_s"])
    if len(dops) == 2:
        a, b = (d["lind"] for d in dops)
        conv["dop853_loose_vs_tight"] = dict(F_T=abs(a["FT"][-1] - b["FT"][-1]), FL_trace=maxabs(a["F"], b["F"]),
                                             rho_T=tdist(a["snaps"][-1], b["snaps"][-1]))
    if conv:
        q["convergence"] = conv
    if sens:
        q["detection_power"] = {
            sv["task"]["variant"]: dict(description=VARIANTS[sv["task"]["variant"]],
                                        F_T=sv["lind"]["FT"][-1], dF_T=abs(sv["lind"]["FT"][-1] - L["FT"][-1]),
                                        dC=abs(min(sv["lind"]["F"]) - min(L["F"])),
                                        dFL_trace=maxabs(sv["lind"]["F"], L["F"]))
            for sv in sens}
    q["independent_traces"] = dict(FL=L["F"], FTL=L["FT"], nbarL=L["nbar"], F_closed=C["F"], F_T_closed=C["FT"])
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", type=int, default=0, help="first K slices of the CS 412 ns pulse, print only")
    a = ap.parse_args()
    t0 = time.time()
    if a.smoke:
        jid = "seed_gkp_geodesic_cs_k1_s11"
        tr = json.loads((RES / "traces80" / f"{jid}.json").read_text())
        st = self_tests()
        print(json.dumps(st, indent=1), flush=True)
        for task in (dict(id=jid, mode="main", nslices=a.smoke), dict(id=jid, mode="fine", nslices=a.smoke),
                     dict(id=jid, mode="dop", rtol=1e-10, atol=1e-12, nslices=a.smoke)):
            o = worker(task)
            n = len(o["lind"]["F"])
            msg = f"{task['mode']}: {o['wall_s']:.1f}s, FL max|d| {maxabs(tr['FL'][:n], o['lind']['F']):.2e}, " \
                  f"FTL max|d| {maxabs(tr['FTL'][:n], o['lind']['FT']):.2e}"
            if "closed" in o:
                msg += f", closed F max|d| {maxabs(tr['F'][:n], o['closed']['F']):.2e}"
            if "nfev" in o["lind"]:
                msg += f", nfev {o['lind']['nfev']}"
            print(msg, flush=True)
        return
    sel = select_runs()
    print("selected:", [s["id"] for s in sel], flush=True)
    tasks = [dict(id=s["id"], mode="main") for s in sel]
    fine_ids = [sel[0]["id"], sel[4]["id"], sel[5]["id"]]          # CS 412, CS 2133, CS flip
    tasks += [dict(id=i, mode="fine") for i in fine_ids]
    dop_ids = [sel[0]["id"], sel[5]["id"]]                         # CS 412, CS flip (most photons)
    tasks += [dict(id=i, mode="dop", rtol=rt, atol=at) for i in dop_ids for rt, at in ((1e-7, 1e-9), (1e-10, 1e-12))]
    tasks += [dict(id=i, mode="sens", variant=v) for i in dop_ids for v in VARIANTS]
    tasks.sort(key=lambda t: -(2 if "k3" in t["id"] else 1) * (3 if t["mode"] == "dop" else 1))
    with mp.Pool(min(a.workers, len(tasks))) as pool:
        results = []
        for o in pool.imap_unordered(worker, tasks):
            print(f"[{time.time() - t0:7.1f}s] done {o['task']} in {o['wall_s']:.1f}s", flush=True)
            results.append(o)
    st = self_tests()
    out = dict(meta=dict(date="2026-10-28", D=D, D_path=D_PATH, SUBL=SUBL, g=G, Delta=DELTA, rates=RATES,
                         numpy=np.__version__, scipy=scipy.__version__, wall_s=time.time() - t0,
                         methods=dict(A="expm_multiply of the vectorized Lindblad generator per slice (9 points per call)",
                                      A2="same, each record interval as 2 chained expm_multiply calls",
                                      B="DOP853 on the matrix-form master equation, restarted per slice")),
               self_tests=st, runs=[])
    for s in sel:
        m = next(o for o in results if o["task"]["id"] == s["id"] and o["task"]["mode"] == "main")
        f = next((o for o in results if o["task"]["id"] == s["id"] and o["task"]["mode"] == "fine"), None)
        dps = sorted([o for o in results if o["task"]["id"] == s["id"] and o["task"]["mode"] == "dop"],
                     key=lambda o: -o["task"]["rtol"])
        sns = [o for o in results if o["task"]["id"] == s["id"] and o["task"]["mode"] == "sens"]
        out["runs"].append(dict(selection=s, **compare(s, m, f, dps, sns)))
    OUT_JSON.write_text(json.dumps(out, indent=1))
    for r in out["runs"]:
        print(r["selection"]["label"], r["selection"]["id"],
              {k: r[k]["absdiff"] for k in ("F_T_noisy", "C_noisy", "F_T_closed", "C_closed")},
              "FL", r["FL_trace"]["max_absdiff"], "TD", r["rho80_lindblad_trace_distance"]["max"], flush=True)
    print(f"wrote {OUT_JSON} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
