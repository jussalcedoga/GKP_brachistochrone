"""Flip coincidence, numerics (Nov 2 2026).

Claim checked.  Among logical flips that keep the qubit encoded (U(t)C = C for all t), the photon number averaged over
the code space is constant in time (Proposition S5), so the averaged exposure is nbar_C T and the averaged first-order
error is T (Gamma_q + kappa nbar_C + kappa_phi V_C).  The photon problem and the error problem therefore reduce to the
time problem, whose solution is the flip brachistochrone omega Y_L at T* = theta/omega.

Sections (every number goes to num_facts.json with the section that produced it)
 A  code space at Delta = 0.4, cutoffs 80 and 90: c, theta, nbar_C, V_C, Bloch form Pi n Pi = nbar_C 1 + b.sigma on C
 B  flip brachistochrone U = exp(-i omega t Y_L): nbar(t) of the six cardinal inputs (SI definition with the
    non-orthogonal |0_L>, |1_L>), of the orthonormal octahedron (a 2-design), of one orthonormal basis, of Haar-random
    inputs, and the exact average 1/2 Tr(U Pi U^dag n); per-input exposures; Var n averages
 C  other flips that keep the qubit encoded: slower brachistochrones, two-leg great-circle detours, random smooth
    wiggles exp(-i Theta(s) Y_L) exp(-i Phi(s) m.sigma) rescaled to the speed limit (versions S1 and S2), the S1 frame
    family exp(-i omega t Y_L) exp(-i chi(t) Z_L/2), and one flip propagated in the full Fock space with an arbitrary
    time-dependent Hamiltonian on the complement of C
 D  tilted flips that leave C (SI Sec. IV.5, formula of the SI), several lambda, cutoffs 80 and 90 at lambda = 0.3
 E  photon-optimal transfer of |0_L> through the vacuum (stored nodes behind Supplementary Table route), applied to the
    whole code space: exposures, average over the code space, gate fidelity
 F  single-input photon optimum within C (|0_L> -> |1_L>, and a generic input), discretized curves on the Bloch sphere
 G  Lindblad check: flips of equal duration that keep the qubit encoded but differ in shape have the same averaged error
    (loss and cavity dephasing, six octahedron states, toggling frame, RK4)
Writes num_facts.json and num_curves.npz next to this script.  Run:
  python simulations/flip_note/num_core.py
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import math
import pathlib
import pickle
import sys
import time

import numpy as np
from scipy.integrate import simpson
from scipy.linalg import eigh, expm
from scipy.optimize import brentq, minimize
from scipy.special import ellipe, ellipeinc

from common import states
from common.params import G_SQ, NOISE, kappa_mu

T0 = time.time()
HERE = _L.vfile(__file__).parent
ROOT = HERE.parents[1]
SCRIPT = "simulations/flip_note/num_core.py"
OM = G_SQ / math.sqrt(2)                       # 1/us
KAP, GQ = kappa_mu("Yale")                     # 1/us
KPHI = 2.0 / NOISE["Yale"]["Tphic"]            # kappa_phi = 2/T_phi,c (SI Table params)
DELTA = 0.4
RNG = np.random.default_rng(20261102)
FACTS = dict(_about=dict(script=SCRIPT, units="times ns, exposures photon ns (int nbar dt), omega = g/sqrt2",
                         omega_per_us=OM, kappa_per_us=KAP, kappa_phi_per_us=KPHI, Gamma_q_per_us=GQ,
                         Gamma_q_inverse_us=1 / GQ))
CURVES = {}
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
I2 = np.eye(2, dtype=complex)
PAULI = (SX, SY, SZ)


def log(*a):
    print(f"[{time.time() - T0:7.1f} s]", *a, flush=True)


def put(sec, key, value, check=""):
    FACTS.setdefault(sec, dict(_script=f"{SCRIPT} section {sec}"))
    FACTS[sec][key] = value if not check else dict(value=value, check=check)


def rot2(phi, m):
    """exp(-i phi m.sigma) for a unit 3-vector m"""
    m = np.asarray(m, float)
    return math.cos(phi) * I2 - 1j * math.sin(phi) * (m[0] * SX + m[1] * SY + m[2] * SZ)


def bloch(al):
    al = np.asarray(al, complex)
    al = al / np.linalg.norm(al)
    z = np.conj(al[0]) * al[1]
    return np.array([2 * z.real, 2 * z.imag, abs(al[0]) ** 2 - abs(al[1]) ** 2])


def bloch_rot(u):
    """3x3 rotation of Bloch vectors induced by the 2x2 unitary u"""
    return np.array([[0.5 * np.real(np.trace(PAULI[i] @ u @ PAULI[j] @ u.conj().T)) for j in range(3)] for i in range(3)])


# ===================================================================================== A  code space
def code(D):
    g0, g1 = states.gkp(DELTA, 0, D), states.gkp(DELTA, 1, D)
    assert np.abs(g0.imag).max() < 1e-13 and np.abs(g1.imag).max() < 1e-13
    u0, u1 = g0.real.copy(), g1.real.copy()
    c = float(u0 @ u1)
    th = math.acos(c)
    s = math.sin(th)
    e1 = (u1 - c * u0) / s
    Q = np.array([u0, e1]).T                                  # D x 2, orthonormal basis {|0_L>, |1_L^perp>}
    n = np.arange(D, dtype=float)
    M = Q.T @ (n[:, None] * Q)                                # Pi n Pi on C
    M2 = Q.T @ (n[:, None] ** 2 * Q)
    nC = 0.5 * np.trace(M)
    VC = 0.5 * np.trace(M2) - (np.trace(M) ** 2 + np.trace(M @ M)) / 6
    b = np.array([M[0, 1], 0.0, 0.5 * (M[0, 0] - M[1, 1])])
    Pi = Q @ Q.T
    Ncom = Pi * n[None, :] - n[:, None] * Pi                  # [Pi, n]
    sv = np.linalg.svd(Ncom, compute_uv=False)
    top = float(np.sum(u0[int(0.75 * D):] ** 2) + np.sum(u1[int(0.75 * D):] ** 2))
    return dict(D=D, u0=u0, u1=u1, c=c, th=th, s=s, e1=e1, Q=Q, n=n, M=M, M2=M2, nC=float(nC), VC=float(VC), b=b, Pi=Pi,
                comm_op=float(sv[0]), comm_hs2=float(np.sum(sv ** 2)), top_quarter_weight=top,
                XL=np.array([[c, s], [s, -c]], complex))


log("A: code space")
CODES = {D: code(D) for D in (80, 90)}
for D, C in CODES.items():
    put("A", f"D{D}", dict(c=C["c"], theta=C["th"], theta_over_half_pi=C["th"] / (math.pi / 2), T_star_ns=1e3 * C["th"] / OM,
                           nbar_0L=float(C["u0"] @ (C["n"] * C["u0"])), nbar_1L=float(C["u1"] @ (C["n"] * C["u1"])),
                           nbar_1Lperp=float(C["M"][1, 1].real), nbar_C=C["nC"], V_C=C["VC"],
                           b=[float(x) for x in C["b"]], b_norm=float(np.linalg.norm(C["b"])),
                           nbar_min_on_C=C["nC"] - float(np.linalg.norm(C["b"])),
                           nbar_max_on_C=C["nC"] + float(np.linalg.norm(C["b"])),
                           M_offdiag_imag=float(abs(C["M"][0, 1].imag)), comm_opnorm=C["comm_op"], comm_hs2=C["comm_hs2"],
                           top_quarter_weight=C["top_quarter_weight"]))
C80, C90 = CODES[80], CODES[90]
put("A", "convergence_80_vs_90", dict(d_nbar_C=C90["nC"] - C80["nC"], d_V_C=C90["VC"] - C80["VC"], d_c=C90["c"] - C80["c"],
                                      d_b=[float(x) for x in C90["b"] - C80["b"]]),
    "nbar_C, V_C, c and b agree between cutoffs 80 and 90 to the quoted digits")
assert abs(C80["nC"] - 2.658) < 5e-4 and abs(C80["VC"] - 9.60) < 5e-3 and abs(C80["c"] - 0.014160) < 5e-7
log("  nbar_C", C80["nC"], C90["nC"], "V_C", C80["VC"], C90["VC"], "b", C80["b"], C90["b"])

C = C80
D, c, th, s, Q, n, M, M2, nC, VC, b = (C[k] for k in ("D", "c", "th", "s", "Q", "n", "M", "M2", "nC", "VC", "b"))
XL = C["XL"]
TS = th / OM                                               # T* in us
EPS_RATE = GQ + KAP * nC + KPHI * VC                       # averaged first-order error per unit time (1/us)


def full(al):
    return Q @ np.asarray(al, complex)


def nbar_full(psi):
    return float(np.real(np.vdot(psi, n * psi)))


def var_full(psi):
    m1 = np.real(np.vdot(psi, n * psi))
    return float(np.real(np.vdot(psi, n ** 2 * psi)) - m1 ** 2)


def cardinal_SI():
    """six cardinal states as in the SI gate metric: a|0_L> + b|1_L>, normalized with the true overlap c (coefficients in
    the orthonormal basis {|0_L>, |1_L^perp>})"""
    out = []
    for lab, (a_, b_) in (("0", (1, 0)), ("1", (0, 1)), ("+", (1, 1)), ("-", (1, -1)), ("+i", (1, 1j)), ("-i", (1, -1j))):
        v = np.array([a_ + b_ * c, b_ * s], complex)
        out.append((lab, v / np.linalg.norm(v)))
    return out


def octahedron():
    r2 = 1 / math.sqrt(2)
    return [("0", np.array([1, 0], complex)), ("1p", np.array([0, 1], complex)), ("+", np.array([r2, r2], complex)),
            ("-", np.array([r2, -r2], complex)), ("+i", np.array([r2, 1j * r2], complex)),
            ("-i", np.array([r2, -1j * r2], complex))]


def end_map(uT):
    """uT on C equals X_L exp(-i phi Z_L/2) up to a global phase?  returns (residual, phi)"""
    W = XL @ uT
    g = np.sqrt(W[0, 0] * W[1, 1]) if abs(W[0, 0]) > 1e-12 else 1.0
    phi = float(np.angle(W[1, 1] / W[0, 0]))
    target = np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])
    ph = np.vdot(target.ravel(), W.ravel())
    ph /= abs(ph)
    return float(np.abs(W - ph * target).max()), phi


# ===================================================================================== B  brachistochrone
log("B: flip brachistochrone")
put("B", "T_star_ns", 1e3 * TS)
tB = np.linspace(0, TS, 2001)


def u_brach(t, om=OM):
    return rot2(om * t, (0, 1, 0))


card = cardinal_SI()
octa = octahedron()
nb_card = np.array([[nbar_full(full(u_brach(t) @ al)) for t in tB] for _, al in card])
nb_octa = np.array([[nbar_full(full(u_brach(t) @ al)) for t in tB] for _, al in octa])
var_octa = np.array([[var_full(full(u_brach(t) @ al)) for t in tB] for _, al in octa])
var_card = np.array([[var_full(full(u_brach(t) @ al)) for t in tB] for _, al in card])
# exact average over the code space: 1/2 Tr(U Pi U^dag n), full D x D unitary U = Q u Q^T + (1 - Pi)
Pi = C["Pi"]
haar_exact = []
for t in tB[::20]:
    U = Q @ u_brach(t) @ Q.T + (np.eye(D) - Pi)
    P = U @ Pi @ U.conj().T
    haar_exact.append(0.5 * np.real(np.trace(P * n[None, :])))
haar_exact = np.array(haar_exact)
basis_avg = nb_octa[:2].mean(0)                             # one orthonormal basis {|0_L>, |1_L^perp>}
octa_avg = nb_octa.mean(0)
card_avg = nb_card.mean(0)
put("B", "nbar_avg_exact_maxdev", float(np.abs(haar_exact - nC).max()), "1/2 Tr(U Pi U^dag n) - nbar_C along T*, full space")
put("B", "nbar_basis_avg_maxdev", float(np.abs(basis_avg - nC).max()), "average over {|0_L>,|1_L^perp>} (1-design)")
put("B", "nbar_octa_avg_maxdev", float(np.abs(octa_avg - nC).max()), "average over the orthonormal octahedron")
put("B", "nbar_SIcardinal_avg_maxdev", float(np.abs(card_avg - nC).max()),
    "six SI cardinal states (non-orthogonal |0_L>,|1_L>), not an exact design: deviation O(c|b|)")
rbar_card = np.mean([bloch(al) for _, al in card], 0)
put("B", "SIcardinal_mean_bloch", [float(x) for x in rbar_card])
put("B", "SIcardinal_avg_range", [float(card_avg.min()), float(card_avg.max())])
# Haar-random inputs: 400 propagated in the full Fock space, 20000 through the 2x2 block
al_h = RNG.normal(size=(20000, 2)) + 1j * RNG.normal(size=(20000, 2))
al_h /= np.linalg.norm(al_h, axis=1, keepdims=True)
tH = tB[::5]
U2 = np.array([u_brach(t) for t in tH])                   # (nt, 2, 2)
psi_h = np.einsum("tij,kj->tki", U2, al_h)                 # (nt, k, 2)
nb_h = np.real(np.einsum("tki,ij,tkj->tk", psi_h.conj(), M, psi_h))
full_h = np.einsum("di,tki->tkd", Q, psi_h[:, :400])
nb_h_full = np.einsum("tkd,d->tk", np.abs(full_h) ** 2, n)
assert np.abs(nb_h_full - nb_h[:, :400]).max() < 1e-12
mc_mean = nb_h.mean(1)
mc_se = nb_h.std(1) / math.sqrt(nb_h.shape[1])
put("B", "haar_mc", dict(n_states=int(nb_h.shape[1]), max_dev=float(np.abs(mc_mean - nC).max()),
                         max_dev_in_standard_errors=float(np.max(np.abs(mc_mean - nC) / mc_se)),
                         full_space_vs_block_maxdiff=float(np.abs(nb_h_full - nb_h[:, :400]).max())),
    "Monte Carlo mean of 20000 Haar inputs within sampling error of nbar_C")
# per-input exposures (photon ns) along the brachistochrone
expo = lambda y, t: 1e3 * simpson(y, x=t)                  # noqa: E731
E_card = {lab: expo(y, tB) for (lab, _), y in zip(card, nb_card)}
E_octa = {lab: expo(y, tB) for (lab, _), y in zip(octa, nb_octa)}
E_h = 1e3 * simpson(nb_h, x=tH, axis=0)
Phi = 2 * th
w = np.array([b[0] * math.sin(Phi) - b[2] * (1 - math.cos(Phi)), 0.0, b[0] * (1 - math.cos(Phi)) + b[2] * math.sin(Phi)])
w *= 1e3 / (2 * OM)                                         # E(r0) = nbar_C T* + w.r0 (photon ns)
E_an = lambda al: 1e3 * nC * TS + w @ bloch(al)             # noqa: E731
put("B", "exposure_avg_ns", 1e3 * nC * TS, "nbar_C T*")
put("B", "exposure_SIcardinal_ns", E_card)
put("B", "exposure_octa_ns", E_octa)
put("B", "exposure_octa_mean_minus_nCT", float(np.mean(list(E_octa.values())) - 1e3 * nC * TS))
put("B", "exposure_analytic_vs_numeric_maxdiff", float(max(abs(E_an(al) - E_card[lab]) for lab, al in card)),
    "E(r0) = nbar_C T* + w.r0 against Simpson on 2001 points")
put("B", "exposure_per_input_range_ns", [float(1e3 * nC * TS - np.linalg.norm(w)), float(1e3 * nC * TS + np.linalg.norm(w))])
put("B", "exposure_per_input_w_ns", [float(x) for x in w])
put("B", "exposure_haar_spread_ns", dict(mean=float(E_h.mean()), sd=float(E_h.std()), sd_analytic=float(np.linalg.norm(w) / math.sqrt(3)),
                                         min=float(E_h.min()), max=float(E_h.max())))
put("B", "nbar_range_per_input", {lab: [float(y.min()), float(y.max())] for (lab, _), y in zip(card, nb_card)})
put("B", "var_octa_avg_maxdev", float(np.abs(var_octa.mean(0) - VC).max()), "octahedron (2-design) average of Var n = V_C")
put("B", "var_basis_avg_range", [float(var_octa[:2].mean(0).min()), float(var_octa[:2].mean(0).max())],
    "average of Var n over one orthonormal basis is not invariant (needs a 2-design)")
put("B", "var_SIcardinal_avg_range", [float(var_card.mean(0).min()), float(var_card.mean(0).max())])
put("B", "eps1_avg", dict(value=EPS_RATE * TS, ancilla=GQ * TS, loss=KAP * nC * TS, dephasing=KPHI * VC * TS))
CURVES.update(tB_ns=1e3 * tB, nb_card=nb_card, card_labels=np.array([lab for lab, _ in card]), haar_exact=haar_exact,
              haar_exact_t_ns=1e3 * tB[::20], octa_avg=octa_avg, card_avg=card_avg, nb_octa=nb_octa,
              mc_mean=mc_mean, mc_t_ns=1e3 * tH, mc_q05=np.quantile(nb_h, 0.05, axis=1), mc_q95=np.quantile(nb_h, 0.95, axis=1))
log("  avg dev", FACTS["B"]["nbar_avg_exact_maxdev"], "E_card", E_card)


# ===================================================================================== C  other flips within C
log("C: other flips that keep the qubit encoded")


def evaluate(us, ts, label, extra=None):
    """us: (K, 2, 2) unitaries on C at times ts (us); returns averaged exposure, per-input exposures, checks"""
    psi_o = np.einsum("tij,kj->tki", us, np.array([al for _, al in octa]))
    full_o = np.einsum("di,tki->tkd", Q, psi_o)
    nb = np.einsum("tkd,d->tk", np.abs(full_o) ** 2, n)
    m2 = np.einsum("tkd,d->tk", np.abs(full_o) ** 2, n ** 2)
    var_avg = (m2 - nb ** 2).mean(1)
    avg = nb.mean(1)
    T = ts[-1]
    E_avg = 1e3 * simpson(avg, x=ts)
    E_0 = 1e3 * simpson(nb[:, 0], x=ts)
    res, phi = end_map(us[-1])
    r = dict(label=label, T_ns=1e3 * T, T_over_Tstar=T / TS, E_avg_ns=E_avg, nCT_ns=1e3 * nC * T,
             rel_dev_E=E_avg / (1e3 * nC * T) - 1, avg_nbar_maxdev=float(np.abs(avg - nC).max()),
             var_avg_maxdev=float(np.abs(var_avg - VC).max()), E_0L_ns=E_0,
             E_inputs_ns={lab: float(1e3 * simpson(nb[:, k], x=ts)) for k, (lab, _) in enumerate(octa)},
             end_map_residual=res, end_phi=phi,
             eps1_avg=float(GQ * T + KAP * simpson(avg, x=ts) + KPHI * simpson(var_avg, x=ts)),
             eps1_formula=float(EPS_RATE * T))
    if extra:
        r.update(extra)
    return r


def speeds(hs, rs):
    """Delta H on the state with Bloch vector r for H = h0 + h.sigma: |h x r|; uniform bound |h|"""
    return np.linalg.norm(np.cross(hs, rs), axis=1), np.linalg.norm(hs, axis=1)


FLIPS = []
# C1 slower brachistochrones
for k in (1.0, 1.1, 1.25, 1.5, 1.75, 2.0):
    ts = np.linspace(0, k * TS, 1201)
    us = np.array([u_brach(t, OM / k) for t in ts])
    FLIPS.append(evaluate(us, ts, f"brachistochrone x{k}", dict(family="slower", dH0_max_over_omega=1 / k, uniform_max_over_omega=1 / k)))
# C2 two-leg great-circle detours through the equatorial point at azimuth phi (phi = 0 is the brachistochrone)
r1 = np.array([math.sin(2 * th), 0.0, math.cos(2 * th)])
for phi_d in np.deg2rad([0, 30, 60, 90, 120, 150, 180, -90]):
    ue = np.array([math.cos(phi_d), math.sin(phi_d), 0.0])
    m1 = np.cross([0, 0, 1.0], ue)
    m1 /= np.linalg.norm(m1)
    a2 = math.acos(np.clip(ue @ r1, -1, 1))
    m2 = np.cross(ue, r1)
    m2 /= np.linalg.norm(m2)
    t1, t2 = (math.pi / 2) / (2 * OM), a2 / (2 * OM)
    ts = np.linspace(0, t1 + t2, 1601)
    u_mid = rot2(OM * t1, m1)
    us = np.array([rot2(OM * t, m1) if t <= t1 else rot2(OM * (t - t1), m2) @ u_mid for t in ts])
    # leg-wise Simpson needs the kink on the grid: integrate both legs separately
    sel1, sel2 = ts <= t1, ts >= t1
    ts1 = np.r_[ts[sel1], t1] if ts[sel1][-1] < t1 else ts[sel1]
    ts2 = np.r_[t1, ts[sel2]] if ts[sel2][0] > t1 else ts[sel2]
    us1 = np.array([rot2(OM * t, m1) for t in ts1])
    us2 = np.array([rot2(OM * (t - t1), m2) @ u_mid for t in ts2])
    r_a = evaluate(us1, ts1, "leg1")
    r_b = evaluate(us2, ts2 - t1, "leg2")
    r = evaluate(us, ts, f"detour via azimuth {np.rad2deg(phi_d):.0f} deg",
                 dict(family="detour", azimuth_deg=float(np.rad2deg(phi_d)), dH0_max_over_omega=1.0, uniform_max_over_omega=1.0))
    r["E_avg_ns"] = r_a["E_avg_ns"] + r_b["E_avg_ns"]
    r["E_0L_ns"] = r_a["E_0L_ns"] + r_b["E_0L_ns"]
    r["rel_dev_E"] = r["E_avg_ns"] / r["nCT_ns"] - 1
    r["E_inputs_ns"] = {k_: r_a["E_inputs_ns"][k_] + r_b["E_inputs_ns"][k_] for k_ in r_a["E_inputs_ns"]}
    FLIPS.append(r)


# C3 random smooth wiggles, rescaled in time to the speed limit (S1: Delta H = omega on the state that starts as |0_L>,
# S2: Delta H <= omega on every code state), and the S2 version also run 1.3x slower
def wiggle(seed, amp=1.0):
    rg = np.random.default_rng(seed)
    K = 4
    a = amp * rg.normal(size=K) * 0.35 / np.arange(1, K + 1)
    bb = amp * rg.normal(size=K) * 0.8 / np.arange(1, K + 1)
    m = rg.normal(size=3)
    m /= np.linalg.norm(m)
    ks = np.arange(1, K + 1)
    Th = lambda x: th * x + np.sin(np.pi * np.outer(x, ks)) @ a                 # noqa: E731
    dTh = lambda x: th + (np.pi * ks * np.cos(np.pi * np.outer(x, ks))) @ a     # noqa: E731
    Ph = lambda x: np.sin(np.pi * np.outer(x, ks)) @ bb                         # noqa: E731
    dPh = lambda x: (np.pi * ks * np.cos(np.pi * np.outer(x, ks))) @ bb         # noqa: E731
    x = np.linspace(0, 1, 4001)
    TH, DTH, PH, DPH = Th(x), dTh(x), Ph(x), dPh(x)
    us, hs, rs = [], [], []
    for i in range(len(x)):
        A = rot2(TH[i], (0, 1, 0))
        Bm = rot2(PH[i], m)
        us.append(A @ Bm)
        RA = bloch_rot(A)
        hs.append(DTH[i] * np.array([0, 1.0, 0]) + DPH[i] * (RA @ m))          # i u' u^dag = Th' Y + Ph' A (m.s) A^dag
        rs.append(bloch_rot(A @ Bm) @ np.array([0, 0, 1.0]))
    return x, np.array(us), np.array(hs), np.array(rs), dict(seed=seed, amp=amp, a=a.tolist(), b=bb.tolist(), m=m.tolist())


WIG = []
WIG_CASES = [(seed, 1.0) for seed in range(1, 13)] + [(seed, amp) for seed in range(1, 7) for amp in (0.1, 0.2, 0.35, 0.5)]
for seed, amp in WIG_CASES:
    x, us, hs, rs, meta = wiggle(seed, amp)
    # check of the generator formula by finite differences
    du = (us[2:] - us[:-2]) / (2 * (x[1] - x[0]))
    hfd = np.array([1j * d @ u.conj().T for d, u in zip(du, us[1:-1])])
    hfd_vec = np.array([[0.5 * np.real(np.trace(P @ h)) for P in PAULI] for h in hfd])
    gen_err = float(np.abs(hfd_vec - hs[1:-1]).max())
    v1, v2 = speeds(hs, rs)
    for ver, v, slow in ((("S1", v1, 1.0), ("S2", v2, 1.0), ("S2", v2, 1.3)) if amp == 1.0 else (("S1", v1, 1.0), ("S2", v2, 1.0))):
        dtdx = slow * v / OM
        ts = np.r_[0, np.cumsum(0.5 * (dtdx[1:] + dtdx[:-1]) * np.diff(x))]
        # exposure in the x variable (exact on the grid of x): int f dt = int f dt/dx dx
        psi_o = np.einsum("tij,kj->tki", us, np.array([al for _, al in octa]))
        nb = np.real(np.einsum("tki,ij,tkj->tk", psi_o.conj(), M, psi_o))
        full_o = np.einsum("di,tki->tkd", Q, psi_o[::200])
        assert np.abs(np.einsum("tkd,d->tk", np.abs(full_o) ** 2, n) - nb[::200]).max() < 1e-12
        m2_ = np.real(np.einsum("tki,ij,tkj->tk", psi_o.conj(), M2, psi_o))
        T = simpson(dtdx, x=x)
        E_avg = 1e3 * simpson(nb.mean(1) * dtdx, x=x)
        E_0 = 1e3 * simpson(nb[:, 0] * dtdx, x=x)
        res, phi = end_map(us[-1])
        r = dict(label=f"wiggle {seed} amp {amp} {ver}" + (" x1.3" if slow != 1 else ""), family="wiggle", version=ver, slow=slow,
                 T_ns=1e3 * T, T_over_Tstar=T / TS, E_avg_ns=E_avg, nCT_ns=1e3 * nC * T, rel_dev_E=E_avg / (1e3 * nC * T) - 1,
                 avg_nbar_maxdev=float(np.abs(nb.mean(1) - nC).max()),
                 var_avg_maxdev=float(np.abs((m2_ - nb ** 2).mean(1) - VC).max()), E_0L_ns=E_0,
                 dH0_max_over_omega=float(np.max(np.where(dtdx > 1e-14, v1 / np.maximum(dtdx, 1e-300), 0)) / OM),
                 uniform_max_over_omega=float(np.max(np.where(dtdx > 1e-14, v2 / np.maximum(dtdx, 1e-300), 0)) / OM),
                 L0_FS=float(simpson(v1, x=x)), end_map_residual=res, end_phi=phi, generator_fd_err=gen_err, **meta)
        FLIPS.append(r)
        if seed == 1 and amp == 1.0 and ver == "S2" and slow == 1.0:
            CURVES.update(wig_t_ns=1e3 * ts, wig_nb=nb, wig_avg=nb.mean(1))
        WIG.append(r)
# C4 the S1 frame family at T*: exp(-i omega t Y_L) exp(-i chi(t) Z_L/2), random smooth chi
for seed in range(1, 7):
    rg = np.random.default_rng(100 + seed)
    a = rg.normal(size=3) * 2.0
    lin = rg.normal() * 3.0
    chi = lambda t: lin * t / TS + np.sin(np.pi * np.outer(t / TS, [1, 2, 3])) @ a           # noqa: E731
    dchi = lambda t: lin / TS + (np.pi / TS * np.array([1, 2, 3]) * np.cos(np.pi * np.outer(t / TS, [1, 2, 3]))) @ a   # noqa: E731
    ts = np.linspace(0, TS, 2001)
    CH, DCH = chi(ts), dchi(ts)
    us = np.array([u_brach(t) @ rot2(0.5 * x_, (0, 0, 1)) for t, x_ in zip(ts, CH)])
    # generator: omega Y + (chi'/2) R_y(2 omega t) z
    rs = np.array([bloch_rot(u) @ np.array([0, 0, 1.0]) for u in us])
    hs = np.array([OM * np.array([0, 1.0, 0]) + 0.5 * d * (bloch_rot(u_brach(t)) @ np.array([0, 0, 1.0])) for t, d in zip(ts, DCH)])
    v1, v2 = speeds(hs, rs)
    ub0 = np.array([u_brach(t)[:, 0] for t in ts])               # |0_L> along the brachistochrone
    path_dev = float(np.max(1 - np.abs(np.sum(ub0.conj() * us[:, :, 0], 1)) ** 2))
    r = evaluate(us, ts, f"S1 frame family {seed}",
                 dict(family="S1frame", chi_T=float(CH[-1]), dH0_max_over_omega=float(v1.max() / OM),
                      dH0_min_over_omega=float(v1.min() / OM), uniform_max_over_omega=float(v2.max() / OM),
                      path_of_0L_infidelity_max=path_dev))
    FLIPS.append(r)
# C5 a detour (azimuth 90 deg) propagated in the full Fock space, with an arbitrary time-dependent Hermitian
# Hamiltonian on the complement of C added (piecewise constant on 600 steps, exact exponentials)
rg = np.random.default_rng(7)
Kc = [rg.normal(size=(D, D)) + 1j * rg.normal(size=(D, D)) for _ in range(2)]
Kc = [0.5 * (k + k.conj().T) for k in Kc]
Qc = np.eye(D) - Pi
Kc = [OM * Qc @ k @ Qc / np.linalg.norm(Qc @ k @ Qc, 2) for k in Kc]
ue = np.array([0, 1.0, 0])
m1 = np.cross([0, 0, 1.0], ue)
m2 = np.cross(ue, r1)
m2 /= np.linalg.norm(m2)
t1, t2 = (math.pi / 2) / (2 * OM), math.acos(np.clip(ue @ r1, -1, 1)) / (2 * OM)
nst = 600
dt = (t1 + t2) / nst
psis = np.array([full(al) for _, al in octa]).T            # D x 6
Ufull = np.eye(D, dtype=complex)
nb_rows, comm_max, leak_max = [], 0.0, 0.0
for k in range(nst + 1):
    tk = k * dt
    nb_rows.append(np.einsum("dk,d->k", np.abs(Ufull @ psis) ** 2, n))
    leak_max = max(leak_max, float(np.linalg.norm(Qc @ Ufull @ Pi, 2)))
    if k == nst:
        break
    tm = tk + 0.5 * dt
    hvec = OM * (m1 if tm < t1 else m2)
    Hc = Q @ (hvec[0] * SX + hvec[1] * SY + hvec[2] * SZ) @ Q.T
    Hfull = Hc + math.cos(9 * tm / (t1 + t2)) * Kc[0] + math.sin(5 * tm / (t1 + t2)) * Kc[1]
    comm_max = max(comm_max, float(np.linalg.norm(Hfull @ Pi - Pi @ Hfull, 2)))
    Ufull = expm(-1j * dt * Hfull) @ Ufull
nb_rows = np.array(nb_rows)
ts = np.linspace(0, t1 + t2, nst + 1)
k1 = int(round(t1 / dt))
E_avg5 = 1e3 * (simpson(nb_rows[:k1 + 1].mean(1), x=ts[:k1 + 1]) + simpson(nb_rows[k1:].mean(1), x=ts[k1:]))
uT_C = Q.T @ Ufull @ Q
res5, phi5 = end_map(uT_C)
ref = next(f_ for f_ in FLIPS if f_["label"] == "detour via azimuth 90 deg")
put("C", "full_space_with_complement_hamiltonian", dict(
    T_ns=1e3 * (t1 + t2), E_avg_ns=E_avg5, nCT_ns=1e3 * nC * (t1 + t2), rel_dev_E=E_avg5 / (1e3 * nC * (t1 + t2)) - 1,
    avg_nbar_maxdev=float(np.abs(nb_rows.mean(1) - nC).max()), commutator_H_Pi_max=comm_max, leak_Q_U_Pi_max=leak_max,
    end_map_residual=res5, end_phi=phi5, E_0L_ns=1e3 * (simpson(nb_rows[:k1 + 1, 0], x=ts[:k1 + 1]) + simpson(nb_rows[k1:, 0], x=ts[k1:])),
    E_0L_without_complement_ns=ref["E_0L_ns"], complement_norm_over_omega=1.0),
    "an arbitrary Hamiltonian on the complement of C changes no cost on C")
put("C", "flips", FLIPS)
cp = [f for f in FLIPS]
put("C", "summary", dict(
    n_flips=len(cp), max_abs_rel_dev_E=float(max(abs(f["rel_dev_E"]) for f in cp)),
    max_avg_nbar_dev=float(max(f["avg_nbar_maxdev"] for f in cp)), max_var_avg_dev=float(max(f["var_avg_maxdev"] for f in cp)),
    max_end_map_residual=float(max(f["end_map_residual"] for f in cp)),
    min_T_over_Tstar=float(min(f["T_over_Tstar"] for f in cp)),
    max_eps1_rel_dev=float(max(abs(f["eps1_avg"] / f["eps1_formula"] - 1) for f in cp if "eps1_avg" in f)),
    max_dH0_over_omega=float(max(f["dH0_max_over_omega"] for f in cp)),
    min_T_over_Tstar_S1=float(min(f["T_over_Tstar"] for f in cp if f.get("version") == "S1")),
    min_T_over_Tstar_S2=float(min(f["T_over_Tstar"] for f in cp if f.get("version") == "S2"))),
    "every flip that keeps the qubit encoded: E_avg = nbar_C T, T >= T*, end map X_L exp(-i phi Z_L/2)")
log("  summary", FACTS["C"]["summary"]["value"])


# ===================================================================================== D  tilted flips that leave C
log("D: tilted flips (SI Sec. IV.5)")


def tilt_family(Cd, lams):
    Dd, Pid, nd = Cd["D"], Cd["Pi"], Cd["n"]
    Nd = np.diag(nd)
    Cm = Pid @ Nd - Nd @ Pid
    sv = np.linalg.svd(Cm, compute_uv=False)
    A = Cm / sv[0]
    ev, V = eigh(1j * A)
    EA = lambda x: (V * np.exp(-1j * x * ev)) @ V.conj().T              # noqa: E731  exp(x A)
    thd = Cd["th"]
    Tst = thd / OM

    def T_dH(lam):
        if lam == 0:
            return Tst
        f = lambda T: OM * T * (2 / np.pi) * ellipe((lam * np.pi / (OM * T)) ** 2) - thd   # noqa: E731
        lo = max(Tst, lam * np.pi / OM * (1 + 1e-12))
        return brentq(f, lo, 3 * Tst, xtol=1e-14)

    out = []
    for lam in lams:
        T = T_dH(lam)
        tt = np.linspace(0, T, 1201)
        nb, vv, lk = [], [], []
        for t in tt:
            E = EA(lam * np.sin(np.pi * t / T))
            Pt = E @ Pid @ E.conj().T
            m1 = np.real(np.trace(Pt @ Nd))
            m2 = np.real(np.trace(Pt @ Nd @ Nd))
            PNP = Pt @ Nd @ Pt
            nb.append(m1 / 2)
            vv.append(m2 / 2 - (m1 ** 2 + np.real(np.trace(PNP @ PNP))) / 6)
            lk.append(1 - np.real(np.trace(Pid @ Pt)) / 2)
        J = simpson(nb, x=tt)
        Vi = simpson(vv, x=tt)
        out.append(dict(lam=lam, T_ns=1e3 * T, T_over_Tstar=T / Tst, E_avg_ns=1e3 * J, nCT_ns=1e3 * Cd["nC"] * T,
                        below_line=1 - J / (Cd["nC"] * T), saving_vs_brach=1 - J / (Cd["nC"] * Tst),
                        eps1=GQ * T + KAP * J + KPHI * Vi, eps1_saving=1 - (GQ * T + KAP * J + KPHI * Vi) / (Tst * (GQ + KAP * Cd["nC"] + KPHI * Cd["VC"])),
                        eps1_noDeph_saving=1 - (GQ * T + KAP * J) / (Tst * (GQ + KAP * Cd["nC"])),
                        max_leak_avg=float(max(lk)), comm_H_Pi_nonzero=bool(lam > 0),
                        nbar_avg_curve=np.array(nb), t_ns=1e3 * tt))
    return out, dict(opnorm=float(sv[0]), hs2=float(np.sum(sv ** 2)), rate=0.5 * float(np.sum(sv ** 2)) / float(sv[0]))


LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
TILT, TILT_META = tilt_family(C80, LAMS)
TILT90, _ = tilt_family(C90, [0.3])
for r in TILT:
    CURVES[f"tilt_{r['lam']:.2f}_t_ns"] = r.pop("t_ns")
    CURVES[f"tilt_{r['lam']:.2f}_nbar"] = r.pop("nbar_avg_curve")
for r in TILT90:
    r.pop("t_ns")
    r.pop("nbar_avg_curve")
put("D", "tilted", TILT)
put("D", "tilt_meta", TILT_META)
put("D", "tilt_0.3_D90", TILT90[0])
t03 = next(r for r in TILT if r["lam"] == 0.3)
assert abs(t03["T_ns"] - 319.86) < 0.05 and abs(t03["E_avg_ns"] - 601.67) < 0.05
log("  lam 0.3", t03["T_ns"], t03["E_avg_ns"], t03["saving_vs_brach"], "D90", TILT90[0]["T_ns"], TILT90[0]["E_avg_ns"])


# ===================================================================================== E  vacuum transfer of |0_L>
log("E: photon-optimal transfer through the vacuum, applied to the code space")


class _Any:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s_):
        self.__dict__.update(s_ if isinstance(s_, dict) else {"state": s_})


class _Finder(pickle.Unpickler):
    def find_class(self, mod, name):
        try:
            return super().find_class(mod, name)
        except Exception:
            return _Any


PKL = ROOT / "manuscript_figures/analytics_audit/cache/photon_flip.pkl"   # same file (md5 480b71a6...)
with open(PKL, "rb") as f:
    ph = _Finder(f).load()
Uph = np.asarray(ph[0] if isinstance(ph, (tuple, list)) else ph, float)
ue0, ue1, ue1p = C80["u0"][::2], C80["u1"][::2], C80["e1"][::2]
ne = 2.0 * np.arange(D // 2)
Pie = np.outer(ue0, ue0) + np.outer(ue1p, ue1p)
xg, wg = np.polynomial.legendre.leggauss(32)


def route(nodes, vecs):
    """rank-two rotations that carry |0_L> along the stored nodes at speed omega, applied to the columns of vecs"""
    V = np.array(vecs, complex).T.copy()                    # dim x k
    L = 0.0
    J = np.zeros(V.shape[1])
    leak = np.zeros(V.shape[1])
    trace_t, trace_nb = [0.0], [np.real(np.einsum("dk,d->k", np.abs(V) ** 2, ne))]
    for pa, pb in zip(nodes[:-1], nodes[1:]):
        pa = pa / np.linalg.norm(pa)
        pb = pb / np.linalg.norm(pb)
        if pa @ pb < 0:
            pb = -pb
        phi_ = math.acos(np.clip(pa @ pb, -1, 1))
        if phi_ < 1e-14:
            continue
        tk = (pb - math.cos(phi_) * pa) / math.sin(phi_)
        va, vt = pa @ V, tk @ V
        rest = V - np.outer(pa, va) - np.outer(tk, vt)
        sq = 0.5 * phi_ * (xg + 1)
        wq = 0.5 * phi_ * wg / OM
        for j in range(V.shape[1]):
            Vs = rest[:, j][None] + np.outer(va[j] * np.cos(sq) - vt[j] * np.sin(sq), pa) + np.outer(va[j] * np.sin(sq) + vt[j] * np.cos(sq), tk)
            J[j] += wq @ ((np.abs(Vs) ** 2) @ ne)
            leak[j] = max(leak[j], float(np.max(1 - np.real(np.einsum("sd,de,se->s", Vs.conj(), Pie, Vs)))))
        V = rest + np.outer(pa, va * math.cos(phi_) - vt * math.sin(phi_)) + np.outer(tk, va * math.sin(phi_) + vt * math.cos(phi_))
        L += phi_
        trace_t.append(L / OM)
        trace_nb.append(np.real(np.einsum("dk,d->k", np.abs(V) ** 2, ne)))
    return dict(T=L / OM, J=1e3 * J, leak=leak, out=V, t=np.array(trace_t), nb=np.array(trace_nb))


oct_e = [ue0 * al[0] + ue1p * al[1] for _, al in octa]
card_e = [ue0 * al[0] + ue1p * al[1] for _, al in card]
RT = route(Uph, [ue0, ue1, ue1p] + oct_e + card_e)
J0, J1, J1p = RT["J"][:3]
J_oct = RT["J"][3:9]
out_card = RT["out"][:, 9:15].T


def six_state(outputs, phis=np.linspace(0, 2 * np.pi, 7201)):
    """SI gate metric: six SI cardinal states against X_L after the best virtual Z rotation on the non-orthogonal basis"""
    best = (-1.0, None)
    coeffs = [(1, 0), (0, 1), (1, 1), (1, -1), (1, 1j), (1, -1j)]
    for p_ in phis:
        F = 0.0
        for (a_, b_), o in zip(coeffs, outputs):
            t_ = b_ * np.exp(0.5j * p_) * ue0 + a_ * np.exp(-0.5j * p_) * ue1
            t_ = t_ / np.linalg.norm(t_)
            F += abs(np.vdot(t_, o)) ** 2
        F /= 6
        if F > best[0]:
            best = (F, p_)
    return best


F6_route, _ = six_state(out_card)
brach_out = [Q[::2] @ (u_brach(TS) @ al) for _, al in card]
F6_brach, _ = six_state(brach_out)
put("E", "vacuum_transfer", dict(
    T_ns=1e3 * RT["T"], T_over_Tstar=RT["T"] / TS, J_0L_ns=float(J0), J_1L_ns=float(J1), J_1Lperp_ns=float(J1p),
    two_input_mean_ns=float(0.5 * (J0 + J1)), avg_over_code_space_ns=float(0.5 * (J0 + J1p)),
    avg_octahedron_ns=float(J_oct.mean()), nCT_at_same_T_ns=1e3 * nC * RT["T"],
    F_1to0=float(abs(np.vdot(ue0, RT["out"][:, 1])) ** 2), F_0to1=float(abs(np.vdot(ue1, RT["out"][:, 0])) ** 2),
    six_state_fidelity_X_L=float(F6_route), six_state_fidelity_brachistochrone=float(F6_brach),
    leak_0L_max=float(RT["leak"][0]), leak_1L_max=float(RT["leak"][1]), pkl=str(PKL.relative_to(ROOT)),
    single_input_saving_vs_brach_0L=float(1 - J0 / E_card["0"]), avg_saving_vs_brach=float(1 - 0.5 * (J0 + J1p) / (1e3 * nC * TS))),
    "matches Supplementary Table route (397 ns, 359, 965, 662, F=0.599)")
assert abs(1e3 * RT["T"] - 397.2) < 0.2 and abs(J0 - 359.3) < 0.3 and abs(J1 - 964.7) < 0.5
CURVES.update(vac_t_ns=1e3 * RT["t"], vac_nb=RT["nb"][:, :3])
log("  vacuum", FACTS["E"]["vacuum_transfer"]["value"])


# ===================================================================================== F  single-input optimum within C
log("F: single-input photon optimum within C")


def wlen(R):
    dots = np.sum(R[:-1] * R[1:], 1)
    crs = np.linalg.norm(np.cross(R[:-1], R[1:]), axis=1)
    ang = np.arctan2(crs, dots)
    mid = R[:-1] + R[1:]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    return 0.5 * np.sum((nC + mid @ b) * ang), 0.5 * np.sum(ang)


def path_init(ra, rb, via=None, K=160):
    """great circle ra -> rb, or two great-circle legs through `via`"""
    def gc(p, q, k):
        a_ = math.acos(np.clip(p @ q, -1, 1))
        e = q - (p @ q) * p
        e /= np.linalg.norm(e)
        sgrid = np.linspace(0, a_, k)
        return np.outer(np.cos(sgrid), p) + np.outer(np.sin(sgrid), e)
    if via is None:
        return gc(ra, rb, K + 1)
    a1, a2 = math.acos(np.clip(ra @ via, -1, 1)), math.acos(np.clip(via @ rb, -1, 1))
    k1_ = max(2, int(round(K * a1 / (a1 + a2))))
    return np.vstack([gc(ra, via, k1_ + 1)[:-1], gc(via, rb, K - k1_ + 1)])


def wlen_grad(R):
    """weighted length and its gradient with respect to the unit vectors R (tangential parts are what matter)"""
    ra_, rb_ = R[:-1], R[1:]
    d = np.sum(ra_ * rb_, 1)
    sn = np.linalg.norm(np.cross(ra_, rb_), axis=1)
    ang = np.arctan2(sn, d)
    s_ = ra_ + rb_
    ns_ = np.linalg.norm(s_, axis=1, keepdims=True)
    mh = s_ / ns_
    wk = nC + mh @ b
    J = 0.5 * np.sum(wk * ang)
    ga = -(rb_ - d[:, None] * ra_) / sn[:, None]
    gb = -(ra_ - d[:, None] * rb_) / sn[:, None]
    gw = (b[None, :] - (mh @ b)[:, None] * mh) / ns_
    G = np.zeros_like(R)
    G[:-1] += 0.5 * (wk[:, None] * ga + ang[:, None] * gw)
    G[1:] += 0.5 * (wk[:, None] * gb + ang[:, None] * gw)
    return J, G


GRAD_CHECK = []


def optimize_path(ra, rb, R0, tag):
    K = len(R0) - 1

    def unpack(x):
        X = x.reshape(K - 1, 3)
        nx = np.linalg.norm(X, axis=1, keepdims=True)
        return np.vstack([ra, X / nx, rb]), nx

    def fg(x):
        R, nx = unpack(x)
        J, G = wlen_grad(R)
        Ri, Gi = R[1:-1], G[1:-1]
        return J, ((Gi - np.sum(Gi * Ri, 1)[:, None] * Ri) / nx).ravel()

    x0 = R0[1:-1].ravel()
    if not GRAD_CHECK:                                            # analytic gradient against central differences
        g_an = fg(x0)[1]
        idx = RNG.choice(len(x0), 12, replace=False)
        h_ = 1e-6
        g_fd = [(fg(x0 + h_ * np.eye(len(x0))[i])[0] - fg(x0 - h_ * np.eye(len(x0))[i])[0]) / (2 * h_) for i in idx]
        GRAD_CHECK.append(float(np.max(np.abs(np.array(g_fd) - g_an[idx]))))
    best = None
    for _ in range(3):                                            # restarts from the last optimum
        r = minimize(fg, x0, jac=True, method="L-BFGS-B",
                     options=dict(maxiter=50000, maxfun=10 ** 7, ftol=1e-16, gtol=1e-12, maxcor=30))
        x0 = r.x
        if best is None or r.fun < best.fun:
            best = r
    Ropt = unpack(best.x)[0]
    return Ropt, float(best.fun), tag


r0N = np.array([0, 0, 1.0])
starts = [("great circle (brachistochrone)", None)]
for az in (60, 90, 135, 180, 225, 270, 300):
    starts.append((f"via azimuth {az}", np.array([math.cos(np.deg2rad(az)), math.sin(np.deg2rad(az)), 0.0])))
res_F = []
for tag, via in starts:
    R0 = path_init(r0N, r1, via)
    if via is not None:                                           # random perturbation of the start
        R0 = R0 + 0.02 * RNG.normal(size=R0.shape) * np.r_[0, np.ones(len(R0) - 2), 0][:, None]
        R0 /= np.linalg.norm(R0, axis=1, keepdims=True)
    Ropt, Jopt, _ = optimize_path(r0N, r1, R0, tag)
    res_F.append(dict(start=tag, J=Jopt, L=wlen(Ropt)[1], max_off_plane_deg=float(np.rad2deg(np.max(np.abs(np.arcsin(np.clip(Ropt[:, 1], -1, 1)))))),
                      min_x=float(Ropt[:, 0].min()), R=Ropt))
Rgc = path_init(r0N, r1, None)
Jgc_disc = wlen(Rgc)[0]
Jgc_exact = th * nC + 0.5 * (b[0] * (1 - math.cos(2 * th)) + b[2] * math.sin(2 * th))
bestF = min(res_F, key=lambda d: d["J"])
put("F", "transfer_0L_to_1L_within_C", dict(
    J_great_circle_exact=Jgc_exact, J_great_circle_discrete=Jgc_disc, exposure_great_circle_ns=1e3 * Jgc_exact / OM,
    exposure_brachistochrone_0L_ns=E_card["0"], b_y=float(b[1]),
    starts=[{k: v for k, v in d.items() if k != "R"} for d in res_F],
    best_J=bestF["J"], best_start=bestF["start"], best_saving_vs_great_circle=1 - bestF["J"] / Jgc_disc,
    best_max_off_plane_deg=bestF["max_off_plane_deg"], best_length_over_theta=bestF["L"] / th,
    gradient_check_maxerr=GRAD_CHECK[0],
    nbar_min_direction=[float(x) for x in -b / np.linalg.norm(b)]),
    "minimize int (nbar_C + b.r) dvartheta over curves on the Bloch sphere of C, 160 segments, L-BFGS-B, 8 starts")
CURVES.update(F_best=bestF["R"], F_gc=Rgc)
log("  F |0_L>:", "J_gc", Jgc_disc, "best", bestF["J"], bestF["start"], "off-plane", bestF["max_off_plane_deg"])


# F2  the same question without discretization: smooth off-plane deformations of the great circle,
# r(l) = cos y(l) g(l) + sin y(l) e_y, g(l) = cos l e_z + sin l e_x, l in [0, 2 theta], y = sum_k c_k sin(k pi l / 2theta),
# J(y) = 1/2 int (nbar_C + b.r) |r'| dl on 2000 Gauss-Legendre nodes (exact for the great circle to 1e-15)
LGL = 2 * th
xq, wq_ = np.polynomial.legendre.leggauss(2000)
lq = 0.5 * LGL * (xq + 1)
wql = 0.5 * LGL * wq_
KMOD = 12
SINS = np.sin(np.pi * np.outer(lq, np.arange(1, KMOD + 1)) / LGL)
COSS = (np.pi * np.arange(1, KMOD + 1) / LGL) * np.cos(np.pi * np.outer(lq, np.arange(1, KMOD + 1)) / LGL)
gq = np.outer(np.cos(lq), [0, 0, 1.0]) + np.outer(np.sin(lq), [1.0, 0, 0])


def J_smooth(cc):
    y, yp = SINS @ cc, COSS @ cc
    r = np.cos(y)[:, None] * gq + np.sin(y)[:, None] * np.array([0, 1.0, 0])
    return 0.5 * wql @ ((nC + r @ b) * np.sqrt(yp ** 2 + np.cos(y) ** 2))


J0s = J_smooth(np.zeros(KMOD))
hh = 1e-4
Hs = np.zeros((KMOD, KMOD))
for i in range(KMOD):
    for j in range(i, KMOD):
        ei, ej = np.eye(KMOD)[i] * hh, np.eye(KMOD)[j] * hh
        Hs[i, j] = Hs[j, i] = (J_smooth(ei + ej) - J_smooth(ei - ej) - J_smooth(-ei + ej) + J_smooth(-ei - ej)) / (4 * hh * hh)
hev = np.linalg.eigvalsh(Hs)
smooth_runs = []
for trial in range(24):
    c0 = RNG.normal(size=KMOD) / np.arange(1, KMOD + 1)
    c0 *= np.deg2rad(5 + 85 * trial / 23) / np.max(np.abs(SINS @ c0))       # largest off-plane angle 5 ... 90 deg
    r_ = minimize(J_smooth, c0, method="BFGS", options=dict(gtol=1e-12, maxiter=20000))
    smooth_runs.append(dict(J0_start=float(J_smooth(c0)), J=float(r_.fun), max_abs_c=float(np.max(np.abs(r_.x))),
                            max_y_start_deg=float(np.rad2deg(np.max(np.abs(SINS @ c0))))))
# a global scan: the two-leg great-circle paths through the equatorial point at azimuth az (az = 0 is the great circle)
Jmer = []
for az in np.deg2rad(np.arange(0, 360, 5)):
    ue_ = np.array([math.cos(az), math.sin(az), 0.0])
    Rm = path_init(np.array([0, 0, 1.0]), r1, ue_, K=2000)
    Jmer.append((float(np.rad2deg(az)), float(wlen(Rm)[0])))
put("F", "transfer_0L_to_1L_smooth", dict(
    J_great_circle=float(J0s), J_great_circle_exact=float(Jgc_exact), hessian_eigenvalues=[float(x) for x in hev],
    hessian_min_eig=float(hev.min()), n_random_starts=len(smooth_runs),
    min_J_over_starts=float(min(d["J"] for d in smooth_runs)), max_J_minus_Jgc=float(max(d["J"] - J0s for d in smooth_runs)),
    min_J_minus_Jgc=float(min(d["J"] - J0s for d in smooth_runs)),
    max_abs_c_at_optimum=float(max(d["max_abs_c"] for d in smooth_runs)),
    largest_start_deviation_deg=float(max(d["max_y_start_deg"] for d in smooth_runs)),
    two_leg_scan_min=min(Jmer, key=lambda z: z[1]), two_leg_scan_max=max(Jmer, key=lambda z: z[1]),
    exposure_great_circle_ns=1e3 * J0s / OM),
    "12 sine modes of off-plane deformation, Hessian by central differences, BFGS from 24 random starts (|y| up to ~90 deg)")
log("  F2: Hessian eig", hev.min(), hev.max(), "min J - Jgc", FACTS["F"]["transfer_0L_to_1L_smooth"]["value"]["min_J_minus_Jgc"])
# discrete optimum of F, re-evaluated without discretization error (resampled to 20000 points along its own arc length)
Rb_ = bestF["R"]
seg = np.arctan2(np.linalg.norm(np.cross(Rb_[:-1], Rb_[1:]), axis=1), np.sum(Rb_[:-1] * Rb_[1:], 1))
sl = np.r_[0, np.cumsum(seg)]
from scipy.interpolate import CubicSpline  # noqa: E402
cs_ = CubicSpline(sl, Rb_, axis=0)
Rfine = cs_(np.linspace(0, sl[-1], 20001))
Rfine /= np.linalg.norm(Rfine, axis=1, keepdims=True)
put("F", "discrete_optimum_resampled", dict(J=float(wlen(Rfine)[0]), minus_Jgc_exact=float(wlen(Rfine)[0] - Jgc_exact)),
    "the best discrete path, resampled, is not below the great circle")
# generic input r0 = (e_z - e_y)/sqrt2 (halfway between |0_L> and |-i_L>): gate path (a small circle at T*) against its
# own single-input optima within C
rg0 = np.array([0, -1.0, 1.0]) / math.sqrt(2)
Rb = bloch_rot(u_brach(TS))
rgT = Rb @ rg0
ts = np.linspace(0, TS, 2001)
gate_path = np.array([bloch_rot(u_brach(t)) @ rg0 for t in ts])
E_gate = 1e3 * simpson(nC + gate_path @ b, x=ts)
Rgc_g = path_init(rg0, rgT, None)
Jg_gc, Lg_gc = wlen(Rgc_g)
Rg_opt, Jg_opt, _ = optimize_path(rg0, rgT, Rgc_g, "generic")
cands = [(Jg_opt, Rg_opt)]
for az in (0, 90, 180, 270):                                      # starts bent to either side of the great circle
    mid = 0.5 * (rg0 + rgT)
    mid /= np.linalg.norm(mid)
    perp = np.cross(rg0, rgT)
    perp /= np.linalg.norm(perp)
    via = mid * math.cos(0.4) + perp * math.sin(0.4) * (1 if az < 180 else -1)
    R_, J_, _ = optimize_path(rg0, rgT, path_init(rg0, rgT, via / np.linalg.norm(via)), "generic bent")
    cands.append((J_, R_))
Jg_opt, Rg_opt = min(cands, key=lambda z: z[0])
_sg = np.arctan2(np.linalg.norm(np.cross(Rg_opt[:-1], Rg_opt[1:]), axis=1), np.sum(Rg_opt[:-1] * Rg_opt[1:], 1))
_sl = np.r_[0, np.cumsum(_sg)]
_Rf = CubicSpline(_sl, Rg_opt, axis=0)(np.linspace(0, _sl[-1], 20001))
_Rf /= np.linalg.norm(_Rf, axis=1, keepdims=True)
Jg_opt = wlen(_Rf)[0]                                           # resampled: free of discretization error
Jg_gc = wlen(path_init(rg0, rgT, None, K=20000))[0]
Lg_gc = wlen(path_init(rg0, rgT, None, K=20000))[1]
nrm_g = np.cross(rg0, rgT)
nrm_g /= np.linalg.norm(nrm_g)
put("F", "generic_input", dict(
    r0=[float(x) for x in rg0], target=[float(x) for x in rgT], gate_T_ns=1e3 * TS, gate_exposure_ns=E_gate,
    gate_speed_over_omega=float(np.linalg.norm(np.cross([0, 1.0, 0], rg0))),
    time_opt_T_ns=1e3 * Lg_gc / OM, time_opt_exposure_ns=1e3 * Jg_gc / OM,
    photon_opt_T_ns=1e3 * wlen(_Rf)[1] / OM, photon_opt_exposure_ns=1e3 * Jg_opt / OM,
    photon_vs_time_opt_saving=1 - Jg_opt / Jg_gc, photon_vs_time_opt_extra_time=wlen(_Rf)[1] / Lg_gc - 1,
    b_normal_to_its_great_circle=float(b @ nrm_g),
    max_dev_from_great_circle_deg=float(np.rad2deg(np.max(np.abs(np.arcsin(np.clip(Rg_opt @ nrm_g, -1, 1))))))),
    "per input the gate path is neither the input's time optimum nor its photon optimum, and within C the two single-input "
    "optima differ only when b has a component normal to the input's great circle")
CURVES.update(G_gate=gate_path, G_opt=Rg_opt, G_gc=Rgc_g)
log("  generic:", FACTS["F"]["generic_input"]["value"])
# |1_L> alone within C: the gate carries it from r1 to R r1 through -x (the side with more photons, b_x < 0) along its
# shortest great circle; the other great circle through +x is longer but has fewer photons
r1T = Rb @ r1
gate1 = np.array([bloch_rot(u_brach(t)) @ r1 for t in ts])
E_gate1 = 1e3 * simpson(nC + gate1 @ b, x=ts)
opts1 = []


def resampled(R_):
    seg_ = np.arctan2(np.linalg.norm(np.cross(R_[:-1], R_[1:]), axis=1), np.sum(R_[:-1] * R_[1:], 1))
    sl_ = np.r_[0, np.cumsum(seg_)]
    Rf = CubicSpline(sl_, R_, axis=0)(np.linspace(0, sl_[-1], 20001))
    return Rf / np.linalg.norm(Rf, axis=1, keepdims=True)


for via in (np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, -1.0, 0])):
    R_, J_, _ = optimize_path(r1, r1T, path_init(r1, r1T, via), "1L")
    Jr_, Lr_ = wlen(resampled(R_))
    opts1.append((Jr_, Lr_, via.tolist(), float(R_[:, 0].min()), float(R_[:, 0].max()),
                  float(np.rad2deg(np.max(np.abs(np.arcsin(np.clip(R_[:, 1], -1, 1))))))))
J1_opt, L1_opt, via1, xmin1, xmax1, offp1 = min(opts1, key=lambda z: z[0])
# the in-plane great circle through +x, exact
phs = np.linspace(2 * th, 4 * th - 2 * np.pi, 20001)              # polar angle from r1 back over +x to R r1
Rxp = np.array([np.sin(phs), 0 * phs, np.cos(phs)]).T
assert np.allclose(Rxp[0], r1) and np.allclose(Rxp[-1], r1T)
J1_xplane, L1_xplane = wlen(Rxp)
put("F", "transfer_1L_within_C", dict(
    gate_T_ns=1e3 * TS, gate_exposure_ns=E_gate1, gate_FS_length=float(th),
    photon_opt_exposure_ns=1e3 * J1_opt / OM, photon_opt_T_ns=1e3 * L1_opt / OM, photon_opt_start=via1,
    photon_opt_x_range=[xmin1, xmax1], photon_opt_max_off_plane_deg=offp1,
    saving_vs_gate=1 - 1e3 * J1_opt / OM / E_gate1, extra_time=L1_opt / th - 1,
    great_circle_via_plus_x=dict(exposure_ns=1e3 * J1_xplane / OM, T_ns=1e3 * L1_xplane / OM,
                                 saving_vs_gate=1 - 1e3 * J1_xplane / OM / E_gate1),
    all_starts=[dict(J_ns_resampled=1e3 * o[0] / OM, T_ns=1e3 * o[1] / OM, via=o[2], max_off_plane_deg=o[5]) for o in opts1]),
    "per input the gate path within C is not the photon optimum: |1_L> would rather pass the side with fewer photons")
log("  |1_L> within C:", FACTS["F"]["transfer_1L_within_C"]["value"])

# ===================================================================================== G  Lindblad check
if "--nolind" not in sys.argv:
    log("G: Lindblad check at equal duration 1.75 T*")
    a_op = np.diag(np.sqrt(np.arange(1, D)), 1).astype(complex)
    Nop = np.diag(n).astype(complex)
    TL = 1.75 * TS
    # flip 1: brachistochrone at omega/1.75; flip 2: detour via azimuth 90 deg slowed to 1.75 T*; flip 3: wiggle 1 (S2,
    # natural duration 1.70 T*) slowed to 1.75 T*.  All three obey Delta H <= omega on every code state.
    ue = np.array([0, 1.0, 0])
    m1 = np.cross([0, 0, 1.0], ue)
    m2 = np.cross(ue, r1)
    m2 /= np.linalg.norm(m2)
    a1, a2 = math.pi / 2, math.acos(np.clip(ue @ r1, -1, 1))
    umid = rot2(0.5 * a1, m1)

    def u_det(t):
        x = t / TL * (a1 + a2)
        return rot2(0.5 * x, m1) if x <= a1 else rot2(0.5 * (x - a1), m2) @ umid

    xw, usw, hsw, rsw, _ = wiggle(1)
    vw = np.linalg.norm(hsw, axis=1)
    tw = np.r_[0, np.cumsum(0.5 * (vw[1:] + vw[:-1]) * np.diff(xw))] / OM

    T_wig_natural = float(tw[-1])                                 # duration of wiggle 1 at Delta H <= omega (S2)
    tw *= TL / tw[-1]
    # closed form of wiggle 1 at arbitrary x (same construction as wiggle())
    rgw = np.random.default_rng(1)
    Kw = 4
    aw = rgw.normal(size=Kw) * 0.35 / np.arange(1, Kw + 1)
    bw = rgw.normal(size=Kw) * 0.8 / np.arange(1, Kw + 1)
    mw = rgw.normal(size=3)
    mw /= np.linalg.norm(mw)
    ksw = np.arange(1, Kw + 1)

    def u_wig(t):
        xx = float(np.interp(t, tw, xw))
        Th_ = th * xx + np.sin(np.pi * xx * ksw) @ aw
        Ph_ = np.sin(np.pi * xx * ksw) @ bw
        return rot2(Th_, (0, 1, 0)) @ rot2(Ph_, mw)

    assert np.abs(u_wig(TL) - usw[-1]).max() < 1e-12
    flips_L = {"brachistochrone_slow": lambda t: u_brach(t, OM / 1.75), "detour_90": u_det, "wiggle_1": u_wig}

    def lindblad(ufun, steps=600):
        dt_ = TL / steps
        rhos = np.array([np.outer(full(al), full(al).conj()) for _, al in octa])
        Id = np.eye(D) - Pi

        def U_of(t):
            return Q @ ufun(t) @ Q.T + Id

        def rhs(t, R):
            U = U_of(t)
            La = U.conj().T @ a_op @ U
            Ln = U.conj().T @ Nop @ U
            o = np.zeros_like(R)
            for L_, g_ in ((La, KAP), (Ln, KPHI)):
                LdL = L_.conj().T @ L_
                o += g_ * (L_ @ R @ L_.conj().T - 0.5 * (LdL @ R + R @ LdL))
            return o

        t = 0.0
        for _ in range(steps):
            k1_ = rhs(t, rhos)
            k2_ = rhs(t + dt_ / 2, rhos + dt_ / 2 * k1_)
            k3_ = rhs(t + dt_ / 2, rhos + dt_ / 2 * k2_)
            k4_ = rhs(t + dt_, rhos + dt_ * k3_)
            rhos = rhos + dt_ / 6 * (k1_ + 2 * k2_ + 2 * k3_ + k4_)
            t += dt_
        inf = [1 - float(np.real(np.vdot(full(al), r_ @ full(al)))) for (_, al), r_ in zip(octa, rhos)]
        tt = np.linspace(0, TL, 1201)
        fo = []
        for _, al in octa:
            vals = []
            for t_ in tt:
                q = full(ufun(t_) @ al)
                m1_ = np.real(np.vdot(q, n * q))
                vals.append(KAP * m1_ + KPHI * (np.real(np.vdot(q, n ** 2 * q)) - m1_ ** 2))
            fo.append(float(simpson(vals, x=tt)))
        return dict(infid_states=inf, infid_avg=float(np.mean(inf)), first_order_states=fo, first_order_avg=float(np.mean(fo)),
                    rel_dev=float(np.mean(inf) / np.mean(fo) - 1))

    LIND = {k: lindblad(f_) for k, f_ in flips_L.items()}
    fo_formula = TL * (KAP * nC + KPHI * VC)
    avgs = [v["infid_avg"] for v in LIND.values()]
    put("G", "lindblad_equal_duration", dict(T_ns=1e3 * TL, first_order_formula=fo_formula, flips=LIND,
                                              wiggle1_natural_T_ns=1e3 * T_wig_natural,
                                              wiggle1_within_speed_limit=bool(T_wig_natural <= TL),
                                              spread_of_averages_rel=float((max(avgs) - min(avgs)) / np.mean(avgs)),
                                              spread_per_state_rel=float(max(max(v["infid_states"]) - min(v["infid_states"]) for v in LIND.values()) / np.mean(avgs))),
        "three flips of equal duration that keep the qubit encoded: equal averaged Lindblad error, oscillator part only")
    log("  Lindblad", {k: (v["infid_avg"], v["rel_dev"]) for k, v in LIND.items()}, "formula", fo_formula)


# ===================================================================================== write
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    return o


FACTS["_about"]["runtime_s"] = time.time() - T0
(HERE / "num_facts.json").write_text(json.dumps(clean(FACTS), indent=1))
np.savez_compressed(HERE / "num_curves.npz", **CURVES, nC=nC, VC=VC, b=b, theta=th, c=c, TS_ns=1e3 * TS, omega=OM,
                    r1=r1, tilt_lams=np.array(LAMS))
log("wrote", HERE / "num_facts.json", HERE / "num_curves.npz")
