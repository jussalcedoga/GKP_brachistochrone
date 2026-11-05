"""Physical parameters of the manuscript, each with its source.

Tags: [LIT] taken from a paper (reference given), [DEF] a definition of this work, [ASSUMPTION] a modelling choice
that is not fixed by a paper (its effect is tested in the Supplement).

Unit convention.  All rates are angular rates in rad/us (equivalently us^-1); times in ns unless stated.  Every
dynamical result is scale-free in units of 1/G_SQ, so a different squeezing rate rescales all times by the same factor.
"""
from __future__ import annotations

import math

# ----------------------------------------------------------------------------- conditional squeezing hardware
G_SQ = 7.5          # [LIT] squeezing rate of the qubit-conditioned squeeze, r = g t; Del Grosso et al., PRA 111, 042606
                    #       (2025): g_d = 7.5 x 10^6 s^-1, r ~ 1.5 in 200 ns.  (The protocol deck used g = 15 rad/us,
                    #       i.e. twice this value; its times convert to ours by a factor 2.)
OMEGA_REF = G_SQ / math.sqrt(2)   # [DEF] Fubini-Study speed of the squeezer on the vacuum, DeltaH = g/sqrt2
                                  #       = 5.303 rad/us: the energy-variance budget at which every analytic path is
                                  #       traversed (T = L / OMEGA_REF), as in the photon-optimal-path note.
RABI_MAX = 10.0 * G_SQ            # [ASSUMPTION] qubit drive bound |Omega| <= 10 g = 75 rad/us (pi pulse in 41.9 ns);
                                  #       the ratio 10 follows the protocol deck; S3 varies it.

# ----------------------------------------------------------------------------- targets (main text)
GKP_DELTA = 0.4     # [ASSUMPTION] finite-energy envelope of the square GKP code (fixes theta_0); the photon-path note
                    #       and the protocol deck use the same value; S2 scans Delta = 0.3 ... 0.5.
CAT_ALPHA = 2.0     # [ASSUMPTION] even cat |alpha> + |-alpha>, alpha = 2 (as in the deck and the note)
FOCK_TARGETS = (2, 4)

# ----------------------------------------------------------------------------- numerics
D_PREP = 60         # [DEF] Fock cutoff for GRAPE (population at n >= 2D/3 penalized); checks at D_CHECK
D_CHECK = 80
D_PATH = 80         # cutoff for analytic paths (converged: top-quarter weight < 1e-5)
F_STAR = 0.99       # [DEF] target fidelity of a feasible pulse

# ----------------------------------------------------------------------------- noise (Lindblad evaluation only)
# rates in rad/us = 1/us.  'mu' is the n-independent error-rate proxy of the analytic first-order cost
# eps = (mu T + kappa int<n>dt)  (report_analytics, Ref. [1] noise model): mu = 1/(2 T_phi,q) + 1/(4 T_1,q).
NOISE = {
    "Yale": dict(T1c=606.0, T1q=280.0, Tphiq=414.0, Tphic=5100.0,
                 source="Sivak et al., Nature 616, 50 (2023), Table S1 / Sec. S2 E (bare cavity and ancilla)"),
    "CSproposal": dict(T1c=200.0, T1q=200.0, Tphiq=10.0, Tphic=None,
                       source="Del Grosso et al., PRA 111, 042606 (2025), Sec. IV (tau_r = tau_q = 200 us, tau_phi = 10 us)"),
    "deck": dict(T1c=1000.0, T1q=1000.0, Tphiq=300.0, Tphic=None, kerr_per_s=100.0,
                 source="38-slide deck 'light loss set' (an assumption of the deck), for the cross-check only"),
}


def kappa_mu(name):
    """(kappa, mu) in 1/us of a noise set: photon loss rate and the n-independent proxy rate."""
    s = NOISE[name]
    kappa = 1.0 / s["T1c"]
    mu = 1.0 / (2 * s["Tphiq"]) + 1.0 / (4 * s["T1q"])
    return kappa, mu


def ns(t_in_units_of_inverse_g, g=G_SQ):
    """convert a dimensionless time g t into ns"""
    return 1e3 * t_in_units_of_inverse_g / g
