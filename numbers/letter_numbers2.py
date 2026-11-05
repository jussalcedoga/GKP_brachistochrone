"""Letter verification, part 2: eps_1 of the transfer with and without cavity dephasing (brachistochrone against the
photon-optimal transfer rebuilt from its stored nodes), the photon-optimal time of |1_L> at Delta = 0.3 (R2), and the
Fig. 1 sphere weight of the photon-optimal transfer. Independent GKP construction as in letter_numbers.py."""
import json
import math
import pathlib

import numpy as np
from scipy.special import gammaln

import layout as _L  # repository layout (src/layout.py)
R = _L.DATA                            # stored results, under their original relative paths
MF = R / "manuscript_figures"
OM = 7.5 / math.sqrt(2) / 1000.0  # 1/ns
ALPHA = math.sqrt(math.pi / 2)


def gkp(mu, Delta, D, M=12):
    n = np.arange(D)
    amp = np.zeros(D, dtype=complex)
    for m in range(-M, M + 1):
        if (m - mu) % 2:
            continue
        for l in range(-M, M + 1):
            z = m * ALPHA + 1j * l * ALPHA
            if z == 0:
                t = np.zeros(D, dtype=complex); t[0] = 1
            else:
                t = np.exp(n * np.log(z) - 0.5 * gammaln(n + 1))
            amp += np.exp(-1j * math.pi * m * l / 2 - abs(z) ** 2 / 2) * t
    amp *= np.exp(-Delta ** 2 * n)
    amp /= np.linalg.norm(amp)
    if amp[0].real < 0:
        amp = -amp
    return amp.real


# rates (Sivak et al., per ns): T1c = 606 us, T2c = 980 us -> kappa_phi = 2/T_phi_c; Gamma_q = (476 us)^-1
kap = 1 / 606e3
kphi = 2 * (1 / 980e3 - 1 / (2 * 606e3))
Gq = 1 / (2 * 414e3) + 1 / (4 * 280e3)
print(f"rates per us: kappa={kap * 1e3:.6f} kappa_phi={kphi * 1e3:.6f} Gamma_q={Gq * 1e3:.6f} (1/Gamma_q = {1 / Gq / 1e3:.0f} us)")

P = np.load(MF / "analytics_audit/photon_flip/results/paths.npz")
o0, o1 = P["opt0"], P["opt1"]
De = o0.shape[1]
nev = 2.0 * np.arange(De)  # even Fock numbers
y0, y1 = gkp(0, 0.4, 2 * De)[::2], gkp(1, 0.4, 2 * De)[::2]
c = y0 @ y1
th = math.acos(c)
p1 = (y1 - c * y0) / math.sin(th)


def integrals(states, dphi):
    nb = np.array([s @ (nev * s) for s in states])
    vr = np.array([s @ (nev ** 2 * s) for s in states]) - nb ** 2
    return nb, vr


# brachistochrone
phis = np.linspace(0, th, 4001)
st = [math.cos(f) * y0 + math.sin(f) * p1 for f in phis]
nb, vr = integrals(st, None)
Tg = th / OM
In_g = np.trapezoid(nb, phis) / OM
Iv_g = np.trapezoid(vr, phis) / OM
# photon-optimal transfer from nodes: reversed preparation of 0_L then preparation of 1_L, great-circle arcs
nodes = np.vstack([o0[::-1], o1[1:]])
In_p = Iv_p = Tp = 0.0
for a_, b_ in zip(nodes[:-1], nodes[1:]):
    a_ = a_ / np.linalg.norm(a_); b_ = b_ / np.linalg.norm(b_)
    ca = np.clip(a_ @ b_, -1, 1)
    ang = math.acos(ca)
    if ang < 1e-14:
        continue
    q = (b_ - ca * a_) / math.sin(ang)
    ff = np.linspace(0, ang, 41)
    s_ = [math.cos(f) * a_ + math.sin(f) * q for f in ff]
    n_, v_ = integrals(s_, None)
    In_p += np.trapezoid(n_, ff) / OM
    Iv_p += np.trapezoid(v_, ff) / OM
    Tp += ang / OM
print(f"brachistochrone: T={Tg:.1f} ns, int nbar dt={In_g:.1f}, int Var dt={Iv_g:.1f}")
print(f"photon optimum : T={Tp:.1f} ns, int nbar dt={In_p:.1f}, int Var dt={Iv_p:.1f}")
RED = {}
for lab, kp in (("without cavity dephasing", 0.0), ("with cavity dephasing", kphi)):
    eg = Gq * Tg + kap * In_g + kp * Iv_g
    ep = Gq * Tp + kap * In_p + kp * Iv_p
    RED[lab] = 1 - ep / eg
    print(f"eps_1 {lab}: brachistochrone {eg:.3e}, photon optimum {ep:.3e}, reduction {100 * (1 - ep / eg):.1f}%")
F3 = json.loads((R / "manuscript/fig3_time_vs_photon/results/fig3.json").read_text())["tradeoff"]["flip"]["eps"]["Yale"]
print(f"stored (fig3.json, Yale, no dephasing): photon {100 * (1 - F3['eps_photon'] / F3['eps_geo']):.1f}%, "
      f"trade-off w* {100 * (1 - F3['at_w_star']['eps'] / F3['eps_geo']):.1f}%")

# Fig. 1 weight of the photon-optimal transfer in span{0_L, 1_L, out-of-plane part of the vacuum}
vac = np.zeros(De); vac[0] = 1
d = vac - (vac @ y0) * y0 - (vac @ p1) * p1
d /= np.linalg.norm(d)
w = [(s @ y0) ** 2 + (s @ p1) ** 2 + (s @ d) ** 2 for s in nodes / np.linalg.norm(nodes, axis=1)[:, None]]
print(f"info  Fig. 1(b) photon-optimal transfer: min weight in span(0_L,1_L,vacuum direction) over nodes {min(w):.5f} "
      "(the vacuum direction, not the figure's; the printed 0.987 is checked in letter_numbers.py)")

# R2: photon-optimal time of |1_L> at Delta 0.3 (second half of the Delta = 0.3 flip)
A2 = json.loads((MF / "analytics_audit/photon_flip/results/audit_photon2.json").read_text())["flip_D0.3"]
gal = json.loads((R / "manuscript/si/state_gallery/results/state_gallery.json").read_text())["targets"]["gkp03"]
T0 = gal["L_ph"] / OM
z1 = gkp(1, 0.3, 90, M=14)
t1 = math.acos(z1[0])
print(f"R2: T_sum={A2['T_sum_ns']:.1f} ns, |0_L>@0.3 photon optimum {T0:.1f} ns (J {gal['J_ph'] / OM / 1e0 if gal['J_ph'] < 10 else gal['J_ph']:.4g}),"
      f" so |1_L>@0.3 photon optimum {A2['T_sum_ns'] - T0:.1f} ns vs T* {t1 / OM:.1f} ns (+{100 * ((A2['T_sum_ns'] - T0) / (t1 / OM) - 1):.1f}%)")
print("gallery gkp03 keys/values:", {k: (round(v, 5) if isinstance(v, float) else type(v).__name__) for k, v in gal.items()})

# ---------------- checks against the values printed in the Letter ----------------
fails = []


def inrange(x, lo, hi, what):
    ok = lo <= x <= hi
    print(f"{'PASS' if ok else 'FAIL'}  value  {what}: {x:.6g} in [{lo}, {hi}]")
    if not ok:
        fails.append(what)


print()
inrange(RED["without cavity dephasing"], 0.215, 0.225, "eps_1 lowered by 22% without cavity dephasing (photon optimum, from its nodes)")
inrange(1 - F3["eps_photon"] / F3["eps_geo"], 0.215, 0.225, "eps_1 lowered by 22% without cavity dephasing (photon optimum, fig3.json)")
inrange(1 - F3["at_w_star"]["eps"] / F3["eps_geo"], 0.255, 0.265, "eps_1 lowered by 26% without cavity dephasing (trade-off w*)")
inrange(RED["with cavity dephasing"], 0.35, 0.45, "eps_1 lowered by about 40% with cavity dephasing (photon optimum)")
print(f"\n{len(fails)} mismatch(es): {fails}" if fails else "\nALL CHECKED LETTER NUMBERS MATCH THE DATA")
raise SystemExit(1 if fails else 0)
