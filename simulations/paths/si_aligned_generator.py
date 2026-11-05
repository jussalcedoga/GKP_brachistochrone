"""SM, aligned generator for the logical flip: does the periodic potential cos(sqrt(pi) p) trace the flip brachistochrone?

On the ideal square GKP code space e^{+-i sqrt(pi) p} both act as X_L (they differ by the stabilizer e^{2 i sqrt(pi) p}), so
H_X = lambda cos(sqrt(pi) p) acts as lambda X_L: e^{-i H_X t}|0_L> = cos(lambda t)|0_L> - i sin(lambda t)|1_L>, a great circle
in span{|0_L>, |1_L>} -- the flip brachistochrone -- and its energy variance on |0_L> is lambda.  For finite-energy code
words (Delta = 0.3, 0.4, 0.5) we set lambda so that Delta H = omega = g/sqrt2 on |0_L> (same budget as every path of the
paper) and report the largest fidelity with |1_L>, its time against T* = theta/omega, the leakage out of the code space, the
photon exposure against the brachistochrone, and the gate test (the same evolution applied to |1_L> must return |0_L>).
cos(sqrt(pi) p) = [D(alpha) + D(-alpha)]/2, alpha = sqrt(pi/2) (a displacement by sqrt(pi) in q), built as expm in the Fock
basis at D = 90 and checked at D = 80.
Writes results/aligned_generator.json, aligned_generator.png/.pdf.   python simulations/paths/si_aligned_generator.py   (~1 min)
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
from scipy.linalg import eigh, expm  # noqa: E402

from common import states  # noqa: E402
from common.params import OMEGA_REF  # noqa: E402

trapz = getattr(np, "trapezoid", None) or np.trapz


def run(Delta, D, n_t=1201):
    a = np.diag(np.sqrt(np.arange(1, D)), 1)
    al = np.sqrt(np.pi / 2)
    Dp = expm(al * (a.T - a))
    C = 0.5 * (Dp + Dp.T)                                  # real symmetric: cos(sqrt(pi) p)
    u0, u1 = states.gkp(Delta, 0, D), states.gkp(Delta, 1, D)
    c = abs(np.vdot(u0, u1))
    th = np.arccos(c)
    m1 = np.real(np.vdot(u0, C @ u0))
    m2 = np.real(np.vdot(C @ u0, C @ u0))
    dC = np.sqrt(m2 - m1 ** 2)
    lam = OMEGA_REF / dC                                   # Delta H = omega on |0_L>
    ev, V = eigh(C)
    Tstar = th / OMEGA_REF
    t = np.linspace(0, 2 * Tstar, n_t)
    Pi = np.linalg.qr(np.array([u0, u1]).T)[0]
    nv = np.arange(D)
    out = {}
    for lab, psi0, tgt in (("0to1", u0, u1), ("1to0", u1, u0)):
        c0 = V.T @ psi0
        X = (V @ (np.exp(-1j * lam * np.outer(ev, t)) * c0[:, None])).T           # (n_t, D)
        F = np.abs(X @ tgt.conj()) ** 2
        leak = 1 - np.sum(np.abs(X.conj() @ Pi) ** 2, 1)
        nb = np.sum(np.abs(X) ** 2 * nv, 1)
        out[lab] = dict(F=F, leak=leak, nbar=nb)
    k = int(np.argmax(out["0to1"]["F"]))
    expo = lambda nb: float(trapz(nb[:k + 1], t[:k + 1]))                       # noqa: E731  us * photons
    s_geo = OMEGA_REF * t[:k + 1]                                                   # brachistochrone at the same time
    n0, n1, N01 = (np.real(np.vdot(x, nv * y)) for x, y in ((u0, u0), (u1, u1), (u0, u1)))
    ng = (np.sin(th - s_geo) ** 2 * n0 + np.sin(s_geo) ** 2 * n1 + 2 * np.sin(th - s_geo) * np.sin(s_geo) * N01) / np.sin(th) ** 2
    return dict(Delta=Delta, D=D, theta=th, T_star_ns=1e3 * Tstar, lam_over_omega=lam / OMEGA_REF, mean_C=m1, dC=dC,
                t_ns=(1e3 * t).tolist(), k_best=k, T_best_ns=1e3 * t[k], F_best=float(out["0to1"]["F"][k]),
                F_gate_1to0=float(out["1to0"]["F"][k]), peak_leak=float(out["0to1"]["leak"][:k + 1].max()),
                expo_ns=1e3 * expo(out["0to1"]["nbar"]), expo_geo_ns=1e3 * float(trapz(ng, t[:k + 1])) if k else 0.0,
                F_0to1=out["0to1"]["F"].tolist(), F_1to0=out["1to0"]["F"].tolist(), leak=out["0to1"]["leak"].tolist(),
                nbar=out["0to1"]["nbar"].tolist())


res = {}
for Delta in (0.3, 0.4, 0.5):
    r = run(Delta, 90)
    r80 = run(Delta, 80)
    r["check_D80"] = dict(F_best=r80["F_best"], T_best_ns=r80["T_best_ns"], F_gate_1to0=r80["F_gate_1to0"])
    res[f"{Delta:.1f}"] = r
    print(f"Delta {Delta}: lambda/omega = {r['lam_over_omega']:.4f}, <C> = {r['mean_C']:.4f}, F(|1_L>) max {r['F_best']:.5f} at "
          f"{r['T_best_ns']:.1f} ns (T* = {r['T_star_ns']:.1f} ns), gate |1_L> -> |0_L> {r['F_gate_1to0']:.5f}, peak leakage "
          f"{100 * r['peak_leak']:.2f}%, exposure {r['expo_ns']:.0f} vs brachistochrone {r['expo_geo_ns']:.0f} photon ns; "
          f"D80: F {r80['F_best']:.5f}, gate {r80['F_gate_1to0']:.5f}", flush=True)
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "aligned_generator.json").write_text(json.dumps(res))

from common import style as S  # noqa: E402,I001
import matplotlib.pyplot as plt  # noqa: E402

S.style(usetex=True)
S.big()
fig, axs = plt.subplots(1, 3, figsize=(14.5, 5.6))
fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.17, wspace=0.42)
r = res["0.4"]
t = np.array(r["t_ns"])
ax = axs[0]
ax.plot(t, np.cos(r["theta"] - OMEGA_REF * t * 1e-3) ** 2, **S.PATH_STYLE["geodesic"], label="brachistochrone")
ax.plot(t, r["F_0to1"], color=S.CRIMSON, lw=3.2, ls=(0, (5, 2)), label=r"$\lambda\cos(\sqrt\pi\,\hat p)$")
ax.axvline(r["T_star_ns"], color=S.DARK, lw=1.5, ls=(0, (4, 3)))
ax.text(r["T_star_ns"] + 8, 0.03, r"$T^*$", fontsize=S.FS["small"], color=S.DARK)
ax.annotate(rf"${r['F_best']:.3f}$", xy=(r["T_best_ns"], r["F_best"]), xytext=(r["T_best_ns"] + 150, 0.90), fontsize=S.FS["small"],
            color=S.CRIMSON, arrowprops=dict(arrowstyle="-", color=S.CRIMSON, lw=1.4))
ax.set_xlim(0, t[-1])
ax.set_ylim(0, 1.04)
ax.set_xlabel(r"time $t$ (ns)")
ax.set_ylabel(r"fidelity with the other code word")
ax.set_title(r"$\Delta=0.4$, $\Delta H=\omega$ (both code words)", fontsize=S.FS["ann"])
ax.legend(loc="upper left", fontsize=S.FS["small"], handlelength=2.2)
S.panel_label(ax, "a")
ax = axs[1]
for d, col in (("0.3", S.NAVY), ("0.4", S.ROYALBLUE), ("0.5", "#9DB4F7")):
    ax.plot(np.array(res[d]["t_ns"]) / res[d]["T_star_ns"], 100 * np.array(res[d]["leak"]), color=col, lw=3.0,
            label=rf"$\Delta={d}$")
ax.axvline(1.0, color=S.DARK, lw=1.5, ls=(0, (4, 3)))
ax.set_xlim(0, 2)
ax.set_ylim(0, None)
ax.set_xlabel(r"$t/T^*$")
ax.set_ylabel(r"leakage out of the code space (\%)")
ax.legend(loc="upper left", fontsize=S.FS["small"])
ax.set_title("leakage of the evolving code word", fontsize=S.FS["ann"])
S.panel_label(ax, "b")
ax = axs[2]
ds = [0.3, 0.4, 0.5]
ax.plot(ds, [1 - res[f"{d:.1f}"]["F_best"] for d in ds], "o-", color=S.CRIMSON, ms=11, mec="k", lw=3.0)
for d in ds:
    e = res[f"{d:.1f}"]
    ax.annotate(rf"$F={e['F_best']:.3f}$", xy=(d, 1 - e["F_best"]), xytext=(10, -6), textcoords="offset points",
                fontsize=S.FS["small"], ha="left", va="top")
ax.set_yscale("log")
ax.set_xlim(0.27, 0.56)
ax.set_ylim(0.015, 0.22)
ax.set_yticks([0.02, 0.05, 0.1, 0.2])
ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2"])
ax.set_xlabel(r"GKP envelope $\Delta$")
ax.set_ylabel(r"infidelity at the best time")
ax.set_title(r"finite-energy error (best time $\approx0.99\,T^*$)", fontsize=S.FS["ann"])
S.panel_label(ax, "c")
S.save(fig, HERE / "aligned_generator")
print("wrote", HERE / "aligned_generator.png")