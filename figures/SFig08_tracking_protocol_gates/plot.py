"""Supplementary Fig. S08, the tracking picture for the three gate sets: GKP preparation (Delta = 0.4) along the
brachistochrone in T = 824 ns, oscillator state in the plane of |0> and the path direction |chi> (as in main Fig. 4a), with
one slice t_k = T/2 highlighted: path state p(t_k), tangent dp/dt, the states of the plane with closeness F_k >= 0.97 (the
polish threshold of Eq. (S loss); circular segment, exact for states of the plane), and the state rho(t_k) of one run (the
start closest to the mean, Lindblad evolution, measured rates).  Every run is kept (mfcommon.usable returns all runs,
trivial SNAP runs included).  Data: manuscript/fig4_cs_path_following/results/traces80.
python figures/SFig08_tracking_protocol_gates/plot.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

import mfcommon as M  # noqa: E402

S = M.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402
import numpy as np  # noqa: E402

from common import brachistochrone as B  # noqa: E402
import compute as CP  # noqa: E402

FS = S.FS
Sv, Tv = CP.transfer("gkp", 80)
_, th0, _ = B.geodesic_parts(Sv, Tv)
G = M.groups()
NK, KK, THR, K_DUR = 8, 4, 0.97, 2           # slice points drawn, highlighted slice (t_k = T/2), threshold, T = 824 ns
ak = th0 * np.arange(NK + 1) / NK
nk = np.c_[np.sin(2 * ak), np.cos(2 * ak)]    # path state cos a|0> + sin a|chi>  ->  (x, z) = (sin 2a, cos 2a)
RHO_OFF = {"cs": (-86, -6), "cd": (-30, -54), "snap": (18, -2)}
info = {}

fig, axs = plt.subplots(1, 3, figsize=(15.6, 6.6))
fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.14, wspace=0.07)
for j, (ax, gate) in enumerate(zip(axs, ("cs", "cd", "snap"))):
    col = M.GATE_COLOR[gate]
    allr = G[("gkp", "geodesic", gate, K_DUR)]
    lst = M.usable(allr)
    a = np.linspace(0, th0, 300)
    t = np.linspace(0, np.pi / 2, 300)
    ax.plot(np.sin(t), np.cos(t), color="#111827", lw=1.4, ls=(0, (2, 2)))
    ax.plot(np.sin(2 * a), np.cos(2 * a), color=S.ROYALBLUE, lw=M.LW + 0.6, ls=(0, (5, 2)), zorder=3)
    phi = np.arccos(2 * THR - 1)                     # F = (1 + r.n_k)/2 >= thr  <=>  r.n_k >= 2 thr - 1
    arc = np.linspace(2 * ak[KK] - phi, 2 * ak[KK] + phi, 100)
    ax.add_patch(Polygon(np.c_[np.sin(arc), np.cos(arc)], closed=True, fc=S.ROYALBLUE, alpha=0.20, ec=S.ROYALBLUE,
                         lw=1.4, zorder=1))
    for r in lst:
        ax.plot(r["xL"], r["zL"], color=col, lw=M.LW_THIN, alpha=0.22, zorder=2)
    rep = M.representative(allr)
    mx, mz, mF = (np.array(rep[k]) for k in ("xL", "zL", "FL"))
    ax.plot(mx, mz, color=col, lw=M.LW, zorder=4)
    i = int(round(KK / NK * (len(mx) - 1)))
    ax.plot([nk[KK, 0], mx[i]], [nk[KK, 1], mz[i]], color=S.INK, lw=1.8, zorder=5)
    ax.plot(mx[i], mz[i], "o", ms=14, color=col, mec="k", mew=1.1, zorder=6)
    ax.annotate(r"$\rho(t_k)$", (mx[i], mz[i]), xytext=RHO_OFF[gate], textcoords="offset points", fontsize=FS["ann"],
                color=col)
    ax.plot(nk[:, 0], nk[:, 1], "o", ms=11, mfc="white", mec=S.ROYALBLUE, mew=2.4, zorder=7)
    tang = np.array([np.cos(2 * ak[KK]), -np.sin(2 * ak[KK])])
    ax.add_patch(FancyArrowPatch(nk[KK], nk[KK] + 0.40 * tang, arrowstyle="-|>", mutation_scale=28, lw=3.2,
                                 color=S.ROYALBLUE, zorder=8))          # the tangent in the path colour, as in Fig. 3(a)
    ax.text(*(nk[KK] + 0.43 * tang + np.array([0.02, 0.03])), r"$\dot p(t_k)$", fontsize=FS["ann"], color=S.ROYALBLUE,
            zorder=8)
    ax.annotate(r"$p(t_k)$", nk[KK], xytext=(10, 12), textcoords="offset points", fontsize=FS["ann"], color=S.ROYALBLUE)
    ax.text(0.0, 1.05, r"$|0\rangle$", fontsize=FS["ann"], ha="center", va="bottom")
    ax.text(nk[-1, 0] + 0.05, nk[-1, 1], r"$|0_L\rangle$", fontsize=FS["ann"], ha="left", va="center")
    nex = len(allr) - len(lst)
    ax.text(0.98, 0.98, rf"$F_k={mF[i]:.2f}$ at $t_k=T/2$" + "\n" + f"{len(lst)} starts" + (f" ({nex} excluded)" if nex else ""),
            transform=ax.transAxes, ha="right", va="top", fontsize=FS["small"] - 1, color=col, linespacing=1.2)
    ax.set_xlim(-0.08, 1.24)
    ax.set_ylim(-0.05, 1.20)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$x=2\,{\rm Re}\langle0|\rho|\chi\rangle$")
    ax.set_title(M.GATE_SHORT[gate], fontsize=FS["title"] + 2, color=col, pad=8)
    if j == 0:
        ax.set_ylabel(r"$z=\langle0|\rho|0\rangle-\langle\chi|\rho|\chi\rangle$")
    else:
        ax.set_yticklabels([])
    S.panel_label(ax, "abc"[j], x=-0.24 if j == 0 else -0.07, y=1.02)
    info[gate] = dict(n=len(allr), usable=len(lst), seed=rep["seed"], F_k=float(mF[i]), C=rep["C_noisy"],
                      F_T=rep["F_T_final_noisy"], C_mean=M.stat(lst, "C_noisy"), F_T_mean=M.stat(lst, "F_T_final_noisy"))
S.save(fig, HERE / "S08_tracking_protocol_gates")
for g, d in info.items():
    print(f"{g}: {d['usable']}/{d['n']} usable; representative seed {d['seed']}: F_k(T/2)={d['F_k']:.3f} C={d['C']:.3f} "
          f"F_T={d['F_T']:.4f}; usable mean C={d['C_mean'][0]:.3f}+-{d['C_mean'][1]:.3f}, F_T={d['F_T_mean'][0]:.4f}+-{d['F_T_mean'][1]:.4f}")
print("wrote", HERE / "S08_tracking_protocol_gates.png")
