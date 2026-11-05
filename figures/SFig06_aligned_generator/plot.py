"""Fig. S5, a generator aligned with the flip: the periodic potential lambda cos(sqrt(pi) p) at Delta H = omega on |0_L>
(lambda X_L on the ideal code space).  Re-plotted from manuscript/si/aligned_generator/results/aligned_generator.json
(written by simulations/paths/si_aligned_generator.py, cutoff 90, checked at 80).
python figures/SFig06_aligned_generator/plot.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

import mfcommon as M  # noqa: E402

S = M.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.params import OMEGA_REF  # noqa: E402

FS = S.FS
res = json.loads((M.MS / "si" / "aligned_generator" / "results" / "aligned_generator.json").read_text())
fig, axs = plt.subplots(1, 3, figsize=(15.0, 5.8))
fig.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.17, wspace=0.40)

# (a) fidelity with the other code word, Delta = 0.4
ax = axs[0]
r = res["0.4"]
t = np.array(r["t_ns"])
ax.plot(t, np.cos(r["theta"] - OMEGA_REF * t * 1e-3) ** 2, **dict(S.PATH_STYLE["geodesic"], lw=M.LW),
        label="brachistochrone")
ax.plot(t, r["F_0to1"], color=S.CRIMSON, lw=M.LW - 0.6, ls=(0, (5, 2)), label=r"$h\cos(\sqrt\pi\,\hat p)$")
ax.axvline(r["T_star_ns"], color="#111827", lw=1.8, ls=(0, (4, 3)), ymax=1.04 / 1.36)   # stops below the legend
ax.text(r["T_star_ns"] + 10, 0.04, r"$T^*$", fontsize=FS["small"], color="#111827")
ax.set_xlim(0, t[-1])
ax.set_ylim(0, 1.36)
ax.set_yticks([0, 0.5, 1])
ax.set_xlabel(r"time $t$ (ns)")
ax.set_ylabel(r"fidelity with $|1_L\rangle$")
ax.set_title(rf"$\Delta=0.4$: max ${r['F_best']:.3f}$ at ${r['T_best_ns'] / r['T_star_ns']:.2f}\,T^*$", fontsize=FS["ann"],
             pad=8)
ax.legend(loc="upper right", fontsize=FS["small"] - 2, handlelength=1.8, handletextpad=0.5, borderaxespad=0.1,
          labelspacing=0.3)
S.panel_label(ax, "a", x=-0.27, y=1.04)

# (b) leakage out of span{|0_L>, |1_L>}
ax = axs[1]
for d, col in (("0.3", S.NAVY), ("0.4", S.ROYALBLUE), ("0.5", "#9DB4F7")):
    ax.plot(np.array(res[d]["t_ns"]) / res[d]["T_star_ns"], 100 * np.array(res[d]["leak"]), color=col, lw=M.LW - 0.6,
            label=rf"$\Delta={d}$")
ax.axvline(1.0, color="#111827", lw=1.8, ls=(0, (4, 3)))
ax.set_xlim(0, 2)
ax.set_ylim(0, 52)
ax.set_xlabel(r"$t/T^*$")
ax.set_ylabel(r"leakage (\%)")
ax.legend(loc="upper left", fontsize=FS["small"] - 1, handlelength=1.6)
ax.set_title("weight outside the code space", fontsize=FS["ann"], pad=8)
S.panel_label(ax, "b", x=-0.27, y=1.04)

# (c) infidelity at the best time against the envelope
ax = axs[2]
ds = [0.3, 0.4, 0.5]
ax.plot(ds, [1 - res[f"{d:.1f}"]["F_best"] for d in ds], "o-", color=S.CRIMSON, ms=14, mec="k", lw=M.LW - 0.8)
for d in ds:
    e = res[f"{d:.1f}"]
    ax.annotate(rf"$F={e['F_best']:.3f}$", xy=(d, 1 - e["F_best"]), xytext=(12, -8), textcoords="offset points",
                fontsize=FS["small"] - 1, ha="left", va="top")
ax.set_yscale("log")
ax.set_xlim(0.27, 0.58)
ax.set_ylim(0.015, 0.25)
ax.set_yticks([0.02, 0.05, 0.1, 0.2])
ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2"])
ax.set_xlabel(r"GKP envelope $\Delta$")
ax.set_ylabel(r"$1-F$ at the best time")
ax.set_title("finite-energy error", fontsize=FS["ann"], pad=8)
S.panel_label(ax, "c", x=-0.27, y=1.04)
S.save(fig, HERE / "S06_aligned_generator")
for d in ("0.3", "0.4", "0.5"):
    e = res[d]
    print(f"Delta={d}: lambda/omega={e['lam_over_omega']:.4f} <cos>={e['mean_C']:.4f} T*={e['T_star_ns']:.1f} ns  best F={e['F_best']:.4f} "
          f"at {e['T_best_ns']:.1f} ns ({e['T_best_ns'] / e['T_star_ns']:.3f} T*)  gate 1->0 {e['F_gate_1to0']:.4f}  "
          f"peak leak {100 * e['peak_leak']:.2f}%  exposure {e['expo_ns']:.0f} vs brachistochrone {e['expo_geo_ns']:.0f} photon ns  "
          f"D80: F {e['check_D80']['F_best']:.5f} (D90 {e['F_best']:.5f})")
print("wrote", HERE / "S06_aligned_generator.png")
