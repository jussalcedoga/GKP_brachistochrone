"""Fig. S (round 3): following against arriving, without a fidelity threshold.  GKP preparation along the
brachistochrone at T = 412, 824 and 2133 ns (panels a-c, one row; Oct 29 one-column SI layout): for every tracking
weight W of the ladder (10 ... 2000), the mean
+- s.d. over the 20 starts of Fig. 4 of the minimum closeness C (closed evolution) and of the target infidelity 1 - F_T
under the Lindblad equation (Yale rates), both at cutoff 60 as stored by the benchmark; trivial SNAP runs included.
Open marker: W = 10 (weakest tracking); filled marker: W = 2000.
Data: analytics_audit/round3/tradeoff.json (tradeoff.py).
python figures/SFig09_tradeoff/plot.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

import mfcommon as M  # noqa: E402

S = M.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

FS = M.FS                                    # same print sizes as the other one-row SI figures (S02, S06, S08)
plt.rcParams.update({"xtick.minor.size": 6.5, "xtick.minor.width": 2.0})
R = json.loads((M.MF / "analytics_audit" / "round3" / "tradeoff.json").read_text())

fig, axs = plt.subplots(1, 3, figsize=(15.6, 5.7), sharex=True, sharey=True)
fig.subplots_adjust(left=0.075, right=0.99, top=0.885, bottom=0.165, wspace=0.07)
for j, (ax, k, letter) in enumerate(zip(axs, (1, 2, 3), "abc")):
    for gate in ("snap", "cd", "cs"):
        o = R[f"{gate}/k{k}"]
        x = 1 - np.array(o["F_mean"])
        y = np.array(o["C_mean"])
        col = M.GATE_COLOR[gate]
        ax.errorbar(x, y, xerr=np.array(o["F_sd"]), yerr=np.array(o["C_sd"]), fmt="none", ecolor=col, elinewidth=1.7,
                    alpha=0.55, zorder=2)
        ax.plot(x, y, "-", color=col, lw=3.8, zorder=3)
        ax.plot(x[1:-1], y[1:-1], "o", ms=8, color=col, zorder=4)
        ax.plot(x[0], y[0], "o", ms=15, mfc="white", mec=col, mew=3.0, zorder=5)
        ax.plot(x[-1], y[-1], "o", ms=15, color=col, mec="white", mew=1.4, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(4e-3, 0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"target infidelity $1-\mathcal{F}_T$")        # Nov 9: script F for the target fidelity
    ax.set_title(f"$T={R[f'cs/k{k}']['T_ns']:.0f}$ ns", fontsize=FS["title"], pad=10)
    if j == 0:
        ax.set_ylabel(r"closeness $C$")
    S.panel_label(ax, letter, x=-0.25 if j == 0 else -0.07, y=1.03)
o = R["cs/k1"]
axs[0].annotate(r"$W=10$", (1 - o["F_mean"][0], o["C_mean"][0]), xytext=(10, -34), textcoords="offset points",
                fontsize=FS["ann"], color=S.INK)
axs[0].annotate(r"$W=2000$", (1 - o["F_mean"][-1], o["C_mean"][-1]), xytext=(12, 2), textcoords="offset points",
                fontsize=FS["ann"], color=S.INK)
hs = [Line2D([], [], color=M.GATE_COLOR[g], lw=3.8, marker="o", ms=9) for g in ("cs", "cd", "snap")]
axs[0].legend(hs, ["CS", "CD", "SNAP"], loc="lower left", fontsize=FS["legend"], frameon=False, handlelength=2.0,
              borderaxespad=0.5, labelspacing=0.35)
S.save(fig, HERE / "S23_tradeoff")
print("wrote", HERE / "S23_tradeoff.png")
