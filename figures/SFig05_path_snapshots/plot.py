"""SI figure: the states along the three optimal paths of the logical flip |0_L> -> |1_L> (Delta = 0.4).
Rows: the time-optimal path (brachistochrone), the photon-optimal path and the trade-off path (minimizer of
int(1 + w*<n>)ds, w* = kappa/mu = 0.79, Yale), each traversed at the speed limit Delta H = omega = g/sqrt2 (T = L/omega).
Columns: Wigner functions of the path state at t/T = 0, 0.2, ..., 1 (|q|,|p| <= 5, common colour scale +-1/pi); last
column: the effect of noise on the path itself, Delta W = W(rho(T)) - W(|1_L>), where rho(T) is the oscillator state
reached when the path is generated arc by arc by its rank-two Hamiltonians under the Lindblad equation with the Yale rates
(photon loss, cavity dephasing; the qubit idles), with 1 - F = 1 - <1_L|rho(T)|1_L> (separate symmetric colour scale).
Path states: the reference paths of the tracking runs (cutoff 80, supplementary_information/_cache).  W and Delta W exact on
the shown window (mfcommon.wigner_grid, review Nov 1, L1-F1), saved to <folder>_wigner.npz.
python manuscript_figures/supplementary_information/<this folder>/plot.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
import mfcommon as M  # noqa: E402,I001

S = M.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

from common import grape, lindblad, tracking  # noqa: E402
from common.params import OMEGA_REF  # noqa: E402

FS = M.FS
CACHE = M.MF / "supplementary_information" / "_cache"
D = 80
FR = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
ROWS = (("geodesic", "time optimal", S.ROYALBLUE), ("photon", "photon optimal", S.CRIMSON),
        ("tradeoff", r"trade-off ($w^*$)", S.TRADEOFF))
Wg = M.wigner_grid(D, 161)                                     # |q|,|p| <= 5, exact on the shown window (L1-F1)
noise = lindblad.rates("Yale")
out = {}
data = {}
for ref, _, _ in ROWS:
    X = np.load(CACHE / f"path_flip_{ref}.npz")["X"][:, :D]
    P = tracking.Path(X)
    T = P.L / OMEGA_REF                                          # us
    G, _ = P.at(np.array(FR))
    Ws = [Wg.of_vectors((g / np.linalg.norm(g))[None]) for g in G]
    pm = tracking.PathModel(P, D, 400)                           # as tracking.path_lindblad_F: rank-two arc generators
    C = np.full((400, 6), 400 / T)
    rho = lindblad.evolve(pm, C, T / 400, grape.joint(P.P[0][:D]), noise, dt_max=2e-3)
    r = rho.reshape(2, D, 2, D)
    ro = r[0, :, 0, :] + r[1, :, 1, :]
    tgt = G[-1] / np.linalg.norm(G[-1])
    F = float(np.real(np.vdot(tgt, ro @ tgt)))
    dW = Wg.of_rho(ro) - Wg.of_vectors(tgt[None])
    data[ref] = (Ws, dW)
    nb = [float(np.real(np.vdot(g, np.arange(D) * g)) / np.vdot(g, g).real) for g in G]
    out[ref] = dict(T_ns=1e3 * T, L=P.L, infidelity_Yale=1 - F, nbar_at_snapshots=nb)
    print(f"{ref}: T={1e3 * T:.1f} ns  1-F (Yale, path itself) = {1 - F:.2e}  <n> at snapshots:",
          " ".join(f"{v:.2f}" for v in nb), flush=True)
dmax = max(float(np.max(np.abs(d[1]))) for d in data.values())
DCMAP = LinearSegmentedColormap.from_list("dW", [S.ROYALBLUE, "#FFFFFF", S.CRIMSON])
PW, GAP, LAB, DGAP = 1.80, 0.06, 0.95, 0.95     # DGAP: space between the Wigner block and the Delta W column
W_IN = LAB + 7 * PW + 6 * GAP + DGAP + 0.95
H_IN = 0.75 + 3 * PW + 2 * GAP + 1.45
fig = plt.figure(figsize=(W_IN, H_IN))


def ax_at(x0, y_top, w, h):
    return fig.add_axes([x0 / W_IN, 1 - (y_top + h) / H_IN, w / W_IN, h / H_IN])


xw = Wg.x
ext = [xw[0], xw[-1], xw[0], xw[-1]]
for i, (ref, name, col) in enumerate(ROWS):
    Ws, dW = data[ref]
    y = 0.75 + i * (PW + GAP)
    axl = ax_at(0.05, y, LAB - 0.1, PW)
    axl.axis("off")
    axl.text(0.55, 0.5, name + "\n" + rf"$T={out[ref]['T_ns']:.0f}$ ns", rotation=90, ha="center", va="center",
             fontsize=FS["small"] - 2, color=col, transform=axl.transAxes, multialignment="center", linespacing=1.2)
    for j in range(7):
        ax = ax_at(LAB + j * (PW + GAP) + (DGAP if j == 6 else 0), y, PW, PW)
        if j < 6:
            im = ax.imshow(Ws[j], origin="lower", extent=ext, cmap=S.RBCMAP, vmin=-S.W_LIM, vmax=S.W_LIM)
            if i == 0:
                ax.set_title(rf"$t/T={FR[j]:.1f}$", fontsize=FS["ann"], pad=6)
        else:
            imd = ax.imshow(dW, origin="lower", extent=ext, cmap=DCMAP, vmin=-dmax, vmax=dmax)
            ax.text(0.035, 0.965, r"$1-F=$" + "\n" + rf"${out[ref]['infidelity_Yale'] * 1e3:.2f}\times10^{{-3}}$",
                    transform=ax.transAxes, fontsize=FS["small"] - 3, ha="left", va="top", linespacing=1.15,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
            if i == 0:
                ax.set_title(r"$\Delta W$ (measured rates)", fontsize=FS["ann"], pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(col)
            sp.set_linewidth(2.6)
y_cb = 0.75 + 3 * PW + 2 * GAP + 0.30
cax = ax_at(LAB, y_cb, 6 * PW + 5 * GAP, 0.20)
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_ticks([-S.W_LIM, 0, S.W_LIM])
cb.set_ticklabels([r"$-1/\pi$", r"$0$", r"$1/\pi$"])
cb.ax.tick_params(labelsize=FS["tick"])
cb.set_label(r"$W(q,p)$", fontsize=FS["label"])
cax2 = ax_at(LAB + 6 * (PW + GAP) + DGAP, y_cb, PW, 0.20)
cb2 = fig.colorbar(imd, cax=cax2, orientation="horizontal")
cb2.set_ticks([-dmax, 0, dmax])
cb2.set_ticklabels([rf"${-dmax * 1e3:.1f}$", r"$0$", rf"${dmax * 1e3:.1f}$"])
cb2.ax.tick_params(labelsize=FS["tick"] - 2)
cb2.set_label(r"$\Delta W\ (\times10^{-3})$", fontsize=FS["label"] - 2)
S.save(fig, HERE / HERE.name)
np.savez_compressed(HERE / f"{HERE.name}_wigner.npz", x=xw, **{ref: np.array(d[0]) for ref, d in data.items()},
                    **{f"{ref}_dW": d[1] for ref, d in data.items()})
(HERE / f"{HERE.name}_facts.json").write_text(json.dumps(out, indent=1))
print("wrote", HERE / f"{HERE.name}.png", f"| max |Delta W| = {dmax:.2e}")
