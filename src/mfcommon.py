"""Shared helpers for manuscript_figures/ (the final figures, the Supplemental Material figures and the draft).

Read-only access to the data of manuscript/ (that folder is never written to), the in-house style at the final print
size, and the run-selection rules used by every tracking figure and every quoted number:

  * trivial(r): the pulse never moved the oscillator off its initial state (max_t |<n>(t) - <n>(0)| < 1e-3 and
    max_t |F_T(t) - F_T(0)| < 1e-4 along the Lindblad evolution).  This happens only for SNAP-displacement control (34 of
    217 runs; 0 of 345 for CS, 0 of 221 for ECD): zero cavity drive is a stationary point of the tracking objective for
    displacement drives on parity-even paths, and L-BFGS-B cannot leave it once a stage drives the amplitude to the
    bound.  These runs are KEPT in every statistic (user decision, Oct 27); their number is stated in the captions.
  * representative(lst): the start closest to the mean (smallest mean-squared distance of its Lindblad (x, z, F)
    trajectory from the average over all starts); thick lines and Wigner panels show this single run, never an average
    over starts (which would smear the Wigner function).
"""
from __future__ import annotations
import layout as _L  # repository layout: data/ and output/ (src/layout.py)

import pathlib
import sys

ROOT = _L.vfile(__file__).parents[1]
MS = ROOT / "manuscript"                     # source data and code (read only)
MF = ROOT / "manuscript_figures"
for p in (MS, MS / "fig4_cs_path_following", MF):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from common import style as S  # noqa: E402

# SNAP: aqua, chosen (no yellow, green or gray) as the colour most distinct from crimson, magenta, royal blue, purple and
# ink in normal and simulated protan/deutan/tritan vision with usable contrast on white (worst-case CIE76 Delta E = 29;
# tools/cvd_palette.py)
GATE_COLOR = {"cs": S.CRIMSON, "cd": "#A8329A", "snap": "#4CC9F0"}
# Nov 4 (user): the gate set is CD (conditional displacement), as in the Letter, ECD is the echoed hardware gate
GATE_NAME = {"cs": "conditional squeezing (CS)", "cd": "conditional displacement (CD)",
             "snap": "SNAP--displacement (SNAP)"}
GATE_SHORT = {"cs": "CS", "cd": "CD", "snap": "SNAP"}
YALE = r"$T_{1c}=606~\mu$s, $T_{\phi c}=5.1$~ms, $T_{1q}=280~\mu$s, $T_{\phi q}=414~\mu$s"
# print sizes: figures are drawn at twice their printed width (14 in for a two-column figure*, 7 in for one column),
# so these point sizes print at half (tick labels 11.5 pt, annotations 11 pt)
FS = dict(tick=23, label=25, title=24, legend=22, ann=22, small=20, panel=32)
LW = 4.6                                     # data lines (single runs, paths)
LW_THIN = 1.0                                # individual random starts


def setup():
    """in-house rcParams with the larger print sizes of the final figures"""
    S.style(usetex=True)
    S.big()
    S.FS.update(FS)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": FS["ann"], "axes.labelsize": FS["label"], "axes.titlesize": FS["title"],
                         "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "legend.fontsize": FS["legend"],
                         "axes.linewidth": 2.8, "lines.linewidth": LW, "xtick.major.width": 2.8,
                         "ytick.major.width": 2.8, "xtick.major.size": 12, "ytick.major.size": 12})
    return S


def trivial(r) -> bool:
    n, F = np.asarray(r["nbarL"]), np.asarray(r["FTL"])
    return bool(np.max(np.abs(n - n[0])) < 1e-3 and np.max(np.abs(F - F[0])) < 1e-4)


def usable(lst):
    """every run is used (no exclusion, Oct 27): runs that stay at the start (trivial) are counted and reported in the
    captions, never dropped"""
    return list(lst)


def n_trivial(lst):
    return sum(1 for r in lst if trivial(r))


def groups():
    """all tracking runs, grouped by (transfer, reference path, gate, duration index k); trivial runs are kept here and
    removed with usable()"""
    import load as L
    return L.groups("traces80")


def representative(lst):
    import load as L
    u = usable(lst)
    return L.representative(u) if u else None


def curve_stat(lst, key, tkey="tL_ns", n=201):
    import load as L
    return L.curve_stat(lst, key, tkey, n)


def stat(lst, key):
    v = np.array([r[key] for r in lst], float)
    return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def rho_file(r):
    if r["job"]["id"].startswith("warm_"):           # warm-started SNAP runs of sims/snap_warm.py (round 3)
        return MF / "sims" / "results" / "snap_warm" / "rho" / f"{r['job']['id']}.npz"
    return MS / "fig4_cs_path_following" / "results" / "rho80" / f"{r['job']['id']}.npz"


WIGNER_LW = 15.0                             # half-width of the window of the y-integral (see wigner_grid)


class _ShownWigner:
    """W(q, p) on the shown window |q|, |p| <= L (npts points), exact there: src/common/wigner.py evaluated on the wide
    window [-Lw, Lw] at the same spacing and cropped to [-L, L].  The transform on [-L, L] alone keeps only the pairs
    x +- y inside the window and so drops every contribution of the wave function beyond it (for example the |1_L>
    peaks at q = +-3 sqrt(pi) = +-5.32, up to 11% of the colour limit 1/pi; review Nov 1, item L1-F1).  With Lw = 15
    every panel of the figures agrees with the displaced-parity formula (1/pi) Tr[rho D(alpha) Pi D(alpha)^dag] to
    2e-9 (manuscript_review/revision_tools/final_nov2/ffix_assess.py); Lw = 10 still leaves 4e-3 for the CS runs,
    whose states reach high Fock levels."""

    def __init__(self, D, L, npts, Lw):
        from common.wigner import WignerGrid
        h = 2 * L / (npts - 1)
        off = int(round((Lw - L) / h))
        self.G = WignerGrid(D, L + off * h, npts + 2 * off)
        self.sl = slice(off, off + npts)
        self.x = self.p = self.G.x[self.sl]
        assert np.allclose(self.x, np.linspace(-L, L, npts))
        self.D, self.L = D, L

    def of_vectors(self, V):
        return self.G.of_vectors(V)[self.sl, self.sl]

    def of_rho(self, rho, tol=1e-8):
        return self.G.of_rho(rho, tol)[self.sl, self.sl]


_WG = {}


def wigner_grid(D, npts=121, L=5.0, Lw=WIGNER_LW):
    """exact Wigner function on the shown grid |q|, |p| <= L (npts points): methods of_vectors, of_rho, grid x (cached)"""
    key = (D, npts, L, Lw)
    if key not in _WG:
        _WG[key] = _ShownWigner(D, L, npts, Lw)
    return _WG[key]


def wigner_of_run(r, fracs, lindblad=True, D=90, L=5.0, npts=121):
    """Wigner functions W(q, p) of the oscillator state of the run r at the stored fractions t/T closest to `fracs`
    (Lindblad states 'rhoL' by default, closed 'rho'), from the states saved by manuscript/.../traces.py (cutoff 90),
    exact on the shown grid |q|, |p| <= L (wigner_grid).
    Returns (x grid, list of W, list of the fractions actually used)."""
    z = np.load(rho_file(r))
    fr = z["fr"]
    Wg = wigner_grid(D, npts, L)
    out, used = [], []
    for f in fracs:
        i = int(np.argmin(np.abs(fr - f)))
        out.append(Wg.of_rho(z["rhoL" if lindblad else "rho"][i].astype(complex)))
        used.append(float(fr[i]))
    return Wg.x, out, used


def closeness_of_run(r, path_states, fracs, lindblad=True):
    """closeness <p|rho|p> of the saved oscillator states of run r to the given path states (rows), at `fracs`"""
    z = np.load(rho_file(r))
    fr = z["fr"]
    out = []
    for f, p in zip(fracs, path_states):
        i = int(np.argmin(np.abs(fr - f)))
        rho = z["rhoL" if lindblad else "rho"][i].astype(complex)
        out.append(float(np.real(np.vdot(p, rho @ p))))
    return out
