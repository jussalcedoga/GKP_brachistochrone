"""In-house figure style (same rcParams as csqrl.style / RL_brach; re-implemented so the manuscript is self-contained).

Text: LaTeX with Helvetica (helvet) and sans-serif math (sfmath).  Thick lines, large inward ticks, large labels.
Colour semantics, fixed for every figure of the manuscript:
    ROYALBLUE solid    optimal time path (brachistochrone)
    CRIMSON  dashed    optimal photon path
    PURPLE   dotted    analytic counterpart of the photon path (closed form / detour through the vacuum)
    AMBER    dash-dot  best time-photon trade-off path (weighted geodesic at w*)
    INK      solid     gate sequence (GRAPE pulse with conditional squeezing)
Target palette (colour-blind safe, no royal blue / crimson): TARGET_COLORS.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

_CACHE = pathlib.Path(tempfile.gettempdir()) / f"manuscript_mpl_{os.getuid() if hasattr(os, 'getuid') else 'win'}" / str(os.getpid())
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))
os.environ.setdefault("TEXMFVAR", str(_CACHE / "texmf-var"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROYALBLUE = "#4169E1"
CRIMSON = "#DC143C"
PURPLE = "#7C3AED"
AMBER = "#D97706"
INK = "#111827"
DARK = "#4B5563"
GRAY = "#9CA3AF"
LIGHT_BLUE = "#B9C9FF"
LIGHT_RED = "#F4A6B7"
TEAL = "#0F9D8A"
RBCMAP = LinearSegmentedColormap.from_list("royalblue_white_crimson", [ROYALBLUE, "#FFFFFF", CRIMSON])
# sequential map for photon numbers (crimson = photons), white at n = 0
NCMAP = LinearSegmentedColormap.from_list("white_crimson", ["#FFFFFF", "#FCE7EC", "#F4A6B7", "#E2637F"])
TRADEOFF = "#8B3FC7"         # the time-photon compromise: between royalblue (time) and crimson (photons)
NAVY = "#1F3B99"
RASPBERRY = "#A3195B"
# target palette, all within the royalblue-crimson family
TARGET_COLORS = {"fock2": ROYALBLUE, "fock4": "#7F9CF5", "cat": CRIMSON, "gkp": NAVY, "flip": RASPBERRY,
                 "binomial": "#8B93B0"}
TARGET_MARKERS = {"fock2": "^", "fock4": "v", "cat": "s", "gkp": "o", "flip": "D", "binomial": "X"}
PATH_STYLE = {
    "geodesic": dict(color=ROYALBLUE, ls="-", lw=3.6),
    "photon": dict(color=CRIMSON, ls=(0, (5, 2.2)), lw=3.4),
    "analytic": dict(color=PURPLE, ls=(0, (1, 1.6)), lw=3.4),
    "tradeoff": dict(color=TRADEOFF, ls=(0, (6, 2, 1.5, 2)), lw=3.4),
    "pulse": dict(color=INK, ls="-", lw=2.4),
}
W_LIM = 1.0 / 3.141592653589793


def style(usetex: bool = True) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
        "font.size": 16, "axes.labelsize": 19, "axes.titlesize": 18,
        "axes.linewidth": 2.2, "lines.linewidth": 3.2,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "xtick.major.size": 8, "ytick.major.size": 8,
        "xtick.major.width": 2.0, "ytick.major.width": 2.0,
        "xtick.minor.size": 4.5, "ytick.minor.size": 4.5,
        "xtick.minor.width": 1.5, "ytick.minor.width": 1.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": False, "ytick.right": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.fontsize": 13.5, "legend.frameon": False,
        "mathtext.fontset": "stixsans", "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb,bm}\usepackage{helvet}\usepackage{sfmath}"
                               r"\renewcommand{\familydefault}{\sfdefault}",
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.facecolor": "white",
        "axes.titlepad": 10, "image.interpolation": "bicubic",
    })


FS = dict(tick=21, label=23, title=22, legend=20, ann=20, small=18, panel=30)


def big() -> None:
    """large print for figures drawn ~14.5 in wide (a PRL figure* is ~7 in: text prints at about half its point size)"""
    plt.rcParams.update({
        "font.size": FS["ann"], "axes.labelsize": FS["label"], "axes.titlesize": FS["title"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "legend.fontsize": FS["legend"],
        "axes.linewidth": 2.6, "lines.linewidth": 4.2, "lines.markersize": 12,
        "xtick.major.size": 11, "ytick.major.size": 11, "xtick.major.width": 2.6, "ytick.major.width": 2.6,
        "xtick.minor.size": 6, "ytick.minor.size": 6, "xtick.minor.width": 1.8, "ytick.minor.width": 1.8,
        "xtick.major.pad": 7, "ytick.major.pad": 7, "axes.labelpad": 6, "axes.titlepad": 10,
        "legend.handlelength": 2.2, "legend.columnspacing": 1.4, "legend.borderaxespad": 0.3,
    })


def panel_label(ax, letter, x=-0.16, y=1.04, size=None):
    """Physical Review style panel label "(a)" at axes coordinates (x, y)"""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=FS["panel"] if size is None else max(size, FS["panel"]),
            va="bottom", ha="left")


def save(fig, path, dpi: int = 300) -> None:
    """PNG at `dpi` and a vector PDF next to it"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.12, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def grid(ax, axis="both"):
    ax.grid(alpha=0.22, axis=axis)
