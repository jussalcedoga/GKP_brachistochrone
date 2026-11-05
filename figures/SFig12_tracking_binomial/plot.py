"""Supplementary Fig. 12 of the paper: As Fig. 4, for the binomial-code state.

Drawn by the shared script `figures/Fig4/plot_fig4.py` with the arguments `--target binomial --snap-warm --landscape`. This wrapper renders exactly this figure, as
`make SFig12` or `python figures/make_figures.py SFig12` do, and copies it to output/paper/si_figures/S11.png.

Usage (from the repository root, after `source env.sh`):  python figures/SFig12_tracking_binomial/plot.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import make_figures as MF  # noqa: E402

r = MF.run(MF.BY_ID["sfig12"])
print(("ok   " if r["rc"] == 0 else "FAIL ") + r["id"], r["seconds"], "s", r["png"])
sys.exit(r["rc"])
