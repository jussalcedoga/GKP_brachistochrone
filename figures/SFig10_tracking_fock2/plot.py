"""Supplementary Fig. 10 of the paper: As Fig. 4, for Fock |2>.

Drawn by the shared script `figures/Fig4/plot_fig4.py` with the arguments `--target fock2 --landscape`. This wrapper renders exactly this figure, as
`make SFig10` or `python figures/make_figures.py SFig10` do, and copies it to output/paper/si_figures/S09.png.

Usage (from the repository root, after `source env.sh`):  python figures/SFig10_tracking_fock2/plot.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import make_figures as MF  # noqa: E402

r = MF.run(MF.BY_ID["sfig10"])
print(("ok   " if r["rc"] == 0 else "FAIL ") + r["id"], r["seconds"], "s", r["png"])
sys.exit(r["rc"])
