"""Supplementary Fig. 14 of the paper: The same at m = 7.27.

Drawn by the shared script `figures/three_paths.py` with the arguments `--target flip --k 3 --landscape`. This wrapper renders exactly this figure, as
`make SFig14` or `python figures/make_figures.py SFig14` do, and copies it to output/paper/si_figures/S22.png.

Usage (from the repository root, after `source env.sh`):  python figures/SFig14_logical_flip_m727/plot.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import make_figures as MF  # noqa: E402

r = MF.run(MF.BY_ID["sfig14"])
print(("ok   " if r["rc"] == 0 else "FAIL ") + r["id"], r["seconds"], "s", r["png"])
sys.exit(r["rc"])
