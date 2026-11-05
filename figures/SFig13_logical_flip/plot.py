"""Supplementary Fig. 13 of the paper: The transfer |0_L> -> |1_L> along three paths (m = 2.81).

Drawn by the shared script `figures/three_paths.py` with the arguments `--target flip --k 2 --landscape`. This wrapper renders exactly this figure, as
`make SFig13` or `python figures/make_figures.py SFig13` do, and copies it to output/paper/si_figures/S16.png.

Usage (from the repository root, after `source env.sh`):  python figures/SFig13_logical_flip/plot.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import make_figures as MF  # noqa: E402

r = MF.run(MF.BY_ID["sfig13"])
print(("ok   " if r["rc"] == 0 else "FAIL ") + r["id"], r["seconds"], "s", r["png"])
sys.exit(r["rc"])
