"""Regenerate the figures of the Letter and of the Supplementary Information from the stored results in data/.

No long simulation is rerun. Each figure is drawn by the script(s) listed in FIGURES below; the PNG is then copied to
output/paper/ under the file name the paper uses (figures/fig1.png ... and si_figures/S01.png ...), and the run times
are written to output/paper/run_log.json. "all" also runs the two renders of AUX (the transfer at m = 1.17 and 1.40),
which are not paper figures but whose *_facts.json hold the m = 1.17 and 1.40 columns of Supplementary Table 12.

Usage (from the repository root, after `source env.sh`):
    python figures/make_figures.py all [-j 4]       every figure of the paper and the two renders of AUX
    python figures/make_figures.py Fig4 SFig10       selected figures (Letter Fig. N = FigN, Supplementary Fig. N = SFigN)
    python figures/make_figures.py aux               the two renders of AUX only
    python figures/make_figures.py --list            the table of figures, scripts and outputs
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import layout as L  # noqa: E402

SI = "manuscript_figures/supplementary_information"
O = str(L.OUT)


def out(rel):
    return f"{O}/{rel}"


# id, file name in the paper, commands (argv after the interpreter, relative to the repository root), produced PNG
# (relative to output/)
FIGURES = [
    ("Fig1", "figures/fig1.png", [["figures/Fig1/plot_fig1.py"]], "manuscript_figures/fig1/fig1.png"),
    ("Fig2", "figures/fig2.png", [["figures/Fig2/compute_fig2.py"], ["figures/Fig2/plot_fig2.py"]],
     "manuscript_figures/fig2/fig2.png"),
    ("Fig3", "figures/fig3.png", [["figures/Fig3/plot_fig3.py"]], "manuscript_figures/fig3/fig3.png"),
    ("Fig4", "figures/fig4.png", [["figures/Fig4/plot_fig4.py", "--wide", "--out", out("manuscript_figures/fig4/fig4")]],
     "manuscript_figures/fig4/fig4.png"),                     # the two-column layout of the Letter (figure*)
    ("SFig1", "si_figures/S02.png", [["figures/SFig01_photon_path_analytic/plot.py"]],
     f"{SI}/S02_photon_path_analytic/S02_photon_path_analytic.png"),
    ("SFig2", "si_figures/S01.png", [["figures/SFig02_state_gallery/plot.py"]],
     f"{SI}/S01_state_gallery/S01_state_gallery.png"),
    ("SFig3", "si_figures/S03.png", [["figures/SFig03_time_photon_comparison/plot.py"]],
     f"{SI}/S03_time_photon_comparison/S03_time_photon_comparison.png"),
    ("SFig4", "si_figures/S05.png", [["figures/SFig04_error_budget/plot.py"]], f"{SI}/S05_error_budget/S05_error_budget.png"),
    ("SFig5", "si_figures/S04.png", [["figures/SFig05_path_snapshots/plot.py"]],
     f"{SI}/S04_path_snapshots/S04_path_snapshots.png"),
    ("SFig6", "si_figures/S06.png", [["figures/SFig06_aligned_generator/plot.py"]],
     f"{SI}/S06_aligned_generator/S06_aligned_generator.png"),
    ("SFig7", "si_figures/S07.png", [["figures/SFig07_squeezer_capacity/plot.py"]],
     f"{SI}/S07_squeezer_capacity/S07_squeezer_capacity.png"),
    ("SFig8", "si_figures/S08.png", [["figures/SFig08_tracking_protocol_gates/plot.py"]],
     f"{SI}/S08_tracking_protocol_gates/S08_tracking_protocol_gates.png"),
    ("SFig9", "si_figures/S23.png", [["figures/SFig09_tradeoff/plot.py"]], f"{SI}/S23_tradeoff/S23_tradeoff.png"),
    ("SFig10", "si_figures/S09.png",
     [["figures/Fig4/plot_fig4.py", "--target", "fock2", "--landscape", "--out", out(f"{SI}/S09_tracking_fock2/S09_tracking_fock2")]],
     f"{SI}/S09_tracking_fock2/S09_tracking_fock2.png"),
    ("SFig11", "si_figures/S10.png",
     [["figures/Fig4/plot_fig4.py", "--target", "cat", "--landscape", "--out", out(f"{SI}/S10_tracking_cat/S10_tracking_cat")]],
     f"{SI}/S10_tracking_cat/S10_tracking_cat.png"),
    ("SFig12", "si_figures/S11.png",
     [["figures/Fig4/plot_fig4.py", "--target", "binomial", "--snap-warm", "--landscape", "--out",
       out(f"{SI}/S11_tracking_binomial/S11_tracking_binomial")]],
     f"{SI}/S11_tracking_binomial/S11_tracking_binomial.png"),
    ("SFig13", "si_figures/S16.png",
     [["figures/three_paths.py", "--target", "flip", "--k", "2", "--landscape", "--out", out(f"{SI}/S16_logical_flip/S16_logical_flip")]],
     f"{SI}/S16_logical_flip/S16_logical_flip.png"),
    ("SFig14", "si_figures/S22.png",
     [["figures/three_paths.py", "--target", "flip", "--k", "3", "--landscape", "--out",
       out(f"{SI}/S20_S22_flip_durations/S22_flip_m727")]],
     f"{SI}/S20_S22_flip_durations/S22_flip_m727.png"),
    ("SFig15", "si_figures/S25.png", [["figures/SFig15_pulses_tracking/plot.py"]], f"{SI}/S25_pulses_tracking/S25.png"),
    ("SFig16", "si_figures/S26.png", [["figures/SFig16_pulses_target/plot.py"]], f"{SI}/S26_pulses_target/S26.png"),
    ("SFig17", "si_figures/S27.png", [["figures/SFig17_pulses_flip/plot.py"]], f"{SI}/S27_pulses_flip/S27.png"),
    ("SFig18", "si_figures/S28.png", [["figures/SFig18_flip_coincidence/plot.py"]], f"{SI}/S28_flip_coincidence/S28_note.png"),
]
# renders that are not paper figures but whose *_facts.json feed Supplementary Table 12 (transfer at m = 1.17, 1.40);
# tables/si_tables_10_12.py reads the stored copies in data/, and tools/compare_figures.py compares the regenerated
# facts with them
AUX = [
    ("aux_flip_m117", None, [["figures/three_paths.py", "--target", "flip", "--k", "0", "--out",
                              out(f"{SI}/S20_S22_flip_durations/S20_flip_m117")]], f"{SI}/S20_S22_flip_durations/S20_flip_m117.png"),
    ("aux_flip_m140", None, [["figures/three_paths.py", "--target", "flip", "--k", "1", "--out",
                              out(f"{SI}/S20_S22_flip_durations/S21_flip_m140")]], f"{SI}/S20_S22_flip_durations/S21_flip_m140.png"),
]
BY_ID = {f[0].lower(): f for f in FIGURES + AUX}


def env():
    e = dict(os.environ)
    paths = [REPO / p for p in ("src", "simulations/tracking", "simulations/gate_sets", "simulations/analysis",
                                "simulations/analysis/review_oct28", "simulations/analysis/photon_flip")]
    e["PYTHONPATH"] = os.pathsep.join([str(p) for p in paths] + ([e["PYTHONPATH"]] if e.get("PYTHONPATH") else []))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        e.setdefault(k, "1")                     # as in production; results do not depend on it beyond round-off
    return e


def run(fig):
    fid, paper, cmds, png = fig
    logd = L.out_dir("paper", "logs")
    t0 = time.time()
    rc = 0
    with open(logd / f"{fid}.log", "w") as log:
        for c in cmds:
            rc = subprocess.call([sys.executable] + c, cwd=REPO, env=env(), stdout=log, stderr=subprocess.STDOUT)
            if rc:
                break
    dt = time.time() - t0
    dst = None
    if rc == 0 and paper:
        dst = L.OUT / "paper" / paper
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(L.OUT / png, dst)
    return dict(id=fid, paper_file=paper, rc=rc, seconds=round(dt, 1), png=str(dst) if dst else str(L.OUT / png))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", default=["all"])
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for fid, paper, cmds, png in FIGURES + AUX:
            print(f"{fid:14s} {str(paper):20s} " + " && ".join("python " + " ".join(c) for c in cmds))
        return
    ids = [i.lower() for i in a.ids]
    if "all" in ids:
        sel = FIGURES + AUX
    elif "aux" in ids:
        sel = AUX
    else:
        unknown = [i for i in ids if i not in BY_ID]
        if unknown:
            sys.exit(f"unknown figure id(s) {unknown}; see --list")
        sel = [BY_ID[i] for i in ids]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        res = list(ex.map(run, sel))
    for r in res:
        print(f"{'ok  ' if r['rc'] == 0 else 'FAIL'} {r['id']:14s} {r['seconds']:7.1f} s  {r['png']}")
    logf = L.OUT / "paper" / "run_log.json"
    old = json.loads(logf.read_text()) if logf.exists() else {}
    old.update({r["id"]: r for r in res})
    logf.write_text(json.dumps(old, indent=1))
    print(f"total {time.time() - t0:.0f} s; logs in {L.OUT / 'paper' / 'logs'}")
    sys.exit(1 if any(r["rc"] for r in res) else 0)


if __name__ == "__main__":
    main()
