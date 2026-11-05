"""Table S-gallery (label tab:gallery): inputs of the closed forms and results for the state gallery, with the closed-form
columns (beta, mu, closed-form excess) recomputed with the consistent coefficient beta = 2 nu/n_chi (analytics audit C06)
by manuscript_figures/fig2/compute_fig2.closed_form; every other column is the stored data of
manuscript/si/state_gallery/results/state_gallery.json (read only).  Writes state_gallery_table.tex (the body of a
ruledtabular tabular) next to this file.
python tables/si_table3_state_gallery.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import importlib.util
import json
import pathlib
import sys

HERE = _L.vfile(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

import mfcommon as M  # noqa: E402
import numpy as np  # noqa: E402


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CF = module("compute_fig2", _L.REPO / "figures" / "Fig2" / "compute_fig2.py")
AU = module("audit", _L.REPO / "simulations" / "analysis" / "audit.py")
R = json.loads((M.MS / "si" / "state_gallery" / "results" / "state_gallery.json").read_text())["targets"]
ROWS = [("sq1", r"squeezed, $r=1$"), ("fock2", r"Fock $\ket2$"), ("fock4", r"Fock $\ket4$"), ("fock6", r"Fock $\ket6$"),
        ("binomial", r"$(\ket0+\ket4)/\sqrt2$"), ("cat15", r"cat, $\gamma=1.5$"), ("cat25", r"cat, $\gamma=2.5$"),
        ("cat4", r"4-component cat, $\gamma=2$"), ("gkp03", r"GKP $\ket{0_L}$, $\Delta=0.3$"),
        ("gkp05", r"GKP $\ket{0_L}$, $\Delta=0.5$"), ("gkpplus", r"GKP $\ket{+_L}$, $\Delta=0.4$")]
GV = AU.gallery_vectors()


def bound(e):
    for k in ("B", "B_lower", "bound"):
        if k in e:
            return e[k]
    raise KeyError(f"no lower-bound key in {sorted(e)}")


def main():
    head = (r"target & $\theta$ & $n_\chi$ & $\sigma_\chi$ & $n_d$ & $\eta$ & $J_{\rm geo}$ & "
            r"$J_{\rm ph}$ & $B$ & closed form (\%) & one-mode (\%) & time (\%) & saving (\%)\\\hline")
    lines = [r"\begin{tabular}{lcccccccccccc}", head]
    excess, modal = {}, {}
    for key, lab in ROWS:
        e = R[key]
        cells = [lab, f"{e['theta0']:.4f}", f"{e['n_chi']:.3f}", f"{e['sigma_chi']:.3f}"]
        if key in GV:
            v = np.real(np.asarray(GV[key]))
            v = v * np.sign(v[0])
            c = CF.closed_form(v)
            assert abs(c["theta0"] - e["theta0"]) < 1e-6 and abs(c["n_chi"] - e["n_chi"]) < 1e-6, key
            assert abs(c["sigma_chi"] - e["sigma_chi"]) < 1e-6, key
            excess[key] = c["J"] / e["J_ph"] - 1
            P = CF.paths.Prep(v.astype(complex))       # round 4: the closed form summed over all normal modes
            modal[key] = CF.pc.exposure(CF.modal_leg(P), P.N) / e["J_ph"] - 1
            cells += [f"{c['nu']:.3f}", f"{c['beta']:.3f}"]       # n_d and eta (closed_form's names: nu, beta)
        else:                                   # |chi> is a Fock state: sigma_chi = 0, the brachistochrone is optimal
            cells += ["--", "--"]
        cells += [f"{e['J_geo']:.4f}", f"{e['J_ph']:.4f}", f"{bound(e):.4f}",
                  f"{100 * modal[key]:.3f}" if key in modal else "--",
                  f"{100 * excess[key]:.1f}" if key in excess else "--",
                  f"{max(100 * e['time_cost'], 0.0):.1f}", f"{max(100 * e['saving'], 0.0):.1f}"]
        lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\end{tabular}")
    (HERE / "state_gallery_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    kmax = max(excess, key=excess.get)
    for k, v in excess.items():
        print(f"{k:8s} closed form {100 * modal[k]:+.3f}%, one-mode {100 * v:.2f}% (superseded beta: "
              f"{100 * R[k]['closed_excess']:.2f}%)")
    print(f"largest closed-form excess: {100 * excess[kmax]:.2f}% ({kmax})")
    print("wrote", HERE / "state_gallery_table.tex")


if __name__ == "__main__":
    main()
