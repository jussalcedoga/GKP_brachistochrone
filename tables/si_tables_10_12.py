"""Appendix tables of the submission manuscript, from the stored facts of the benchmark figures (no simulation):
  tab_prep.tex   preparation of Fock |2>, the cat C_2^+, the binomial word and GKP |0_L> by the three gate sets at
                 412, 824 and 2133 ns (target fidelity under Lindblad noise and minimum closeness to the brachistochrone)
  tab_paths.tex  CS along the three paths of GKP and cat preparation (and CD, SNAP along the brachistochrone)
  tab_flip.tex   the transfer |0_L> -> |1_L> along the three paths at m = 1.17, 1.40, 2.81 and 7.27 times their minimal time
Writes the tabular bodies next to this file.   python tables/si_tables_10_12.py"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import pathlib

MF = _L.DATA / "manuscript_figures"    # stored facts
HERE = _L.vfile(__file__).parent
SI = MF / "supplementary_information"
G3 = ("cs", "cd", "snap")


def load(rel):
    return json.loads((MF / rel).read_text())


def ft(e, best=True):
    s = f"{e['F_T_mean']:.3f}({round(1000 * e['F_T_sd']):d})" if e["F_T_sd"] >= 5e-4 else f"{e['F_T_mean']:.3f}"
    if best and "F_T_max" in e:
        s += f" [{e['F_T_max']:.3f}]"
    return s


def cc(e):
    return f"{e['C_mean']:.2f}"


def prep():
    rows = [r"target & $T$ (ns) & \multicolumn{2}{c}{CS} & \multicolumn{2}{c}{CD} & \multicolumn{2}{c}{SNAP}\\",
            r" & & $F_T$ & $C$ & $F_T$ & $C$ & $F_T$ & $C$\\\hline"]
    for rel, lab in (("supplementary_information/S09_tracking_fock2/S09_tracking_fock2_facts.json", r"Fock $\ket2$"),
                     ("supplementary_information/S10_tracking_cat/S10_tracking_cat_facts.json", r"cat $\ket{C_2^+}$"),
                     ("supplementary_information/S11_tracking_binomial/S11_tracking_binomial_facts.json",
                      r"binomial $\frac{\ket0+\ket4}{\sqrt2}$"),
                     ("fig4/fig4_facts.json", r"GKP $\ket{0_L}$")):
        c = load(rel)["columns"]
        for i, k in enumerate(("1", "2", "3")):
            cells = [lab if i == 0 else "", f"{c[k]['T_ns']:.0f}"]
            for g in G3:
                e = c[k][g]
                cells += [ft(e), cc(e)]
            rows.append(" & ".join(cells) + r"\\")
        rows[-1] += r"\hline" if lab != r"GKP $\ket{0_L}$" else ""
    return "\n".join(rows)


def paths():
    rows = [r"target, $m$ & path & $T$ (ns) & CS $F_T$ & CS $C$ & CS exposure & CD $F_T$ / $C$ & SNAP $F_T$ / $C$\\\hline"]
    for rel, lab in (("supplementary_information/S12_gkp_three_paths/S12_gkp_three_paths_facts.json", "GKP"),
                     ("supplementary_information/S14_cat_three_paths/S14_cat_three_paths_facts.json", "cat")):
        d = load(rel)
        for i, ref in enumerate(("geodesic", "tradeoff", "photon")):
            c = d["columns"][ref]
            cs = c["cs"]
            cells = [f"{lab}, {d['m_pace']:.2f}" if i == 0 else "",
                     {"geodesic": "time optimal", "tradeoff": "trade-off", "photon": "photon optimal"}[ref],
                     f"{c['T_ns']:.0f}", ft(cs, False), cc(cs), f"{cs['expo_L_ns']:.0f}"]
            for g in ("cd", "snap"):
                e = c.get(g)
                cells.append(f"{ft(e, False)} / {cc(e)}" if e and e.get("n_usable") else "--")
            rows.append(" & ".join(cells) + r"\\")
        rows[-1] += r"\hline" if lab == "GKP" else ""
    return "\n".join(rows)


def flip():
    rows = [r"$m$ & path & $T$ (ns) & CS $F_T$ & CS $C$ & CS exposure & CD $F_T$ & CD $C$ & SNAP $F_T$ & SNAP $C$\\\hline"]
    files = (("supplementary_information/S20_S22_flip_durations/S20_flip_m117_facts.json"),
             ("supplementary_information/S20_S22_flip_durations/S21_flip_m140_facts.json"),
             ("supplementary_information/S16_logical_flip/S16_logical_flip_facts.json"),
             ("supplementary_information/S20_S22_flip_durations/S22_flip_m727_facts.json"))
    for j, rel in enumerate(files):
        d = load(rel)
        for i, ref in enumerate(("geodesic", "tradeoff", "photon")):
            c = d["columns"][ref]
            cells = [f"{d['m_pace']:.2f}" if i == 0 else "",
                     {"geodesic": "time optimal", "tradeoff": "trade-off", "photon": "photon optimal"}[ref],
                     f"{c['T_ns']:.0f}", ft(c["cs"], False), cc(c["cs"]), f"{c['cs']['expo_L_ns']:.0f}"]
            for g in ("cd", "snap"):
                cells += [ft(c[g], False), cc(c[g])]
            rows.append(" & ".join(cells) + r"\\")
        if j < len(files) - 1:
            rows[-1] += r"\hline"
    return "\n".join(rows)


for name, fn in (("tab_prep", prep), ("tab_paths", paths), ("tab_flip", flip)):
    (HERE / f"{name}.tex").write_text(fn() + "\n", encoding="utf-8")
    print(f"--- {name}\n" + fn())
