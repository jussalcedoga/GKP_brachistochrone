"""Repository layout: where the stored results (data/) and the regenerated outputs (output/) live.

The scripts of this repository were written for the working tree in which the results of the paper were produced. In
that tree every script found its inputs, and wrote its outputs, relative to its own location. Here the code is grouped by
role (src/, simulations/, figures/, numbers/, tables/, tests/), and the stored results keep their original relative
paths under data/ (data/manuscript/..., data/manuscript_figures/..., data/manuscript_review/...).

vfile(__file__) returns a script's original location, mapped
  into data/    for the simulations and analyses, whose products are the stored results, and
  into output/  for the figure, number and table scripts, whose products are regenerated.
Only paths are mapped. The numerics of every script are unchanged.

Environment variables
  GKP_DATA       folder of the stored results (default: <repository>/data)
  GKP_OUT        folder of the outputs (default: <repository>/output)
  GKP_PAPER_TEX  optional folder with main.tex and supplement.tex of the paper, for the wording checks in numbers/
"""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(os.environ.get("GKP_DATA", REPO / "data")).resolve()
OUT = pathlib.Path(os.environ.get("GKP_OUT", REPO / "output")).resolve()

# repository path -> (original path in the working tree, "data" | "out")
REGISTRY = {
    'figures/Fig1/compute_sphere.py': ('manuscript_figures/fig1/compute_sphere.py', 'out'),
    'figures/Fig1/plot_fig1.py': ('manuscript_figures/fig1/plot_fig1.py', 'out'),
    'figures/Fig2/compute_fig2.py': ('manuscript_figures/fig2/compute_fig2.py', 'out'),
    'figures/Fig2/plot_fig2.py': ('manuscript_figures/fig2/plot_fig2.py', 'out'),
    'figures/Fig3/plot_fig3.py': ('manuscript_figures/fig3/plot_fig3.py', 'out'),
    'figures/Fig4/plot_fig4.py': ('manuscript_figures/fig4/plot_fig4.py', 'out'),
    'figures/SFig01_photon_path_analytic/plot.py': ('manuscript_figures/supplementary_information/S02_photon_path_analytic/plot.py', 'out'),
    'figures/SFig02_state_gallery/plot.py': ('manuscript_figures/supplementary_information/S01_state_gallery/plot.py', 'out'),
    'figures/SFig03_time_photon_comparison/plot.py': ('manuscript_figures/supplementary_information/S03_time_photon_comparison/plot.py', 'out'),
    'figures/SFig04_error_budget/plot.py': ('manuscript_figures/supplementary_information/S05_error_budget/plot.py', 'out'),
    'figures/SFig05_path_snapshots/plot.py': ('manuscript_figures/supplementary_information/S04_path_snapshots/plot.py', 'out'),
    'figures/SFig06_aligned_generator/plot.py': ('manuscript_figures/supplementary_information/S06_aligned_generator/plot.py', 'out'),
    'figures/SFig07_squeezer_capacity/plot.py': ('manuscript_figures/supplementary_information/S07_squeezer_capacity/plot.py', 'out'),
    'figures/SFig08_tracking_protocol_gates/plot.py': ('manuscript_figures/supplementary_information/S08_tracking_protocol_gates/plot.py', 'out'),
    'figures/SFig09_tradeoff/plot.py': ('manuscript_figures/supplementary_information/S23_tradeoff/plot.py', 'out'),
    'figures/SFig15_pulses_tracking/plot.py': ('manuscript_figures/supplementary_information/S25_pulses_tracking/plot.py', 'out'),
    'figures/SFig16_pulses_target/plot.py': ('manuscript_figures/supplementary_information/S26_pulses_target/plot.py', 'out'),
    'figures/SFig17_pulses_flip/plot.py': ('manuscript_figures/supplementary_information/S27_pulses_flip/plot.py', 'out'),
    'figures/SFig18_flip_coincidence/plot.py': ('manuscript_figures/supplementary_information/S28_flip_coincidence/plot_note.py', 'out'),
    'figures/three_paths.py': ('manuscript_figures/supplementary_information/_three_paths.py', 'out'),
    'numbers/letter_numbers.py': ('manuscript_review/revision_tools/final_nov2/verify/letter/letter_numbers.py', 'out'),
    'numbers/letter_numbers2.py': ('manuscript_review/revision_tools/final_nov2/verify/letter/letter_numbers2.py', 'out'),
    'numbers/pulses_text_facts.py': ('manuscript_figures/sims/pulses_text_facts.py', 'data'),
    'simulations/analysis/audit.py': ('manuscript_figures/analytics_audit/audit.py', 'data'),
    'simulations/analysis/cs_proposal/cs_proposal_hamiltonian.py': ('manuscript_figures/analytics_audit/round6/cs_proposal/cs_proposal_hamiltonian.py', 'data'),
    'simulations/analysis/photon_flip/audit_photon.py': ('manuscript_figures/analytics_audit/photon_flip/audit_photon.py', 'data'),
    'simulations/analysis/photon_flip/audit_photon2.py': ('manuscript_figures/analytics_audit/photon_flip/audit_photon2.py', 'data'),
    'simulations/analysis/review_oct28/lindblad_crosscheck.py': ('manuscript_figures/analytics_audit/review_oct28/lindblad_crosscheck.py', 'data'),
    'simulations/analysis/review_oct28/robustness.py': ('manuscript_figures/analytics_audit/review_oct28/robustness.py', 'data'),
    'simulations/analysis/round3/codespace.py': ('manuscript_figures/analytics_audit/round3/codespace.py', 'data'),
    'simulations/analysis/round3/cs_half_rate.py': ('manuscript_figures/analytics_audit/round3/cs_half_rate.py', 'data'),
    'simulations/analysis/round3/flip_gate_summary.py': ('manuscript_figures/analytics_audit/round3/flip_gate_summary.py', 'data'),
    'simulations/analysis/round3/onesided_summary.py': ('manuscript_figures/analytics_audit/round3/onesided_summary.py', 'data'),
    'simulations/analysis/round3/pauli_frame.py': ('manuscript_figures/analytics_audit/round3/pauli_frame.py', 'data'),
    'simulations/analysis/round3/rates_parity.py': ('manuscript_figures/analytics_audit/round3/rates_parity.py', 'data'),
    'simulations/analysis/round3/route_test.py': ('manuscript_figures/analytics_audit/round3/route_test.py', 'data'),
    'simulations/analysis/round3/stabilizers_along.py': ('manuscript_figures/analytics_audit/round3/stabilizers_along.py', 'data'),
    'simulations/analysis/round3/tradeoff.py': ('manuscript_figures/analytics_audit/round3/tradeoff.py', 'data'),
    'simulations/analysis/round4/transfer_closed.py': ('manuscript_figures/analytics_audit/round4/transfer_closed.py', 'data'),
    'simulations/flip_note/num_core.py': ('manuscript_review/flip_coincidence/num_core.py', 'data'),
    'simulations/gate_sets/capability_noisy.py': ('manuscript_figures/sims/capability_noisy.py', 'data'),
    'simulations/gate_sets/capability_pulses.py': ('manuscript_figures/sims/capability_pulses.py', 'data'),
    'simulations/gate_sets/cs_onesided.py': ('manuscript_figures/sims/cs_onesided.py', 'data'),
    'simulations/gate_sets/flip_gate.py': ('manuscript_figures/sims/flip_gate.py', 'data'),
    'simulations/gate_sets/pulses_check_B.py': ('manuscript_figures/sims/pulses_check_B.py', 'data'),
    'simulations/gate_sets/pulses_closed_target.py': ('manuscript_figures/sims/pulses_closed_target.py', 'data'),
    'simulations/gate_sets/pulses_extract.py': ('manuscript_figures/sims/pulses_extract.py', 'data'),
    'simulations/gate_sets/pulses_facts.py': ('manuscript_figures/sims/pulses_facts.py', 'data'),
    'simulations/gate_sets/pulses_facts_B.py': ('manuscript_figures/sims/pulses_facts_B.py', 'data'),
    'simulations/gate_sets/pulses_fig_check.py': ('manuscript_figures/sims/pulses_fig_check.py', 'data'),
    'simulations/gate_sets/pulses_figlib.py': ('manuscript_figures/sims/pulses_figlib.py', 'data'),
    'simulations/gate_sets/pulses_reproduce.py': ('manuscript_figures/sims/pulses_reproduce.py', 'data'),
    'simulations/gate_sets/pulses_smooth.py': ('manuscript_figures/sims/pulses_smooth.py', 'data'),
    'simulations/gate_sets/pulses_stats.py': ('manuscript_figures/sims/pulses_stats.py', 'data'),
    'simulations/gate_sets/restart_trivial.py': ('manuscript_figures/sims/restart_trivial.py', 'data'),
    'simulations/gate_sets/snap_warm.py': ('manuscript_figures/sims/snap_warm.py', 'data'),
    'simulations/paths/fig1_brachistochrone.py': ('manuscript/fig1_brachistochrone/compute.py', 'data'),
    'simulations/paths/fig2_photon_optimal.py': ('manuscript/fig2_photon_optimal/compute.py', 'data'),
    'simulations/paths/fig3_time_vs_photon.py': ('manuscript/fig3_time_vs_photon/compute.py', 'data'),
    'simulations/paths/si_aligned_generator.py': ('manuscript/si/aligned_generator/compute.py', 'data'),
    'simulations/paths/si_squeezer_capacity.py': ('manuscript/si/squeezer_capacity/compute.py', 'data'),
    'simulations/paths/si_state_gallery.py': ('manuscript/si/state_gallery/compute.py', 'data'),
    'simulations/tracking/analyze.py': ('manuscript/fig4_cs_path_following/analyze.py', 'data'),
    'simulations/tracking/compute.py': ('manuscript/fig4_cs_path_following/compute.py', 'data'),
    'simulations/tracking/load.py': ('manuscript/fig4_cs_path_following/load.py', 'data'),
    'simulations/tracking/traces.py': ('manuscript/fig4_cs_path_following/traces.py', 'data'),
    'simulations/tracking/wigner_snapshots.py': ('manuscript/fig4_cs_path_following/wigner_snapshots.py', 'data'),
    'src/common/__init__.py': ('manuscript/common/__init__.py', 'data'),
    'src/common/brachistochrone.py': ('manuscript/common/brachistochrone.py', 'data'),
    'src/common/geometry.py': ('manuscript/common/geometry.py', 'data'),
    'src/common/grape.py': ('manuscript/common/grape.py', 'data'),
    'src/common/lindblad.py': ('manuscript/common/lindblad.py', 'data'),
    'src/common/params.py': ('manuscript/common/params.py', 'data'),
    'src/common/paths.py': ('manuscript/common/paths.py', 'data'),
    'src/common/photon_core.py': ('manuscript/common/photon_core.py', 'data'),
    'src/common/states.py': ('manuscript/common/states.py', 'data'),
    'src/common/style.py': ('manuscript/common/style.py', 'data'),
    'src/common/tracking.py': ('manuscript/common/tracking.py', 'data'),
    'src/common/wigner.py': ('manuscript/common/wigner.py', 'data'),
    'src/mfcommon.py': ('manuscript_figures/mfcommon.py', 'data'),
    'tables/letter_table1_tracking.py': ('manuscript_figures/review_round/table_tracking.py', 'out'),
    'tables/si_table1_truncation_order.py': ('manuscript_figures/analytics_audit/round5/order_table_cutoff.py', 'out'),
    'tables/si_table3_state_gallery.py': ('manuscript_figures/supplementary_information/S01_state_gallery/table.py', 'out'),
    'tables/si_tables_10_12.py': ('manuscript_figures/review_round/tables.py', 'out'),
    'tests/test_common.py': ('manuscript/tests/test_common.py', 'data'),
    'tests/test_snap.py': ('manuscript/tests/test_snap.py', 'data'),
    'tests/test_tracking.py': ('manuscript/tests/test_tracking.py', 'data'),
}
_CODE = {old: new for new, (old, _) in REGISTRY.items()}


def vfile(f):
    """original location of the script or module f (its __file__), mapped into data/ or output/"""
    rel = pathlib.Path(f).resolve().relative_to(REPO).as_posix()
    old, mode = REGISTRY[rel]
    p = (OUT if mode == "out" else DATA) / old
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def code(old):
    """repository file of the script whose original path is old"""
    return REPO / _CODE[old]


def out_dir(*parts):
    p = OUT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def paper_tex_dir():
    d = os.environ.get("GKP_PAPER_TEX")
    return pathlib.Path(d).resolve() if d else None


def paper_text(name):
    """text of main.tex or supplement.tex if GKP_PAPER_TEX is set, otherwise the table rows of the paper that the
    figure and pulse scripts parse to check that the data reproduce the printed digits (data/paper_quotes/)"""
    d = paper_tex_dir()
    if d is not None:
        return (d / name).read_text(encoding="utf-8")
    return (DATA / "paper_quotes" / name.replace(".tex", "_quoted_rows.tex")).read_text(encoding="utf-8")


def paper_text_full(name):
    """whitespace-normalized full text of the paper source, or None without GKP_PAPER_TEX"""
    d = paper_tex_dir()
    return " ".join((d / name).read_text(encoding="utf-8").split()) if d is not None else None
