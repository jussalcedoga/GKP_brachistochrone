# Figures

One folder per figure of the paper, named by its number (`FigN` for the Letter, `SFigNN_...` for the Supplementary Information). Each folder has a `README.md` (what the figure shows, the command, the input data and the checks) and the script that draws it. `make_figures.py` runs them all (`make figures`), and `reference_png.json` holds the md5 and pixel size of every published PNG for `tools/compare_figures.py`.

| figure | folder | file in the paper | command | time |
|---|---|---|---|---|
| Fig. 1 | [`Fig1/`](Fig1/) | `figures/fig1.png` | `make Fig1` | 1 min |
| Fig. 2 | [`Fig2/`](Fig2/) | `figures/fig2.png` | `make Fig2` | 40 s |
| Fig. 3 | [`Fig3/`](Fig3/) | `figures/fig3.png` | `make Fig3` | 30 s |
| Fig. 4 | [`Fig4/`](Fig4/) | `figures/fig4.png` | `make Fig4` | 70 s |
| Supplementary Fig. 1 | [`SFig01_photon_path_analytic/`](SFig01_photon_path_analytic/) | `si_figures/S02.png` | `make SFig1` | 35 s |
| Supplementary Fig. 2 | [`SFig02_state_gallery/`](SFig02_state_gallery/) | `si_figures/S01.png` | `make SFig2` | 55 s |
| Supplementary Fig. 3 | [`SFig03_time_photon_comparison/`](SFig03_time_photon_comparison/) | `si_figures/S03.png` | `make SFig3` | 35 s |
| Supplementary Fig. 4 | [`SFig04_error_budget/`](SFig04_error_budget/) | `si_figures/S05.png` | `make SFig4` | 50 s |
| Supplementary Fig. 5 | [`SFig05_path_snapshots/`](SFig05_path_snapshots/) | `si_figures/S04.png` | `make SFig5` | 40 s |
| Supplementary Fig. 6 | [`SFig06_aligned_generator/`](SFig06_aligned_generator/) | `si_figures/S06.png` | `make SFig6` | 30 s |
| Supplementary Fig. 7 | [`SFig07_squeezer_capacity/`](SFig07_squeezer_capacity/) | `si_figures/S07.png` | `make SFig7` | 40 s |
| Supplementary Fig. 8 | [`SFig08_tracking_protocol_gates/`](SFig08_tracking_protocol_gates/) | `si_figures/S08.png` | `make SFig8` | 25 s |
| Supplementary Fig. 9 | [`SFig09_tradeoff/`](SFig09_tradeoff/) | `si_figures/S23.png` | `make SFig9` | 25 s |
| Supplementary Fig. 10 | [`SFig10_tracking_fock2/`](SFig10_tracking_fock2/) | `si_figures/S09.png` | `make SFig10` | 65 s |
| Supplementary Fig. 11 | [`SFig11_tracking_cat/`](SFig11_tracking_cat/) | `si_figures/S10.png` | `make SFig11` | 65 s |
| Supplementary Fig. 12 | [`SFig12_tracking_binomial/`](SFig12_tracking_binomial/) | `si_figures/S11.png` | `make SFig12` | 65 s |
| Supplementary Fig. 13 | [`SFig13_logical_flip/`](SFig13_logical_flip/) | `si_figures/S16.png` | `make SFig13` | 85 s |
| Supplementary Fig. 14 | [`SFig14_logical_flip_m727/`](SFig14_logical_flip_m727/) | `si_figures/S22.png` | `make SFig14` | 90 s |
| Supplementary Fig. 15 | [`SFig15_pulses_tracking/`](SFig15_pulses_tracking/) | `si_figures/S25.png` | `make SFig15` | 25 s |
| Supplementary Fig. 16 | [`SFig16_pulses_target/`](SFig16_pulses_target/) | `si_figures/S26.png` | `make SFig16` | 20 s |
| Supplementary Fig. 17 | [`SFig17_pulses_flip/`](SFig17_pulses_flip/) | `si_figures/S27.png` | `make SFig17` | 20 s |
| Supplementary Fig. 18 | [`SFig18_flip_coincidence/`](SFig18_flip_coincidence/) | `si_figures/S28.png` | `make SFig18` | 35 s |

Not paper figures: `make figures` also renders the transfer at m = 1.17 and 1.40 (`three_paths.py --k 0`, `--k 1`), whose `*_facts.json` hold the m = 1.17 and 1.40 columns of Supplementary Table 12.
