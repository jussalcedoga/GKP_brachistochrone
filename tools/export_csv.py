"""Write CSV copies of the result tables in data/csv/, for auditing without Python or NumPy.

The CSV files hold the same numbers as the stored JSON results (written with 17 significant digits, so every double is
reproduced exactly); nothing is recomputed. The JSON, JSON-lines and NumPy files in data/ remain the primary record.

  data/csv/tracking_runs.csv        one row per tracking run (833 runs): target, reference path, gate set, duration,
                                    random start, cutoff, tracking weight, final fidelity and minimum closeness under
                                    closed and Lindblad evolution, photon exposure (Fig. 4, Table I, Supplementary
                                    Figs. 8-14, Supplementary Tables 10 and 12)
  data/csv/target_only_runs.csv     one row per target-only start (Supplementary Fig. 16, SI Sec. VIII)
  data/csv/flip_gate_runs.csv       one row per flip-gate optimization (Supplementary Table 11, Supplementary Fig. 17)
  data/csv/figure_facts/*.csv       every number each figure shows (the *_facts.json of the figure scripts), flattened
                                    to (key, value) rows

Usage (from the repository root):  source env.sh && python tools/export_csv.py
"""
import csv
import json
import pathlib

import layout as L

D = L.DATA
OUT = D / "csv"
TR = D / "manuscript" / "fig4_cs_path_following" / "results" / "traces80"
TGS = (2.58, 3.09, 6.18, 16.0)       # grid jobs: index k of their duration (as simulations/tracking/load.py)


def num(x):
    return repr(float(x)) if isinstance(x, float) else x


def flat(x, pre="", out=None, maxlen=8):
    """nested dicts to dotted keys; short lists of scalars to indexed keys; long arrays are left out"""
    out = {} if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            flat(v, f"{pre}.{k}" if pre else str(k), out, maxlen)
    elif isinstance(x, list):
        if len(x) <= maxlen and all(not isinstance(v, (list, dict)) for v in x):
            for i, v in enumerate(x):
                out[f"{pre}[{i}]"] = num(v)
    else:
        out[pre] = num(x)
    return out


def write(path, rows, first=()):
    keys = list(first) + sorted({k for r in rows for k in r} - set(first))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.relative_to(D)}: {len(rows)} rows, {len(keys)} columns")


# tracking runs: the job settings and every scalar result; the traces (x, z, F, F_T, nbar along t) stay in the JSON
rows = []
for p in sorted(TR.glob("*.json")):
    r = json.loads(p.read_text())
    jb = r["job"]
    k, seed = jb.get("k"), jb.get("seed")
    if jb["set"] in ("main", "si") and jb.get("Tg") in TGS:
        k, seed = TGS.index(jb["Tg"]), 2
    row = dict(file=p.name, run_id=jb.get("id"), batch=jb["set"], target=jb.get("name"), path=jb.get("ref"),
               gate_set=jb.get("gate"), k=k, seed=seed, noise_model=jb.get("noise"))
    row.update({key: num(v) for key, v in r.items() if not isinstance(v, (list, dict))})
    rows.append(row)
write(OUT / "tracking_runs.csv", rows, first=("file", "run_id", "batch", "target", "path", "gate_set", "k", "seed",
                                              "T_ns", "N", "D", "noise_model", "w_star", "F_T_final", "F_T_final_noisy",
                                              "C", "C_noisy"))

# target-only preparation and the flip gate (the slice variables X stay in the JSON lines)
S = D / "manuscript_figures" / "sims" / "results"
for src, dst, first in (("capability_noisy.jsonl", "target_only_runs.csv", ("gate", "T_ns", "seed")),
                        ("flip_gate.jsonl", "flip_gate_runs.csv", ("mode", "gate", "T_ns", "N", "seed"))):
    rows = [flat({k: v for k, v in json.loads(l).items() if k != "X"}) for l in open(S / src) if l.strip()]
    write(OUT / dst, rows, first=first)

# the numbers every figure shows
FF = OUT / "figure_facts"
FF.mkdir(parents=True, exist_ok=True)
n = 0
for p in sorted(D.glob("manuscript_figures/**/*_facts.json")):
    f = flat(json.loads(p.read_text()), maxlen=64)
    with open(FF / (p.stem + ".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value"])
        w.writerows(sorted(f.items()))
    n += 1
print(f"wrote csv/figure_facts/: {n} files")

(OUT / "README.md").write_text("""# CSV copies of the result tables

Written by [`tools/export_csv.py`](../../tools/export_csv.py) from the stored JSON results in `data/`, for auditing
with any spreadsheet or language. Numbers carry 17 significant digits, so every stored double is reproduced exactly.
Nothing here is recomputed; the JSON, JSON-lines and NumPy files remain the primary record.

| file | rows | content |
|---|---|---|
| `tracking_runs.csv` | one per tracking run | `target` (gkp, fock2, cat, binomial, flip), `path` (geodesic = brachistochrone, photon, tradeoff), `gate_set` (cs, cd, snap), duration `T_ns` and its index `k`, random start `seed`, slices `N`, Fock cutoff `D` of the Lindblad evaluation, selected tracking weight `w_star`; `F_T_final` and `C` under closed evolution, `F_T_final_noisy` and `C_noisy` under the Lindblad equation (the values in the figures and tables), photon exposure `expo_ns` (closed) and `expo_L_ns` (Lindblad). The traces along $t$ are in the JSON file named in `file` (`data/manuscript/fig4_cs_path_following/results/traces80/`) |
| `target_only_runs.csv` | one per start | target-only GKP preparation: `F60` (closed, cutoff 60), `F90_closed`, `F90_noisy` (cutoff 90, Lindblad) |
| `flip_gate_runs.csv` | one per optimization | the logical flip as a gate: settings, scores at cutoff 60 (`F_T_D60[i]`, `C_D60[i]`), and the evaluation at cutoff 90 (`eval.*`) |
| `figure_facts/<figure>_facts.csv` | (key, value) | every number a figure shows, from its `*_facts.json` (nested keys joined by dots, list items by `[i]`) |

Which runs enter which figure: Fig. 4 and Table I use `target=gkp`, `path=geodesic`, `k` = 1, 2, 3 (412, 824,
2133 ns), 20 starts per gate set and duration (batches `main` and `seed`, see `src/mfcommon.py` and
`simulations/tracking/load.py`).
""")
