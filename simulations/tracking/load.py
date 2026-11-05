"""Group the pulse traces by (transfer, reference path, gate set, duration index k) over the random starts.

Sources: results/<traces>/<id>.json for the grid jobs of the first production run (sets main/si, seed 2; used only on
the brachistochrone, where the grid duration equals the path-paced duration) and for the seed jobs (set seed, seeds 1-5,
path-paced durations T_k = m_k L_path/omega).  Durations k = 0..3 (deck grid gT = 2.58, 3.09, 6.18, 16.0).
"""
import layout as _L  # repository layout: data/ and output/ (src/layout.py)
import json
import os
import pathlib
from collections import defaultdict

import numpy as np

HERE = _L.vfile(__file__).parent
TGS = (2.58, 3.09, 6.18, 16.0)


def groups(traces="traces80"):
    out = defaultdict(list)
    layout_test = os.environ.get("TRACK_LAYOUT_TEST") == "1"     # layout tests only: grid jobs of every path
    for p in sorted((HERE / "results" / traces).glob("*.json")):
        r = json.loads(p.read_text())
        jb = r["job"]
        if jb["set"] in ("main", "si"):
            if (jb["ref"] != "geodesic" and not layout_test) or jb["Tg"] not in TGS:
                continue
            k, seed = TGS.index(jb["Tg"]), 2
        elif jb["set"] == "seed":
            k, seed = jb["k"], jb["seed"]
        else:
            continue
        r["seed"] = seed
        out[(jb["name"], jb["ref"], jb["gate"], k)].append(r)
    return out


def stat(lst, key, fn=None):
    """mean and standard deviation over the random starts of a scalar"""
    v = np.array([fn(r) if fn else r[key] for r in lst], float)
    return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def representative(lst, keys=("xL", "zL", "FL"), tkey="tL_ns", n=201):
    """the start closest to the mean: smallest mean-squared distance of its (x, z, F) trajectory (Lindblad evolution, on a
    common grid of fractions of T) from the average over the starts.  A single physical run, unlike the mean state,
    which is a mixture of different pulses and smears out the evolution."""
    if len(lst) == 1:
        return lst[0]
    d = np.zeros(len(lst))
    for key in keys:
        _, m, _, Y = curve_stat(lst, key, tkey, n)
        d += np.mean((Y - m[None, :]) ** 2, axis=1)
    return lst[int(np.argmin(d))]


def curve_stat(lst, key, tkey="t_ns", n=201):
    """mean and standard deviation of a time trace over the starts, on a common grid of fractions of T"""
    s = np.linspace(0, 1, n)
    Y = []
    for r in lst:
        t = np.array(r[tkey])
        Y.append(np.interp(s, t / t[-1], np.array(r[key])))
    Y = np.array(Y)
    T = float(np.mean([r[tkey][-1] for r in lst]))
    return s * T, Y.mean(0), (Y.std(0, ddof=1) if len(Y) > 1 else 0 * Y[0]), Y
