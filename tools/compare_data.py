"""Compare two data trees (for example data/ and a copy in which an analysis or simulation was rerun).

For every file present in both trees: JSON and JSON-lines numerically (relative tolerance --rtol; run times and script
names are ignored, and lists of records with an "id" are matched by id), .npz array by array
(numpy.allclose with --rtol and --atol), anything else byte by byte.

Usage:  python tools/compare_data.py data /path/to/rerun/data [--only manuscript_figures/analytics_audit/round3]
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys


IGNORE = {"wall_s", "wall", "elapsed_s", "seconds", "time_s", "runtime_s", "_script", "script", "pkl"}  # run times, script and input file names (metadata)


def _by_id(x):
    """lists of records with an "id" are compared by id (the order of parallel runs is arbitrary)"""
    if isinstance(x, list) and x and all(isinstance(r, dict) and "id" in r for r in x):
        d = {str(r["id"]): r for r in x}
        if len(d) == len(x):
            return d
    return x


def num_diff(a, b, rtol, path=""):
    a, b = _by_id(a), _by_id(b)
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted((set(a) | set(b)) - IGNORE):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing on one side")
            else:
                out += num_diff(a[k], b[k], rtol, f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in num_diff(x, y, rtol, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return []
        if math.isfinite(a) and math.isfinite(b) and abs(a - b) <= rtol * max(abs(a), abs(b)):
            return []
        return [f"{path}: {a!r} vs {b!r}"]
    return [] if a == b else [f"{path}: {str(a)[:50]!r} vs {str(b)[:50]!r}"]


def compare(p, q, rtol, atol):
    if p.suffix == ".json":
        return num_diff(json.loads(p.read_text()), json.loads(q.read_text()), rtol)
    if p.suffix == ".jsonl":
        a = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        b = [json.loads(l) for l in q.read_text().splitlines() if l.strip()]
        return num_diff(a, b, rtol)
    if p.suffix == ".npz":
        import numpy as np
        A, B = np.load(p, allow_pickle=False), np.load(q, allow_pickle=False)
        out = [f"key {k} missing on one side" for k in set(A.files) ^ set(B.files)]
        for k in sorted(set(A.files) & set(B.files)):
            x, y = A[k], B[k]
            if x.shape != y.shape:
                out.append(f"{k}: shape {x.shape} vs {y.shape}")
            elif x.dtype.kind in "fc" and not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
                out.append(f"{k}: max |diff| {np.max(np.abs(x - y)):.3e}")
            elif x.dtype.kind not in "fc" and not np.array_equal(x, y):
                out.append(f"{k}: differs")
        return out
    return [] if hashlib.md5(p.read_bytes()).digest() == hashlib.md5(q.read_bytes()).digest() else ["bytes differ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=pathlib.Path)
    ap.add_argument("b", type=pathlib.Path)
    ap.add_argument("--only", default="")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12)
    x = ap.parse_args()
    bad = n = 0
    for p in sorted((x.a / x.only).rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(x.a)
        q = x.b / rel
        if not q.exists():
            continue
        n += 1
        d = compare(p, q, x.rtol, x.atol)
        if d:
            bad += 1
            print(f"DIFFERS {rel}: " + "; ".join(d[:4]))
    print(f"{n} files compared, {bad} differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
