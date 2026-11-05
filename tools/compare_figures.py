"""Check the regenerated figures against the published ones.

1. Every PNG in output/paper/ is compared with figures/reference_png.json (md5, width and height of the PNG files of the
   paper): identical bytes, or the same pixel size and the largest per-pixel difference (needs --ref).
2. With --ref DIR (a folder holding the paper's figures/ and si_figures/), the pixels are compared as well.
3. Every *_facts.json written by the figure scripts (the numbers each figure shows) is compared with the stored copy in
   data/ (relative tolerance 1e-9 for numbers, exact for strings).

Usage:  python tools/compare_figures.py [--ref DIR]
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import layout as L  # noqa: E402


def png_size(p):
    b = p.read_bytes()[:24]
    return int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")


def pixel_diff(a, b):
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return "pixel comparison needs pillow"
    x = np.asarray(Image.open(a).convert("RGB"), float)
    y = np.asarray(Image.open(b).convert("RGB"), float)
    if x.shape != y.shape:
        return f"shape {x.shape} vs {y.shape}"
    d = np.abs(x - y).max(axis=2)
    return f"max |diff| = {d.max():.0f}/255, fraction of pixels with |diff| > 8: {np.mean(d > 8):.1e}"


def same(a, b, path=""):
    """list of differences between two JSON values"""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'output' if k in a else 'data'}")
            else:
                out += same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += same(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (a == b or (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1e-300))
                or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=pathlib.Path, help="folder with the paper's figures/ and si_figures/")
    a = ap.parse_args()
    ref = json.loads((REPO / "figures" / "reference_png.json").read_text())
    bad = 0
    print("== figures (output/paper/ against the paper's PNG files)")
    for name, r in ref.items():
        p = L.OUT / "paper" / name
        if not p.exists():
            print(f"MISSING  {name}")
            bad += 1
            continue
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        w, h = png_size(p)
        if md5 == r["md5"]:
            status = "IDENTICAL"
        elif (w, h) == (r["width"], r["height"]):
            status = "same size"
        else:
            status = "SIZE DIFFERS"
            bad += 1
        extra = pixel_diff(p, a.ref / name) if a.ref and (a.ref / name).exists() else ""
        print(f"{status:12s} {name:20s} {w}x{h} (paper {r['width']}x{r['height']}) {r.get('label', '')} {extra}")
    print("\n== facts (numbers shown in the figures) against the stored copies in data/")
    n = 0
    for p in sorted(L.OUT.rglob("*_facts.json")):
        rel = p.relative_to(L.OUT)
        q = L.DATA / rel
        if not q.exists():
            continue
        n += 1
        d = same(json.loads(p.read_text()), json.loads(q.read_text()))
        print(f"{'same' if not d else 'DIFFERS':8s} {rel}" + ("" if not d else "  " + "; ".join(d[:5])))
        bad += bool(d)
    print(f"\n{n} facts files compared; {bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
