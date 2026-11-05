"""Check the integrity of data/ against data/SHA256SUMS (SHA-256 of every stored file).

Pure Python, no third-party packages; equivalent to `cd data && sha256sum -c SHA256SUMS`. Reports missing, changed and
unlisted files; exit status 1 if any file is missing or changed.

Usage:  python tools/verify_data.py [data folder]      (default: data/ of this repository, or $GKP_DATA)
"""
import hashlib
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GKP_DATA", REPO / "data")).resolve()
listed, bad, missing = {}, [], []
for line in (D / "SHA256SUMS").read_text().splitlines():
    h, p = line.split("  ", 1)
    listed[p] = h
for p, h in listed.items():
    f = D / p
    if not f.is_file():
        missing.append(p)
        continue
    sha = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    if sha.hexdigest() != h:
        bad.append(p)
extra = sorted(f.relative_to(D).as_posix() for f in D.rglob("*") if f.is_file()
               and f.relative_to(D).as_posix() not in listed and f.name not in ("README.md", "SHA256SUMS")
               and "/results/smoke/" not in f.as_posix())
print(f"{len(listed)} files listed in {D / 'SHA256SUMS'}: {len(listed) - len(bad) - len(missing)} OK, "
      f"{len(bad)} changed, {len(missing)} missing; {len(extra)} files not listed")
for name, lst in (("changed", bad), ("missing", missing), ("not listed", extra)):
    for p in lst[:20]:
        print(f"  {name}: {p}")
print("DATA INTEGRITY OK" if not (bad or missing) else "DATA INTEGRITY FAILED")
sys.exit(1 if (bad or missing) else 0)
