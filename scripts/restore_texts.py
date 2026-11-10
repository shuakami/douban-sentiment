"""Restore full-text data files for local experiments.

The git-ignored mirror lives at ``data/_fulltext/`` and mirrors the
repo layout (``data/_fulltext/data/...``, ``data/_fulltext/annotation/...``).
This script copies them back over the de-identified tracked files.

    python scripts/restore_texts.py            # restore
    python scripts/deidentify_public_texts.py  # re-strip before committing

Only run this on machines that legitimately hold the source data
(Moviedata-10M / your own scrape).  Never commit the restored files.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "_fulltext"


def restore() -> int:
    n = 0
    for src in FULL.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(FULL)          # e.g. data/foo.jsonl
        parts = rel.parts
        if parts[0] not in ("data", "annotation") or len(parts) < 2:
            continue
        if parts == ("annotation", "annotation.html"):
            continue  # tracked file with its own de-identification pass
        dst = ROOT.joinpath(*parts)          # data/foo.jsonl
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    return n


def main() -> None:
    n = restore() if FULL.exists() else 0
    print(f"restored {n} files")
    if n == 0:
        print("no _fulltext mirror found; see DATA_LICENSE.md for data access")


if __name__ == "__main__":
    sys.exit(main())
