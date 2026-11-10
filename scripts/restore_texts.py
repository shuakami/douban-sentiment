"""Restore full-text data files for local experiments.

Copies the git-ignored originals from ``data/_fulltext/`` and
``annotation/_fulltext/`` back over the de-identified tracked files.

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
AFULL = ROOT / "annotation" / "_fulltext"


def restore(src_root: Path, dst_root: Path) -> int:
    n = 0
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    return n


def main() -> None:
    n = restore(FULL, ROOT / "data") if FULL.exists() else 0
    m = restore(AFULL, ROOT / "annotation") if AFULL.exists() else 0
    print(f"restored {n} data files, {m} annotation files")
    if n + m == 0:
        print("no _fulltext mirror found; see DATA_LICENSE.md for data access")


if __name__ == "__main__":
    sys.exit(main())
