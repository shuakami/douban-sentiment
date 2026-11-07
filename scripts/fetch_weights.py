#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch the published fine-tuned weights into the layout web/manifest.json expects.

Downloads Shuakami/douban-sentiment-weights from the Hugging Face Hub and
copies the files under models/ (gitignored):

    macbert_{3c,bin,neu}/  ->  models/macbert_{3c_distill,bin_distill,neu_corpus}/
    heads/*.pt             ->  models/*.pt

Also caches the frozen judge backbone
IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment (~2.6 GB) so the demo can
run fully offline afterwards. hfl/chinese-macbert-base is only needed for
retraining, not inference.
"""
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "Shuakami/douban-sentiment-weights"
BACKBONE = "IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment"
MODELS = ROOT / "models"

DIR_MAP = {
    "macbert_3c": "macbert_3c_distill",
    "macbert_bin": "macbert_bin_distill",
    "macbert_neu": "macbert_neu_corpus",
}


def main():
    snap = Path(snapshot_download(REPO, ignore_patterns=["papers/*", ".gitattributes"]))
    MODELS.mkdir(exist_ok=True)
    for src_name, dst_name in DIR_MAP.items():
        src, dst = snap / src_name, MODELS / dst_name
        if not src.is_dir():
            sys.exit(f"missing {src_name} in {REPO}")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"{src_name}/ -> models/{dst_name}/")
    heads = snap / "heads"
    for pt in heads.glob("*.pt"):
        shutil.copy2(pt, MODELS / pt.name)
        print(f"heads/{pt.name} -> models/{pt.name}")
    snapshot_download(BACKBONE)
    print(f"backbone {BACKBONE} cached")
    print("done — the demo can now run with HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
