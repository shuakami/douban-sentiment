#!/usr/bin/env bash
# 下载所需模型到 models/ 目录
set -e
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p models

python3.11 - <<'PY'
import os
from huggingface_hub import snapshot_download

targets = [
    ("bert-base-chinese", "models/bert-base-chinese"),
    ("IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment", "models/erlangshen-110m-sentiment"),
]
for repo, dst in targets:
    print("downloading", repo)
    # 注意：Erlangshen 仓库只有 pytorch_model.bin，没有 safetensors
    snapshot_download(repo_id=repo, local_dir=dst,
                      allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors", "*.model"])
    print("  ->", dst)
PY
