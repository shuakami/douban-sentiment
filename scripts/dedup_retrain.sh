#!/usr/bin/env bash
# Dedup-592 gold-OOF retrain driver — serialized GPU jobs.
# Replicates the deployed-pipeline channels on reviews_600_dedup592.jsonl
# with Leak-B-consistent teacher targets (dedup-aligned probs).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="D:/Project/douban-sentiment-env/venv/Scripts/python.exe"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="D:/Project/douban-sentiment-env/hf"
DATA=reviews_600_dedup592.jsonl
LOG=out/dedup_retrain.log
: > "$LOG"

run() { echo "=== $* ===" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

# --- distill students (GPU, ~2 min/fold-set each) ---
run $PY scripts/distill.py --model hfl/chinese-macbert-base --task 3class \
    --alpha 0.5 --epochs 8 --tag dedup_distill_3c_a50 --data $DATA \
    --teacher ens_3class_allch_dedup592.json
run $PY scripts/distill.py --model hfl/chinese-macbert-base --task 3class \
    --alpha 1.0 --epochs 8 --tag dedup_distill_3c_a100 --data $DATA \
    --teacher ens_3class_allch_dedup592.json
run $PY scripts/distill.py --model hfl/chinese-macbert-base --task binary \
    --alpha 0.5 --epochs 8 --tag dedup_distill_bin_a50 --data $DATA \
    --teacher ens_binary_allch2_dedup592.json
run $PY scripts/distill.py --model hfl/chinese-macbert-base --task binary \
    --alpha 1.0 --epochs 8 --tag dedup_distill_bin_a100 --data $DATA \
    --teacher ens_binary_allch2_dedup592.json

# --- frozen-feature MLP heads (fast) ---
run $PY scripts/distill_head.py 3class feats_mb13b_m256_dedup592.npy \
    --tag dedup_head_3c_a50 --alpha 0.5 --epochs 300 --data $DATA \
    --teacher ens_3class_allch_dedup592.json
run $PY scripts/distill_head.py 3class feats_mb13b_m256_dedup592.npy \
    --tag dedup_head_3c_a100 --alpha 1.0 --epochs 300 --data $DATA \
    --teacher ens_3class_allch_dedup592.json
run $PY scripts/distill_head.py binary feats_mb13b_m256_dedup592.npy \
    --tag dedup_head_bin_a50 --alpha 0.5 --epochs 300 --data $DATA \
    --teacher ens_binary_allch2_dedup592.json
run $PY scripts/distill_head.py binary feats_mb13b_m256_dedup592.npy \
    --tag dedup_head_bin_a100 --alpha 1.0 --epochs 300 --data $DATA \
    --teacher ens_binary_allch2_dedup592.json
run $PY scripts/distill_head.py neu feats_mb13b_m256_dedup592.npy \
    --tag dedup_head_neu_a50 --hidden 256 --alpha 0.5 --epochs 300 --data $DATA \
    --teacher ens_neu_teacher_dedup592.json
run $PY scripts/distill_head.py neu feats_mb13b_m256_dedup592.npy \
    --tag dedup_head_neu_a100 --hidden 256 --alpha 1.0 --epochs 300 --data $DATA \
    --teacher ens_neu_teacher_dedup592.json

echo "=== ALL DONE ===" | tee -a "$LOG"
