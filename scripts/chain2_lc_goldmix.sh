#!/bin/bash
# Post-r3 queue: learning curve (stu3c pseudo, per-class sweep) +
# gold-mixed r3 student + silver-standard eval. GPU-serialized.
set -e
DATA="${DOUBAN_DATA_ROOT:-E:/GB77_data}"
PY="${PYTHON:-python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
cd "$(dirname "$0")/.."

echo "[chain2] waiting for r3 goldprobs..."
for i in $(seq 1 300); do
  [ -f "$DATA/models/macbert_3c_r3pseudo_e2_goldprobs.npy" ] && break
  sleep 60
done

echo "[chain2] learning curve: per-class 334/1667/6667/16667"
for pc in 334 1667 6667 16667; do
  $PY scripts/train_bigdata.py --task 3class --pseudo stu3c --conf 0.7 \
    --per-class $pc --epochs 2 --tag stu3c_pc${pc}_e2
  $PY scripts/gold_probs_bigdata.py --tag stu3c_pc${pc}_e2 --task 3class
  $PY -X utf8 scripts/eval_checkpoint.py --tag stu3c_pc${pc}_e2 --task 3class | head -2
done

echo "[chain2] gold-mixed r3 student"
$PY scripts/train_bigdata.py --task 3class --pseudo stu3c_v3g50 --conf 0.5 \
  --per-class 55000 --epochs 2 --tag macbert_3c_r3g50_e2
$PY scripts/gold_probs_bigdata.py --tag macbert_3c_r3g50_e2 --task 3class
$PY -X utf8 scripts/eval_checkpoint.py --tag macbert_3c_r3g50_e2 --task 3class | head -2

echo "[chain2] silver eval on r3 student"
$PY scripts/silver_eval.py --model "$DATA/models/macbert_3c_r3pseudo_e2" \
  --tag r3pseudo --n-per-rating 800
$PY scripts/silver_eval.py --model "$DATA/models/macbert_3c_pseudo_p55k_e2" \
  --tag stu3c --n-per-rating 800

echo "[chain2] DONE"
