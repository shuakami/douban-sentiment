#!/bin/bash
# r3 flywheel chain: wait for pool feats -> detector pool probs ->
# v3 ensemble probs -> r3 student train -> gold eval. GPU-serialized.
set -e
DATA="${DOUBAN_DATA_ROOT:-E:/GB77_data}"
PY="${PYTHON:-python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
cd "$(dirname "$0")/.."

echo "[chain] waiting for mb13b pool feats..."
for i in $(seq 1 240); do
  [ -f "$DATA/features/mb13b_pool_c07_feats.npy" ] && break
  sleep 60
done
[ -f "$DATA/features/mb13b_pool_c07_feats.npy" ] || { echo "TIMEOUT"; exit 1; }

echo "[chain] detector pool probs"
$PY scripts/pool_probs.py --model "$DATA/models/neu_pseudo_p60k_e2" \
  --tag neu_corpus_pool --batch 128 --conf 0.7

echo "[chain] v3 ensemble corpus probs"
$PY scripts/ensemble_corpus_probs.py

echo "[chain] gold-mix pool prep"
$PY scripts/mix_gold_into_pool.py --pseudo stu3c_v3 --repeat 50 --tag stu3c_v3g50

echo "[chain] r3 student training"
$PY scripts/train_bigdata.py --task 3class --pseudo stu3c_v3 --conf 0.5 \
  --per-class 55000 --epochs 2 --tag macbert_3c_r3pseudo_e2

echo "[chain] gold probs + eval"
$PY scripts/gold_probs_bigdata.py --tag macbert_3c_r3pseudo_e2 --task 3class
$PY -X utf8 scripts/eval_checkpoint.py --tag macbert_3c_r3pseudo_e2 --task 3class

echo "[chain] DONE"
