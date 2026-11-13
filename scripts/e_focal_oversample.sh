#!/usr/bin/env bash
# Experiment E: focal loss + class-balanced oversampling on the 3-class
# distill student (gold-600 only, deployed recipe a0.5/t2/e8/m256/s42/cw1).
# Serialized GPU work — do NOT run alongside other GPU jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=D:/Project/douban-sentiment-env/hf
export DOUBAN_DATA_ROOT=E:/GB77_data
PY=D:/Project/douban-sentiment-env/venv/Scripts/python.exe
MODEL=hfl/chinese-macbert-base

echo "=== E-OOF focal_gamma=2 ==="
"$PY" scripts/distill.py --model "$MODEL" --task 3class --alpha 0.5 --tau 2 \
  --epochs 8 --max_len 256 --seed 42 --class_weight 1 --focal_gamma 2 \
  --tag distill_macbert_3c_a50_t2_e8_m256_focal2

echo "=== E-OOF balance_classes ==="
"$PY" scripts/distill.py --model "$MODEL" --task 3class --alpha 0.5 --tau 2 \
  --epochs 8 --max_len 256 --seed 42 --class_weight 1 --balance_classes 1 \
  --tag distill_macbert_3c_a50_t2_e8_m256_bal

echo "=== E-final base (gold-only control, human400) ==="
"$PY" scripts/train_final_distill.py --model "$MODEL" --task 3class \
  --alpha 0.5 --tau 2 --epochs 8 --max_len 256 --seed 42 --class_weight 1 \
  --out eexp_base
"$PY" scripts/predict_human400.py --model models/eexp_base --arm base

echo "=== E-final focal ==="
"$PY" scripts/train_final_distill.py --model "$MODEL" --task 3class \
  --alpha 0.5 --tau 2 --epochs 8 --max_len 256 --seed 42 --class_weight 1 \
  --focal_gamma 2 --out eexp_focal
"$PY" scripts/predict_human400.py --model models/eexp_focal --arm focal

echo "=== E-final balance ==="
"$PY" scripts/train_final_distill.py --model "$MODEL" --task 3class \
  --alpha 0.5 --tau 2 --epochs 8 --max_len 256 --seed 42 --class_weight 1 \
  --balance_classes 1 --out eexp_bal
"$PY" scripts/predict_human400.py --model models/eexp_bal --arm bal

echo "=== E DONE ==="
