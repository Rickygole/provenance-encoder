#!/usr/bin/env bash
# From-scratch pipeline, no backbone anywhere.
#   1. Pretrain the encoder with MLM on MIMIC note bundles (saves weights now)
#   2. Fine-tune on multi-note mortality
#   3. Ablate: pretrained vs random init, block mask vs vanilla
set -u
cd "$(dirname "$0")/.."
OUT=/Users/rickygole/mimic_local_only/ours.log
: > "$OUT"

echo "=== PRETRAIN: hard block mask ===" >> "$OUT"
python3 scripts/pretrain.py --arm A3 --size 10M --steps 3000 --batch 8 --seq-len 256 \
  --seg-tokens 36 --seed 0 --eval-every 1000 --eval-batches 30 --shuffle-segments \
  --hard-block >> "$OUT" 2>&1

echo "=== PRETRAIN: vanilla ===" >> "$OUT"
python3 scripts/pretrain.py --arm A1 --size 10M --steps 3000 --batch 8 --seq-len 256 \
  --seg-tokens 36 --seed 0 --eval-every 1000 --eval-batches 30 --shuffle-segments >> "$OUT" 2>&1

R=/Users/rickygole/mimic_local_only/runs
echo "=== FINETUNE: pretrained + block mask ===" >> "$OUT"
python3 scripts/finetune_ours.py --ckpt "$R/A3_10M_st3000_s0_hardblk_shuf" \
  --tag blockpre --epochs 3 >> "$OUT" 2>&1

echo "=== FINETUNE: pretrained vanilla ===" >> "$OUT"
python3 scripts/finetune_ours.py --ckpt "$R/A1_10M_st3000_s0_shuf" \
  --tag vanillapre --epochs 3 >> "$OUT" 2>&1

echo "=== FINETUNE: random init (no pretraining) ===" >> "$OUT"
python3 scripts/finetune_ours.py --ckpt none --tag randinit --epochs 3 >> "$OUT" 2>&1

echo "ours complete" >> "$OUT"
