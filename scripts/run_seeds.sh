#!/usr/bin/env bash
# Seeds 1 and 2 for every downstream arm. The headline claim -- "pretraining
# contributes nothing" -- rests on 0.8014 vs 0.7998, which one seed cannot
# separate. Pretraining seed is varied too, not just the fine-tuning seed,
# because pretraining-seed variance is the relevant noise for that claim.
set -u
cd "$(dirname "$0")/.."
OUT=/Users/rickygole/mimic_local_only/seeds.log
: > "$OUT"
R=/Users/rickygole/mimic_local_only/runs

for s in 1 2; do
  echo "=== PRETRAIN block seed=$s ===" >> "$OUT"
  python3 scripts/pretrain.py --arm A3 --size 10M --steps 3000 --batch 8 --seq-len 256 \
    --seg-tokens 36 --seed "$s" --eval-every 3000 --eval-batches 30 --shuffle-segments \
    --hard-block >> "$OUT" 2>&1
  echo "=== PRETRAIN vanilla seed=$s ===" >> "$OUT"
  python3 scripts/pretrain.py --arm A1 --size 10M --steps 3000 --batch 8 --seq-len 256 \
    --seg-tokens 36 --seed "$s" --eval-every 3000 --eval-batches 30 --shuffle-segments \
    >> "$OUT" 2>&1

  echo "=== FINETUNE blockpre seed=$s ===" >> "$OUT"
  python3 scripts/finetune_ours.py --ckpt "$R/A3_10M_st3000_s${s}_hardblk_shuf" \
    --tag blockpre --seed "$s" --epochs 2 >> "$OUT" 2>&1
  echo "=== FINETUNE vanillapre seed=$s ===" >> "$OUT"
  python3 scripts/finetune_ours.py --ckpt "$R/A1_10M_st3000_s${s}_shuf" \
    --tag vanillapre --seed "$s" --epochs 2 >> "$OUT" 2>&1
  echo "=== FINETUNE randinit seed=$s ===" >> "$OUT"
  python3 scripts/finetune_ours.py --ckpt none --tag randinit --seed "$s" --epochs 2 >> "$OUT" 2>&1
done
echo "seeds complete" >> "$OUT"
