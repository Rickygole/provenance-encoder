#!/usr/bin/env bash
# Does intra-note attention masking beat standard concatenation on PubMedBERT?
# Same backbone, same weights, same data order. Only the attention mask differs.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/Users/rickygole/mimic_local_only/hf
OUT=/Users/rickygole/mimic_local_only/task.log
: > "$OUT"
for seed in 0 1; do
  for arm in "" "--block"; do
    echo "=== seed=$seed arm=${arm:-concat} ===" >> "$OUT"
    python3 scripts/finetune_pubmedbert.py --epochs 1 --batch 4 --accum 4 \
      --seed "$seed" $arm >> "$OUT" 2>&1
  done
done
echo "task complete" >> "$OUT"
