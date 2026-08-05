#!/usr/bin/env bash
# Waits for the live matched run, then works the remaining plan in dependency
# order. Readmission is first: it needs no pretrained checkpoint and it is the
# only task measured to have headroom above the floor (0.678 vs mortality 0.874).
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/Users/rickygole/mimic_local_only/hf
OUT=/Users/rickygole/mimic_local_only/queue2.log
: > "$OUT"
say () { echo "=== $* ===" >> "$OUT"; }

while pgrep -f "run_matched.py --systems metadata" >/dev/null; do sleep 60; done
say "matched2 finished; starting queue"

# 1. READMISSION -- highest value, no dependency on any checkpoint.
say "READMISSION: floors + ours_random + PubMedBERT"
python3 scripts/run_matched.py --task-dir local_only/task_readmit30 \
  --systems metadata,tfidf,ours_random,pubmedbert --max-len 512 \
  --chars-per-note 1200 --epochs 3 --lr-grid 2e-5,5e-5 --seeds 2 >> "$OUT" 2>&1

# 2. Tokenizer on the decontaminated corpus -- unblocks all pretraining.
say "retrain tokenizer (decontaminated corpus)"
python3 scripts/train_tokenizer.py >> "$OUT" 2>&1

# 3. Re-pretrain so a checkpoint exists that records its input convention.
#    Every prior checkpoint was deleted and the new code refuses to guess.
say "re-pretrain block-mask arm (records input_convention)"
python3 scripts/pretrain.py --arm A3 --size 10M --steps 3000 --batch 8 \
  --seq-len 256 --seg-tokens 36 --seed 0 --eval-every 3000 --eval-batches 30 \
  --shuffle-segments --hard-block >> "$OUT" 2>&1

# 4. Pretrained vs random, now under the correct convention, on BOTH tasks.
for T in local_only/task local_only/task_readmit30; do
  say "pretrained-vs-random on $T"
  python3 scripts/run_matched.py --task-dir "$T" \
    --systems ours_random,ours_pretrained \
    --ckpt local_only/runs/A3_10M_st3000_s0_hardblk_shuf \
    --max-len 256 --chars-per-note 600 --epochs 3 --lr-grid 2e-5,5e-5 \
    --seeds 2 >> "$OUT" 2>&1
done
say "queue2 complete"
