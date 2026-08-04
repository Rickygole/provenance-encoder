#!/usr/bin/env bash
# Runs after the matched-protocol job. Order matters: the lineage guard blocks
# pretraining until the tokenizer is retrained on the decontaminated corpus.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/Users/rickygole/mimic_local_only/hf
OUT=/Users/rickygole/mimic_local_only/queue.log
: > "$OUT"
while pgrep -f run_matched.py >/dev/null; do sleep 60; done
echo "=== matched job finished, starting queue ===" >> "$OUT"

# 1. DIAGNOSTIC: is PubMedBERT's collapse (0.8223 -> 0.6376) the task or the
#    input budget? Same fixed task, seq 512 / 1200 chars ~= the original budget.
echo "=== DIAGNOSTIC: PubMedBERT seq512 on fixed task ===" >> "$OUT"
python3 scripts/run_matched.py --systems pubmedbert --max-len 512 \
  --chars-per-note 1200 --epochs 2 --lr-grid 2e-5 --seeds 1 >> "$OUT" 2>&1

# 2. Retrain tokenizer on the decontaminated corpus (unblocks pretraining).
echo "=== retrain tokenizer on decontaminated corpus ===" >> "$OUT"
python3 scripts/train_tokenizer.py >> "$OUT" 2>&1

# 3. THE TASK THAT HAS HEADROOM: readmission. Floor is only 0.678 there.
echo "=== READMISSION: all systems ===" >> "$OUT"
python3 scripts/run_matched.py --task-dir local_only/task_readmit30 \
  --systems metadata,tfidf,ours_random,pubmedbert --max-len 512 \
  --chars-per-note 1200 --epochs 2 --lr-grid 2e-5,5e-5 --seeds 2 >> "$OUT" 2>&1
echo "queue complete" >> "$OUT"
