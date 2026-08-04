#!/usr/bin/env bash
# Label-budget ladder: fine-tune at {500, 2000, 8000, full} labeled training
# examples, for our pretrained arm, our random-init arm, and TF-IDF -- with
# the metadata floor always alongside too.
#
# The question this answers: pretraining's benefit is supposed to live in the
# LOW-label regime. If pretraining still does nothing at 500 labels, the null
# measured at full label budget is strong. If it helps in the low-label
# regime, the honest claim shrinks to "labels wash out pretraining" rather
# than "pretraining does nothing."
#
# Subsampling is subject-group-disjoint and prevalence-preserving (see
# subsample_group_disjoint in scripts/run_matched.py) and only ever touches
# the TRAIN split -- val/test are never subsampled or otherwise touched.
#
# Every rung is one scripts/run_matched.py invocation, so every table
# includes the non-neural floor by construction and reports mean+/-std over
# --ft-seeds.
#
# Env overrides (all optional):
#   CKPT                pretrained checkpoint dir (must contain model.pt).
#                       MUST be a checkpoint pretrained on the DECONTAMINATED
#                       corpus -- see scripts/build_bundles.py -- otherwise
#                       this ladder inherits the same task-split leakage it
#                       is trying to characterize.
#   LABEL_BUDGETS       space-separated budgets; "full" means no subsampling
#                       (default: 500 2000 8000 full)
#   FT_SEEDS            fine-tuning seeds passed to run_matched.py (default 0,1,2)
#   SUBSAMPLE_SEED       seed for the group-disjoint subsample itself (default 0)
#   LR_GRID, EPOCHS     fine-tuning protocol (default 2e-5,5e-5 / 2)
#   DEVICE              torch device (default auto)
#   TASK_DIR            default local_only/task
#   TAG_PREFIX          results/ file tag prefix (default labelladder)
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${CKPT:-local_only/runs/A3_10M_st3000_s0_hardblk_shuf}"
LABEL_BUDGETS="${LABEL_BUDGETS:-500 2000 8000 full}"
FT_SEEDS="${FT_SEEDS:-0,1,2}"
SUBSAMPLE_SEED="${SUBSAMPLE_SEED:-0}"
LR_GRID="${LR_GRID:-2e-5,5e-5}"
EPOCHS="${EPOCHS:-2}"
DEVICE="${DEVICE:-auto}"
TASK_DIR="${TASK_DIR:-local_only/task}"
TOKENIZER_DIR="${TOKENIZER_DIR:-local_only/tokenizer}"   # must match the tokenizer CKPT was pretrained with
TAG_PREFIX="${TAG_PREFIX:-labelladder}"
LIMIT_EVAL="${LIMIT_EVAL:-0}"   # smoke-test only; 0 = full downstream val set

if [ ! -f "$CKPT/model.pt" ]; then
  echo "CKPT=$CKPT has no model.pt -- point CKPT at a real pretrained checkpoint" \
       "(e.g. produced by scripts/pretrain.py or scripts/run_token_ladder.sh)." >&2
  exit 1
fi
echo "label ladder: ckpt=$CKPT budgets=[$LABEL_BUDGETS] ft_seeds=$FT_SEEDS device=$DEVICE"

for BUDGET in $LABEL_BUDGETS; do
  SUB_FLAGS=()
  if [ "$BUDGET" != "full" ]; then
    SUB_FLAGS=(--subsample-train "$BUDGET" --subsample-seed "$SUBSAMPLE_SEED")
  fi
  echo
  echo "=== rung: label budget $BUDGET ==="
  python3 scripts/run_matched.py \
    --systems metadata,tfidf,ours_random,ours_pretrained \
    --ckpt "$CKPT" --seeds "$FT_SEEDS" --lr-grid "$LR_GRID" --epochs "$EPOCHS" \
    --device "$DEVICE" --task-dir "$TASK_DIR" --tokenizer-dir "$TOKENIZER_DIR" \
    --limit-eval "$LIMIT_EVAL" \
    ${SUB_FLAGS[@]+"${SUB_FLAGS[@]}"} \
    --tag "${TAG_PREFIX}_${BUDGET}"
done

echo
python3 scripts/analyze_label_ladder.py --tag-prefix "$TAG_PREFIX"
