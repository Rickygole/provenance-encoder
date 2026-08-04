#!/usr/bin/env bash
# Token-budget ladder: pretrain the SAME arm at {1M, 6M, 25M, 60M} MLM tokens
# and fine-tune each resulting checkpoint on the downstream task, printing MLM
# val loss and downstream AUROC side by side at every rung.
#
# The question this answers: we measured a 1.320 nat MLM improvement produce a
# 0.0004 AUROC change, and the obvious rebuttal is "you undertrained -- 6M
# tokens for a 10.6M-param model is ~0.6 tokens/param, ~30x under
# compute-optimal." If the downstream line stays flat across a 60x token
# range while MLM loss keeps falling, that rebuttal is answered rather than
# dodged.
#
# Fresh process per pretraining run -- scripts/pretrain.py documents an MPS
# memory leak across configs, so this script never invokes it twice in one
# process. Every downstream evaluation is routed through
# scripts/run_matched.py, which ALWAYS computes the non-neural floor
# (metadata-only LR, TF-IDF) alongside any neural arm and reports mean+/-std
# over the fine-tuning seeds in --ft-seeds, so no rung's table can omit the
# floor by construction.
#
# Env overrides (all optional):
#   ARM, ARM_FLAGS      pretraining arm + its flags (default: A3, the
#                       hard-block-mask winning arm from the README headline)
#   SIZE                model size preset (default 10M)
#   BATCH, SEQ_LEN, SEG_TOKENS   pretraining shape (default 8, 256, 36 --
#                       matches the existing pretraining protocol)
#   TOKEN_BUDGETS       space-separated token counts (default: 1M 6M 25M 60M)
#   PRETRAIN_SEED       pretraining seed (default 0)
#   FT_SEEDS            fine-tuning seeds passed to run_matched.py (default 0,1,2)
#   LR_GRID, EPOCHS     fine-tuning protocol (default 2e-5,5e-5 / 2)
#   DEVICE              torch device for BOTH pretrain and fine-tune (default auto)
#   TASK_DIR, BUNDLES_DIR, TOKENIZER_DIR   data locations (default local_only/*)
#   TAG_PREFIX          results/ file tag prefix (default tokladder)
#   KEEP_CKPTS          if "1", keep each rung's pretrained weights on disk
#                       after fine-tuning consumes them (default: delete them --
#                       disk is at ~22GB free and per-epoch checkpoints are
#                       explicitly the thing not to accumulate)
set -euo pipefail
cd "$(dirname "$0")/.."

ARM="${ARM:-A3}"
ARM_FLAGS="${ARM_FLAGS:---hard-block --shuffle-segments}"
SIZE="${SIZE:-10M}"
BATCH="${BATCH:-8}"
SEQ_LEN="${SEQ_LEN:-256}"
SEG_TOKENS="${SEG_TOKENS:-36}"
TOKEN_BUDGETS="${TOKEN_BUDGETS:-1000000 6000000 25000000 60000000}"
PRETRAIN_SEED="${PRETRAIN_SEED:-0}"
FT_SEEDS="${FT_SEEDS:-0,1,2}"
LR_GRID="${LR_GRID:-2e-5,5e-5}"
EPOCHS="${EPOCHS:-2}"
DEVICE="${DEVICE:-auto}"
TASK_DIR="${TASK_DIR:-local_only/task}"
BUNDLES_DIR="${BUNDLES_DIR:-local_only/bundles}"
TOKENIZER_DIR="${TOKENIZER_DIR:-local_only/tokenizer}"
TAG_PREFIX="${TAG_PREFIX:-tokladder}"
KEEP_CKPTS="${KEEP_CKPTS:-0}"
LIMIT_TRAIN="${LIMIT_TRAIN:-0}"   # smoke-test only; 0 = full downstream train set
LIMIT_EVAL="${LIMIT_EVAL:-0}"     # smoke-test only; 0 = full downstream val set

TOKENS_PER_STEP=$((BATCH * SEQ_LEN))
echo "token ladder: arm=$ARM ($ARM_FLAGS) size=$SIZE budgets=[$TOKEN_BUDGETS] " \
     "tokens/step=$TOKENS_PER_STEP device=$DEVICE"

for BUDGET in $TOKEN_BUDGETS; do
  STEPS=$(( (BUDGET + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
  echo
  echo "=== rung: token budget $BUDGET (~$STEPS pretraining steps) ==="
  LOG="$(mktemp)"
  python3 scripts/pretrain.py --arm "$ARM" $ARM_FLAGS --size "$SIZE" \
    --steps "$STEPS" --batch "$BATCH" --seq-len "$SEQ_LEN" --seg-tokens "$SEG_TOKENS" \
    --seed "$PRETRAIN_SEED" --eval-every "$STEPS" --eval-batches 30 --device "$DEVICE" \
    --tag "${TAG_PREFIX}_b${BUDGET}" \
    --task-dir "$TASK_DIR" --bundles-dir "$BUNDLES_DIR" --tokenizer-dir "$TOKENIZER_DIR" \
    2>&1 | tee "$LOG"
  RUN_DIR="$(grep -o 'saved weights -> .*/model\.pt' "$LOG" | sed 's/saved weights -> //; s#/model\.pt##')"
  rm -f "$LOG"
  if [ -z "$RUN_DIR" ]; then
    echo "pretrain.py did not report saved weights for budget $BUDGET -- aborting ladder" >&2
    exit 1
  fi
  echo "  checkpoint: $RUN_DIR"

  python3 scripts/run_matched.py \
    --systems metadata,tfidf,ours_random,ours_pretrained \
    --ckpt "$RUN_DIR" --seeds "$FT_SEEDS" --lr-grid "$LR_GRID" --epochs "$EPOCHS" \
    --device "$DEVICE" --task-dir "$TASK_DIR" --tokenizer-dir "$TOKENIZER_DIR" \
    --limit-train "$LIMIT_TRAIN" --limit-eval "$LIMIT_EVAL" \
    --tag "${TAG_PREFIX}_b${BUDGET}"

  if [ "$KEEP_CKPTS" != "1" ]; then
    # Delete only the weights (the disk-heavy part) -- summary.json and
    # metrics.jsonl are pure scalars (no note text) and analyze_token_ladder.py
    # needs summary.json's MLM loss for this exact rung after run_matched.py
    # has already consumed model.pt above.
    rm -f "$RUN_DIR/model.pt"
    echo "  deleted $RUN_DIR/model.pt (kept summary.json for the ladder table; " \
         "set KEEP_CKPTS=1 to retain full weights)"
  fi
done

echo
python3 scripts/analyze_token_ladder.py --tag-prefix "$TAG_PREFIX"
