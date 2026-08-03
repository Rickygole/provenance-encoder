#!/usr/bin/env bash
# Does A4's 0.84-nat gain come from PROVENANCE, or from any per-segment partition?
#
# All arms use --shuffle-segments so role is decorrelated from bundle position
# (bundles are chronological; segment index alone predicts role at 45.1% vs a
# 32.6% majority baseline).
#
#   A4 real      : true role labels                 -- the claim
#   A5 segidx    : ordinal position in bundle       -- POSITIVE competing hypothesis
#   A4 random    : independent random role/segment  -- destroys consistency
#   A4 length    : note-length bucket               -- informative, no authority meaning
#   A1 vanilla   : reference under the same shuffling
#
# A5 ~ A4  => segmentation, not provenance. Thesis dead, mechanism identified.
# A5 ~ A1  => role identity does real work.
#
# NOTE: --permute-roles is deliberately absent. A global derangement is VACUOUS:
# `A` is a free KxK table, so the model relearns the permuted table and the loss
# is unchanged by construction. It has zero statistical power.
set -u
cd "$(dirname "$0")/.."
OUT=/Users/rickygole/mimic_local_only/controls.log
: > "$OUT"
run () {
  echo "=== $1 ===" >> "$OUT"
  shift
  python3 scripts/pretrain.py --size 10M --steps 1000 --batch 8 --seq-len 256 \
    --seg-tokens 36 --seed 0 --eval-every 500 --eval-batches 30 "$@" >> "$OUT" 2>&1
}
run "A1 vanilla (shuffled)"       --arm A1 --shuffle-segments
run "A4 real roles (shuffled)"    --arm A4 --shuffle-segments
run "A5 segment-index roles"      --arm A4 --shuffle-segments --segment-index-roles
run "A4 random roles"             --arm A4 --shuffle-segments --random-roles
run "A4 length-bucket roles"      --arm A4 --shuffle-segments --length-roles
echo "controls complete" >> "$OUT"
