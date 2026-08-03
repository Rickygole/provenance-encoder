#!/usr/bin/env bash
# THE decisive control. A4 gained 0.84 nats over vanilla at 1000 steps. Is that
# provenance, or is it a full-rank attention bias shared across every layer?
#
#   real      -- true role labels
#   permuted  -- derangement of role labels: structure kept, semantics destroyed
#   random    -- independent random role per segment: structure destroyed too
#
# If the gain survives permuted, it is not provenance. If it survives random, it
# is capacity or structure and nothing to do with document authority.
set -u
cd "$(dirname "$0")/.."
OUT=/Users/rickygole/mimic_local_only/controls.log
: > "$OUT"
for spec in "A4:" "A4:--permute-roles" "A4:--random-roles" "A3:--random-roles"; do
  arm="${spec%%:*}"; flag="${spec#*:}"
  echo "=== arm=$arm control='${flag:-real}' ===" >> "$OUT"
  python3 scripts/pretrain.py --arm "$arm" --size 10M --steps 1000 --batch 8 \
    --seq-len 256 --seg-tokens 36 --seed 0 --eval-every 500 --eval-batches 30 \
    $flag >> "$OUT" 2>&1
done
echo "controls complete" >> "$OUT"
