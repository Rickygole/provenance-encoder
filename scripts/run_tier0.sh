#!/usr/bin/env bash
# Tier 0 kill-shots plus the full arm set, 3 seeds. Each can end the project.
#
# Already settled: A1-global-positions = 4.5122, WORSE than local (4.3465), so
# the "crippled baseline" hypothesis is refuted -- the segment gain is not the
# model buying back global position.
set -u
cd "$(dirname "$0")/.."
OUT=/Users/rickygole/mimic_local_only/tier0.log
: > "$OUT"
run () { echo "=== $1 (seed $2) ===" >> "$OUT"; shift 2
  python3 scripts/pretrain.py --size 10M --steps 1000 --batch 8 --seq-len 256 \
    --seg-tokens 36 --eval-every 500 --eval-batches 30 --shuffle-segments \
    "$@" >> "$OUT" 2>&1; }
for s in 0 1 2; do
  run "A1 vanilla"          "$s" --arm A1 --seed "$s"
  run "T5-RPE"              "$s" --arm A3 --seed "$s" --rpe
  run "ALiBi"               "$s" --arm A3 --seed "$s" --alibi
  run "hard block mask"     "$s" --arm A3 --seed "$s" --hard-block
  run "misaligned blocks"   "$s" --arm A4 --seed "$s" --misaligned-blocks --segment-index-roles
  run "A5 segment index"    "$s" --arm A4 --seed "$s" --segment-index-roles
  run "A4 true roles"       "$s" --arm A4 --seed "$s"
done
echo "tier0 complete" >> "$OUT"
