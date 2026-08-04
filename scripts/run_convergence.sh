#!/usr/bin/env bash
# Does the segmentation-bias advantage survive longer training?
#
# At 1000 steps A5 beat vanilla by 0.92 nats, and the gap WIDENED from 0.83 at
# step 500. A pure convergence-speed effect should narrow as the baseline catches
# up. Train 4x longer with frequent evals and read the trajectory.
#
# A1 = vanilla, A5 = segment-index bias (the strongest arm, and the one with no
# provenance semantics -- so this measures the mechanism, not the thesis).
set -u
cd "$(dirname "$0")/.."
OUT=/Users/rickygole/mimic_local_only/convergence.log
: > "$OUT"
for spec in "A1:" "A4:--segment-index-roles"; do
  arm="${spec%%:*}"; flag="${spec#*:}"
  echo "=== arm=$arm ${flag:-vanilla} steps=4000 ===" >> "$OUT"
  python3 scripts/pretrain.py --arm "$arm" --size 10M --steps 4000 --batch 8 \
    --seq-len 256 --seg-tokens 36 --seed 0 --eval-every 250 --eval-batches 30 \
    --shuffle-segments $flag >> "$OUT" 2>&1
done
echo "convergence complete" >> "$OUT"
