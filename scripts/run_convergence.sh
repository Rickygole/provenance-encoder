#!/usr/bin/env bash
# Does the A4 advantage survive training to convergence, or is it convergence SPEED?
#
# At 1000 steps A4 beat A1 by ~0.84 nats. A fixed-step comparison cannot tell a
# real asymptotic advantage from a head start -- which is the single most likely
# way this project produces another artifact. Train 4x longer and see whether the
# gap narrows, holds, or widens.
set -u
cd "$(dirname "$0")/.."

STEPS=${STEPS:-4000}
OUT=${OUT:-/Users/rickygole/mimic_local_only/convergence.log}
: > "$OUT"

for arm in A1 A4 A2 A3; do
  echo "=== arm=$arm steps=$STEPS ===" >> "$OUT"
  python3 scripts/pretrain.py --arm "$arm" --size 10M --steps "$STEPS" \
    --batch 8 --seq-len 256 --seg-tokens 36 --seed 0 \
    --eval-every 500 --eval-batches 30 >> "$OUT" 2>&1
done
echo "convergence sweep complete" >> "$OUT"
