#!/usr/bin/env bash
# Four arms x three seeds, fresh process each (MPS accumulates memory across
# configs in one process, and seed control requires a clean RNG state).
# Identical steps, batch, sequence length and data order across arms.
set -u
cd "$(dirname "$0")/.."

STEPS=${STEPS:-1000}
BATCH=${BATCH:-8}
SEQ=${SEQ:-256}
SEG=${SEG:-36}
SIZE=${SIZE:-10M}

echo "sweep: size=$SIZE steps=$STEPS batch=$BATCH seq=$SEQ"
for seed in 0 1 2; do
  for arm in A1 A2 A3 A4; do
    echo "=== arm=$arm seed=$seed ==="
    python3 scripts/pretrain.py --arm "$arm" --size "$SIZE" --steps "$STEPS" \
      --batch "$BATCH" --seq-len "$SEQ" --seg-tokens "$SEG" --seed "$seed" \
      --eval-every 250 --eval-batches 30 2>&1 | grep -E "FINAL|\[eval\]|moved|Error|error"
  done
done
echo "sweep complete"
