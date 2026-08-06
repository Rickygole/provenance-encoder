#!/bin/bash
# Defect-free control: run all seeds (fresh process each) + baselines +
# final aggregation/DeLong report. Run from this directory.
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=(42 123 2024)
EPOCHS=3

for SEED in "${SEEDS[@]}"; do
    echo "=== seed $SEED ==="
    python3 train.py --seed "$SEED" --epochs "$EPOCHS"
done

echo "=== baselines ==="
python3 baselines.py

echo "=== final report ==="
python3 report.py

echo "=== done ==="
