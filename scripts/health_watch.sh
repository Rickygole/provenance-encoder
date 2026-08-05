#!/usr/bin/env bash
# Emits a line ONLY when the machine enters a state that has already cost this
# project ~2.5 hours: memory contention driving swap, which slowed training 6x
# (14.6 min/config -> 167.5 min/config). Diagnosed as swap thrash, not thermal:
# `pmset -g therm` recorded no thermal warnings throughout.
#
# Thresholds: swap >5200MB or free memory <20% or >1 GPU job running.
set -u
while true; do
  swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1)
  freepct=$(memory_pressure 2>/dev/null | sed -nE 's/.*free percentage: ([0-9]+)%.*/\1/p' | tail -1)
  jobs=$(pgrep -fc "run_matched.py|pretrain.py" 2>/dev/null || echo 0)
  ts=$(date +%H:%M)
  [ "${swap:-0}" -gt 5200 ] 2>/dev/null && \
    echo "$ts SWAP HIGH ${swap}MB -- training will slow ~6x; stop ollama / avoid concurrent jobs"
  [ -n "${freepct:-}" ] && [ "$freepct" -lt 20 ] 2>/dev/null && \
    echo "$ts MEMORY LOW ${freepct}% free -- swap thrash imminent"
  [ "${jobs:-0}" -gt 1 ] 2>/dev/null && \
    echo "$ts CONCURRENCY ${jobs} GPU jobs -- measured to cause stalls on 16GB"
  sleep 120
done
