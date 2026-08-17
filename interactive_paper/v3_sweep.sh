#!/bin/bash
# v3 live re-run (8z follow-up, 2026-08-16): 7 pools x 3 escalating tiers,
# sequential so worst-case expert concurrency stays at 3 (probation cap).
# Never arms are probe-independent and reused at report time.
cd /f/Claude_Proj/minimcp_test/interactive_paper || exit 1
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=v3_sweep.log
: > "$LOG"
for B in frozen striviaqa swebq sdqa sllama sreason valpaca; do
  for T in conservative balanced aggressive; do
    echo "=== $(date '+%H:%M:%S') $B $T ===" >> "$LOG"
    modal run modal_bench.py::run_live --bench "$B" --tier "$T" \
      --art-path "/data/gate_v3_${B}.json" --suffix _v3 >> "$LOG" 2>&1 \
      || echo "!!! FAILED $B $T" >> "$LOG"
  done
done
echo "=== SWEEP DONE $(date) ===" >> "$LOG"
grep -E "tier .* complete|FAILED" "$LOG"
