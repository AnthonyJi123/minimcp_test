#!/bin/bash
# Autonomous Phase-2 rerun supervisor (all-public pools), 2026-07-08.
# Waits for run_signals to finish, then label -> calibrate -> pull roc.png.
SIG_OUT="/c/Users/rhe95/AppData/Local/Temp/claude/F--Claude-Proj-minimcp-test/d25f0bd5-5f58-43bb-9098-eca39b312e1d/tasks/b32359yjr.output"
cd /f/Claude_Proj/minimcp_test || exit 1
LOG="interactive_paper/pipeline_watch.log"
say() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }

say "supervisor started; waiting for run_signals (task b32359yjr)"
for i in $(seq 1 90); do
  if grep -q "collected signals for 600 queries" "$SIG_OUT" 2>/dev/null; then
    say "signals complete (600/600)"; break
  fi
  if [ "$i" -eq 90 ]; then say "TIMEOUT waiting for signals after 90min — aborting"; exit 1; fi
  sleep 60
done

say "running label (gpt-5.4-mini judge, 600 calls)..."
PYTHONUTF8=1 modal run interactive_paper/modal_app.py::label >> "$LOG" 2>&1
if ! grep -q ">>> labeled 600" "$LOG"; then
  say "WARNING: label did not report 600 rows — check log above"; fi

say "running calibrate..."
PYTHONUTF8=1 modal run interactive_paper/modal_app.py::calibrate >> "$LOG" 2>&1

say "pulling roc.png..."
PYTHONUTF8=1 modal volume get --force gate-data roc.png interactive_paper/figures/roc.png >> "$LOG" 2>&1

say "=== SUPERVISOR DONE ==="
grep -E "(PHASE-2 VERDICT|BEST signal|escalate_rate|AUC=)" "$LOG" | tail -30 >> "$LOG.summary" 2>/dev/null
say "summary written to $LOG.summary"
