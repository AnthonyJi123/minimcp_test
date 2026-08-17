#!/bin/bash
# v3 report chain: judge heard answers per pool (never arms reused,
# probe-independent), then OAB official-judge rejudge for the three
# OpenAudioBench pools. Sequential; judge concurrency bounded per run.
cd /f/Claude_Proj/minimcp_test/interactive_paper || exit 1
export PYTHONUTF8=1 MSYS_NO_PATHCONV=1
LOG=v3_report.log
: > "$LOG"
run() { echo "=== $(date '+%H:%M:%S') $* ===" >> "$LOG"
        modal run "$@" >> "$LOG" 2>&1 || echo "!!! FAILED $*" >> "$LOG"; }

run modal_bench.py::report --bench frozen --suffix _v3 \
    --never-glob "/data/gated_traces_v2.jsonl.never.shard*"
run modal_bench.py::report --bench striviaqa --suffix _v3 \
    --never-glob "/data/striviaqa_traces.jsonl.never.shard*"
run modal_bench.py::report --bench swebq --suffix _v3 \
    --never-glob "/data/swebq_traces.jsonl.never.shard*"
run modal_bench.py::report --bench sdqa --suffix _v3 \
    --never-glob "/data/sdqa_traces.jsonl.never.shard*"
run modal_bench.py::report --bench sllama --suffix _v3 \
    --never-glob "/data/sllama_v2_traces.jsonl.never.shard*"
run modal_bench.py::report --bench sreason --suffix _v3 \
    --never-glob "/data/sreason_v2_traces.jsonl.never.shard*"
run modal_bench.py::valpaca_report --suffix _v3 \
    --never-glob "/data/valpaca_v2_traces.jsonl.never.shard*"
run modal_bench.py::oab_rejudge_live --bench striviaqa --suffix _v3
run modal_bench.py::oab_rejudge_live --bench swebq --suffix _v3
run modal_bench.py::oab_rejudge_live --bench sllama --suffix _v3
echo "=== REPORTS DONE $(date) ===" >> "$LOG"
grep -E "heard accuracy|official|score \(1-5\)|\[.*\] n=|FAILED" "$LOG"
