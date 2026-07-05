# MiniCPM-o 4.5 — Full-Duplex Capability Probes

Model: `openbmb/MiniCPM-o-4_5` (9B) on a single H100 80GB, bf16, sdpa.
Each probe drives the duplex loop at 1 Hz, injects a controlled event at a
known chunk **T**, and derives a metric from the model's per-second
`is_listen` / `text` / `end_of_turn` / `kv_cache_length` stream.
These cover the audio+vision+speech full-duplex behaviors that the paper's
only quantitative full-duplex benchmark (LiveSports-3K-CC, vision-only) omits.

| # | Capability | Headline result |
|---|-----------|-----------------|
| 1 | Backchannel (listen vs. barge in during monologue) | barges_in (interrupted_user rate=1.0, seg_chars=33.4) |
| 2 | Online Correction (revise prior claim on new evidence) | narrate-new-count latency=3.0 chunk(s); explicit-correction rate=0.0; causal rate=0.6 |
| 3 | Long-Horizon Streaming (stability over time) | 120.0/120.0 chunks, crashed rate=0.0, listen-drift=33.6ms, speak-max=808.3ms, RT-violations=0.0, kv/chunk=80.3, vram+1404.4MB |
| 4 | Memory Budget (retain key fact across distance) | recall@q1 rate=0.2, recall@q2 rate=0.2; eff dist q2=19.0 chunk(s) |

## Per-probe detail

### p5_backchannel — Backchannel (listen vs. barge in during monologue)
```json
{
  "_runs": 5,
  "last_user_audio_chunk": {
    "mean": 20.0,
    "min": 20,
    "max": 20,
    "n": 5,
    "null_runs": 0
  },
  "first_speaking_onset_chunk": {
    "mean": 19.2,
    "min": 19,
    "max": 20,
    "n": 5,
    "null_runs": 0
  },
  "interrupted_user_rate": 1.0,
  "speaking_segment_chars": {
    "mean": 33.4,
    "min": 31,
    "max": 35,
    "n": 5,
    "null_runs": 0
  },
  "is_short_backchannel_rate": 0.0,
  "verdict": "barges_in",
  "segment_text": "嗯听上去你今天早上遇到的事情确实是挺多的，一连串的事情发生确实会让人",
  "_probe": "p5_backchannel",
  "_capability": "Backchannel (listen vs. barge in during monologue)",
  "_wall_s": 49.6
}
```
Full chunk-level trace: `results/p5_backchannel.jsonl`

### p7_online_correction — Online Correction (revise prior claim on new evidence)
```json
{
  "_runs": 5,
  "change_chunk_T": {
    "mean": 7.0,
    "min": 7,
    "max": 7,
    "n": 5,
    "null_runs": 0
  },
  "stated_2_before_change_rate": 0.8,
  "narrated_new_count_5_chunk": {
    "mean": 10.0,
    "min": 8,
    "max": 12,
    "n": 3,
    "null_runs": 2
  },
  "narration_latency_chunks": {
    "mean": 3.0,
    "min": 1,
    "max": 5,
    "n": 3,
    "null_runs": 2
  },
  "explicitly_corrected_rate": 0.0,
  "control_said_5_without_change_rate": 0.0,
  "revision_is_causal_rate": 0.6,
  "transcript": "画面里有 两个圆点。 现在有 五个圆点。",
  "_probe": "p7_online_correction",
  "_capability": "Online Correction (revise prior claim on new evidence)",
  "_wall_s": 34.7
}
```
Full chunk-level trace: `results/p7_online_correction.jsonl`

### p8_long_horizon — Long-Horizon Streaming (stability over time)
```json
{
  "_runs": 5,
  "intended_chunks": {
    "mean": 120.0,
    "min": 120,
    "max": 120,
    "n": 5,
    "null_runs": 0
  },
  "completed_chunks": {
    "mean": 120.0,
    "min": 120,
    "max": 120,
    "n": 5,
    "null_runs": 0
  },
  "crashed_rate": 0.0,
  "speaking_chunks": {
    "mean": 36.4,
    "min": 33,
    "max": 40,
    "n": 5,
    "null_runs": 0
  },
  "listen_latency_head_mean_ms": {
    "mean": 196.7,
    "min": 196.1,
    "max": 197.4,
    "n": 5,
    "null_runs": 0
  },
  "listen_latency_tail_mean_ms": {
    "mean": 230.4,
    "min": 230.0,
    "max": 230.7,
    "n": 5,
    "null_runs": 0
  },
  "listen_latency_drift_ms": {
    "mean": 33.6,
    "min": 33.0,
    "max": 34.5,
    "n": 5,
    "null_runs": 0
  },
  "speak_latency_mean_ms": {
    "mean": 690.4,
    "min": 675.7,
    "max": 705.3,
    "n": 5,
    "null_runs": 0
  },
  "speak_latency_max_ms": {
    "mean": 808.3,
    "min": 773.5,
    "max": 830.6,
    "n": 5,
    "null_runs": 0
  },
  "real_time_violation_chunks": {
    "mean": 0.0,
    "min": 0,
    "max": 0,
    "n": 5,
    "null_runs": 0
  },
  "kv_start": {
    "mean": 169.0,
    "min": 169,
    "max": 169,
    "n": 5,
    "null_runs": 0
  },
  "kv_end": {
    "mean": 9719.4,
    "min": 9698,
    "max": 9736,
    "n": 5,
    "null_runs": 0
  },
  "kv_tokens_per_chunk": {
    "mean": 80.3,
    "min": 80.1,
    "max": 80.4,
    "n": 5,
    "null_runs": 0
  },
  "kv_bounded": null,
  "vram_mb_start": {
    "mean": 23933.8,
    "min": 23882.2,
    "max": 24009.9,
    "n": 5,
    "null_runs": 0
  },
  "vram_mb_end": {
    "mean": 25338.2,
    "min": 25300.0,
    "max": 25382.5,
    "n": 5,
    "null_runs": 0
  },
  "vram_growth_mb": {
    "mean": 1404.4,
    "min": 1290.1,
    "max": 1435.4,
    "n": 5,
    "null_runs": 0
  },
  "_probe": "p8_long_horizon",
  "_capability": "Long-Horizon Streaming (stability over time)",
  "_wall_s": 214.9
}
```
Full chunk-level trace: `results/p8_long_horizon.jsonl`

### p9_memory_budget — Memory Budget (retain key fact across distance)
```json
{
  "_runs": 5,
  "query1_chunk": {
    "mean": 15.0,
    "min": 15,
    "max": 15,
    "n": 5,
    "null_runs": 0
  },
  "recall_at_q1_rate": 0.2,
  "kv_at_q1": {
    "mean": 1402.6,
    "min": 1388,
    "max": 1415,
    "n": 5,
    "null_runs": 0
  },
  "effective_distance_q1": {
    "mean": 6.4,
    "min": 2,
    "max": 8,
    "n": 5,
    "null_runs": 0
  },
  "query2_chunk": {
    "mean": 32.0,
    "min": 32,
    "max": 32,
    "n": 5,
    "null_runs": 0
  },
  "recall_at_q2_rate": 0.2,
  "kv_at_q2": {
    "mean": 2804.0,
    "min": 2788,
    "max": 2826,
    "n": 5,
    "null_runs": 0
  },
  "effective_distance_q2": {
    "mean": 19.0,
    "min": 9,
    "max": 25,
    "n": 5,
    "null_runs": 0
  },
  "model_restated_passphrase_rate": 1.0,
  "answer_q1": "会。  今天的 暗号是蓝 色河马 7392。 ",
  "answer_q2": "马739 2。再确 认一次，暗 号是什么？ 嗯，没 问题。 今天",
  "caveat": "no fixed KV budget reached; effective distance reflects model self-restatement",
  "_probe": "p9_memory_budget",
  "_capability": "Memory Budget (retain key fact across distance)",
  "_wall_s": 119.9
}
```
Full chunk-level trace: `results/p9_memory_budget.jsonl`

