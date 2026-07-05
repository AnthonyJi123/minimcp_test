# MiniCPM-o 4.5 — Full-Duplex Capability Probes

Model: `openbmb/MiniCPM-o-4_5` (9B) on a single H100 80GB, bf16, sdpa.
Each probe drives the duplex loop at 1 Hz, injects a controlled event at a
known chunk **T**, and derives a metric from the model's per-second
`is_listen` / `text` / `end_of_turn` / `kv_cache_length` stream.
These cover the audio+vision+speech full-duplex behaviors that the paper's
only quantitative full-duplex benchmark (LiveSports-3K-CC, vision-only) omits.

| # | Capability | Headline result |
|---|-----------|-----------------|
| 1 | Audio-Visual Full Duplex (act on speech while speaking) | comply latency=2.5 chunk(s); causal rate=0.0 |
| 2 | User Interruption (stop after 停) | sustained-stop after 停: latency=3.5 chunk(s), ack rate=0.4, causal rate=0.8 |
| 3 | Visual Interruption (revise in-progress speech on scene change) | speaking@T rate=1.0; revise latency=3.0 chunk(s); causal rate=1.0 |
| 4 | Cross-modal Conflict (trust vision vs audio) | trusts vision both directions rate=1.0 (main=vision, mirror=vision) |
| 5 | Backchannel (listen vs. barge in during monologue) | barges_in (interrupted_user rate=1.0, seg_chars=33.4) |
| 6 | Proactive Timing (silent when idle, alert on salient event) | idle-silent rate=1.0, proactive latency=0.5 chunk(s); causal rate=0.8 |
| 7 | Online Correction (revise prior claim on new evidence) | narrate-new-count latency=2.6 chunk(s); explicit-correction rate=0.0; causal rate=1.0 |
| 8 | Long-Horizon Streaming (stability over time) | 300/300 chunks, crashed rate=False, listen-drift=167.1ms, speak-max=1659.2ms, RT-violations=72, kv/chunk=84.9, vram+4290.8MB |
| 9 | Memory Budget (retain key fact across distance) | recall@q1 rate=0.4, recall@q2 rate=0.4; eff dist q2=14.0 chunk(s) |

## Per-probe detail

### p1_av_fullduplex — Audio-Visual Full Duplex (act on speech while speaking)
```json
{
  "_runs": 5,
  "precondition_met_speaking_cn_before_T_rate": 1.0,
  "inject_chunk_T": {
    "mean": 8.0,
    "min": 8,
    "max": 8,
    "n": 5,
    "null_runs": 0
  },
  "switched_to_english_chunk": {
    "mean": 10.5,
    "min": 10,
    "max": 12,
    "n": 4,
    "null_runs": 1
  },
  "comply_latency_chunks": {
    "mean": 2.5,
    "min": 2,
    "max": 4,
    "n": 4,
    "null_runs": 1
  },
  "control_switched_spontaneously_rate": 1.0,
  "valid_causal_rate": 0.0,
  "transcript": "画面 当中是一个黄 色的正方形，它 被放置 在了深蓝 色的 The video shows a yellow square in the center of the frame, placed against a dark blue background. The square is solid and uniform in color, creating a strong contrast with the background. There are no",
  "_probe": "p1_av_fullduplex",
  "_capability": "Audio-Visual Full Duplex (act on speech while speaking)",
  "_wall_s": 62.3
}
```
Full chunk-level trace: `results/p1_av_fullduplex.jsonl`

### p2_user_interruption — User Interruption (stop after 停)
```json
{
  "_runs": 5,
  "precondition_speaking_before_stop_rate": 1.0,
  "inject_chunk_T": {
    "mean": 9.0,
    "min": 9,
    "max": 9,
    "n": 5,
    "null_runs": 0
  },
  "sustained_stop_chunk": {
    "mean": 12.5,
    "min": 11,
    "max": 14,
    "n": 4,
    "null_runs": 1
  },
  "stop_latency_chunks": {
    "mean": 3.5,
    "min": 2,
    "max": 5,
    "n": 4,
    "null_runs": 1
  },
  "acknowledged_stop_rate": 0.4,
  "control_natural_stop_latency_chunks": null,
  "stop_is_causal_rate": 0.8,
  "transcript_around_stop": "白色的圆 形。",
  "_probe": "p2_user_interruption",
  "_capability": "User Interruption (stop after 停)",
  "_wall_s": 47.4
}
```
Full chunk-level trace: `results/p2_user_interruption.jsonl`

### p3_visual_interruption — Visual Interruption (revise in-progress speech on scene change)
```json
{
  "_runs": 5,
  "precondition_speaking_at_T_rate": 1.0,
  "swap_chunk_T": {
    "mean": 6.0,
    "min": 6,
    "max": 6,
    "n": 5,
    "null_runs": 0
  },
  "described_B_after_swap_chunk": {
    "mean": 9.0,
    "min": 8,
    "max": 10,
    "n": 5,
    "null_runs": 0
  },
  "revise_latency_chunks": {
    "mean": 3.0,
    "min": 2,
    "max": 4,
    "n": 5,
    "null_runs": 0
  },
  "still_described_A_after_swap": {
    "mean": 6.5,
    "min": 6,
    "max": 7,
    "n": 4,
    "null_runs": 1
  },
  "control_mentioned_B_without_swap_rate": 0.0,
  "valid_causal_rate": 1.0,
  "transcript_before": "现在 看到的是一个黑 色背景的画",
  "transcript_after": "面，中间有一个 红色的圆。接 着画面变 成了深绿色背 景，中央显 示着一个亮绿 色的三角形。这 个三角形是 等边三角形，它 的颜色明 显比背景要 鲜艳一些。",
  "_probe": "p3_visual_interruption",
  "_capability": "Visual Interruption (revise in-progress speech on scene change)",
  "_wall_s": 58.6
}
```
Full chunk-level trace: `results/p3_visual_interruption.jsonl`

### p4_crossmodal_conflict — Cross-modal Conflict (trust vision vs audio)
```json
{
  "_runs": 5,
  "main_vision_red_audio_green": "vision",
  "mirror_vision_green_audio_red": "vision",
  "trusts_vision_both_directions_rate": 1.0,
  "transcript_main": "现在画 面是红色 的。",
  "transcript_mirror": "现在画 面是绿色的。",
  "_probe": "p4_crossmodal_conflict",
  "_capability": "Cross-modal Conflict (trust vision vs audio)",
  "_wall_s": 16.8
}
```
Full chunk-level trace: `results/p4_crossmodal_conflict.jsonl`

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
    "mean": 19.4,
    "min": 19,
    "max": 20,
    "n": 5,
    "null_runs": 0
  },
  "interrupted_user_rate": 1.0,
  "speaking_segment_chars": {
    "mean": 33.4,
    "min": 27,
    "max": 38,
    "n": 5,
    "null_runs": 0
  },
  "is_short_backchannel_rate": 0.0,
  "verdict": "barges_in",
  "segment_text": "哎呀，你这一连串的事情发生，确实让人心疼。地铁停运、路况拥堵，还",
  "_probe": "p5_backchannel",
  "_capability": "Backchannel (listen vs. barge in during monologue)",
  "_wall_s": 45.2
}
```
Full chunk-level trace: `results/p5_backchannel.jsonl`

### p6_proactive_timing — Proactive Timing (silent when idle, alert on salient event)
```json
{
  "_runs": 5,
  "event_chunk_T": {
    "mean": 10.0,
    "min": 10,
    "max": 10,
    "n": 5,
    "null_runs": 0
  },
  "idle_false_positive_alerts": {
    "mean": 0.0,
    "min": 0,
    "max": 0,
    "n": 5,
    "null_runs": 0
  },
  "stayed_silent_when_idle_rate": 1.0,
  "proactive_alert_chunk": {
    "mean": 10.5,
    "min": 10,
    "max": 12,
    "n": 4,
    "null_runs": 1
  },
  "proactive_latency_chunks": {
    "mean": 0.5,
    "min": 0,
    "max": 2,
    "n": 4,
    "null_runs": 1
  },
  "control_alerted_without_event_rate": 0.0,
  "proactive_is_causal_rate": 0.8,
  "transcript_after_event": "注意！画 面中出现了火 焰警告标志。",
  "_probe": "p6_proactive_timing",
  "_capability": "Proactive Timing (silent when idle, alert on salient event)",
  "_wall_s": 39.5
}
```
Full chunk-level trace: `results/p6_proactive_timing.jsonl`

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
  "stated_2_before_change_rate": 1.0,
  "narrated_new_count_5_chunk": {
    "mean": 9.6,
    "min": 8,
    "max": 10,
    "n": 5,
    "null_runs": 0
  },
  "narration_latency_chunks": {
    "mean": 2.6,
    "min": 1,
    "max": 3,
    "n": 5,
    "null_runs": 0
  },
  "explicitly_corrected_rate": 0.0,
  "control_said_5_without_change_rate": 0.0,
  "revision_is_causal_rate": 1.0,
  "transcript": "画面里有两 个黄色的圆 点。 现在画 面中有五 个圆点。",
  "_probe": "p7_online_correction",
  "_capability": "Online Correction (revise prior claim on new evidence)",
  "_wall_s": 31.2
}
```
Full chunk-level trace: `results/p7_online_correction.jsonl`

### p8_long_horizon — Long-Horizon Streaming (stability over time)
```json
{
  "intended_chunks": 300,
  "completed_chunks": 300,
  "crashed": false,
  "speaking_chunks": 155,
  "listen_latency_head_mean_ms": 201.6,
  "listen_latency_tail_mean_ms": 368.7,
  "listen_latency_drift_ms": 167.1,
  "speak_latency_mean_ms": 942.3,
  "speak_latency_max_ms": 1659.2,
  "real_time_violation_chunks": 72,
  "kv_start": 169,
  "kv_end": 25557,
  "kv_tokens_per_chunk": 84.9,
  "kv_bounded": null,
  "vram_mb_start": 23892.0,
  "vram_mb_end": 28182.8,
  "vram_growth_mb": 4290.8,
  "_probe": "p8_long_horizon",
  "_capability": "Long-Horizon Streaming (stability over time)",
  "_n_chunks": 300,
  "_wall_s": 183.9
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
  "recall_at_q1_rate": 0.4,
  "kv_at_q1": {
    "mean": 1404.2,
    "min": 1395,
    "max": 1412,
    "n": 5,
    "null_runs": 0
  },
  "effective_distance_q1": {
    "mean": 9.0,
    "min": 7,
    "max": 15,
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
  "recall_at_q2_rate": 0.4,
  "kv_at_q2": {
    "mean": 2805.8,
    "min": 2786,
    "max": 2820,
    "n": 5,
    "null_runs": 0
  },
  "effective_distance_q2": {
    "mean": 14.0,
    "min": 1,
    "max": 24,
    "n": 5,
    "null_runs": 0
  },
  "model_restated_passphrase_rate": 0.8,
  "answer_q1": "肉面，下 午 还要开个会。 请问 今天的暗 号是什么？蓝 色",
  "answer_q2": "的暗 号是什么？ 再 确认 一次，暗 号 是什么？",
  "caveat": "no fixed KV budget reached; effective distance reflects model self-restatement",
  "_probe": "p9_memory_budget",
  "_capability": "Memory Budget (retain key fact across distance)",
  "_wall_s": 91.6
}
```
Full chunk-level trace: `results/p9_memory_budget.jsonl`

