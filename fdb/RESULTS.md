# MiniCPM-o 4.5 on Full-Duplex-Bench v1.0

Run date: 2026-07-04. Model: `openbmb/MiniCPM-o-4_5` (duplex mode, audio-only,
SDPA, transformers 4.51.0) on Modal serverless H100 (6-way sharded).
Benchmark: [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
(arXiv [2503.04721](https://arxiv.org/abs/2503.04721)), all 727 v1.0 samples,
official evaluation scripts + official ASR (`nvidia/parakeet-tdt-0.6b-v2`).

## Results vs paper baselines

Paper numbers from arXiv v2/v3 Table III (cross-checked against both revisions).

### Pause handling — TOR ↓ (don't speak during the user's mid-turn pause)

| Model | Candor | Synthetic |
|---|---|---|
| dGSLM | 0.935 | 0.934 |
| Moshi | 0.980 | 0.985 |
| Freeze-Omni | 0.481 | 0.642 |
| Gemini Live | 0.310 | 0.255 |
| **MiniCPM-o 4.5** | **0.125** | **0.117** |

Best result of any model by a wide margin — it almost never barges into a pause.

### Backchannel (ICC) — TOR ↓, Frequency ↑, JSD ↓

| Model | TOR | Freq | JSD |
|---|---|---|---|
| dGSLM | 0.691 | 0.015 | 0.934 |
| Moshi | 1.000 | 0.001 | 0.957 |
| Freeze-Omni | 0.636 | 0.001 | 0.997 |
| Gemini Live | 0.091 | 0.012 | 0.896 |
| **MiniCPM-o 4.5** | **0.000** | **0.000** | **1.000** |

Never wrongly takes the turn (best TOR), but produces **zero backchannels**
(worst Freq/JSD): during continuous English user speech it stays fully silent.
Spot-checked transcripts confirm genuinely empty outputs, not a pipeline
artifact. Same qualitative finding as our probe harness (no backchannel
ability), though here it under-talks rather than barging in.

### Smooth turn-taking (Candor) — TOR ↑, latency ↓

| Model | TOR | Latency (s) |
|---|---|---|
| dGSLM | 0.975 | 0.352 |
| Moshi | 0.941 | 0.265 |
| Freeze-Omni | 0.336 | 0.953 |
| Gemini Live | 0.655 | 1.301 |
| **MiniCPM-o 4.5** | **1.000** | **1.541** |

Takes the turn on **every** sample (best TOR of any model) but is the slowest
to start speaking. Note the architectural floor: the model runs a 1 Hz
listen/speak loop and our harness plays chunk *i*'s audio from tick *i+1*
(earliest causal playback), so latency is quantized to ~1 s ticks —
sub-second latency is structurally impossible for this loop design.

### User interruption (Synthetic) — TOR ↑, GPT score ↑, latency ↓

| Model | TOR | GPT score | Latency (s) |
|---|---|---|---|
| dGSLM | 0.917 | 0.201 | 2.531 |
| Moshi | 1.000 | 0.765 | 0.257 |
| Freeze-Omni | 0.867 | 3.615 | 1.409 |
| Gemini Live | 0.891 | 3.376 | 1.183 |
| **MiniCPM-o 4.5** | **0.915** | *pending* | **0.904** |

Responds to interruptions competitively (TOR 0.915, latency 0.90 s — second
fastest after Moshi). The GPT relevance rating requires an `OPENAI_API_KEY`
(official judge = `gpt-4-turbo` via the benchmark's `evaluate.py`); run
`modal run modal_fdb_eval.py::run_eval --openai-key sk-... --subsets
synthetic_user_interruption` to fill it in.

## Takeaway

MiniCPM-o 4.5's time-division 1 Hz duplex gives it a **conservative,
turn-respecting profile**: state-of-the-art pause handling, perfect
turn-taking rate, decent interruption handling — at the cost of ~1.5 s
response latency (tick-quantized) and a complete absence of backchanneling.
It is qualitatively the opposite of Moshi (fast, always-talking, poor pause
handling).

## Method notes / caveats

- Inference: `fdb/infer.py` streams each `input.wav` as 1 s / 16 kHz chunks
  through `DuplexView.prefill → generate → finalize` with `frame_list=None`;
  output audio (base64 float32 @ 24 kHz, exactly 1 s per speaking tick) is
  placed on a time-aligned 24 kHz track (drain-before-process), trimmed to the
  input duration — matching the reference (Freeze-Omni) inference scripts and
  the benchmark's example data.
- System prompt + reference voice: the official MiniCPM-o-Demo audio-duplex
  **English Call** preset (`assets/presets/audio_duplex/english_call.yaml`,
  `ref_en_dlc_1.wav`).
- ASR: the repo's current official `get_transcript/asr.py`
  (parakeet-tdt-0.6b-v2). The paper's original runs credited a different ASR
  (code comments reference CrisperWhisper), so decimal-level comparison with
  the paper table carries that grain of salt.
- Single pass (no repeats); the paper also reports single-run numbers.
- Runtime: ~0.15 s wall per 1 s chunk on H100 (RTF ≈ 0.15); full 727-sample
  run ≈ 1 GPU-hour total across 6 workers.

## Reproduce

```bash
modal run modal_fdb.py::download_dataset        # Drive -> fdb-data volume
modal run modal_fdb.py::smoke                   # 2 samples/subset sanity
modal run modal_fdb.py::run_infer --workers 6   # 727 samples on H100s
modal run modal_fdb_eval.py::run_asr            # parakeet -> output.json
modal run modal_fdb_eval.py::run_eval           # official metrics
```

Raw eval logs: `fdb-data` volume under `eval_logs/`; per-sample outputs and
per-chunk traces under `exp/minicpm-o-4_5/{subset}/{id}/`.
