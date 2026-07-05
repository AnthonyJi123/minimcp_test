# MiniCPM-o 4.5 — Full-Duplex Capability Eval Harness

A probe suite that measures the **audio + vision + speech** full-duplex behaviors
that MiniCPM-o 4.5 claims but the paper does **not** quantify. The only quantitative
full-duplex experiment in the paper is **LiveSports-3K-CC** (vision-only, audio-free,
win-rate 54.4 vs GPT-4o). This harness probes the rest.

## Why this works

The duplex model runs a **~1 Hz decision loop**. Each step:

```
duplex.prefill(audio_waveform=<1s 16kHz mono>, frame_list=[PIL.Image])
result = duplex.generate()   # -> is_listen, text, audio_data, end_of_turn, cost_*
duplex.finalize()            # REQUIRED before the next prefill
```

`is_listen` is the model's per-second **speak-vs-stay-silent** decision — directly
observable. That turns "interruption", "backchannel", and "proactivity" into
**measurable latencies**, not subjective impressions. We also read
`UnifiedProcessor.kv_cache_length` per chunk (for the memory-budget probe) and
`torch.cuda.memory_allocated` (for long-horizon stability).

## Layout

| File | Role |
|------|------|
| `session.py` | Loads the model once (`UnifiedProcessor`→`DuplexView`); drives the per-chunk loop; records a JSONL trace row per chunk. |
| `timeline.py` | `Scenario` = ordered 1-s chunks `(audio, frame, label, event)`; named events mark the reference point **T**. |
| `stimuli.py` | Reproducible stimuli: synthetic frames (`card`, `dots`, scene-swaps) + edge-tts user utterances sliced to 1-s 16 kHz chunks. |
| `judge.py` | Rule-based keyword scoring (offline); optional Anthropic LLM judge if `ANTHROPIC_API_KEY` is set. |
| `probe_base.py` | `Probe` contract + latency helpers. |
| `probes/p1..p9.py` | One probe per under-evaluated capability. |
| `run_eval.py` | Loads the model once, runs probes, writes `results/<probe>.jsonl` + `report.md`. |

## The 9 probes

| # | Probe | Injected event @ T | Metric |
|---|-------|--------------------|--------|
| 1 | `p1_av_fullduplex` | spoken "switch to English" while speaking | latency to comply (proves input-while-speaking) |
| 2 | `p2_user_interruption` | spoken "停" mid-utterance | seconds until `is_listen`/`end_of_turn` |
| 3 | `p3_visual_interruption` | scene A→B swap | chunks until speech reflects B |
| 4 | `p4_crossmodal_conflict` | image RED vs audio "it's green" | follows vision or audio |
| 5 | `p5_backchannel` | long user monologue | speaking ratio during monologue |
| 6 | `p6_proactive_timing` | idle scene → salient alert | idle false-positives + proactive latency |
| 7 | `p7_online_correction` | 2 dots → 5 dots | corrects prior count? latency |
| 8 | `p8_long_horizon` | N continuous chunks | latency drift + VRAM creep + crash |
| 9 | `p9_memory_budget` | passphrase early, queried later | recall@distance + KV length |

## Run

```bash
# on the pod, inside the venv, from harness/
source /workspace/venv/bin/activate
python run_eval.py                 # all 9 probes
python run_eval.py p2 p3           # a subset
LONGHORIZON_CHUNKS=2000 python run_eval.py p8   # ~33 min stability run
python run_eval.py --report-only   # rebuild report.md from saved metrics
```

Outputs land in `/workspace/minicpm-o-eval/results/`:
`<probe>.jsonl` (chunk-level trace), `<probe>.metric.json`, and `report.md`.

## Environment (pod)

- `openbmb/MiniCPM-o-4_5` weights at `/workspace/models/MiniCPM-o-4_5`
- venv at `/workspace/venv` (`--system-site-packages`): **transformers==4.51.0**
  (4.52+ breaks the custom `Resampler` weight-init), `minicpmo-utils[all]`,
  `setuptools<81` (librosa 0.9 needs `pkg_resources`), edge-tts.
- Reference voice: `MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav`.

## Notes / limitations

- Stimuli are deliberately **synthetic and reproducible** (solid colors, shapes,
  dot counts) so metrics are unambiguous. Swapping in real video clips
  (`assets/omni_duplex1.mp4`) is a drop-in `frame_track` change.
- Semantic probes (4/7) use keyword rules by default; set `ANTHROPIC_API_KEY`
  to enable the LLM judge for fuzzier transcripts.
- 1 chunk = 1 second of stream, so "latency in chunks" ≈ "latency in seconds".
