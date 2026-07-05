# MiniCPM-o 4.5 — Full-Duplex Capability Eval

Deploy **MiniCPM-o 4.5** on an H100 and probe the **audio + vision + speech**
full-duplex behaviors that its paper claims but does **not** quantify. The only
quantitative full-duplex experiment in the paper is **LiveSports-3K-CC**
(vision-only, audio-free, win-rate 54.4 vs GPT-4o); this repo probes the rest:
interruption, visual interruption, cross-modal conflict, backchannel, proactive
timing, online correction, long-horizon stability, and memory.

## Repo layout

```
harness/            the eval harness (see harness/README.md for the design)
  session.py        loads the model once; drives the 1 Hz duplex loop; records JSONL
  timeline.py       Scenario = ordered 1-s chunks (audio, frame, label, event)
  stimuli.py        synthetic frames + edge-tts utterances, 16 kHz / 1-s chunks
  judge.py          rule-based scoring (negation/language-aware) + optional LLM judge
  probe_base.py     Probe contract + latency/precondition helpers
  probes/p1..p9.py  one probe per under-evaluated capability (each with controls)
  run_eval.py       runs probes (+ control arms) -> results/*.jsonl + report.md
  show_trace.py     pretty-print a probe's chunk trace
results/            traces, per-probe metrics, report.md  (committed sample run)
VALIDITY.md         adversarial validity review of the suite + what each result shows
```

## Deployment (H100 / RunPod)

The model runs on the pod; harness source lives here and is `scp`'d over.

```bash
# 1. weights (~19 GB bf16, fits H100 80GB; put on a persistent volume)
hf download openbmb/MiniCPM-o-4_5 --local-dir /workspace/models/MiniCPM-o-4_5

# 2. env (venv with --system-site-packages to reuse the image's torch)
python -m venv --system-site-packages /workspace/venv && . /workspace/venv/bin/activate
pip install "minicpmo-utils[all]" transformers==4.51.0 accelerate==1.12.0 \
            "setuptools<81" "pydantic>=2.11" PyYAML edge-tts opencv-python-headless
git clone https://github.com/OpenBMB/MiniCPM-o-Demo   # provides the duplex API

# 3. run
cd harness && python run_eval.py            # all probes + controls
```

**Gotchas (the non-obvious ones):**
- **`transformers==4.51.0` is required** — 4.52+ breaks MiniCPM's custom `Resampler`
  weight-init (`'Resampler' object has no attribute '_initialize_weights'`).
- `minicpmo-utils` pins **librosa 0.9.0**, which imports `pkg_resources` → keep
  **`setuptools<81`**.
- Frames passed to the duplex API must be **PIL Images**, not numpy (else ~18 GB blowup).
- The duplex loop is `prefill(audio, frame_list) → generate() → finalize()`;
  `finalize()` must run before the next `prefill()`.

## What the harness exploits

The duplex model makes a **per-second speak-vs-listen decision** (`is_listen`),
and each `generate()` returns `text`, `end_of_turn`, `kv_cache_length`, and
per-stage latencies. That turns interruption / backchannel / proactivity into
**measurable latencies**, and `kv_cache_length` drives the memory/long-horizon
probes. See `harness/README.md` for full detail and `VALIDITY.md` for an honest
assessment of which results are causally established vs. suggestive.
