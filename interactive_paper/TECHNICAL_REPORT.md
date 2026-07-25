# When Does a Small Model Know to Hand Off?
## Zero-Training Escalation Gates for Full-Duplex Speech Models

**Technical Report — v3** (v1: 2026-07-13, probe gate; v2: 2026-07-14, audit +
p(True) + 3-backbone replication; v3: 2026-07-24, adds Phases 5c–6d + system
profiling — the duplex-damage mechanism story, audio-input replication, the
thinking ablation, and the fork/overlap latency analysis)
Date: 2026-07-24 · Seed: 42 · Status: Phases 0–6 complete; Phase 7 (step-2
result injection) open

---

## Abstract

This is **step 1 of a larger plan**: teach a small full-duplex conversational
model to know *when to hand a query off* to a large cloud model (step 2 — how
the big model's result comes back into the live session — is future work). We
attach zero-training escalation gates to MiniCPM-o 4.5 (9B omni/duplex, Qwen3
backbone) and evaluate them against small-model failure labels on 600 frozen
public-benchmark queries (SimpleQA, MMLU-Pro, TriviaQA, GSM8K/MATH-500,
dolly/alpaca), with matched-pair controls across 10 backbones/fine-tunes and
both text and speech input. Main findings:

1. **A last-layer hidden-state probe looks good (AUC 0.822) but is mostly a
   query-type shortcut** — pool-identity oracle 0.715; leave-one-pool-out it
   inverts on math (0.372). Verbalized self-eval ("p(True)": read P(Yes) off
   one token of "Would you answer this correctly?") beats it with zero
   training on most backbones, and catches 100%-fail long-tail traps the
   probe misses — *if asked before the model answers*.
2. **Duplex fine-tuning damages self-knowledge readouts, not self-knowledge.**
   Matched pairs (Qwen3-8B → MiniCPM-o 4.5; Qwen2.5-7B → Qwen2.5-Omni /
   MiniCPM-o 2.6) show LOPO probe transfer degrade raw > omni-streaming >
   duplex, label-coverage controlled. A full layer × position sweep localizes
   the damage: mid-layer transfer is near-raw (LOPO math **0.931** at L22 vs
   **0.366** at L35) — the duplex fine-tune **overwrites the late-layer
   last-token readout** (where a streaming head encodes turn control), it
   does not destroy the information.
3. **Each readout breaks in one modality and survives in the other.** The
   late-layer cliff is *text-input-specific* (audio input: L35 = 0.936, no
   cliff; unchanged under the speak-mode template — it tracks input
   modality, not output mode). Conversely, pre-answer verbalized
   introspection *collapses on audio input* (trap p_yes 0.055 → 0.556) on
   both MiniCPM duplex generations while the omni-streaming and
   frozen-backbone controls stay intact — duplex-specific again. Controlled
   arms (filler-audio, audio+text-dup) and an ASR audit pin the mechanism:
   the verbal self-check runs over *text-token pathways*; audio embeddings
   don't feed it, though a probe reads the same information at 0.93+.
4. **The modality-shared mid-layer core is created by end-to-end training.**
   Text↔audio probe transfer is ~0.86 on every end-to-end model, ≈ dead
   (0.52–0.60) on Freeze-Omni's frozen backbone with identical text weights.
   Four quadrants: e2e training builds the shared core; duplex training
   damages the readouts; omni-streaming gets both right; a frozen adapter
   gets neither.
5. **The gate wins as a system.** Mid-layer probe = best pre-decode signal
   (test area +0.064 vs final-layer +0.054); it decides in **20 ms (text) /
   45 ms (audio) — before the first output token**. Against MiniCPM-o 4.5's
   own `enable_thinking`, gated-cloud escalation dominates on BOTH axes at
   every rate (e.g. @33%: acc .787 vs .637 at 3.4× lower latency). Cloud
   round-trip overlaps the talker's floor time: the expert result is ready
   before the local answer ends for 40–58% of the test mix, though only
   20–31% of *gate-escalated* queries (they skew short-answer) — stall
   phrases of P50 ~2–3 s bridge the rest.

**Step-1 answer:** zero-training self-knowledge exists, is strong, and is
*modality- and layer-fragile in systematic, duplex-induced ways*. A deployable
gate reads a mid-layer (~60% depth) at prefill — the one signal that survives
every condition tested — with verbalized p(True) as the zero-plumbing
alternative on text input and post-draft check where a draft is affordable.

---

## 1. Setup

| Component | Value |
|---|---|
| Target model | MiniCPM-o 4.5 (9B, Qwen3-8B backbone, omni + full-duplex FT), bf16, SDPA |
| Matched-pair controls | Qwen3-8B (raw); Qwen2.5-7B (raw) vs Qwen2.5-Omni-7B (streaming omni) vs MiniCPM-o 2.6 (duplex); Qwen2-7B (raw) vs Qwen2-Audio-7B (audio FT) vs Freeze-Omni (frozen backbone + adapter); Mistral-7B-v0.3, GLM4-9B-chat (raw diversity) |
| Infra | Modal, H100; torch 2.8.0, transformers 4.51.0 pinned (4.57.6 for omni; 4.45.2 for Freeze-Omni) |
| Judge | `gpt-5.4-mini` (blind, structured outputs); validated by gpt-5.5 re-judge: agreement .962 text / .945 audio, headline deltas ≤ .010 |
| Escalation target | `gpt-5.5` |
| Audio arm | frozen pool TTS-rendered (OpenAI tts-1, voice alloy; content unchanged — public benchmarks only, only modality synthetic, per Spoken-SQuAD/VoiceBench practice) |
| Determinism | greedy, seed 42; 600 queries frozen, 360 calib / 240 test |

**Query pool (600, public benchmarks only):** trap = SimpleQA 50 (MiniCPM
fails 100%); hard-knowledge = MMLU-Pro 150; easy-fact = TriviaQA 100;
easy-chat = dolly-15k + alpaca-zh 150; hard-math = GSM8K tail + MATH-500 150.
Text fail rates: trap 1.00 / knowledge .48 / fact .34 / chat .21 / math .19
(overall .358). Audio adds a modality tax of +5 to +15 pts per pool.

**Candidate gate signals (all zero-training):** decode-scalar (entropy /
margin), linear probe on hidden states (position × layer swept), verbalized
self-eval p(True) = P(Yes) off the first token of a pre-answer ("Would you
answer this correctly?") or post-draft ("Is this answer correct?") prompt.

---

## 2. RQ1 on text input: which signal knows the model will fail?

### 2.1 Headline AUCs (MiniCPM-o 4.5, calib, probe = 5-fold OOF)

| signal | AUC | training needed |
|---|---:|---|
| **ptrue_post** | **0.899** | none |
| probe on h_prompt (final layer) | 0.822 [0.777, 0.863] | LR on calib |
| **ptrue_pre** | 0.807 | none |
| max_entropy@4 (best scalar) | 0.696 | none |

### 2.2 The probe's number is a type shortcut (audit)

h_prompt classifies the source pool at 95.8%; a pool-identity oracle alone
reaches AUC 0.715; appending true pool labels to the probe adds nothing
(0.822 → 0.821). LOPO: math **inverts** (0.372 — the probe scores terse
MATH-500 "safer" than verbose GSM8K: style familiarity), trap mean score
0.232 on 100%-fail questions. Entropy ≈ chance on 2 of 3 raw backbones.

### 2.3 p(True): ask before it answers

Pre-answer trap introspection is 3-for-3 on raw/duplex text: on 100%-fail
SimpleQA the model says "No, I won't get this right" (mean escalation score
0.945) — asked *after* answering it endorses its own confident-wrong answer
(0.945 → 0.360). ptrue_post beats the trained probe on 4 of 5 backbones
tested (exception GLM4: 0.685 < probe 0.798 — "p(True) always wins" is
softened to "usually; depends on self-eval calibration quality").

---

## 3. The duplex effect: overwritten readouts, intact knowledge

### 3.1 Matched pairs — damage tracks duplex-ness, not audio (5c)

LOPO transfer of the standard (final-layer, last-token) probe, one backbone
family (Qwen2.5-7B), label-coverage matched (`lopo_matched2`):

| LOPO | qwen2.5-7b raw | qwen2.5-omni (streaming) | minicpm-o26 (duplex) | [o4.5, Qwen3 base] |
|---|---:|---:|---:|---:|
| hard-math | **0.809** | 0.745 | **0.538 ≈ chance** | 0.372 inverts |
| hard-knowledge | 0.680 | 0.613 | **0.526 ≈ chance** | 0.61 |

Raw matched to o2.6's exact fail rates keeps math at 0.825 → label coverage
ruled out on a second backbone; the **representation** changed. Gradient:
raw > omni-streaming > duplex. (Qwen2-Audio pair: capability collapse
confound, documented negative; raw baselines vary — the defensible statistic
is the within-pair Δ.)

### 3.2 Layer × position sweep — the cliff (5d)

Prefill-only capture of every decoder layer × {last-token, mean-pool}.
Final-layer numbers reproduce the generation-time hooks 5-for-5.

| model | best mid-layer (LOPO math, last-tok) | final layer | shape |
|---|---:|---:|---|
| qwen3-8b raw | 0.964 (L33/36) | 0.958 | plateau, no cliff |
| **minicpm-o45 duplex** | **0.931 (L22/36)** | **0.366** | cliff L31→L34 (.757→.357), inverts |
| qwen2.5-7b raw | 0.893 (L21/28) | 0.809 | mild dip |
| qwen2.5-omni streaming | 0.794 (L16/28) | 0.746 | diffuse mild depression, no cliff |
| **minicpm-o26 duplex** | **0.822 (L21/28)** | **0.540** | cliff L21→L27 |

Mid-network, the duplex model carries near-raw transferable difficulty info
(within-pair Δ −0.03 at best layer vs −0.59 at the readout). Mean-pooling
survives to the final layer on duplex models → the damage is **(late-layer ×
last-token)-specific** — exactly where a streaming head must encode
turn-control state. Duplex damage is severe but LOCAL; omni-streaming damage
is mild but DIFFUSE. This *revises* the v2 conclusion: the self-knowledge
signal exists and is strong; the standard readout just can't see it on
duplex models.

### 3.3 Does the mid-layer probe convert into a better gate? (5e)

Frozen-test tradeoff areas (same protocol as Phase 5): probe_final +0.054 <
ptrue_pre +0.059 < **midlayer_L22 +0.064** < ptrue_post +0.068 (post-draft
only). L20–L30 all ≥ +0.056 (not knife-edge); mean-pool uniformly worse.
The in-mix area *under-sells* the mid-layer probe: its real advantage is
LOPO robustness (math .93 vs .37), which a same-mix test can't show. Caveat:
probe quantile-threshold transfer is weak (needs Phase-3-style
C-compression calibration if deployed); p(True) transfers rates much better.

---

## 4. Speech input: each readout has one blind modality

### 4.1 The cliff is text-input-specific (6a)

Same frozen pool, TTS-rendered, pure-audio content. Audio→audio LOPO math:
mid-layers L12–L16 hit .93–.96, and the final layer **does not invert:
L35 = .936** (text .366). Speak-mode template control (`use_tts_template=True`
on text input): cliff unchanged (L35 .362) → the operative variable is the
modality of the *context* (text tokens vs audio embeddings), not the output
mode. The duplex fine-tune re-purposed late-layer last-position processing
of text-token contexts specifically.

### 4.2 Cross-modal probe transfer works — on e2e-trained models (6a, 6d)

Probes trained on TEXT hiddens, scored on AUDIO hiddens (and vice versa):
early layers are modality-specific (.54–.60), from ~50% depth the
representation is modality-shared — text→audio **.855 at the deployed L22**
(plateau .82–.87). Replicates on o2.6 (onset ~46% depth, plateau .74–.80)
and cross-family on qwen2.5-omni (onset ~25%, plateau .80–.83). But on
**Freeze-Omni** — identical frozen Qwen2-7B weights, audio via adapter —
transfer is ≈ dead (text→audio .52–.60, audio→text .34–.54): **the shared
mid-layer core is CREATED by training the backbone on the modality**, an
adapter aligns well enough to converse but audio-context hiddens live off
the text manifold. Practical recipe validated: calibrate the gate on cheap
text data, deploy on speech — for end-to-end models only.

### 4.3 Verbalized introspection collapses on audio — duplex-specific (6a/6b)

| model | FT type | trap p_yes pre, text → audio | audio ptrue_pre AUC |
|---|---|---|---:|
| minicpm-o45 | duplex | .055 → **.556** (collapse) | .786, trap dead |
| minicpm-o26 | duplex | .196 → .345, all pools compress to ~.5 | **.491 ≈ chance** |
| qwen2.5-omni | omni-streaming | .279 → **.213 intact** | .727 |
| Freeze-Omni | frozen + adapter | .165 → **.112 intact** | (.612; capability-collapse caveat) |

Mechanism nailed by three controls on o4.5: (1) ASR audit — trap WER .074
(heard fine), p(True) on the model's OWN transcript snaps back to .074 ≈
text; well-heard subset still collapses; corr(WER, p_yes) ≈ 0 → not
perception. (2) Filler-audio arm — irrelevant audio does NOT inflate p_yes
(depresses it) → not a context prior/persona. (3) Audio+text-dup arm — text
tokens of the same question alongside the audio fully restore introspection
(.034) → **binding: the verbal instance-check runs over text-token
pathways; audio embeddings don't feed it**, while the probe reads the same
instance info at .93+. Log-odds decomposition agrees: the audio shift is
graded (chat +0.19 → trap +4.37), i.e. verbal self-assessment regresses to
type-level priors on audio. Cheap fix confirmed twice: any text
re-presentation (own transcript, dup) restores the signal
("repeat-then-judge").

Elegant symmetry: the probe's readout is text-fragile, p(True)'s readout is
audio-fragile — in both cases the knowledge survives and a *readout* breaks,
and in both cases the breakage is duplex-FT-specific.

---

## 5. Mechanism synthesis: four quadrants

| | shared mid-layer core | faithful readouts |
|---|---|---|
| raw text backbone | (text only) | ✅ |
| omni-streaming e2e (qwen2.5-omni) | ✅ built | ✅ intact |
| **duplex e2e (MiniCPM-o ×2)** | ✅ built | ❌ overwritten (probe: late-layer/text; verbal: audio) |
| frozen backbone + adapter (Freeze-Omni) | ❌ absent | ✅ verbal survives (capability craters instead) |

End-to-end multimodal training BUILDS the shared semantic core; duplex-style
training additionally DAMAGES the readouts; omni-streaming gets both right;
an adapter alone gets neither. For gate design this means: the mid-layer
last-token probe is the one signal that survives every quadrant that can
converse.

---

## 6. RQ2/RQ3 — the gate as a system

### 6.1 Accuracy tradeoff (frozen test n=240)

small-only **0.588**; big-only (gpt-5.5) **0.917**; paraphrase-relayed big
0.879 (relay tax 1–4 pts). Every gate beats random escalation at every
operating point; balanced tier @33% escalation → **0.779–0.787**, recovering
~58% of the small→big gap at 1/3 the cost ($1.12/100q big-only).

### 6.2 Against the model's own thinking mode (6c)

MiniCPM-o 4.5 ships `enable_thinking`. Policy table (test n=240; latency =
per-pool measured medians, think per-query, cloud per-query):

| policy | acc | lat mean s | P50 | P95 |
|---|---:|---:|---:|---:|
| fast-only | .588 | 3.0 | 3.5 | 4.2 |
| all-THINK (o4.5's own) | .637 | 22.4 | 17.2 | 60.1 |
| gated-think @.33 | .613 | 12.1 | 3.6 | 47.6 |
| **gated-cloud @.15** | .688 | 5.3 | 3.6 | 10.1 |
| **gated-cloud @.33** | **.787** | **6.5** | 3.6 | 20.8 |
| gated-cloud @.50 | .858 | 7.0 | 3.6 | 24.4 |

**Gated-cloud dominates all-THINK on both axes at every escalation rate.**
Gated-think is weak for a mechanistic reason: the gate predominantly flags
knowledge/trap failures, which thinking cannot fix (thinking helps math
.187→.127, HURTS chat .207→.233) — external escalation is *necessary*, not
just better.

### 6.3 The fork: gate decision lands before the first token (6c/7a)

The probe is one 4096-d dot product on a hidden state already being
computed — the "fork" at layer k costs microseconds and the main forward
never waits (Jisen's parallel-branch design is free by construction; the
real question is at which depth the signal is sufficient). Measured on H100
(CUDA-synced, truncated-forward methodology):

| quantity | text | audio |
|---|---:|---:|
| L22 truncated decision (the gate) | **20 ms** | **45 ms** |
| TTFT (full prefill + 1st token) | 36 ms | 68 ms |
| ptrue_pre (extra short prefill) | 39 ms | 67 ms |
| full answer (= ptrue_post entry fee) | 1 855 ms | 3 507 ms |

The decision exists **before decoding starts** and sits far inside the
200–300 ms voice turn-taking budget. Per-layer truncated-forward curves
(`prefill_timing` / `fork_report`, figures/fork_pareto.png): L22's hidden
is available at **57% of prefill wall-time** (text; 61% audio) and L22 is
simultaneously the in-mix quality peak (OOF .866 > final layer .835) —
deciding early costs nothing, and escalation can launch while the last
~40% of prefill plus decode still runs. Forking at ~L10 is too early
(.794, and pre-50%-depth layers are modality-specific per §4.2); the fork
belongs at ~55–60% depth. On audio, the encoder front-end (~20 ms) is paid
before layer 0 regardless of fork depth.

### 6.4 Cloud round-trip overlaps the talker's floor time (7a)

If the gate fires at prefill and the cloud call launches immediately, is
the expert result ready before the talker finishes speaking (so the next
turn can deliver it)? Per-query gpt-5.5 latencies (P50 3.0 s / P95 24.4 s)
vs pool-matched measured local answer durations (`overlap_report`,
figures/overlap.png):

| P(result ready before local answer ends + slack) | +0 s | +2 s | +5 s |
|---|---:|---:|---:|
| whole test mix, text | .40 | .65 | .81 |
| whole test mix, audio | .58 | .75 | .84 |
| gate-escalated @.33, text | .20 | .39 | .60 |
| gate-escalated @.33, audio | .31 | .47 | .63 |

The catch: **escalated queries skew toward short local answers** (trap and
knowledge — trap overlap ≈ 0), so pure overlap covers only ~a fifth to a
third of escalations; the residual wait after the local answer ends is
**P50 2–3 s** (one stall sentence) but P90 ~27 s — the tail is gpt-5.5's
own reasoning latency, which argues for a fast-expert tier or streamed
partial results in step 2. Audio helps (longer utterances buy time).

![Escalation timelines](figures/timeline_scenarios.png)

**Figure — escalation timelines (all measured medians, text arm).**
(a) The fork: the L22 probe decision (20 ms, 57% of prefill) precedes the
first output token (36 ms); the cloud call launches while prefill is still
running, so the talker never waits on the gate. (b) Audio-channel occupancy
across four scenarios: pre-answer routing (RouteLLM-style) leaves 2.7 s of
dead air before anything is spoken; gated hard-math fully overlaps the
cloud round-trip (expert P50 2.7 s < draft 3.5 s — zero silence); gated
easy-fact bridges with one 1.8 s stall sentence; gated trap is the
structural worst case — gpt-5.5 is slowest exactly on the SimpleQA traps
(P50 8.2 s, P90 54 s) while the local draft is shortest (0.8 s), leaving a
7.4 s gap that stalling cannot bridge — the step-2 problem. Vector version:
`figures/timeline_scenarios.pdf`; script `figures/timeline_scenarios.py`.

---

## 7. Positioning vs LLM routing

RouteLLM-class routers (Ong et al., 2406.18665) decide *before* generation
from query-side features (embeddings, preference-trained matchers) — the
big model then answers the user directly, so no result-injection problem
exists. Two reasons that shape doesn't transfer to full-duplex voice, which
is why this project is step 1 of a two-step design: (1) the user talks to
ONE voice — the talker must speak whatever comes back, and the session
state (audio context, duplex KV) lives in the small model; (2) query-side
routing is exactly the "type recognition" our audit exposed — our probe's
oracle-gap analysis (pool oracle 0.715 vs probe 0.822) quantifies how much
a query-feature router can know in-distribution, and LOPO shows that
component fails across distribution shift while the model-internal
mid-layer signal transfers (math .93). LLMRouterBench (2601.07206) reports
most routers fail to beat simple baselines under a standardized protocol
and attributes the oracle gap to model-recall failures — consistent with
our "the signal must come from the answering model itself, not the query"
conclusion; its protocol is the natural home for a head-to-head
query-feature-router vs internal-signal-gate comparison under distribution
shift (future work).

---

## 8. Design consequences + step 2 preview

**Two-stage gate, final form:**

- **Stage 1 (prefill, free):** mid-layer (~60% depth) last-token probe —
  20/45 ms, before the first token, survives duplex FT and both input
  modalities; needs score-scale calibration for threshold transfer.
- **Stage 2 (optional, post-draft):** ptrue_post on the draft — the
  strongest overall signal; the draft doubles as the distilled escalation
  query and the fallback answer. On audio input, run p(True) on the model's
  own transcript ("repeat-then-judge"), never on raw audio context.

**Step 2 (result injection) is now scoped by data:** the streaming smoke
verified the official injection point (`teacher_forcing_text` per chunk) and
surfaced the first real problem — the model *pushes back* on injected expert
results that conflict with its own reasoning rather than relaying them. The
overlap analysis (6.4) adds the timing constraint: P50 one stall sentence,
P90 dominated by the expert's own latency. Injection protocol design +
authority framing is the next phase.

## 9. Remaining gaps (honest list)

- p(True) prompt sensitivity: one phrasing each (pre/post).
- Audio arm is TTS speech (one voice); SD-QA real-speech validation open.
- Mid-layer probe threshold (rate) transfer needs deployment-grade
  calibration; area/AUC claims unaffected.
- Finding 1's audio side is MiniCPM-scoped (no second runnable duplex
  family); a new open-weight full-duplex model (e.g. if Qwen3.5-Omni or
  DuplexOmni weights land) is a pre-registered prediction test: it should
  show the late-layer text-input cliff.
- Test split reused across signal comparisons (protocol identical,
  signals pre-specified; a fresh test set would be cleaner).
- Judge variance now bounded (gpt-5.5 re-judge, deltas ≤ .010) but single
  judge family; no human labels.
- Total spend ≈ **$290** of the $2000 budget.

## 10. Artifacts

`RESULTS.md` (full log, Phases 0–7a), `PLAN.md`, `gate_config.json`, `src/`
(gate, decode, hf_decode, layers, escalate, distill, inject, queries,
signals), `modal_app.py` (text pipelines + reports), `modal_audio.py` (audio
arm, thinking ablation, latency + fork + overlap benches), `modal_freeze.py`
(Freeze-Omni). Figures: roc, tradeoff, tradeoff_ptrue, tradeoff_midlayer,
layer_sweep, fork_pareto, overlap, timeline_scenarios (+pdf). Volumes:
`gate-data` (signals, layers npz,
features incl. gpt-5.5 re-judge, ptrue shards, audio pool wavs, benches),
`minicpm-o45-weights` (+ all control-model snapshots), `fdb-data`,
`bench-data`.
