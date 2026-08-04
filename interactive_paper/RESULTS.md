# RESULTS — Zero-Training Escalation Gate for MiniCPM-o 4.5

Execution log. One section per Phase: numbers, figures, decisions, gotchas,
and abandoned routes. This is the raw material for the eventual paper.

Seed fixed at **42** everywhere. Models: small = MiniCPM-o 4.5 (9B); judge =
`gpt-5.4-mini`; escalation target = `gpt-5.5`.

> **2026-07-08 — provider switch: Anthropic → OpenAI.** The user has OpenAI
> credit, not Anthropic. Judge + query-gen moved to `gpt-5.4-mini`, escalation
> target to `gpt-5.5` (GPT-5.x reasoning models; `reasoning_effort="low"`,
> generous `max_completion_tokens`). `src/escalate.py` ported to the OpenAI SDK
> (Structured Outputs `response_format` json_schema, strict). Modal secret is now
> `openai`; the separate app is `think-gate-gen`; `build_claude_queries` →
> `build_gen_queries`; `queries_claude.jsonl` → `queries_gen.jsonl`. Sections
> below written before this date still describe the original Claude design.

---

## Setup / environment (2026-07-07)

Lives as a **subdir** inside the existing `dyyfk/minimcp_test` repo
(`interactive_paper/`), reusing that project's Modal infra:

- **Modal** workspace `rhe9527`, client 1.5.1. Invoke via the `modal` CLI
  (the anaconda base python has no modal module; Python310's does). Prefix
  `PYTHONUTF8=1` on Windows or the CLI's ✓ output crashes on cp936.
- **Weights reused**: MiniCPM-o 4.5 already on the `minicpm-o45-weights` Volume
  (downloaded 2026-06-30 by the sibling `modal_app.py::download_weights`), so
  Phase 0 needs no HF pull — mounts read-only at `/workspace/models/MiniCPM-o-4_5`.
- **Validated stack** (from the sibling project, unchanged): torch 2.8.0+cu128,
  transformers **4.51.0** (4.52+ breaks MiniCPM's Resampler init),
  `minicpmo-utils[all]` (librosa 0.9.0), `setuptools<81`, SDPA (no flash-attn).
  `image` in `modal_app.py` adds `openai` + `scikit-learn` + `pandas` +
  `pyarrow` on top; `src/` mounted at `/workspace/gate` (add_local_dir last).
- **`gate-data` Volume** created for query pool + feature store + preds.
- **OpenAI model ids** (pricing/models pages, July 2026): judge/gen
  `gpt-5.4-mini`, escalation `gpt-5.5`. GPT-5.x reasoning models — hidden
  reasoning tokens bill as output; calls use `reasoning_effort="low"` +
  `max_completion_tokens`. If an id 404s, pin a dated snapshot.

### BLOCKER (Phase 2): openai Modal secret

Phase 2's judge labeling and easy-chat/trap query generation need OpenAI inside
the container. User action required (once):

```
modal secret create openai OPENAI_API_KEY=sk-...
```

Phases 0–1 (small model only) proceed without it.

---

## Phase 0 — Modal env + model smoke test ✅ GO (2026-07-07)

`modal run interactive_paper/modal_app.py::smoke` — image built in 69s, ran on H100.

- **Load**: 16.1 s | **VRAM 16.4 GB** with `init_vision=False, init_audio=False,
  init_tts=False` — text-only load is ~4 GB lighter than the full duplex model,
  leaving the H100 nearly empty. Text `model.chat(content=[prompt])` works fine
  with the audio/vision encoders uninitialized.
- **Model internals** (for the Phase-1 hook):
  - top module `MiniCPMO`, single child `llm` = **`Qwen3ForCausalLM`**
  - `num_hidden_layers=36`, `hidden_size=4096`, `vocab_size=151748`
  - `llm.named_children = ['model', 'lm_head']` → backbone `model.llm.model`,
    last decoder layer `model.llm.model.layers[35]`, head `model.llm.lm_head`.
- **Decode speed** (greedy, bf16, SDPA):

  | probe | out tok | s | tok/s |
  |-------|--------:|--:|------:|
  | chat (octopus) | 50 | 3.7 | 13.6 |
  | math (GSM8K)   | 94 | 2.8 | 33.6 |
  | gpqa (QM)      | 81 | 2.5 | 32.8 |

  The `chat` 13.6 tok/s is a short-output artifact (fixed prefill/decode overhead
  amortized over only 50 tokens); **sustained rate ≈ 33 tok/s**, well above the
  ≥ 20 tok/s gate.
- **Coherence**: all three answers fluent, correct English (3-hearts octopus fact;
  GSM8K worked step-by-step; commutator → Heisenberg uncertainty). No Chinese
  probe run yet, but the duplex project already confirmed bilingual output.

**Verdict: GO.** No downgrade needed — MiniCPM-o 4.5 hooks cleanly (standard
Qwen3 backbone under `model.llm`), so the plan's fallback to 2.6 / Qwen3-8B is
not triggered.

---

## Phase 1 — signal extraction (hook-based capture) ✅ GO (2026-07-07)

Design decision: **don't re-implement MiniCPM's chat template / decode loop**
(it lives in remote code and is fragile). Instead let `model.chat()` generate
greedily as normal and *passively observe* via two forward hooks —
`model.llm.model.layers[35]` (last-layer hidden) and `model.llm.lm_head`
(logits). `src/decode.py::chat_with_signals`. Signals therefore come from the
model's real generation; the returned text IS the model's answer.

Per forward pass we grab the last-position hidden + last-position logits. Forward
0 = prefill → `h_prompt` (whole-prompt representation) + the 1st-token
distribution; forwards 1..K = per-step. Store first-K scalar signals
(entropy/margin) + `h_prompt` [4096] + `h_mean8` (mean of first ≤8 step hiddens).

`modal run interactive_paper/modal_app.py::signal_check`:

- **Faithfulness**: hooked text **identical** to plain `chat()` output → hooks
  don't perturb generation. ✓
- **Overhead**: reported −28% (hooked 49.9 vs plain 35.9 tok/s) — this is
  warmup-order noise (the plain call ran first and paid CUDA kernel warmup). The
  real conclusion is the hooks add **no meaningful slowdown** (a `.float().cpu()`
  copy + a float32 softmax over the vocab, only for the first 16 steps). Well
  inside the < 30 % gate. ✓
- **Sanity** (entropy in nats, first-K steps):

  | query | n_forward | mean entropy@16 | entropy[0] |
  |-------|----------:|----------------:|-----------:|
  | easy ("capital of France?") | 9 | 0.185 | 0.54 |
  | hard (1-D well ⟨x²⟩ derivation) | 17 | **0.885** | **3.86** |

  Hard entropy ≈ 4.8× easy; the first generated token is already strongly
  discriminative (0.54 vs 3.86). Direction matches intuition — wiring is correct. ✓

**Verdict: GO.** Signal capture is faithful, cheap, and directionally sane.

---

## Phase 2 — calibration dataset + discriminative analysis (⭐ GO/NO-GO GATE)

**Status: ✅ COMPLETE (2026-07-08) — verdict GO, see below.** (Was paused at
the secret blocker; resolved by the OpenAI switch + `openai` secret.)

Pipeline (all in `modal_app.py`, pools/formatting in `src/queries.py`, Claude in
`src/escalate.py`):

1. `build_public_queries` (CPU, **no secret**) — GSM8K tail + MATH-500 (hard-math
   150), MMLU-Pro (+GPQA if reachable) (hard-knowledge 150), TriviaQA (easy-fact
   100) → `queries_public.jsonl`.
2. `build_claude_queries` (CPU, **needs secret**) — easy-chat 150 (中英各半) +
   trap 50 via `claude-opus-4-8` → `queries_claude.jsonl`.
3. `finalize_queries` — merge, assign `id`, stratified 60/40 calib/test split
   (seed 42) → `queries.jsonl`. **Run once over all 600** (ids are sequential, so
   re-finalizing would renumber and invalidate any already-collected signals).
4. `run_signals` (H100 ×4 shards) — `chat_with_signals` per query → per-shard
   parquet with answer + entropy/margin/h_prompt/h_mean8.
5. `label` (CPU, **needs secret**) — `claude-opus-4-8` judge → `adequate`;
   `escalate_label = 1 - adequate` → `calib_features.parquet`.
6. `calibrate` (CPU, sklearn) — ROC-AUC per signal (scalar entropy/margin @K∈{4,8,16},
   logistic probes on `h_prompt` / `h_mean8`, combined LR), per-pool AUC
   (the trap-pool "probe beats entropy" story), `roc.png`, and the go/no-go verdict.

### GOTCHA — Modal resolves `Secret.from_name` for *every* function in an app

First `build_public_queries` run failed with **"Secret 'anthropic' not found"**
even though it makes no LLM call: Modal hydrates the secrets of *all*
`@app.function`s when you run *any* function in that app. Fix: put the two
LLM-dependent functions on a **separate app object** (`gen_app =
modal.App("think-gate-gen")`) in the same file — `modal run …::build_public_queries`
then hydrates only `app` and skips the `openai` secret. (Creating even a
placeholder secret is intentionally gated behind the user in auto mode.)

### BLOCKER (unchanged) — the user must create the real secret

```
modal secret create openai OPENAI_API_KEY=sk-...
```

After that, the gate is reached with:
`build_gen_queries` → `finalize_queries` → `run_signals` → `label` → `calibrate`.

### ⭐ FINAL VERDICT (2026-07-08, all-public rerun): **GO — best AUC 0.828**

**Rerun with 100% public datasets** (user decision: no LLM-generated eval
queries — the GPT-generated pools are gone). easy-chat = dolly-15k (75 en,
short no-context instructions) + shibing624/alpaca-zh (75 zh); trap =
**SimpleQA** (basicv8vc/SimpleQA, 50). 600 total, 360 calib / 240 test
(seed 42, ids re-frozen — this supersedes the 597-query run below).
Supervised end-to-end by `supervisor.sh` (log: `pipeline_watch.log`).

- signals: 600/600 on 4×H100, 0 failures. label: 600/600, 0 judge errors.
- Escalate rate by pool: **trap 1.000 (!)**, hard-knowledge .480, easy-fact
  .340, easy-chat .207, hard-math .187; overall **.358**.
- **SimpleQA fixed the trap pool** — MiniCPM failed ALL 50 (GPT-generated
  traps: only .102). Confirms the "use public benchmarks" call. Side effect:
  single-class pool → within-trap AUC undefined; the "probe rescues entropy
  on traps" story is still untestable within-pool (would need a trap set with
  some successes — e.g. PopQA stratified by popularity, for later).

ROC-AUC (600, 5-fold CV):

| signal | AUC (vs 597-run) |
|---|---|
| **probe_h_prompt** | **0.828** (0.812) |
| combined | 0.823 (0.790) |
| probe_h_mean8 | 0.776 (0.805) |
| max_entropy@4 (best scalar) | 0.696 (0.624) |

Conclusions carried / updated:
1. Probe-on-h_prompt remains the signal; conclusion ROBUST to the dataset
   swap (0.812 → 0.828). Entropy stays far behind (best scalar 0.696).
2. h_prompt now clearly beats h_mean8 (0.828 vs 0.776) — pre-decode gating
   (fire before the first token) is the design for Phase 3.
3. Per-pool: probe wins hard-math (.891) and hard-knowledge (.779); entropy
   wins easy-fact (.845 vs .762) and easy-chat (.644 vs .535 — chat failures
   are near-invisible to the probe; caveat for the paper).
4. combined (0.823) no longer hurts vs probe (0.828) but adds nothing —
   Phase-3 gate stays probe-only.

Cost of the rerun: ~$6 GPU + ~$2.5 API. Stopped at the gate per plan.

### Overfit audit (2026-07-08, user challenge — calib-only, test untouched)

User asked whether the probe overfits to our pool. `modal_app.py::audit`
(CPU, calib 360 only — also fixes a process slip: `calibrate` had been doing
CV over all 600 incl. test; the plan-compliant headline is calib-only):

| check | result |
|---|---|
| in-sample AUC | 1.000 (p≫n memorizes, as expected) |
| **calib-only 5-fold CV** | **0.821** (headline barely moves vs 0.828 full) |
| **pool-identity-only oracle** | **0.715** — pool membership alone buys most of the aggregate AUC |
| LOPO easy-chat / easy-fact | 0.704 / 0.717 (transfers OK) |
| LOPO hard-knowledge | 0.606 (weak) |
| **LOPO hard-math** | **0.372 — WORSE than chance; inverts** |
| **LOPO trap** | mean escalate-score **0.232** on 100%-fail questions — would NOT escalate them |

**Verdict: the user's suspicion is substantially correct.** The probe is
honest *within* the calibration distribution (0.821 out-of-fold), but a large
share of the aggregate number is composition shortcut (oracle 0.715), and it
does **not** transfer to unseen query types: trained without math it actively
misranks math failures (0.372), and trained without traps it scores
looks-simple-but-fatal SimpleQA questions as LOW-risk (0.232) — exactly the
"reads surface familiarity, not self-knowledge" failure mode.

Implications recorded for the paper & Phase 3:
- Claim must be scoped: "zero-training gate calibrated on a deployment-like
  query mix", NOT "universal difficulty detector". The LOPO table itself is
  an honest & interesting result (probe generalization is distribution-bound).
- Phase-3 gate remains viable for in-distribution use; deployment requires
  the calibration mix to resemble traffic (or periodic recalibration).
- Candidate stronger baselines to try later: verbalized self-eval / p(True)
  (also zero-training, may transfer better), per-pool calibration, more
  diverse calibration pools.

---

### [SUPERSEDED] first run (597, with GPT-generated pools): GO — AUC 0.812

Full Phase-2 execution (post provider switch, all five steps in one session):

- `build_gen_queries` ✅ — **197** generated (easy-chat 148, trap 49;
  gpt-5.4-mini under-delivered 2+1 vs the requested 150+50 — accepted).
  Validated key + model id + Structured Outputs in one shot (~$0.50).
- `finalize_queries` ✅ — **597 total** (hard-math 150, hard-knowledge 150,
  easy-fact 100, easy-chat 148, trap 49), split **358 calib / 239 test**
  (seed 42). Ids `q0000…` now FROZEN — do not re-run finalize.
- `run_signals` ✅ — 4× H100, 597/597 rows, 0 failures →
  `signals.shard{0..3}.parquet`.
- `label` ✅ — gpt-5.4-mini judged 597/597, **0 judge errors**.
  Escalate rate by pool: hard-knowledge .500, easy-fact .350, hard-math .187,
  easy-chat .128, **trap .102**; overall **.271**.
- `calibrate` ✅ → `roc.png` (pulled to `figures/roc.png`).

ROC-AUC (predicting small-model failure), full 597:

| signal | AUC |
|---|---|
| **probe_h_prompt** | **0.812** |
| probe_h_mean8 | 0.805 |
| combined (entropy+margin+probe LR) | 0.790 |
| max_entropy@16 (best scalar) | 0.626 |
| mean_margin@16 (worst) | 0.490 |

Findings vs expectations:

1. **The linear probe on the prompt representation is the whole story** —
   0.812 vs 0.626 for the best scalar. Entropy/margin are far weaker than
   hoped; the *combined* LR (0.790) is actually WORSE than the probe alone,
   i.e. the scalars add noise, not signal. Phase 3's gate score should be
   probe-only (one dot product — still zero-cost online).
2. **h_prompt ≈ h_mean8** (0.812 vs 0.805): the prompt's last hidden state
   already carries the difficulty signal — decode-time hiddens add nothing.
   Big practical win: the gate can fire **before the first token is decoded**.
3. **The trap pool failed its design** (escalate rate 0.102, expected high):
   gpt-5.4-mini's "confident-but-wrong" traps were mostly answerable by
   MiniCPM. With ~5 positives in n=49, its per-pool AUCs (entropy .855 /
   probe .555) are noise — the planned "probe rescues entropy on traps"
   narrative is UNSUPPORTED on this data. Either regenerate traps harder
   (stronger gen model / verify small-model-fails before accepting) or drop
   that storyline.
4. Per-pool (where n gives signal): probe > entropy on easy-chat
   (.769/.593) and hard-knowledge (.749/.673); entropy > probe on easy-fact
   (.859/.692); tie on hard-math (~.86 both). Entropy is good exactly where
   failures are knowledge-retrieval flavored; the probe is more uniform.

Per PLAN discipline #5: **stopped here and reported to the user.** Phase 3
(threshold gate) is unblocked on a GO.

## Phase 3 — online threshold gate ✅ (2026-07-09)

**Goal**: turn the Phase-2 probe into a real-time trigger + pick deployable
thresholds. Done entirely on **CPU** — every `h_prompt` is already in
`calib_features.parquet`, so the online score is a deterministic replay; no GPU
decode was spent to validate the trigger logic.

### Design decisions (both forced by the data, deviating from PLAN §Phase-3)

1. **Pre-decode single-shot, not streaming EMA.** PLAN designed the gate as
   EMA + k-consecutive hysteresis over per-step scores — sensible for a scalar
   that evolves during decode (entropy/margin). But Phase 2 found the winning
   signal is the probe on `h_prompt`, a **single score available at prefill**
   (h_prompt 0.828 > decode-time h_mean8 0.776). So the headline gate fires
   **before the first token** from one score. `src/gate.py::EscalationGate` still
   implements the full EMA/hysteresis/cooldown machinery (pure-Python, unit-tested,
   needed for the duplex Phase 6); the headline runs it in single-shot mode
   (`k_consecutive=1, ema_alpha=1.0`), where it degenerates to `score >= threshold`.
2. **Tiers by escalation BUDGET, not precision target.** At base rate 0.322 and
   AUC 0.83, PLAN's "precision >= 0.80 default" is only reachable at ~0 recall
   (degenerate — first two attempts pinned all thresholds to 1.0). Escalation rate
   is the real cost knob and the exact axis Phase 5 sweeps, so tiers are set at
   target escalate rates {conservative .15 / balanced .30 / aggressive .50}.

### Overfit-aware threshold calibration (the non-obvious part)

The shipped probe is fit on all 360 calib rows, but with 4096 dims / n=360 the
data is **linearly separable → in-sample AUC 1.000 at every C** (regularization
shrinks score magnitudes, not in-sample ranking). Picking thresholds on in-sample
scores is therefore meaningless. Fixes in `modal_app.py::fit_gate`:

- **thresholds live on 5-fold OOF scores** (seed 42), which reflect deployment
  generalization (~0.83), not the memorized 1.0.
- **C-regularization sweep** picks C by OOF AUC (tie → smaller C):
  C=0.001 (OOF **0.828**) > 0.01 (.822) > 1.0 (.821) > 0.1 (.818). Heavy L2 both
  maximizes OOF AUC *and* compresses the shipped probe's score scale so an
  OOF-quantile threshold transfers to it.

`gate_config.json` (pulled to repo, 4096-float probe + 3 thresholds) is the
artifact Phases 4–5 load. `src/gate.py` = `Probe` (sigmoid(w·h+b), one dot
product) + `EscalationGate`; `src/test_gate.py` = 28 pure-Python checks (pass).

### Realized operating points on calib (OOF scores, `gate_eval`)

| tier | thr | escalate | precision | recall |
|------|----:|---------:|----------:|-------:|
| conservative | 0.933 | 0.150 | 0.722 | 0.336 |
| balanced     | 0.475 | 0.300 | 0.657 | 0.612 |
| aggressive   | 0.070 | 0.500 | 0.539 | 0.836 |

Per-pool trigger rate (conservative / balanced / aggressive):

| pool | esc-rate | cons | bal | aggr |
|------|---------:|-----:|----:|-----:|
| **trap** (SimpleQA, 100% fail) | 1.00 | **0.80** | 1.00 | 1.00 |
| hard-knowledge | 0.40 | 0.18 | 0.42 | 0.73 |
| easy-fact | 0.23 | 0.08 | 0.22 | 0.43 |
| hard-math | 0.22 | 0.09 | 0.20 | 0.32 |
| easy-chat | 0.18 | 0.01 | 0.10 | 0.32 |

Reads exactly as hoped: even the **conservative** tier catches **80% of the
100%-fail trap questions** while false-triggering easy-chat only 1%. Trigger rate
tracks pool failure rate everywhere **except hard-math** (under-caught: 0.09/0.20/
0.32, below its difficulty) — the same "probe reads knowledge-difficulty better
than math-difficulty" weakness the Phase-2 LOPO audit exposed, now visible
in-distribution too. Caveat carries to the paper.

**Validation**: `EscalationGate.from_config` (single-shot) reproduced the
`score >= threshold` decision on all 360 calib rows → deployment trigger logic
is correct. Phase-3 cost ≈ $0 (3 short CPU runs).

**Next (Phase 4)**: escalation chain E2E — trigger → distilled query → GPT-5.5 →
inject/paraphrase. `chat_gated` (live pre-decode stop on the H100) is deferred to
Phase 4, where the escalation chain actually consumes the trigger; Phase 3's CPU
replay already validates the gate numerically.

---

## Phase 4 — escalation chain E2E ✅ (2026-07-09)

**Goal**: trigger → distilled query → gpt-5.5 → inject/paraphrase, end-to-end.
Built `src/distill.py` (`distill_query`), `src/inject.py` (`paraphrase`),
`src/escalate.py::ask_expert`/`ask_expert_many` (gpt-5.5, error-safe, token-usage
capture), and `decode.generate`. `modal_app.py::e2e_demo` runs the full chain on
hard test queries and prints a readable trace.

Verified on the trace (PLAN Phase-4 go/no-go): **distilled queries are faithful**
(single-turn queries are already standalone, so distillation is near-identity —
its real payoff is multi-turn/duplex, Phase 6; the Phase-5 eval therefore
escalates the *original* query) and **the paraphrase relays the expert answer
accurately**. Example — a Sn(gray→white) equilibrium-temperature question: small
model went down the wrong equation; gate fired (score 0.988); gpt-5.5 returned the
reference `C. −3.5 °C`; small model paraphrased it faithfully.

**Gotcha**: gpt-5.x hidden reasoning bills as output and can consume the whole
`max_completion_tokens` before the visible answer (empty content,
`finish_reason=length`). Raised the expert cap 4096→**8192** and made `ask_expert`
error-safe (returns `error` instead of raising). At 8192 the full 240-query eval
hit **0 truncation errors**.

---

## Phase 5 — system evaluation ✅ (2026-07-09) ⭐ RQ2 answered

**Goal**: the accuracy-vs-escalation-rate tradeoff on the frozen **test split**
(240). Four conditions; judge = `gpt-5.4-mini`, blind to source.

Pipeline: `eval_expert` (gpt-5.5 answers all 240, judged → big-only) →
`eval_paraphrase` (small model relays each expert answer, judged → hybrid outcome)
→ `eval_assemble` (probe-scores the stored test `h_prompt`, sweeps the gate
threshold to draw the curve, compares to random escalation, computes latency/cost,
writes `figures/tradeoff.png`). Small-only answers/labels were already in
`calib_features.parquet` from Phase 2 (no recompute).

### Headline numbers (test, n=240)

| condition | accuracy | escalation |
|---|---:|---:|
| small-only (MiniCPM-o 4.5) | **0.588** | 0% |
| big-only (gpt-5.5, raw) | **0.917** | 100% |
| big-only relayed via small (paraphrase) | 0.879 | 100% |
| hybrid-gate **conservative** | 0.679 / 0.671¹ | 14.2% |
| hybrid-gate **balanced** | 0.779 / 0.767¹ | 32.9% |
| hybrid-gate **aggressive** | 0.833 / 0.821¹ | 52.9% |

¹ accuracy as *expert-inject* / *small-model-paraphrase*.

### Key results

1. **The gate beats random escalation at every operating point** (the central
   RQ2 claim). Area between the gate curve and the random-escalation line
   (∫(acc_gate − acc_rand) d rate) = **+0.054**. Concretely at 33% escalation the
   gate reaches **0.779** vs random's ~0.696 (+8.3 pts); +4.4 pts at 14%, +7.1 at
   53%. `figures/tradeoff.png` — both gate curves bow well above the diagonal and
   rise steeply early (the gate escalates the highest-risk queries first).
2. **Escalating 33% of traffic recovers ~58% of the small→big accuracy gap**
   (0.588→0.779 of the 0.588→0.917 span). The gate buys most of the big model's
   accuracy at a third of its cost/latency.
3. **Paraphrase (relay) tax ≈ 1–4 pts**: routing the expert answer back through
   the small model costs 0.917→0.879 at full escalation (and ~1 pt per tier). It's
   the price of natural spoken relay; a deployment that can surface the expert
   answer directly avoids it.
4. **big-only is not a ceiling of 1.0**: gpt-5.5 scores 0.917 overall — perfect on
   hard-math (1.00) but only **0.65 on trap** (SimpleQA long-tail facts stump even
   the big model). So the trap pool caps how much *any* escalation can help there.

### Latency & cost

- Latency (s): expert gpt-5.5 **P50 3.0 / P95 24.4**; small-model paraphrase
  **P50 1.0 / P95 5.8**. Small-only decode ≈ 33 tok/s (Phase 0). The escalation
  chain's latency is dominated by the expert call.
- Cost: gpt-5.5 big-only = **$1.12 / 100 queries** (30.5k in + 84.4k out tokens
  over 240, at $5/$30 per M). Hybrid scales ~linearly with escalation rate, so
  balanced ≈ $0.37 / 100q for the expert calls.

**Phase-5 spend** ≈ $3 API (240 expert + ~480 judge) + ~$3 GPU (paraphrase +
demo). Total project spend to date ≈ **$32**.

### Caveats carried to the paper

- Accuracy figures are single-run greedy (seed 42); no judge-variance bars.
- The gate curve is drawn by thresholding stored test `h_prompt` scores — a true
  online run would recompute the identical score at prefill (Phase-3 established
  the deployment scorer reproduces it). `chat_gated`'s live decode-stop was not
  needed for the accuracy eval and remains unimplemented (a latency optimization).
- hard-math is under-escalated by the gate (Phase 2/3 weakness), yet gpt-5.5 would
  answer those perfectly — the biggest missed opportunity the gate leaves on the
  table. A math-aware signal is the clearest follow-up.

### Pool-oracle baseline added to the figure (2026-07-29, grilling session)

The figure's only opponent was random escalation — the weakest possible —
while the Phase-2 audit already showed pool identity alone buys AUC 0.715
of 0.821. `eval_assemble` now draws the system-level version: a
**pool-oracle router** (score = the query's pool CALIB fail rate; true type
labels, no instance information, no test leakage; ties within a pool =
straight segments between pool-boundary points). Result:

- oracle-vs-random area **+0.042** vs the gate's **+0.054** → the internal
  signal's in-distribution residual is **+0.012 area (~22%)** — the
  accuracy-curve counterpart of the AUC decomposition, and slightly
  harsher (AUC said ~1/3 was instance-level).
- calib pool fail rates driving the oracle: trap 1.00 > hard-knowledge .40
  > easy-fact .23 > hard-math .22 > easy-chat .18.
- `figures/tradeoff.png` regenerated (purple line hugs the gate curve);
  paper `system.tex` text + caption updated same session. The paper claim
  now rests explicitly on transfer (LOPO), not the in-distribution margin.
- Open: is +0.012 even significant at n=240? Folded into the
  statistics-hardening todo (paired bootstrap gate-vs-oracle).

---

## Phase 5b — audit deep-dive + p(True) baseline ⭐ (2026-07-14)

User unblocked budget ("$2000 Modal credit, use at your own discretion").
Three experiments closing the report's biggest holes. All numbers below.

### audit2 — math-inversion root cause + evidence-chain closure + CIs (CPU, ~$0)

`modal_app.py::audit2`, calib-only (360), test untouched except pre-stored
Phase-5 outcomes for CIs.

**[A] Math LOPO inversion root-caused.** Within math calib (n=90): gsm8k
fail=0.04, math500 fail=0.55 — failure ≈ "is it MATH-500" (corr +0.592). The
LOPO probe (trained without math) scores MATH-500 *lower* risk than GSM8K
(corr(score, is_math500) = −0.246): trained on knowledge/fact/chat/trap, it
reads terse symbolic competition problems as "safe" and verbose GSM8K stories
as riskier. Not length-driven (corr ≈ 0). Within-source LOPO AUC: math500
0.704 (OK!), gsm8k 0.455 — the inversion is mostly a *between-source* ranking
error. Confirms: the probe reads surface style, not solve-ability.

**[B] Evidence chain closed: h_prompt ⊇ pool identity.** 5-way pool classifier
on h_prompt: **95.8%** 5-fold accuracy. Dummies-only OOF AUC 0.678; h_prompt
0.822; **dummies+h_prompt 0.821 — adding explicit pool identity to the probe
adds NOTHING**, i.e. h_prompt already contains the full type shortcut.
Within-pool AUC of the probe's OOF scores (type shortcut controlled):
easy-chat .693 / easy-fact .793 / hard-knowledge .634 / hard-math .847,
macro-mean **0.742** vs aggregate 0.822.

**[C] Bootstrap 95% CIs (2000 resamples).** calib OOF AUC 0.822 [.777, .863].
Test headline: small 0.588 [.525, .650], big 0.917 [.879, .950], paraphrase
0.879 [.833, .917]; hybrid cons/bal/aggr 0.679 [.617, .733] / 0.779 [.725,
.829] / 0.833 [.783, .879]. **Gate-vs-random area +0.0541 [+0.0399, +0.0677]**
— significantly > 0. (Judge variance still unbounded — single judge.)

### ⭐ p(True) verbalized self-eval — the probe was reading the wrong signal

`collect_ptrue` (4×H100, ~$1): two zero-training, zero-calibration signals per
query, both = P(Yes) read off the first-token logits (Yes/No token mass;
median mass 1.0, so the read is clean):
- **ptrue_pre**: "Would you answer this correctly?" *before* answering.
- **ptrue_post**: "Is this proposed answer correct?" given its stored answer.

`ptrue_analyze` (calib rows for probe comparability):

| signal | AUC | needs training? |
|---|---:|---|
| **ptrue_post** | **0.899** | no |
| pre+post mean | 0.877 | no |
| probe OOF | 0.821 | probe fit on calib |
| **ptrue_pre** | 0.807 | no |

Per-pool (the transfer test — p(True) has no calibration pool, so this IS its
LOPO): easy-chat .708/.812 (pre/post), easy-fact .764/.855, hard-knowledge
.688/.797, **hard-math .809/.944 (probe LOPO: 0.377)**. No inversion anywhere.

**Trap (100%-fail SimpleQA): ptrue_pre mean score 0.945** vs 0.309 non-trap —
100% of traps rank above the non-trap P70. The probe-LOPO scored them 0.328
(would not escalate). **The model KNOWS it doesn't know these long-tail facts
— you just have to ask it before it answers.** ptrue_post degrades on traps
(0.360): after generating a confident-wrong answer the model starts believing
it. Introspect BEFORE answering.

### p(True) tradeoff on frozen test (n=240) — better AUC → better curve

`ptrue_gate_eval` (CPU; thresholds = calib score quantiles, no fitting):

| signal | area vs random | acc @ ~30% esc | thr transfer calib→test |
|---|---:|---:|---|
| probe | +0.0541 | 0.667 (esc 0.12!) | poor (rates drift) |
| ptrue_pre | +0.0594 | 0.779 (esc 0.36) | good |
| **ptrue_post** | **+0.0682** | **0.821 (esc 0.38)** | good |

`figures/tradeoff_ptrue.png`. Deployment note: ptrue_pre costs one extra short
prefill pre-decode (fits the streaming design); ptrue_post needs the full
small answer first (draft-then-check — natural for the System-1/2 loop, and
what step-2 injection consumes anyway).

**Step-1 conclusion REVISED**: genuine zero-training self-knowledge EXISTS and
is verbalizable; the h_prompt linear probe was reading query-type familiarity
instead. The gate should be built on verbalized self-eval (pre-answer for
pre-decode firing, post-answer check where latency allows), with the probe as
a free auxiliary at best.

### Cross-backbone replication (in flight)

`run_signals_hf` / `label_hf` / `run_ptrue_hf` / `xmodel_report` added
(`src/hf_decode.py` = vanilla-HF hook mirror of decode.py). Backbones:
qwen3-8b (MiniCPM's family, raw; thinking disabled) + mistral-7b-instruct-v0.3
(different family). Judge = same gpt-5.4-mini rubric; same 600 queries.

**qwen3-8b (600/600 signals, 0 judge errors).** Fail rates: chat .387,
fact .470, knowledge .507, math .180, trap .980 (49/50 — one success, so trap
AUC computable). Raw no-think Qwen3-8B fails much MORE on the easy pools than
MiniCPM (.387 vs .207 chat) — omni fine-tune + system prompt differences.

| check | MiniCPM-o 4.5 | qwen3-8b | replicates? |
|---|---:|---:|---|
| probe OOF AUC (calib) | 0.822 | **0.838** | ✅ |
| pool-oracle AUC | 0.715 | 0.704 | ✅ type shortcut |
| pool-classifier acc | 0.958 | 0.958 | ✅ |
| max_entropy@4 AUC | 0.696 | **0.468 (useless)** | ❌ entropy is fragile |
| LOPO hard-math | **0.372 (inverts)** | **0.961 (fine!)** | ❌ **does NOT replicate** |
| LOPO trap | score 0.23 (miss) | AUC 0.966 | ❌ |
| LOPO chat/fact/knowledge | .70/.72/.61 | .78/.60/.73 | ~ |
| ptrue_post AUC | 0.899 | **0.897** | ✅ almost exactly |
| ptrue_pre AUC | 0.807 | 0.736 (knowledge .563 weak) | ~ |
| ptrue_pre on trap | 0.945 mean score | **AUC 0.939** | ✅ knows-it-doesn't-know |

**The headline surprise: the LOPO transfer failure is MiniCPM-SPECIFIC.** On
raw Qwen3-8B the h_prompt probe transfers fine across pools (math 0.961, trap
0.966) — no inversion anywhere. So "linear probes can't transfer across query
types" is NOT a universal law; MiniCPM's omni fine-tuning (or its different
failure profile — qwen fails 37–51% on every non-math pool, giving LOPO
training much broader positive coverage) restructures what the probe can read.
Caveat: the two models' label distributions differ a lot, so representation
vs. label-coverage explanations are confounded — resolved by `lopo_matched`
below.

**`lopo_matched` — confound resolved: it's the REPRESENTATION.** Subsampling
qwen's LOPO training pools to MiniCPM's exact per-pool fail rates (chat .21,
fact .34, knowledge .48, trap 1.0; matched train n=246) leaves qwen's
LOPO-math at **0.962–0.968 over 5 subsample seeds** (unmatched 0.961). Label
coverage is ruled out; raw Qwen3-8B's h_prompt linearly encodes transferable
difficulty that MiniCPM-o's (same architecture, omni fine-tuned) does not.

**What replicates cleanly: p(True).** ptrue_post ≈ 0.90 on BOTH backbones, and
pre-answer trap introspection holds (0.939 AUC) — the "model knows it doesn't
know long-tail facts if you ask before it answers" finding is now 2-for-2.
Entropy, meanwhile, collapsed to chance on qwen (0.468) — scalar uncertainty
is the least portable signal of all.

**mistral-7b (600/600, 0 judge errors).** Much weaker model (fail .519
overall; knowledge .711, math .556). Download fought back (unauthenticated
Xet stall → partial snapshot → 403 on `consolidated.safetensors` → missing
sentencepiece; fixes: HF_HUB_DISABLE_XET=1, no config.json short-circuit,
ignore `consolidated*`, sentencepiece in the GPU image).

| check | MiniCPM-o | qwen3-8b | mistral-7b |
|---|---:|---:|---:|
| probe OOF AUC | 0.822 | 0.838 | 0.758 |
| pool-oracle AUC | 0.715 | 0.704 | **0.730 (probe adds only +.03!)** |
| pool-classifier acc | 0.958 | 0.958 | 0.953 |
| max_entropy@4 | 0.696 | 0.468 | 0.501 |
| LOPO math / fact | .372 / .717 | .961 / .596 | .817 / **.445** |
| ptrue_post AUC | **0.899** | **0.897** | **0.814** |
| ptrue_pre AUC | 0.807 | 0.736 | 0.723 |
| ptrue on trap | pre .945 score | pre .939 AUC | post .901 AUC |

### Cross-backbone synthesis (3 models)

1. **ptrue_post is the most portable signal**: 0.899/0.897/0.814 — beats the
   trained probe on ALL THREE backbones, with zero training/calibration.
2. **The type shortcut is universal**: h_prompt encodes pool identity at ~95%
   on all three; oracle AUC 0.70–0.73. On mistral the probe's aggregate edge
   over the oracle is a mere +0.028 — the probe ≈ a type classifier there.
3. **Probe transferability is a property of the BACKBONE, not the method**:
   LOPO transfer is fine on qwen (all ≥ .60, math .96), partial on mistral
   (fact .445 inverts), catastrophic on MiniCPM (math .372, trap missed).
   `lopo_matched` rules out label coverage — it's the representation.
4. **Entropy is the least portable signal**: ≈ chance on 2 of 3 backbones.
5. **Pre-answer trap introspection holds on all three** — "the model knows it
   doesn't know long-tail facts if asked before answering" is now 3-for-3.

Phase-5b total spend ≈ $12 GPU + $6 API. Project total ≈ **$50**.

---

## Phase 5c — duplex-generalization matched pairs ⭐ (2026-07-17)

**Framing correction (user):** this project is FOR full-duplex models —
qwen3-8b/mistral-7b (and every model below) are **controls for the RQ1 signal
findings, never system baselines**. Question: does "the omni fine-tune
destroys probe transferability" generalize across duplex models, or is it
MiniCPM-specific? Design: matched pairs — one raw backbone vs its own
audio/omni/duplex fine-tunes, so each pair is its own control.

**Models** (all ungated, downloaded to the weights volume): Qwen2.5-7B family
= raw `qwen2.5-7b` vs `qwen2.5-omni-7b` (streaming talker–thinker omni FT) vs
`minicpm-o26` (true duplex FT, same Qwen2.5-7B base). Second pair:
`qwen2-7b` raw vs `qwen2-audio-7b` (audio-understanding FT, no duplex).
Second family raw: `glm4-9b-chat-hf`. Rejected: Moshi (no raw counterpart,
can't answer text queries); deferred: Kimi-Audio (dual-stream forward breaks
vanilla generate), GLM-4-Voice (ChatGLM custom layout + likely capability
confound, see below). New plumbing: `omni_image` (transformers 4.57.6) +
`run_signals_omni`/`run_ptrue_omni` (Qwen2.5-Omni Thinker, Qwen2-Audio
`.language_model`), `run_signals_mo`/`run_ptrue_mo` (o2.6 via decode.py;
needs pre-seeding the transformers_modules cache — 4.51's copier misses
image_processing_minicpmv.py), `lopo_matched2` (generalized fail-rate
matching, both directions).

### Headline: LOPO transfer degrades with duplex-ness of the fine-tune

| LOPO (h_prompt probe) | qwen2.5-7b raw | qwen2.5-omni | minicpm-o26 | [o4.5, Qwen3 base] |
|---|---:|---:|---:|---:|
| hard-math | **0.809** | 0.745 | **0.538 (≈chance)** | 0.372 (inverts) |
| hard-knowledge | 0.680 | 0.613 | **0.526 (≈chance)** | 0.61 |
| probe OOF AUC | 0.798 | 0.724 | 0.752 | 0.822 |
| ptrue pre/post | .744/.857 | .749/.703 | .604/.777 | .807/.899 |

Gradient on ONE backbone: raw > omni-streaming > duplex. It is not the audio
modality per se — the closer the fine-tune is to full-duplex training, the
more transferable difficulty info is washed out of h_prompt.

**`lopo_matched2` deconfound (CPU, ~$0):** subsampling raw qwen2.5-7b's LOPO
training pools to o2.6's exact fail rates leaves math at **0.825 [.801,.840]**
(unmatched 0.809; o2.6 actual 0.538); knowledge 0.667 vs o2.6's 0.526. Label
coverage is ruled out on a SECOND backbone — representation damage is now
deconfounded 2-for-2 (Qwen3 pair in 5b, Qwen2.5 pair here).

### Honest complications

1. **Raw baselines vary a lot** (LOPO math: qwen2 .552, glm4 .670, qwen2.5
   .809, mistral .817, qwen3 .961) — the defensible statistic is the
   within-pair Δ, not absolute AUC. "Inversion" (<0.5) remains duplex-only;
   raw models are 5-for-5 non-inverted (mistral fact .445 the one exception).
2. **Qwen2-Audio pair is confounded and uninformative**: the audio FT crushed
   capability itself (math fail .256→.744, knowledge .60→.79), so the failure
   distribution changed under the probe (math LOPO .552→.672, chat .777→.533;
   matched rerun unstable [.34,.61]). Capability collapse ≈ floor effect —
   documented as a negative, not evidence either way. GLM-4-Voice was skipped
   for the same expected confound + adaptation cost.
3. **GLM-4 breaks the p(True) streak**: glm4-9b-chat-hf ptrue_post 0.685 <
   probe 0.798 (was 4-for-4 the other way). "p(True) beats the probe" softens
   to "on most backbones; it depends on self-eval calibration quality."
   Also qwen2.5-omni is the first model with ptrue pre (.749) > post (.703),
   and o2.6 inverts o4.5's trap pattern (pre .604 weak, post .875 strong).

**Step-1 narrative upgrade:** for the paper's target audience (duplex-model
builders) this is a directly actionable caution — hidden-state probes that
work on a raw backbone degrade to chance after duplex fine-tuning (matched-
pair, label-matched evidence on 2 backbones), while verbalized self-eval,
though also dented, stays usable. Gate design conclusion unchanged: behavior-
level signals (p(True)) over representation-level probes for duplex targets.

Phase-5c spend ≈ $25 GPU + $8 API. Project total ≈ **$85**.

---

## Phase 5d — layer × position sweep: destroyed vs relocated ⭐ (2026-07-20)

5c established that the (last layer, last prompt token) probe's transfer
degrades with duplex-ness, but read only that ONE point of the network. Rival
explanations: (a) **destroyed** — duplex training washes difficulty info out
of the model; (b) **relocated** — duplex training repurposes the late-layer /
last-token readout (streaming turn control lives there) and the info survives
elsewhere. User hypothesis going in: (b).

**Method:** prefill-only forward per query (no generation, labels reused from
the 5b/5c judge runs → $0 API), hooks on EVERY decoder layer capturing both
the last-prompt-token hidden and the mean over all prompt positions
(`src/layers.py`, `collect_layers_{hf,omni,mo}`, `layer_sweep_report`,
float16 npz on the volume). Per layer × pooling: OOF AUC + LOPO, calib rows,
same estimators as `xmodel_report`. **Faithfulness check passed:** final-layer
numbers reproduce 5c's generation-time hooks exactly (qwen2.5-7b math .809=.809,
o2.6 .540≈.538, o4.5 .366≈.372, omni .746≈.745, qwen3 .958≈.961).

### Headline: (b) relocated — more precisely, OVERWRITTEN AT THE READOUT

LOPO hard-math, last-token pooling:

| model | best mid-layer | final layer | shape |
|---|---:|---:|---|
| qwen3-8b (raw) | 0.964 (L33/36) | 0.958 | plateau L10→end, no cliff |
| **minicpm-o45 (duplex)** | **0.931 (L22/36)** | **0.366** | cliff in last 4 layers, INVERTS |
| qwen2.5-7b (raw) | 0.893 (L21/28) | 0.809 | mild late dip |
| qwen2.5-omni (streaming) | 0.794 (L16/28) | 0.746 | whole curve depressed, no cliff |
| **minicpm-o26 (duplex)** | **0.822 (L21/28)** | **0.540** | cliff in last ~5 layers |

- **o4.5's famous math inversion (.372) is a readout artifact.** Mid-network,
  the duplex model carries near-raw transferable difficulty info (within-pair
  Δ at best layer −0.03; at final layer −0.59). The collapse is sharply
  localized: L31 .757 → L32 .654 → L33 .492 → L34 .357.
- **o2.6 same signature** (L21 .822 → L27 .540); knowledge likewise (L19 .748
  → L27 .526 ≈ chance).
- **Mean-pooling survives to the end** on the duplex models (o4.5 math ~.80
  at L35; o2.6 .674 at L27) → the damage is position-specific (last token)
  as well as depth-specific (late layers). Both raw models keep last-token
  transfer through the final layer, so the cliff is a fine-tune effect, not
  an architecture generic.
- **Fine-tune types differ in damage profile:** true duplex = severe but
  LOCAL (mid-layers intact, readout cliff/inversion); omni-streaming =
  mild but DIFFUSE (all layers depressed ~.1, no cliff). Also of note: even
  raw backbones peak mid-network, not at the readout (qwen2.5 .893 vs .809).
- in-mix OOF stays high at the duplex readout (o4.5 L35 .835) — consistent
  with 5b's "the readout still supports type recognition + in-distribution
  probing; it's the *transferable self-knowledge* component that's gone".

### Revisions to earlier conclusions

1. 5b's "probe ≈ query-type recognition, math inversion suggests no
   self-knowledge signal" → **the self-knowledge signal exists and is strong
   (.93 LOPO math), the standard readout just can't see it on duplex models.**
2. 5c's "duplex FT washes difficulty info out of h_prompt" → **"duplex FT
   overwrites the late-layer last-token readout; mid-network info is intact."**
   (Consistent with the mechanism: that readout is exactly where a streaming
   head must encode turn-control state.)
3. Gate design for duplex targets: representation probes are BACK on the
   table — read a mid-layer (~60% depth) instead of the final layer. p(True)
   remains the zero-plumbing option; the mid-layer probe is the zero-latency
   option (no extra forward). Two-stage design unchanged otherwise.

Figure: `figures/layer_sweep.png` (5 models × {last,mean} × {LOPO math,
LOPO knowledge, OOF}). Curves: `layer_sweep_{tag}.json` on the volume.

Phase-5d spend ≈ $8 GPU + $0 API. Project total ≈ **$93**.

---

## Phase 5e — mid-layer probe in the RQ2 tradeoff (2026-07-20)

Does 5d convert into a better gate? `midlayer_gate_eval` (CPU, $0): train a
probe on CALIB rows at a mid layer of o4.5 (layer chosen from 5d's calib-only
curves — L22 = calib LOPO-math peak; L18–L30 swept for sensitivity), score
the frozen test split, same curve/area protocol + the same rows as
`ptrue_gate_eval`. Deployed final-layer probe and p(True) re-evaluated
in-run as references.

| signal (test n=240) | area | stage |
|---|---:|---|
| probe_final (deployed cfg) | +0.054 | pre-decode |
| ptrue_pre | +0.059 | pre-decode |
| **midlayer_L22 (last-token)** | **+0.064** | **pre-decode** |
| ptrue_post | +0.068 | post-draft |

- **The mid-layer probe is the best PRE-DECODE signal** — beats both the
  deployed final-layer probe (+0.054→+0.064, closes ~68% of the gap to
  ptrue_post) and ptrue_pre. ptrue_post keeps the overall crown but needs
  the full draft answer first (latency = a whole generation) + an extra
  forward; the mid-layer probe costs literally nothing at prefill.
- Sensitivity: L20–L30 all ≥ +0.056 (L22 best; L18 +0.053) — not knife-edge.
  Mean-pooling uniformly worse in-mix (L22 +0.058) — mid-depth **last-token**
  is the sweet spot, matching 5d.
- Caveat: quantile-threshold transfer is still probe-weak (esc 0.12 realized
  at nominal .15; p(True) transfers rates much better) — a deployed mid-layer
  gate needs Phase-3-style score-scale calibration (C-compression). Area
  (threshold-free) is the headline metric here.
- Note the in-mix area ranking compresses the 5d story: test has the same
  pool mix as calib, where even the damaged readout scores +0.054 via type
  recognition. The mid-layer probe's LOPO robustness (math .93 vs .37) is
  the bigger deployment argument and doesn't show in this table.

**Two-stage gate design, final form:** stage 1 (prefill, free) = mid-layer
probe — now the best zero-latency signal on the duplex target; stage 2
(post-draft, optional) = ptrue_post draft-check. Figure:
`figures/tradeoff_midlayer.png`.

Phase-5e spend ≈ $0. Project total ≈ **$93**.

---

## Phase 6a — audio-input replication ⭐ (2026-07-20)

Does the signal stack survive when the SAME frozen 600-query pool enters
through o4.5's audio channel? Arm A = TTS matched pairs (user-approved:
OpenAI `tts-1`, voice `alloy`, en+zh, 0 truncated; query CONTENT unchanged —
public-benchmark pool, only the modality is synthetic, matching
Spoken-SQuAD/VoiceBench practice). Arm B (SD-QA real-speech validation) still
open. New code: `modal_audio.py` (tag `minicpm-o45-audio`, same file formats
as the text pipeline so `label_hf`/`layer_sweep_report` ran unchanged);
`audio_report` adds paired fail rates + per-layer cross-modal transfer.
Smoke: chat accepts raw 16 kHz numpy; **pure-audio content (no text
instruction) answers the question** (no transcription behavior) → collection
used `content=[audio]` only. Judge: 0 errors on 600.

### Headline 1: the 5d readout cliff is TEXT-INPUT-SPECIFIC

Audio→audio LOPO hard-math (last-token): mid-layers L12–L16 hit **.93–.96**
(peak L16 .960) — the self-knowledge signal is fully present on audio input.
But the final layer **does not invert on audio: L35 = .936** (text input:
L35 = .366). The duplex fine-tune's late-layer/last-token damage — the
central 5d finding — appears only when the prompt is text. On the model's
NATIVE modality the standard readout survives end-to-end. Mechanism note:
the audio chat template may append wrapper text tokens after the audio
(last-prompt-token position ≠ last audio token); un-audited — flagged as the
first follow-up before leaning on this in the paper.

### Headline 2: cross-modal probe transfer works (calibrate on text, deploy on speech)

Per-layer probes trained on TEXT calib rows, scored on AUDIO calib rows
(audio labels), and vice versa:

| region | text→audio | audio→text |
|---|---:|---:|
| early (L00–L08) | .54–.60 | .57–.71 |
| mid (L19–L25) | **.82–.87** (L23 .867) | .73–.86 (L23 .857) |
| late (L28–L35) | .84–.86 | .83–.86 |
| deployed layer L22 | **.855** | .797 |

Early layers are modality-specific; from ~55% depth the representation is
modality-shared and a text-calibrated probe reads audio hiddens at ~.86 AUC.
The cheap deployment recipe (calibrate on text data, run on speech) is
viable at the 5e gate layer.

### Headline 3: the modality tax, and p(True)'s trap introspection collapses

Paired fail rates (same queries, n=600): audio input costs +5 to +15 pts —
easy-chat .207→.307, easy-fact .340→.390, hard-knowledge .480→.627,
hard-math .187→.247, trap 1.00→.96. Audio p(True): pre AUC .786 (text .807),
post .860 (text .899) — mild degradation overall, BUT the 5b trap story does
not survive the modality: **trap mean p_yes_pre = .556 on audio** (text
≈ .055, i.e. escalation score .945 → .444). Asked before answering, the
model no longer knows it doesn't know when the trap question arrives as
speech — near coin-flip. Role reversal vs the text story: on audio the
probe is the robust signal and pre-answer p(True) is the fragile one.

Artifacts: `signals_/layers_/features_/ptrue_minicpm-o45-audio.*`,
`layer_sweep_minicpm-o45-audio.json`, `audio_xmodal_minicpm-o45-audio.json`
on gate-data; wavs in `/data/audio_pool/`. Pipeline (cwd=interactive_paper,
PYTHONUTF8=1): tts_pool → run_audio_signals → label_hf --tag
minicpm-o45-audio → run_audio_ptrue → layer_sweep_report → audio_report.
Gotcha: Modal auto-mounts only the entry module — modal_audio.py's images
add modal_app.py via `add_local_file` or containers die on import.

### ASR audit: perception vs introspection (user challenge, same day)

Could the trap collapse just be the model MIS-HEARING the question (rare
entities + TTS)? Three-arm test (`collect_asr`/`asr_report`): the model
transcribes each wav, then TEXT ptrue_pre runs on its OWN transcript.

| pool | WER mean/med | p_yes text | transcript | audio |
|---|---|---:|---:|---:|
| easy-chat | .077/.000 | .728 | .715 | .800 |
| easy-fact | .040/.000 | .663 | .632 | .860 |
| hard-knowledge | .224/.114 | .498 | .410 | .606 |
| hard-math | .131/.083 | .867 | .771 | .953 |
| **trap** | **.074/.058** | **.055** | **.074** | **.556** |

**Perception hypothesis REFUTED for trap:** (1) trap WER .074 — heard
almost perfectly; (2) on its own transcript p_yes snaps back to .074 ≈ text
.055 — the self-knowledge is THERE and accessible the moment the same heard
content is re-presented as text; (3) the well-heard subset (WER≤.15, n=43)
still collapses (audio p_yes .582), the misheard 7 are actually LOWER
(.394); (4) corr(WER, p_yes_audio) = −.115 ≈ none. → The collapse is in the
audio-input introspection READOUT, not perception. Elegant symmetry with 5d:
probe's late readout is text-fragile, p(True)'s verbalized readout is
audio-fragile — in both cases the knowledge survives, a readout breaks.

Secondary: p_yes_audio > p_yes_text on EVERY pool (+.07 to +.50).
hard-knowledge is the one pool where perception DOES contribute (WER .224,
long MCQ options — consistent with its +14.7pt fail-rate jump). Possible
cheap fix for audio p(True): "repeat-then-judge" (transcribe, then text
ptrue_pre on the transcript) — recovers trap introspection at
~question-length extra decode.

**Log-odds decomposition (`ptrue_shift_report`, $0): the audio shift is NOT
a uniform prior.** Per-query paired Δlogit(p_yes) medians: easy-chat +0.19,
hard-knowledge +0.63, easy-fact +1.19, hard-math +2.50, **trap +4.37**
(global median +1.12; trap excess over global **+3.25**; 30/50 traps flip
from the No side to the Yes side). A single "audio makes it overconfident"
logit bias is refuted — the shift is graded. Revised mechanism hypothesis:
in audio context the VERBALIZED self-assessment regresses to TYPE-level
priors ("chat → easy", "math → I can do math", "factual question → sure"),
and INSTANCE-level evidence (which specific entity) fails to reach the
verbal judgment — while instance-level info demonstrably stays in the
representation (audio LOPO within held-out pools .93+; the probe ranks
instances fine). Explains the full gradient: shift magnitude tracks how
much the correct judgment depends on instance vs type (chat: type suffices;
trap: instance is everything). Discriminating experiments RUN (same day,
`collect_ptrue_arms`/`arms_report`, n=600):

| trap p_yes | text | filler-audio+text (ctx) | audio+text-dup (dup) | audio |
|---|---:|---:|---:|---:|
| | .055 | **.001** | **.034** | .556 |

Δlog-odds vs text: ctx −3.28, dup **−0.44 (full recovery)**, audio +4.37.
**Both arms land on the binding hypothesis:** (ctx) irrelevant audio in
context does NOT inflate p_yes — the context-prior/persona story is refuted
(if anything filler audio depresses p_yes everywhere: fact .663→.268, chat
.728→.528 — audio context biases toward caution, the opposite of
overconfidence); (dup) giving the SAME question as text tokens alongside
the audio fully restores trap introspection (.034 ≈ .055) even though the
audio is still present. Mechanism, final form: **the verbalized
self-assessment performs its instance check (do I know THIS entity?) over
text-token pathways; audio-embedding tokens don't feed it** — while the
instance evidence demonstrably sits in the shared representation (probe
reads it at .93+). Practical fix confirmed twice over: any text
re-presentation of the question (ground-truth dup here, own-transcript in
the ASR audit at .074) restores the signal.

### TTS-template control: the cliff tracks INPUT MODALITY, not speak mode

Mechanism probe (`collect_layers_ttstpl`, tag `minicpm-o45-ttstpl`): same
600 TEXT queries, but prefilled under the speak-mode template
(`use_tts_template=True`, no TTS weights needed). If the 5d cliff came from
"prepare-to-speak" processing, it should move with the template flag.

LOPO hard-math, last-token, late layers:

| input | L31 | L32 | L33 | L34 | L35 | mid peak |
|---|---:|---:|---:|---:|---:|---:|
| text, plain template (5d) | .757 | .654 | .492 | .357 | .366 | .931 (L22) |
| text, TTS template | .689 | .586 | .519 | .468 | **.362** | .949 (L19) |
| audio (6a) | .903 | .901 | .918 | .920 | **.936** | .960 (L16) |

**The cliff is unchanged under the speak-mode template (L35 .362 ≈ .366)**
— it does not track the output-mode flag. Combined with audio's clean L35,
the operative variable is the modality of the CONTEXT (text tokens vs audio
embeddings), not the template or the speaking intent. Mean-pooling again
survives in all three conditions (ttstpl mean L35 math .801). This also
weighs against the trivial "audio's last position is just a template text
token" artifact story: if late layers damaged all text-token processing at
the readout position, the audio template's text wrapper tail would show the
cliff too — it doesn't. Revised mechanism claim: **the duplex fine-tune
re-purposed late-layer last-position processing of text-token contexts
specifically; audio-token contexts (protected by heavy ASR/understanding
training pressure) retain the faithful readout.** Prompt-tail audit stays
open but demoted (this control covers its main scenario).

### Judge validation: gpt-5.5 re-judge (user request, 2026-07-21)

Both o4.5 answer sets re-judged with gpt-5.5 (`rejudge`, JUDGE_MODEL
monkeypatched, max_tokens 8192; 0 errors). **Agreement with gpt-5.4-mini:
text 0.962, audio 0.945** — disagreement concentrated exactly in easy-chat
(no-reference subjective pool: .907 text / .853 audio, flips balanced in
both directions); reference-backed pools ≥ .95, easy-fact and trap at/near
1.0 (audio trap: 5.5 says 1.000 fail vs mini .960). Escalate rates move
≤4pts in any pool. `rescore55` under 5.5 labels: audio ptrue pre .786→.794,
post .860→.864, L22 probe OOF .815→.805 — **every headline number moves
≤.010; no conclusion changes.** Verdict: mini judge validated, keep
gpt-5.4-mini as default (5.5 labels stored in features_gpt55_{tag}.parquet).
Closes the "judge variance" open gap from 5b. Cost ≈ $25.

### Phase-6 streaming feasibility smoke ✅ (2026-07-21)

Headless duplex loop works on our pinned image — NO demo framework needed
(`streaming_smoke` in modal_audio.py, all 5 stages green):

1. **API surface**: remote code ships `streaming_prefill(session_id, msgs,
   omni_mode=True, is_last_chunk, ...)`, `streaming_generate(...,
   teacher_forcing_text='')`, `get_sys_prompt(mode='omni')`,
   `reset_session()`. **`teacher_forcing_text` = the official control point
   for the stall-phrase injection** — the biggest Phase-6 unknown, solved.
2. 14×1s chunks of a TTS wav prefill cleanly (gotcha: tail chunk must be
   zero-padded to 1s — a <0.1s residual under-fills the apm conv (kernel 3)
   and crashes; also pass `is_last_chunk=True` on the final chunk).
3. End-of-turn `streaming_generate` answers the HEARD math question
   correctly, yielding (text, is_final) increments.
4. **Gate insertion point verified**: L22 hook fires once per chunk prefill,
   shape (1, 18, 4096) — ~18 tokens/s of audio; per-chunk mid-layer probe +
   Phase-3 EMA gate is implementable as designed.
5. Same-session follow-up TEXT turn works (the `<result>` relay analog) —
   with a caveat that IS the step-2 problem: injected "expert result: 42"
   conflicting with the model's own $100 calculation → the model pushed back
   and asked to reconcile rather than relaying. Naive injection is not a
   straight relay; the inject prompt (or teacher forcing) must carry
   authority/formatting. First empirical contact with step 2.

Remaining Phase-6 work is now pure design/engineering (no unknowns):
per-chunk probe scores → EMA/hysteresis gate → teacher-forced stall phrase →
expert call → result injection; latency timers per segment.

Open: (a) SD-QA arm B; (b) prompt-tail audit (demoted, see above);
(c) audio latency numbers (audio prefill is longer — the mid-layer
early-exit argument gets stronger).

Phase-6a spend ≈ $45 incl. audits (TTS $1 + 4×H100 collection/ptrue/asr/
ttstpl + judge). Project total ≈ **$138**.

---

## Phase 6b — do the audio findings generalize? (2026-07-21, in progress)

User challenge: o2.6 is same-family — weak generalization evidence. Plan:
o2.6 = within-family robustness; **qwen2.5-omni = the cross-family test**
(finding 2 + the finding-3 duplex-vs-generic discriminator); qwen2-audio =
optional non-duplex control. Moshi/GLM-4-Voice/Kimi documented as blocked
(no text path / architecture). Pre-registered: finding 1's audio side may
stay MiniCPM-scoped (no other duplex family is runnable); finding 3's o2.6
replication has limited power (its TEXT trap introspection was already weak,
p_yes .196 vs o4.5's .055). `modal_audio.py` parametrized (`mtag`);
`audio_report` generalized; omni audio path = new `omni_image_au` +
`Qwen2_5OmniProcessor` (gotchas: needs pillow AND torchvision — the
processor loads image/video processors too; smoke: audio math answered
correctly, hooks 28×3584 OK).

### o2.6 replication (same 600 wavs; collection+label+ptrue+sweep, ~$35)

- **Finding 1 ✅ direction replicates:** audio last-token LOPO math — mid
  peak L22 .761, **final L27 .664** vs text final **.540** (text cliff
  L21 .822→.540). The text-side cliff is absent on audio (mild −.10 dip,
  no approach to chance). Signal overall weaker than o4.5 (.76 vs .96
  peak — weaker model, higher fail rates).
- **Finding 2 ✅ replicates:** cross-modal transfer onset ~L13/28 (~46%
  depth), plateau .74–.80 (peak text→audio L18 .799; o4.5 plateau ~.86).
  Early layers .58–.67. Same shape, lower ceiling.
- **Finding 3 ✅ broad direction, different signature:** audio ptrue_pre
  AUC **.491 ≈ chance** (text .604 was already weak); ptrue_post .805.
  Per-pool p_yes: non-trap pools DROP (chat .712→.585, fact .652→.471,
  math .699→.511) while trap RISES (.196→.345) — everything compresses
  toward ~.5: on the weaker duplex model the audio verbal self-assessment
  loses discrimination entirely, rather than o4.5's trap-specific collapse
  with preserved type ranking. Unified claim: **audio input degrades
  pre-answer verbalized self-assessment on both duplex generations** (o4.5:
  instance component lost; o2.6: all discrimination lost).
- Modality tax o2.6: chat +4.7, fact +4.0, knowledge +14.0, **math +16.0**,
  trap .96→1.00 — larger than o4.5's, consistent with a weaker audio
  front-end.

### qwen2.5-omni cross-family results ⭐ (same 600 wavs, ~$30)

**The finding-3 discriminator came back clean: the omni-streaming control
does NOT collapse — the audio introspection failure is DUPLEX-SPECIFIC.**

| model | FT type | trap p_yes pre text→audio | audio ptrue_pre AUC (text) |
|---|---|---|---|
| minicpm-o45 | duplex | .055 → **.556** collapse | .786 (.807) trap dead |
| minicpm-o26 | duplex | .196 → .345 | **.491 ≈ chance** (.604) |
| **qwen2.5-omni** | omni-streaming | .279 → **.213 INTACT** | .727 (.749) −.02 only |

Omni's audio p_yes actually moves DOWN on every pool except math (chat
.547→.460, fact .570→.457, trap .279→.213) — no overconfidence shift, no
discrimination loss. Same duplex-vs-omni gradient as 5c/5d. **Unified paper
claim now fully supported: duplex fine-tuning damages self-knowledge
READOUTS — the probe's late-layer readout in its text blind spot (5d) and
the verbalized readout in its audio blind spot (6a) — while the omni
control keeps both and the mid-layer signal survives everywhere.**
(Omni quirk persists: ptrue_post < pre on audio too, .599 < .727 — it was
already the only pre>post model on text.)

- **Finding 2 replicates cross-family**: transfer onset ~L06-08/28 (~25%
  depth — EARLIER than MiniCPM's ~50%, consistent with omni's tighter
  audio-text alignment and no duplex damage), plateau L18–L27 ≈ .80–.83
  both directions (L23 text→audio .826).
- **Finding-1 consistency**: omni audio sweep has no cliff (last-token math
  final .757 ≈ text .746; peaks .82 mid) — matches "no duplex FT → no
  cliff in either modality"; its text side was diffuse-depressed, audio
  similar.
- Modality tax omni: chat +14.0, knowledge +10.0, math +6.0, fact +5.0,
  trap 0 (.98=.98).

**Phase 6b verdict:** finding 2 = MiniCPM×2 + omni (cross-family) ✅;
finding 3 = duplex-specific (two duplex generations collapse, omni control
intact) ✅ — now the same shape as finding 1's raw>omni>duplex gradient;
finding 1 audio-side = MiniCPM-scoped as pre-registered (no other runnable
duplex family).

---

## Phase 6c — ablation vs o4.5's own thinking + latency ⭐ (2026-07-22)

User requirement: prove the solution ≥ o4.5's built-in capability
(`enable_thinking`), and benchmark latency end-to-end per request.

### Component latency bench (`latency_bench`, 50 q × text/audio, CUDA-synced,
3-warmup excluded, per-query interleaved; P50/P95 ms)

| config | text | audio |
|---|---:|---:|
| **L22 truncated decision (the gate)** | **20 / 25** | **45 / 104** |
| TTFT (full prefill + 1st token) | 36 / 47 | 68 / 144 |
| ptrue_pre (short prefill + 1 tok) | 39 / 63 | 67 / 169 |
| full answer (= ptrue_post entry fee) | 1855 / 7013 | 3507 / 7391 |

**The gate decides BEFORE the first token** (20 < 36 ms text; 45 < 68 ms
audio) — pre-TTFT escalation is real, and all pre-decode signals sit far
inside the 200–300 ms voice turn-taking budget. Audio latency tax ≈ 2×.
Decode 36.3 tok/s (matches Phase 0). Closes 6a open item (c).

### Thinking ablation (`collect_think` 600 q, enable_thinking=True,
max 2048 tok, gpt-5.4-mini judge 0 errors)

Per-pool: thinking helps math (fail .187→.127) and knowledge (.480→.427),
does nothing on trap (.98), and HURTS easy-chat (.207→.233). Overhead
P50 +8 to +24 s per query (P95 up to +63 s). think_used .57–.92 (the
hybrid mode skips thinking on some chat).

### End-to-end policy table (`e2e_latency_report`; test n=240; fast
latencies = per-pool bench medians [n_forward store is k-capped, unusable];
think = per-query measured; cloud = per-query expert_latency from Phase 5)

| policy | acc | lat mean | P50 | P95 |
|---|---:|---:|---:|---:|
| fast-only | .588 | 3.0 | 3.5 | 4.2 |
| **all-THINK (o4.5's own)** | .637 | **22.4** | 17.2 | 60.1 |
| gated-think @.33 | .613 | 12.1 | 3.6 | 47.6 |
| **gated-cloud @.15** | **.688** | **5.3** | 3.6 | 10.1 |
| **gated-cloud @.33** | **.787** | **6.5** | 3.6 | 20.8 |
| gated-cloud @.50 | .858 | 7.0 | 3.6 | 24.4 |

**Verdict: proven, with domination.** gated-cloud beats all-THINK on BOTH
axes at every escalation rate — already at 15% escalation: +5.1 acc pts at
4.2× lower mean latency; at 33%: +15.0 pts at 3.4× lower. gated-think (the
self-escalation tier) is weak (.613): the gate predominantly flags
knowledge/trap failures, which thinking cannot fix (thinking's gains are
execution/math) — a clean mechanistic reason why external escalation is
necessary, not just better. Thinking-tier idea documented and closed.

Phase-6c spend ≈ $45 (think 600 incl. long generations + judge + bench).

---

## Phase 6d — Freeze-Omni: the frozen-backbone control (2026-07-22)

Freeze-Omni (arXiv 2411.00774; speech encoder + adapter → FROZEN
Qwen2-7B-Instruct + state-head duplex) separates "duplex operation" from
"backbone weight updates". Integration: `modal_freeze.py` (their pins torch
2.2/transformers 4.45.2 — audioLLM manipulates legacy tuple KV; ptrue =
text tokens + chat_template['suffix'] appended to the audio KV via
DynamicCache.from_legacy_cache — without the suffix close the Yes/No mass
is 0.00; audioEncoderProcessor vendored to avoid the flask import chain;
text side = the `qwen2-7b` tag verbatim, same weights).

### Primary readout test: CONFOUNDED by capability collapse (pre-registered)

Audio fail rates explode vs the same weights on text: math .240→**.713**,
chat .333→.720, knowledge .600→.927 (fact +.04, trap .96→.98 only ones
stable). Audio probe: OOF peaks ~.70 (type recognition), **LOPO math
.44–.55 ≈ chance at every layer** — but with labels this collapsed the
readout question is unanswerable here (same verdict class as qwen2-audio
in 5c: documented negative).

### Two informative residues

1. **Finding-3 control still holds**: trap p_yes text .165 → audio **.112**
   — the verbal knows-it-doesn't-know SURVIVES audio on the frozen
   backbone (moves toward honesty, like omni; opposite of both duplex
   models). Third non-duplex model without the collapse. (Caveat: with
   audio capability collapsed, "No" is also the calibrated easy answer —
   pre AUC only .612, post .842.)
2. **Cross-modal transfer is ≈ DEAD on identical weights**: text→audio
   .52–.60, audio→text .34–.54 (inverts at L20) — versus .80–.86 on every
   end-to-end-trained model. **The modality-shared mid-layer core (finding
   2) is not free: it is CREATED by training the backbone on the modality.**
   An adapter alone aligns well enough to converse, but audio-context
   hiddens live off the text manifold — probes don't transfer, and task
   capability craters.

Combined three-way story: end-to-end multimodal training BUILDS the shared
semantic core (Freeze-Omni lacks it); duplex-style training additionally
DAMAGES the readouts (MiniCPM×2); omni-streaming gets both right (core
present, readouts intact). The paper's mechanism section now has all four
quadrants populated.

Phase-6d spend ≈ $45 (download + smoke ×4 + 600 chunked-streaming
collection + judge + qwen2-7b text layers). Project total ≈ **$290**.

---

## Phase 7a — collaborator follow-ups: fork profiling + escalation overlap (2026-07-24)

Meeting feedback (Jisen, Changyi) triaged into: (1) fork-at-layer-k
profiling — Jisen's "branch at ~L10, probe in parallel, both finish
together"; (2) result-feedback overlap — Jisen's "traditional routing
doesn't need to feed results back to a talker; can the cloud result be
ready by the next turn?"; (3) qwen3-omni probe test (Changyi) — assessed,
deferred (below); (4) routing-literature positioning (Changyi) — written
into TECHNICAL_REPORT §7. Items 1–2 run same-day; step-2 injection design
is the next real phase.

### Fork profiling: the decision is ready at 57% of prefill — at the quality peak

`prefill_timing` (H100, 25 q × text/audio × all 36 layers, truncated-forward
= same methodology as 6c's timed_l22, CUDA-synced, 2-query warmup) +
`fork_report` (joins 5d layer-sweep quality; figures/fork_pareto.png;
prefill_timing.parquet on volume). Architecture note first: the "fork" is
free by construction — the probe is one 4096-d dot product (~µs) on a
hidden state the forward computes anyway; nothing waits. The real question
is WHERE the signal is sufficient:

| layer | text ms (% of prefill+1tok) | text OOF | audio ms (%) | audio OOF |
|---|---:|---:|---:|---:|
| L11 (Jisen's ~L10) | 8.1 (35%) | .794 | 24.1 (48%) | .760 |
| L16 | 10.4 (45%) | .797 | 26.4 (53%) | .780 |
| **L22 (deployed)** | **13.2 (57%)** | **.866 = peak** | 30.6 (61%) | .815 |
| L23 | 13.7 (59%) | .850 | 29.3 (59%) | **.822 = peak** |
| L35 (final) | 19.3 (84%) | .835 | 36.6 (73%) | .813 |
| prefill+1tok | 23.2 | — | 50.1 | — |

- **L22 is simultaneously the earliest-useful AND the best layer** (in-mix
  OOF .866 beats the final layer's .835; 5d showed the same for LOPO .93 vs
  .37) — deciding at 57% of prefill costs nothing in quality. Escalation
  can therefore LAUNCH while the last 43% of prefill + decode still runs.
- L10-ish is too early on this model: .794 (−.07 from peak) and early
  layers are modality-specific (6a: cross-modal transfer dead before ~50%
  depth) — the fork belongs at ~55–60% depth, not 30%.
- Audio pays the encoder front-end (~17 ms: audio L1 21.4 ms vs text L1
  3.8 ms) at EVERY fork depth; L1→L22 then adds only ~9 ms. (Both arms'
  L0 rows are inflated by per-query first-call overhead — the L0 point in
  fork_pareto.png is an artifact, ignore it.)
- Absolute times here (23 ms prefill) are lower than 6c's chat-path bench
  (TTFT 36 ms) — different call overhead; the robust statistic is the
  ratio. Both agree the decision predates the first output token.

### Escalation overlap: Jisen's "result by next turn" — yes for the mix, not for escalated traps

`overlap_report` (CPU $0): per-query Phase-5 gpt-5.5 latencies (P50 3.0 s /
P95 24.4 s) vs pool-matched measured local answer durations (6c bench);
timeline = gate fires at l22 → cloud call in parallel with local decode.
P(expert result ready before the talker finishes + slack), figures/overlap.png:

| pool | text+0s | text+2s | text+5s | audio+0s | audio+2s | audio+5s |
|---|---:|---:|---:|---:|---:|---:|
| easy-chat | .43 | .66 | .86 | .77 | .86 | .92 |
| easy-fact | .02 | .62 | .94 | .56 | .87 | .95 |
| hard-knowledge | .48 | .61 | .74 | .44 | .60 | .73 |
| hard-math | .66 | .91 | .93 | .71 | .91 | .93 |
| trap | **.00** | .00 | .26 | .02 | .09 | .38 |
| ALL (test mix) | .40 | .65 | .81 | .58 | .75 | .84 |
| **escalated @.33** | **.20** | **.39** | **.60** | **.31** | **.47** | **.63** |

Stall needed after the local answer ends (escalated @.33): text P50 3.1 s /
P90 28.9 s; audio P50 1.9 s / P90 26.8 s (@.50: 1.8/0.3 s P50).

- **The overlap story works for the traffic mix (40–58%) but the gate
  selects against it**: escalated queries skew trap/knowledge, whose local
  answers are SHORT (trap overlap ≈ 0 — the talker finishes "…is X" in ~1 s
  while gpt-5.5 thinks for 3+). Math is the good case (.66–.91): long local
  answers buy the cloud time.
- Design consequence: same-turn delivery needs only **P50 one stall
  sentence (~2–3 s)**; the P90 tail (~27 s) is gpt-5.5's own reasoning
  latency, not our plumbing — argues for a fast-expert tier and/or streamed
  partial results in step 2. Audio deployment is structurally friendlier
  (utterances are ~2× longer).
- Caveats: expert latency measured Modal-us-east→OpenAI (includes RTT);
  local durations at max_new_tokens=256 (mild underestimate for the
  longest answers); bench n=10/pool → cross-product estimate.
- Paper figure: `figures/timeline_scenarios.{png,pdf}`
  (`figures/timeline_scenarios.py`, runs locally on the pulled parquets'
  medians) — panel (a) ms-scale fork (decision at 20 ms < TTFT 36 ms),
  panel (b) audio-channel occupancy: pre-answer routing (2.7 s dead air)
  vs gated hard-math (full overlap) / easy-fact (1.8 s stall) / trap
  (7.4 s gap, unbridgeable — gpt-5.5 trap P50 8.2 s is the slowest pool
  while trap drafts are the shortest, the structural worst case).

### Changyi's qwen3-omni proposal — disposition

The logic ("if a non-duplex omni's last layers are probeable, the damage is
duplex not modality") is exactly the already-run qwen2.5-omni control:
no cliff in either modality (5d text final .746 ≈ audio .757, 6b), plus 6a's
within-model converse (o4.5 audio L35 .936 vs text .366 — same weights, no
cliff on the native modality) and the ttstpl control. Qwen3-Omni-30B-A3B
(turn-based streaming per model card, premise correct) would add a
same-generation-as-o4.5 omni control (n=2), ~$25 + MoE/thinker integration
(transformers from source) — worthwhile as reinforcement, deferred by
priority call 2026-07-24. Stronger version of the same test: a NEW
open-weight full-duplex model as a pre-registered prediction ("it should
show the text-input late-layer cliff") — watch Qwen3.5-Omni (2026-03,
weights unconfirmed) and DuplexOmni (arXiv 2606.09186).

Phase-7a spend ≈ $2. Project total ≈ **$292**.

---

## Phase 7b — SD-QA real-speech validation ⭐ arm B closes 6a (2026-07-25)

The promised arm B: 200 REAL human recordings (VoiceBench sd-qa, USA split;
10 more dialect splits available for future robustness work), NQ-style
factoid questions with reference answers. Same questions run BOTH as typed
text and as the human recording (matched pairs); judged by gpt-5.4-mini
(0 errors both arms). New code in modal_audio.py: build_sdqa /
collect_sdqa / collect_ptrue_sdqa / sdqa_report. Note: the new OpenAI key
was verified first (openai_ping, pinned us-east per the post-suspension
policy) — the old key died with the suspension.

**Verdict: the audio-side findings are NOT TTS artifacts — all three
replicate on real speech.**

1. **Modality tax replicates:** fail rate .400 text → .450 audio (+.05,
   consistent with fact/knowledge-type content).
2. **Audio overconfidence replicates:** paired p_yes_pre shift +.089
   (audio > text on 62% of queries); on the failure subset, mean p_yes_pre
   .415 text vs .581 audio — asked before answering, real speech makes the
   model claim ability it doesn't have, exactly as with TTS.
3. **Layer structure replicates** (frozen-pool probes → sdqa = transfer
   across BOTH content and modality, a strictly harder test than 6a's
   matched pairs):
   - early layers dead (≤L11 all arms ~.42–.55, au→au even sub-.5),
     rising sharply after ~50% depth — finding 2's boundary;
   - audio late layers stay usable through L35 (au→sdqa-audio .74–.77 at
     L28–L35) — no audio-side collapse on real speech;
   - cross-modal deploy recipe (text-calib → real-speech audio): band
     .76–.80 at L22–L26 (peak .800 @ L25), then declining toward L35
     (.714) — mid-layer is where to read, same design rule as 6a
     (magnitudes below 6a's .86 as expected: content is also OOD here).
4. p(True) on real speech: pre AUC .769 text / .771 audio (discrimination
   preserved); post .813 text / .743 audio (post degrades on audio, same
   direction as 6a).

Honest scope note: the TEXT-side math cliff was not re-testable here
(sd-qa has no math; tx→sdqa-text late layers hold .79–.80, consistent with
the cliff being sharpest on math per 5d) — but the cliff claim never
involved synthetic audio in the first place, so it needed no arm-B
protection. My pre-registered criterion "text side collapses" was
miscalibrated for a factoid-only pool; the criteria that matter (audio
late-layer usability, mid-layer cross-modal band, overconfidence) all pass.

Phase-7b spend ≈ $18 (GPU collection ~$15 + judge ~$2). **Part 1 (the gate)
is now closed:** design final (mid-layer probe stage 1 + p_post stage 2),
mechanism characterized to the observational limit, latency measured (6c),
thinking ablated (6c), cross-family checked (6b/6d), real-speech validated
(7b).

---

## Phase 8 — chat_gated: the gate goes live ⭐ Part-2 milestone 1 (2026-07-25)

First end-to-end LIVE duplex loop (`modal_stream.py`): stream the wav in 1s
chunks → per-chunk L22 probe → EMA gate → mid-stream fire → expert races in
a background thread while the model keeps listening → canned stall phrase →
authority-framed `<result>` injection in the same session → relay. n=25
test queries (5/pool), balanced tier. Artifact: `midlayer_gate_audio.json`
(L22 audio-calib probe, OOF AUC .843, quantile thresholds — new, distinct
from the final-layer `gate_config.json`).

### What worked (the demo exists)

- **Mid-stream firing is real**: hard-knowledge 5/5 escalated at 12–66% of
  the audio (mean 40%) — the gate interrupts WHILE the user is talking,
  exactly as designed. easy/math 0/15 escalated (good restraint).
- **Stall via assistant-role prefill: P50 24 ms.** (Engineering find #1:
  `teacher_forcing_text` KEEPS GENERATING after the forced text — the model
  answered instead of stalling. Fix: prefill the stall as an
  assistant-role message = canned-filler semantics, ~24 ms, then
  `tts_filler` plays it in deployment.)
- **Relay works**: inject → spoken-form answer in P50 642 ms, expert
  answers relayed faithfully (spot-checks correct).
- **Real-time overlap delivers 5/6**: replay feeds audio faster than real
  time, so raw `expert_wait` is a replay artifact (engineering find #2);
  simulating the true timeline (remaining listening + 3 s stall):
  q0150 4.1s expert vs 21.5s budget ✓, q0152 15.1 vs 28.9 ✓, q0160 4.2 vs
  44.5 ✓, q0163 5.2 vs 16.0 ✓, q0164 12.9 vs 55.6 ✓ — **the expert answer
  is ready before the model needs to speak in 5/6 cases**.
- The 6th is the predicted failure shape (7a live-confirmed): **trap =
  short question (4.1s) × late fire (80%) × slow expert (16.4s reasoning
  on an obscure entity) → 12.6s dead air after the stall.** The step-2
  design problem in one row.

### Honest gaps (next milestone's worklist)

1. **Chunk-score calibration** (engineering find #3): offline thresholds
   are quantiles of FULL-prefill scores; live per-chunk scores are noisier
   and the zero-padded tail chunk dilutes. Shipped k=1/raw-score as the
   milestone fix; trap under-fires (1/5 vs offline expectation ~5/5) —
   needs proper chunk-level threshold calibration (collect chunk-score
   distributions on calib, re-quantile).
2. Heard-accuracy small-n: overall .360 (esc .500 n=6, local .316 n=19;
   includes 4 missed traps ≈ all wrong + judged easy-chat). Not a
   headline number at n=25 — the RQ2-style curve needs the calibrated
   gate + larger n.
3. Milestone shortcuts to lift later: expert query = pool text form (live
   ASR distill next), text-only output (no TTS), no barge-in, stall
   assumed 3 s spoken.

Fixes for the trap gap to evaluate next: progressive filler ("let me look
that up… one moment"), expert effort=low for entity lookups (Phase-5
showed trap needs retrieval not reasoning), or partial-answer streaming.

### Chunk-threshold calibration cycle (same day)

Step-1 fix for find #3: ran the EXACT live streaming procedure on all 360
calib wavs (`collect_chunk_scores`), calibrated the firing statistic on the
resulting per-chunk sequences (`calibrate_chunk_gate`), re-ran the same 25
live sessions.

- **Live chunk signal is weaker than offline: best statistic (mean_top2)
  AUC .764 vs offline full-prefill .843** — streaming 1s-boundary reads
  lose ~.08 AUC. (max .759, max_unpadded .757, last_unpadded .637.)
- Re-run @balanced (thr .52): escalation 24%→40% (target 30%; overshoot
  partly a semantics mismatch — thresholds quantiled on mean_top2 but the
  gate fires on single-chunk max). Knowledge 5/5 fire@27% (earlier),
  math 1/5, fact 3/5 (small-n overshoot vs predicted .12).
- **Heard accuracy 0.360 → 0.560** (escalated .700, local .467) — the
  calibrated gate converts directly into user-heard accuracy.
- **Trap still 1/5.** Calibration was necessary but not sufficient:
  predicted trap fire even on calib is only .40 — the live per-chunk trap
  signal is intrinsically weak (short audio, entity lands at the end next
  to the padded tail; offline's strong trap catch came from the full-
  prefill read after the WHOLE question). Trap under-fire is now a
  signal-placement problem, not a threshold problem.

Next design decision (not auto-iterated): an end-of-turn second read
(pre-TTFT verdict check approximating the offline h_prompt position)
vs unpadded-tail scoring vs accepting trap → p(True)-style stage-2.
Cycle spend ≈ $12.

Phase-8 spend ≈ $27 incl. calibration cycle. Project total ≈ **$353**.

---

## Phase 8b — conflict injection: the pushback problem quantified ⭐ (2026-07-30)

Grilling-session decisions (2026-07-30, all recorded in Notion "Full duplex
issues" + todo.tex): Part-2 acceptance = paper-grade live tradeoff curve;
end-of-turn read becomes the deciding gate with mid-stream chunk gate demoted
to speculative prefetch (+ veto); dead-air fix = deadline-aware expert effort
selection; expert query = talker self-transcript; prefetch = query-level
speculative execution (completeness curve = acceptance rate); offline
query-feature router baseline added to the worklist. This experiment is
worklist item ① — it had to run BEFORE the full live sweep so relay
non-compliance can't contaminate the live curve.

Design (`plan_conflict` / `conflict_sessions` / `conflict_report` in
modal_stream.py): 172 live audio sessions on 270 eligible calib queries
(ref + wav + audio label; easy-chat excluded — no references). Conflicting
"expert answers" fabricated by within-pool derangement of reference answers
(deterministic, zero LLM generation). Four framings: F0 neutral, F1 = the
deployed RELAY_TMPL, F2 strong-override, F3 assistant-seeding
(teacher_forcing_text speaks "the answer is: X" and continues). Judge
gpt-5.4-mini, structured {comply | pushback | lip_service | other}.

**Results:**

1. **The deployed channel is clean: model-wrong × correct-inject (component
   C, the actual live workload) comply = 1.00 (n=24).** The live curve is
   not contaminated by relay refusal.
2. **Layer-0 hypothesis CONFIRMED** (component A, conflict × F1 across the
   gate-score range): comply rises .42 → .67 → .75 → .75 across score
   quartiles; resistance (pushback+lip) is .25 in the most-confident
   quartile vs .08 elsewhere. The same signal that triggers escalation
   predicts relay compliance — "a model that knows it doesn't know is
   willing to listen." In the high-score half, hard-knowledge (the dominant
   escalation type) complies 1.00 (n=13).
3. **Framing ladder on the worst case** (confident-correct × conflict,
   component B): F0 comply .00 (other .54) / F1 .53 / F2 .79 (zero
   pushback+lip) / **F3 seeding BACKFIRES: comply .25, lip_service .62** —
   the model speaks the seeded answer then walks it back in its own
   continuation. Ladder layer 2 (assistant-prefill seeding) is empirically
   REFUTED for conflicting content: the model treats its own forced speech
   as revisable.
4. **Silent-override on math**: "other" concentrates in hard-math — the
   model neither relays nor disputes; it re-derives its own solution and
   states its own number (F1 conflict: math comply .57 / silent .26 vs
   non-math .66 / .20). Bare numeric answers carry no authority against
   the model's own computation chain. Harmless with correct experts
   (agreement is indistinguishable from compliance) but rules out
   "verbatim number relay" claims on math.
5. **Swallow rate** (model-wrong × conflict, D, n=11): comply .64,
   pushback .00 — an unconfident model has no internal resistance to wrong
   expert answers. Expert quality is the accuracy ceiling; now quantified.

**Decision for the full sweep: keep F1_authority.** The deployed regime
(C-like) is already perfect with F1; F2 only helps the rare
wrong-expert × confident-model corner and would plausibly raise the D
swallow rate (untested at F2 — deliberate); F3 rejected. F2 stays on the
shelf as the knob if live relay non-compliance appears.

Phase-8b spend ≈ $6 (GPU ~$5 + judge ~$1). Project total ≈ **$359**.

---

## Phase 8c — end-of-turn read + completeness curve: the trap fix works, prefetch dies ⭐ (2026-07-30)

Worklist ②/③a (collect_eot_scores / eot_gate_report / completeness_curve in
modal_stream.py; 360 calib replays ~$3 + 947 mini-judge calls <$1).

**1. The end-of-turn read RECOVERS AND EXCEEDS the offline discrimination:
AUC .887** (offline full-prefill reference .843; live chunk stats
max .759 / mean_top2 .764). Implementation: after the last audio chunk,
prefill an assistant turn with a single space — the L22 hook then reads the
assistant-start position, the streaming analogue of the offline h_prompt
read. (Empty-string assistant content raises; single space works.)

**2. Trap is saved: fire@balanced .90** (fail rate .97; was .20 live /
.40 predicted with chunk thresholds). Full per-pool fire@balanced:
trap .90, knowledge .41 (fail .56), math .22 (fail .24), easy-chat .16,
easy-fact .17. eot quantile thresholds written to midlayer_gate_audio.json
(conservative/balanced/aggressive = .846/.713/.555).

**3. Completeness curve kills partial-question prefetch** (the pre-agreed
wait-k decision rule): acceptance ("would an expert's answer to the partial
transcript answer the full question?") is .07 / .19 / .51 at 25/50/75% of
the words. At the chunk gate's typical knowledge fire point (27–40% of
audio) acceptance is ~.2–.4 → 6–8 of every 10 speculative expert calls
would be wasted. Per pool @25/50/75: knowledge .22/.41/.72, math
.02/.10/.61, trap .00/.10/.27, easy-fact .00/.17/.35, easy-chat
.00/.02/.24. **Decision (auto-executed per the agreed rule): drop
partial-question speculative escalation; the expert starts at end of
turn.** The prefetch threshold scan agrees from the signal side: 80%
early-fire recall costs 50% prefetch rate with 25% waste.

**4. Coherence finding (paper-worthy): trap's acceptance is the worst
(.00/.10/.27) — these questions back-load their semantic core, which is
the SAME underlying property that made the mid-stream probe under-fire on
trap** (Phase 8 signal-placement diagnosis). Two independent measurements
— hidden-state discrimination and semantic content — agree that the
information arrives at the end. One property, two manifestations.

System consequences: live loop v2 simplifies (no speculative expert calls
→ probation pressure drops; end-of-turn read decides pre-TTFT; on fire:
self-transcribe (~1s, overlapped with the stall TTS) → expert). Dead-air
now rests entirely on deadline-aware effort selection (worklist ④ = the
critical path). Milestone-1's "5/6 overlap" is retired as oracle-inflated.

Phase-8c spend ≈ $4. Project total ≈ **$363**.

---

## Phase 8d — effort × query-form characterization: effort is free, ASR is the leak ⭐ (2026-07-30)

Worklist ④+③b (`effort_characterize` / `effort_report`). Subset = the 72
test queries above the balanced offline-score quantile (the gate's
escalation population). Arms: transcript×{low, medium} + gold×low (~216
gpt-5.5 calls, concurrency 3, cached, spaced batches); gold×medium = the
frozen Phase-5 answers (free). Judge gpt-5.4-mini. Transcripts = the
talker's own 6a ASR outputs (deployed query form, decision 4).

**Four-column decomposition (acc, n=72):**

| pool | gold-med | gold-low | tr-med | tr-low | ASR tax | effort tax |
|---|---|---|---|---|---|---|
| easy-chat (5) | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| easy-fact (8) | .88 | .88 | .88 | .88 | 0 | 0 |
| hard-knowledge (32) | .84 | .84 | .53 | .56 | **+.31** | ≈0 |
| hard-math (10) | 1.00 | **1.00** | .30 | .30 | **+.70** | **0** |
| trap (17) | .71 | .59 | .53 | .53 | +.18 | 0 (gold −.12) |
| ALL | .85 | .82 | .57 | .58 | **+.28** | **≈0** |

1. **The effort tax is ≈ zero everywhere — including gold math (low 1.00 =
   med 1.00).** GSM8K-class problems are solved by gpt-5.5 at any effort;
   the only hint of effort sensitivity is trap on gold (.71→.59, ≈2
   queries, n=17 — noise-level). The grilling's effort-sensitivity
   prediction ("math = execution-bound, effort helps") is REFUTED at
   expert scale on this pool mix: the expert is so far above the task that
   effort never binds. Combined with 6c (talker's own thinking also can't
   fix escalated failures), the unified statement: **on gate-escalated
   traffic, extra compute buys nothing at either scale — the failures are
   knowledge-bound.**
2. **Latency, however, doubles at medium: transcript P50 5.8→9.1s, P95
   23.9→78.8s.** Medium effort = 2–3× the wait for zero accuracy.
3. **Policy decision (auto, per the measurement): fixed-LOW effort for all
   escalations.** Dead-air replay (transcribe 1s + stall 3s cover):
   fixed-low P50 3.8s / P90 16.8s / silent>8s 29%, vs fixed-med 7.1 /
   41.2 / 49%. Score-conditioned effort is REJECTED as unnecessary
   complexity (same acc as fixed-low, worse latency). The deadline-aware
   design collapses to the simplest possible policy — which is what
   measuring the dimensions was for.
4. **The ASR-distill tax is the new dominant leak: −.28 overall, −.70 on
   math, −.31 on knowledge.** The talker's own transcription (WER .074
   trap / .224 knowledge per 6a) halves expert accuracy on the escalated
   subset (.85→.57): small slips are fatal for math (a lost number kills
   the problem; +.70 tax despite decent-looking WER) and costly for
   knowledge entities. **WER is a poor predictor of downstream damage.**
   This caps the live curve's escalated-arm accuracy well below the
   Phase-5 oracle-text numbers — the honest deployment ceiling.

**Post-hoc scope correction (failure inspection):** the failing math rows
are MATH-500 formula problems — the TTS reads LaTeX aloud and the
round-trip destroys the symbolic structure (`31/11111` → `31/111`;
`x^4+5x^3+9x^2` → "x cubed x four plus five x squared";
`x^10+(13x-1)^10` → `(x-10)^10+13(x-10)^5`). GSM8K-style word problems
transcribe cleanly (see measure_asr_timing samples). So the math +.70 is
largely a **TTS-of-LaTeX artifact — nobody speaks LaTeX at a voice
assistant — and overstates deployment harm; the deployment-real ASR tax is
the knowledge-entity −.31** (and trap −.18). Paper scope note: the speech
channel is evaluated on speakable content; formula-symbolic math is out of
scope for the audio arm (it already carries the text-side cliff story).
Live ASR timing measured: P50 2.2s / P95 3.4s (math-length utterances,
conservative) — replaces the 1s placeholder in dead-air accounting.

**ASR-tax attribution COMPLETE (user then scoped ASR out of the paper —
"channel property, not our contribution"; user decision 2026-07-30 late).**
Five-arm table (acc, n=72, expert low): gold .82 / self-1best .58 /
self+robust-prompt .58 / self-kbest-GER .56 / **Whisper .62**. Verdicts:
(1) both self-rescues NEGATIVE — the robust prompt changes nothing, and
k-best GER trades a small trap gain (.53→.59) for a knowledge LOSS
(.56→.44: five divergent long transcripts confuse complex MCQs more than
they repair); (2) external ASR recovers a modest fraction (knowledge
.56→.62, trap →.59 = gold-low level); (3) math is .30 in EVERY
transcript arm — the TTS-of-LaTeX artifact, four-way consistent; (4) the
knowledge gap Whisper can't close (.62 vs .84) points upstream: the TTS
pronunciation of obscure entities is itself lossy — channel-inherent.
Paper treatment: dual-view curve (end-to-end honest heard-acc + a
channel-controlled expert-inject view from frozen gold answers, $0) with
the difference labeled as channel cost; ASR robustness cited as
orthogonal work. Consequence: the always-live arm is CANCELLED (its
~168 calls unneeded — the channel-controlled ceiling and random line
synthesize from frozen data). Also measured today and now moot as a
direction: expert reasoning effort ≈ zero accuracy effect at either
query form — escalated failures are knowledge-bound at both scales.

**Mitigation (a) tested same day (user-approved, 72 calls): NEGATIVE.**
A recognition-errors warning prepended to the transcript
(`ROBUST_PREFIX`, form=robust) changes nothing: ALL .58 = plain .58
(knowledge .59 vs .56, trap .47 vs .53, math .30 unchanged — all within
±1 query). Interpretation: the damaging errors are entity substitutions
that leave an internally-coherent transcript — no textual residue to
correct from. **The information is destroyed at transcription, not
noised; the ASR tax is not prompt-fixable.** Remaining options: (b)
better transcription decoding, (c) an external-ASR decomposition arm
(Whisper transcripts × low, ~72 calls — would quantify how much of the
tax is MiniCPM's ASR quality vs inherent audio ambiguity), or (d) accept
and document as the honest cascaded-escalation ceiling. User to pick;
the sweep currently ships the honest self-transcript pipeline.

Phase-8d spend ≈ $8 (expert ~$6 + judge ~$2). Project total ≈ **$371**.

### 8e — live sweep day 1: floor + conservative (2026-07-30)

v2 smoke n=25 passed (overall .440; the 512-token cap fix lifted local
.294→.412 — the default streaming cap was truncating long math answers,
our artifact not the model's stopping). Latencies: eot read 22ms, stall
26ms, relay 678ms. Local answers after the probe turn are clean (no
history contamination). Then the first two full 240-query live arms:

| arm | esc | heard-acc | escalated | local |
|---|---|---|---|---|
| **never** (live floor) | 0% | **.375** | — | .375 |
| **conservative** | 14% | **.450** | .471 | .447 |

- The honest live floor is .375 — well below the offline chat-mode
  small-only .588 (text) because it stacks the audio modality tax AND the
  streaming-answer mode (speech-style short answers that plan aloud then
  stop; real deployment behavior, now measured at n=240). This is exactly
  why the never endpoint had to be run live rather than replay-synthesized.
- conservative (+14% escalation) buys +7.5 pts over the floor.
- Two engineering incidents burned ~$25 GPU, both documented in code:
  cross-app hydration (gen_app entrypoint required) and concurrent
  same-file volume appends (last-committer-wins; per-shard trace files
  now, matching the collect_* convention).

**Same-day continuation (user said keep going): all four arms landed.**

| tier | esc | heard-acc | escalated arm | local arm |
|---|---|---|---|---|
| never (floor) | 0% | **.400** | — | .400 |
| conservative | 14% | **.446** | .441 | .447 |
| balanced | 35% | **.529** | .536 | .526 |
| aggressive | 55% | **.633** | .621 | .648 |

- The live curve rises monotonically, +.23 accuracy over 55% escalation.
  Against the offline text curve (.588 floor → .679/.779/.833 at
  14/33/53%) the live curve sits ≈.19–.20 lower at every operating point
  — the measured price of real streaming deployment, decomposable with
  the day's experiments: audio modality tax + streaming-answer tax
  (floor .588→.400) and ASR-distill tax on the escalated arm
  (expert .85→.57 on the escalation population).
- Judge-variance note: never was judged .375 in the first (partial)
  report run and .400 in the final run — gated_report re-judges each
  invocation (±.02–.03 single-judge noise). The persisted
  gated_traces_v2.parquet freezes the final labels; bootstrap/DeLong must
  run on THAT, not on fresh judgings.
- Still open: the always-escalate ceiling arm (needs ~168 non-subset
  transcript expert calls — user decision pending) — also needed for a
  proper live random-escalation reference line; bootstrap CIs; the
  offline-vs-live figure.

Day-1 total spend ≈ $75 (GPU ~$60 + expert ~$8 + judge ~$7). Project
total ≈ **$446**.

### 8f — query-feature router baseline: the type-shortcut, isolated (2026-07-30)

`router_baseline` ($0, frozen data): TF-IDF (word 1-2gram + char 3-5gram)
→ LR on calib TEXT labels — a RouteLLM-style external router that sees
only the query surface. Three readouts, all as the grilling predicted:

- **in-mix**: OOF AUC .669 — below even the pool-oracle (.715), far below
  the probe (.828). Test tradeoff area **+.040** vs probe_final +.054 /
  midlayer_L22 +.064 / ptrue_post +.068 (same test set, same frozen
  expert answers). Internal signals beat the external router at every
  operating point (@30%: .717 vs probe-era .779).
- **LOPO: total collapse** — easy-chat .466, easy-fact .384,
  hard-knowledge .505, hard-math .574 (vs mid-layer probe .931); trap
  (single-class) mean score .230 = it MISSES the 100%-fail pool.
  Query-feature routing IS the type shortcut and nothing else; held-out
  types kill it, exactly the structural argument (7a positioning) now
  with numbers.
- Scope: a same-data router, not RouteLLM's 100k-preference-trained one —
  the full LLMRouterBench protocol comparison stays future work; this
  baseline isolates the information source (query surface vs internal
  state) under identical data.

Project total ≈ **$426** (router run $0).

### 8g — false-premise pool audition: the gate's third blind spot (2026-07-30)

User challenge: the trap pool conflates knowledge difficulty with
transcription difficulty (obscure entities are hard to hear by design) —
design a transcription-fair trap. WER-stratifying the existing pool was
REJECTED as cherry-picking (endogenous selection: the stratum is defined
by the system's own ASR; the dual-view counterfactual curve answers the
same question with zero item selection). The principled candidate:
**false-premise questions** (failure mode = premise checking; every word
common by design — criterion fixed before data selection). Pool:
thunlp/FalseQA (ACL 2023, public, 687 FPQ test rows), n=60 audition,
standard protocol (build_falseqa_pool / tts_falseqa / falseqa_eval /
falseqa_report, ~$8).

**Audition verdict: FAILS as a trap pool — and the failure is a
finding.** Small-model fail rate .63 text / .47 audio (substantial, not
SimpleQA's 1.00); expert adequacy .80 (escalation would pay); but the
**end-of-turn gate is blind to these failures: fire@balanced .18 (vs
trap's .90), and eot scores barely separate failed from passed audio
answers (.568 vs .507)**. The model answering along a false premise does
not "feel" unable — the mid-layer signal reads missing-knowledge, not
broken-question. Third documented blind spot (easy-chat failures, decode-
time math, now premise checks): the claim sharpens to "the pre-answer
signal detects retrieval-type failures specifically." FalseQA therefore
goes into the signal-boundary analysis, NOT into the tradeoff pools.
Curiosity noted (n=60, not pushed): audio fails LESS than text (.47 vs
.63) — the voice mode's shorter, more hedged answers may accidentally
challenge premises more often.

Phase-8g spend ≈ $8. Project total ≈ **$462**.

### 8h — bootstrap + dual-view figure: Part-2 closeout ($0, 2026-07-30)

`live_dualview` in modal_stream.py (CPU, frozen data only — reads
`gated_traces_v2.parquet` + `eval_expert.parquet`, NO re-judging, no
expert calls; 240 ids common to all four arms; paired bootstrap 10k
resamples, seed 42):

| tier | esc | heard-acc [95% CI] | Δ vs floor [CI] | gold-inject [CI] | channel cost [CI] |
|---|---:|---|---|---|---|
| never | 0% | .400 [.338,.463] | — | .400 | — |
| conservative | 14% | .446 [.383,.508] | +.046 [−.008,+.100] n.s. | .500 [.438,.562] | +.054 [+.029,+.083] * |
| balanced | 35% | .529 [.467,.592] | +.129 [+.067,+.188] * | .637 [.575,.700] | +.108 [+.071,+.150] * |
| aggressive | 55% | .633 [.571,.692] | +.233 [+.171,+.296] * | .767 [.713,.821] | +.133 [+.088,+.179] * |
| always (synth) | 100% | — | — | .917 [.879,.950] | — |

- **Balanced and aggressive beat the live floor significantly; the
  conservative delta (+.046) is n.s. at n=240** — stated in the paper for
  honesty. Channel cost is significant at every escalating arm.
- Dual-view figure `figures/live_dualview.png` (+ numbers in
  `live_dualview.json`, both fetched into the repo + paper/figures/):
  honest heard-acc curve (blue, CIs) vs channel-controlled gold-inject
  counterfactual (green, escalated rows re-scored with frozen gold expert
  answers) + synthesized always endpoint + random reference (pairs with
  the green view — its outcomes are gold), offline text curve for
  context. Two annotated gaps = audio+streaming tax (floor .588→.400)
  and speech-channel cost (ASR-distill + relay).
- In the channel-controlled view the gate clears random at every arm
  (+.03/+.06/+.08).
- **Paper sync (same session): new §"The Gate Goes Live" (sections/
  live.tex: loop design + 8b conflict injection + live curve
  Table~tab:live + dual-view figure + FalseQA boundary), router-baseline
  paragraph in system.tex, discussion rewritten (pushback resolved by the
  gate's own signal + three-failure-species taxonomy), abstract/intro
  updated to "both steps executed", limitations updated (live-loop
  text-mode output, conservative n.s.), todo.tex items marked DONE.
  TECHNICAL_REPORT.md bumped to v4 (§8b).**

---

### 2.1 public pools ✅ (2026-07-07)

`build_public_queries` → **400 queries**: `hard-math` 150 (GSM8K test tail 100 +
MATH-500 50), `hard-knowledge` 150 (MMLU-Pro; **GPQA 401-gated** with no HF token
→ gracefully topped up from MMLU-Pro), `easy-fact` 100 (TriviaQA
unfiltered.nocontext). Dataset-loading + formatting code validated end-to-end.
Remaining 200 (easy-chat 150 + trap 50) are Claude-generated → need the secret.

Pure helpers (`src/queries.py`) unit-tested locally: MCQ formatting, GSM8K
reference extraction, and stratified 60/40 split (deterministic, seed 42) all pass.
