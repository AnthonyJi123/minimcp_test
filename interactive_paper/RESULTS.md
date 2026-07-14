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

### 2.1 public pools ✅ (2026-07-07)

`build_public_queries` → **400 queries**: `hard-math` 150 (GSM8K test tail 100 +
MATH-500 50), `hard-knowledge` 150 (MMLU-Pro; **GPQA 401-gated** with no HF token
→ gracefully topped up from MMLU-Pro), `easy-fact` 100 (TriviaQA
unfiltered.nocontext). Dataset-loading + formatting code validated end-to-end.
Remaining 200 (easy-chat 150 + trap 50) are Claude-generated → need the secret.

Pure helpers (`src/queries.py`) unit-tested locally: MCQ formatting, GSM8K
reference extraction, and stratified 60/40 split (deterministic, seed 42) all pass.
