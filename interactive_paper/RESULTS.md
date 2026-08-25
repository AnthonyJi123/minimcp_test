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

### 8i — latency profile of the live sweep + real-session timelines ($0, 2026-08-05)

User ask: "what did escalation actually cost us — mean / P95 / P99 — and
show real timestamped sessions." All from the frozen
`gated_traces_v2.parquet` per-session timers (no re-runs). Scripts:
`figures/latency_profile.py` (numbers → `latency_profile.{txt,json}`),
`figures/timeline_live.py` (figure). Reconstruction: local total =
eot_read + answer decode; escalated total = eot_read + max(stall prefill,
expert round-trip) + relay decode (the expert thread launches at the gate
decision, concurrent with the stall prefill). `expert_latency_s` is the
TRUE API latency (cache-corrected; min 0.98 s, no timeouts, no ~0 s cache
artifacts in the 250 escalated rows). Text-mode pipeline; speech
synthesis not included.

**Component timers (rows where the stage ran; seconds):**

| stage | n | mean | P50 | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| eot gate read | 960 | .03 | .03 | .03 | .05 |
| stall prefill | 250 | .03 | .03 | .04 | .19 |
| local answer decode | 710 | 4.57 | 2.34 | 15.90 | 17.86 |
| relay decode | 250 | 1.14 | 0.69 | 3.64 | 8.61 |
| expert round-trip (gpt-5.5 low) | 250 | 7.28 | 4.78 | 20.77 | 32.80 |

**Per-arm total response latency (query end → answer text done; s), and
the loss vs the never floor at the same percentile:**

| arm | esc | mean | P50 | P95 | P99 | Δmean | ΔP50 | ΔP95 | ΔP99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| never | 0% | 4.61 | 2.02 | 15.76 | 17.79 | — | — | — | — |
| conservative | 14% | 4.93 | 2.76 | 16.31 | 20.32 | +0.32 | +0.74 | +0.55 | +2.53 |
| balanced | 35% | 6.27 | 4.00 | 18.06 | 30.37 | +1.66 | +1.98 | +2.30 | +12.58 |
| aggressive | 55% | 6.58 | 4.69 | 18.01 | 32.94 | +1.97 | +2.67 | +2.25 | +15.15 |

- **The average price is small; the tail price is real.** Balanced buys
  +.13 heard-acc for +1.7 s mean / +2.0 s P50 — but the P99 doubles
  (17.8 → 30.4 s), and the P99 loss is entirely the expert tail: within
  escalated rows the expert round-trip is 85–88% of total wall time
  (relay decode P50 ≈ 0.7 s, gate + stall prefill ≈ 50 ms combined are
  noise). P95 moves little (+2.3 s) because at 35% escalation the 95th
  percentile is still mostly local long-decode rows; P99 (n=240, ≈2
  worst rows — noisy) is where escalation shows.
- Escalated-row totals: conservative P50 9.2 / P95 21.8; balanced P50
  6.1 / P95 22.4; aggressive P50 5.4 / P95 21.9 (conservative escalates
  the hardest queries → slowest experts, mean 8.9 s vs aggressive 6.7 s).
- **Figure `figures/timeline_live.png`** — three REAL balanced-arm
  sessions, every event a recorded timer: (a) q0254 escalated, expert
  ready at 1.9 s while the talker is still voicing the stall → seamless
  handoff, zero dead air; (b) q0593 (trap) escalated, expert 10.3 s
  outlives the stall → 5.9 s dead air (the 8e/6c prediction,
  live-confirmed on a real session); (c) q0388 not escalated → local
  answer, first token 0.4 s. Speech bars are the one estimated quantity
  (150 wpm; the live loop outputs text — RESULTS 8e).
- **Paper sync (same session): latency paragraph + Table~tab:latency +
  Figure~fig:timeline-live added to sections/live.tex
  (§"The live curve"); figure copied to paper/figures/.**

### 8j — router training receipt + RouterBench grounding (~$1, 2026-08-05)

User ask: "what is the router's training accuracy — give a receipt, and a
score on a routing bench." Two runs, both CPU-only: `router_baseline`
re-run with a full receipt, and new `router_bench` on **RouterBench**
(withmartian/routerbench 0-shot, public — no self-made data).

**Training receipt (8f router, `router_baseline.json` → `receipt`):**
TF-IDF (word 1-2gram + char_wb 3-5gram, min_df=2) → LR (C=1.0, L2,
lbfgs, max_iter=3000), 5-fold stratified OOF, seed 42. n_train=360 calib
(escalate rate .322), 15,103 features.

| split | logloss | acc | majority |
|---|---:|---:|---:|
| train (in-sample) | .377 | .917 | .678 |
| calib OOF (= eval) | .588 | **.678** | **.678** |
| test | .629 | .613 | .588 |

- **The headline answer: at the 0.5 threshold the router's eval accuracy
  exactly equals the majority-class rate** (and test is +.025 over it).
  360 queries of surface text buy a weak ranking signal (OOF AUC .669)
  and no usable classifier — train↔OOF loss gap .38→.59 is textbook
  small-n overfit. AUCs reproduce 8f exactly.

**RouterBench (`router_bench.json`): n=36,497, pair mixtral-8x7b-chat
(correct .568) → gpt-4-1106-preview (.843), label = weak incorrect
(escalate rate .432), same TF-IDF+LR recipe.**

- **In-domain (trained on RouterBench, 5-fold OOF): AUC .710, acc .660
  (majority .568), logloss .615; deferral area over random +.033
  (pair acc @30% escalation .691).** 100× the training data buys
  .669→.710 — the recipe lands at our pool-oracle's level (.715) and
  stays far below the probe (.828). **The 8f baseline is not
  data-starved, it is information-starved — not a strawman.**
- **Leave-one-benchmark-out (the LOPO mirror on public data): the
  format-disjoint benchmarks sit at chance** — hellaswag .502
  (n=10,042), grade-school-math .509 (n=7,450), winogrande .498,
  arc-challenge .581; Chinese_character_riddles .112 (fail rate .98 —
  misses the near-100%-fail pool, the trap-pool signature again). MMLU
  subjects read .55–.69 only because the other 56 subjects stay in
  training (within-format type prior transfers; cross-format it dies).
- **Zero-shot transfer (our calib-trained router → RouterBench): AUC
  .440, area −.022** — below chance; the 8f router learned our pools'
  surface regularities and nothing portable.
- Scope: pair routing readout (AUC/acc/deferral curve), not the full
  RouterBench cost-quality AIQ protocol; that and preference-trained
  routers (RouteLLM 100k) remain the LLMRouterBench future-work item.
- **Paper sync (same session): receipt + RouterBench numbers written
  into the trained-router paragraph in sections/system.tex
  (\citep{hu2024routerbench} already in refs.bib); todo.tex item
  updated; TECHNICAL_REPORT.md bumped to v5 (§8b.5).**

Project total ≈ **$447** (RouterBench run ~$1: 16 CPU + 32 GB, ~25 min).

### 8k — the same receipt for our probes ($0, 2026-08-05)

User follow-up on 8j: "then how does OUR trained probe do on this
accounting?" `probe_receipt` — the identical receipt (train/OOF/test
logloss, acc@0.5 vs majority, AUC, budget-threshold classification acc)
for the two shipped internal gates, exact shipped recipes
(`probe_receipt.json`):

| signal | OOF AUC | OOF acc (maj) | test AUC | test acc (maj) |
|---|---:|---:|---:|---:|
| router 8f (query surface) | .669 | .678 (**= .678**) | .721 | .613 (.588) |
| text h_prompt probe, LR C=.001 | .828 | **.772** (.678) | .819 | **.779** (.588) |
| audio L22 probe (live gate) | .843 | **.764** (.592) | .879 | **.800** (.512) |

- **The probes pass the accounting the router failed**: at the same 0.5
  threshold they clear majority by +.09/+.19 (text, OOF/test) and
  +.17/+.29 (audio) where the router cleared it by +.000/+.025. Same
  n=360 labels, same LR machinery — the difference is purely the input
  representation. (Prediction miss, recorded: I expected C=0.001 score
  compression to pin probe acc@0.5 at majority too — wrong; the internal
  separation survives even heavy regularization.)
- Audio rows use the audio-modality label set (calib esc rate .408, test
  .488 → majority .512) — not the same labels as the text rows; compare
  within-row, not across.
- Budget-threshold classification acc (test): text .704/.775/.700 at
  15/30/50%; audio .692/.750/.800 (realized rates track targets:
  .142/.329/.529 text, .204/.329/.496 audio).
- Honesty notes: the text probe memorizes in-sample even at C=0.001
  (train logloss .009, acc 1.000; 4096 dims ≫ n=360) — all headline
  numbers are OOF/test. Audio probe test logloss .446 — the
  best-calibrated signal we have. AUC headlines reproduce (.828 text
  OOF, .843 audio OOF, .879 audio test).
- **Figure `figures/receipt_compare.png`** (`receipt_figure`, reads both
  receipt jsons — no hardcoded numbers): (a) acc@0.5 vs majority per
  split, (b) train→OOF→test logloss per signal. Loss verdicts: audio
  probe healthy (.278/.483/.446 — eval below base-rate entropy, test
  below OOF, best-calibrated); text probe memorizes in-sample
  (.009 train vs .688 OOF — OOF logloss WORSE than predicting the base
  rate, i.e., uncalibrated probabilities, but held-out acc/AUC hold and
  the gate thresholds on score quantiles, not probabilities); router
  .377/.588 — normal gap, low ceiling.
- **Paper sync (same session): probe-receipt sentence added to the
  trained-router paragraph in sections/system.tex +
  Figure~fig:receipt (receipt_compare.png copied to paper/figures/);
  TECHNICAL_REPORT §8b.5 extended (v5).**

### 8l — router fairness sweep ($0, 2026-08-05)

User challenge: "prove you didn't train a deliberately weak router to
flatter the probe." `router_sweep` (`router_sweep.json`): 24-config grid
— features {word, char, word+char} × min_df {1, 2} × C {.01, .1, 1, 10},
identical 5-fold OOF protocol on the same 360 calib labels.

- **Every config lands in .625–.689; best .689** (word+char, min_df=1,
  C=0.1) vs the shipped .669 (+.020, within small-n noise). Probe: .828.
- The query-surface ceiling on this data is ≈.69 by exhaustive grid,
  ≈.71 by 100× public data (8j RouterBench in-domain) — two independent
  routes to the same ceiling; the shipped config is not a tuning
  artifact. Written into the system.tex router paragraph.

### 8m — RouteLLM released checkpoints, zero-shot on our pool (~$1, 2026-08-05)

User ask: "have you compared against an actually-trained router (e.g.
RouteLLM's)?" `routellm_baseline` (`routellm_baseline.json`): the two
released preference-trained routers — `bert_gpt4_augmented` and
`mf_gpt4_augmented`, trained on ~100k GPT-4-vs-mixtral preference pairs
— scored zero-shot on all 600 labeled queries
(score = calculate_strong_win_rate, their own inference code via the
routellm package).

| router | calib AUC | test AUC | area | acc@30% |
|---|---:|---:|---:|---:|
| RouteLLM BERT | .584 | .523 | +.011 | .688 |
| RouteLLM MF | .602 | .533 | +.007 | .688 |
| our same-data TF-IDF router (8f) | .669 | .721 | +.040 | .717 |
| probe (h_prompt / L22) | .828 / .843 | .819 / .879 | +.054… | .779… |

- **The real preference-trained routers land near chance on our labels
  (test .52–.53)** — below even the 360-sample same-data router. 100k
  preference pairs of "is this hard for mixtral vs GPT-4" carry almost
  nothing about "will MiniCPM-o-4.5 fail this" — the exact mirror of our
  router's .440 transfer TO RouterBench (8j). Routing knowledge is
  model-pair-specific; it does not port across talkers in either
  direction.
- **Trap-pool blindness, again**: BERT ranks trap (.444 mean score)
  BELOW hard-knowledge (.642) and hard-math (.668); MF likewise (trap
  .274 ≤ hard .31). The 100%-fail pool looks "easy" to a router trained
  on another model's preferences — the type-shortcut failure mode
  reproduced on the strongest available external router.
- Scope: zero-shot released checkpoints (their intended deployment
  mode); retraining them on our 360 labels is the same-data condition
  8f already covers. The standardized LLMRouterBench protocol remains
  future work.
- **Same scores vs the AUDIO (TTS) labels** (user ask, apples-to-apples
  with the audio probe): BERT calib .559 / test .600; MF .603 / .581 —
  vs audio L22 probe .843 / .879 and our audio-label TF-IDF router
  .743 / .814 (8n). Context from their own paper (APGR, random=.500):
  even on RouteLLM's own bench their routers reach only .53–.62 on
  MMLU/GSM8K-style objective tasks (strong only on MT-Bench chat,
  .68–.80), so .52–.60 on our pool is consistent with their published
  profile, not an artifact of our setup.
- **Paper sync (same session): system.tex router paragraph — future-work
  clause replaced with the zero-shot RouteLLM numbers; TECHNICAL_REPORT
  §8b.5 extended.** Project total ≈ **$448**.

### 8n — audio-modality router baseline ($0, 2026-08-05)

User: "is there an audio router?" There wasn't — every router number so
far was text-modality. `router_audio` (`router_audio.json`): the 8f
recipe vs the AUDIO labels (the live gate's label set, esc rate
.408 calib / .488 test), two inputs, both with 600/600 coverage:
gold query text, and the talker's own ASR transcript (what an external
router would actually see live).

| signal (audio labels) | OOF AUC | oof acc (maj .592) | test AUC | test acc (maj .512) |
|---|---:|---:|---:|---:|
| router, gold text | .743 | .706 | .814 | .696 |
| router, self-ASR transcript | .738 | .689 | .805 | .704 |
| audio L22 probe | **.843** | **.764** | **.879** | **.800** |

- **Honest headline: on audio labels the surface router is STRONGER
  than on text labels** (.743/.814 vs .669/.721) — the audio channel
  makes hard pools fail harder and more uniformly, so failure is more
  type-correlated and the type shortcut buys more. Reported as-is.
- The probe still leads every readout: AUC +.10 OOF / +.065 test,
  accuracy +10 points (.800 vs .696). The residual is exactly what the
  surface cannot carry: instance-level knowledge state + whether THIS
  utterance was heard correctly.
- **ASR input costs the router almost nothing** (−.005/−.009 AUC vs
  gold text): its handicap is not transcription quality; it is the
  information source, plus the structural live constraint (full
  utterance + ASR needed; cannot fire mid-stream — todo.tex note).
- **Paper sync (same session): audio-router sentences added to the
  system.tex router paragraph; TECHNICAL_REPORT §8b.5 extended.**

### 8m — channel-cost trace-level decomposition: relay exonerated ($0, 2026-08-05)

User hypotheses for the blue↔green gap: (H1, Changyi) talker context too
short to hold the expert micro-turn answer; (H2) talker doesn't follow /
second-guesses the relay instruction. Both testable from
`gated_traces_v2.parquet` alone (250 escalated rows across the three
partial tiers, 108 heard-fail; ref-string containment + query↔transcript
similarity, `difflib` ratio):

- **H1 REFUTED.** Expert micro-turn answers are short: median 76 chars,
  p90 376, max 1948 (~500 tokens) — nowhere near any context limit; the
  single >1500-char answer relayed successfully (heard_ok=1).
- **H2 REFUTED (again).** Only **5/108** fails have the reference inside
  the expert answer but missing from the relay; 8b already measured
  deployed-channel comply = 1.00 (n=24). One degenerate-repetition relay
  observed (Taylor MCQ) — on an already-wrong expert answer.
- **The loss is upstream of the relay (~95%).** Fail split: **69/108
  (64%) corrupted transcript** (fails' query↔transcript sim median .809
  vs .995 on heard-ok rows; MCQ options and formulas garbled —
  hard-knowledge 39, hard-math 27 dominate) → the expert answers the
  wrong question; **26/108 (24%) clean transcript but expert wrong**
  (trap 17 — expert knowledge limit, shared with the gold arm, so partly
  not channel loss at all); 5 relay-drop; ~8 ref/judge noise (7 blank
  refs + 1 in-both-but-judged-fail; one easy-fact ref is broken —
  "first Boston Marathon finishers" keyed to "$85,000").
- Magnitude cross-check: per-escalated gap at aggressive = gold-inject
  .864 − heard .621 = .243 ≈ the five-arm ASR-distill gap (gold .82 −
  self-1best .58 = .24). The relay adds ≈ nothing on top.
- **Consequence: "better relay paradigm" / "keep the talker silent" /
  seq-length ablations target ≤5% of the loss.** The lever is what the
  expert *receives* (the question uplink), not how the answer is spoken.
  Paper sync same session: decomposition sentence added to the
  speech-channel-cost bullet in sections/live.tex.

### 8o — acc × latency joined: the Pareto tradeoff figure ($0, 2026-08-09)

User ask (Aug-3 comment): "a matrix of how much latency bought how much
acc + a Pareto-frontier figure, cherry-picking allowed." Pure join of
frozen readouts — acc + CIs from `live_dualview.json` (8h), per-arm
latency from `latency_profile.json` (8i); no re-runs. Script
`figures/pareto_latency.py` → `pareto_latency.{png,pdf}` (repo +
paper/figures).

**Marginal exchange rates (heard-acc, P50 view; per-arm table in 8i):**

| segment | Δacc | ΔP50 |
|---|---:|---:|
| never → conservative | +4.6 pts (n.s.) | +0.7 s |
| conservative → balanced | +8.3 pts | +1.2 s |
| balanced → aggressive | +10.4 pts | +0.7 s |

- **balanced→aggressive is the cheapest segment per second**, for two
  measured reasons: the marginal escalations are easier queries whose
  experts return faster (escalated-row expert mean 8.9/7.5/6.7 s across
  the tiers), and each escalation displaces a local decode that itself
  averages ~4.6 s.
- Sanctioned cherry-picks in the figure: x = P50 (typical experience);
  the P99 expert tail stays in the table + a figure footnote. Both views
  drawn — gold-inject shares x positions (rescoring changes outcomes,
  not timing). Always ceiling = asymptote (synthesized, no live
  latency).
- **Random reference deliberately NOT drawn in latency space**: random
  arms were never run live, and random@matched-rate would have slightly
  *lower* latency than the gate arms (the gate escalates harder queries
  with slower experts), so any placement from frozen data would either
  flatter the gate or require unmeasured expert latencies on
  never-escalated queries. Gate-vs-random lives in rate space
  (fig:dualview) where it is exact.
- **Paper sync (same session): exchange-rate sentence + 
  Figure~fig:pareto-latency added to the latency paragraph in
  sections/live.tex; figure copied to paper/figures/.**
- **Legend relabel (2026-08-09, after teammate + user both misread the
  two curves):** "heard-acc (honest)" / "gold-inject (channel-
  controlled)" → "deployed: expert answers the talker's transcript" /
  "counterfactual: expert answers the gold text", and the gap annotation
  now states the 8m attribution (95% upstream of the relay: corrupted
  transcripts). The misreading being corrected: green is NOT the
  always-big arm (that is the .917 asymptote) — it is the same sessions
  at the same escalation rates with escalated rows re-scored on
  gold-text expert answers; the blue↔green gap is an *uplink* property,
  not a relay/injection property (8m: 64% corrupted transcript / 24%
  expert-wrong-shared-with-gold / 5 relay drops of 108).

### 8p — wav-pool integrity audit + listening pack ($0, 2026-08-12)

Meeting follow-up ("先解决TTS的问题 — 真正听一下现在读出来的声音"):
before re-rendering anything, audit whether the frozen pool's TTS files
are physically sound, and package the worst uplink failures for human
listening. Modal scan `scan_wavs.py` (stdlib peak/lead-silence over all
of `audio_pool`) → `figures/wav_audit.json`; listening pack at
`data/listen_pack/` (14 wavs + `cases.json` gold-vs-transcript +
README), served for phone listening at
https://rhe9527--tts-listen-web.modal.run/62dc5cd9 (`listen_app.py`,
`modal deploy`; scales to zero, stop with `modal app stop tts-listen`).
Pack = 2 good cases (q0578/q0588 — rare names + version numbers
transcribed verbatim, so the channel CAN be clean) + 6 true mishears + 6
broken/not-mishearing controls. Gotcha for anyone reusing the wavs: they
were written streaming, so RIFF/data sizes are 0xFFFFFFFF and players
show a 24-day duration and refuse to seek — patch the header to the real
byte length (librosa in the pipeline is unaffected).

- **File-level TTS is fine: 1/601 wavs broken** — `q0208` is 49 s of
  pure digital silence (peak=0, streaming render failure). The talker's
  "audio appears to be completely silent" transcript on that row was
  CORRECT, not a hallucination. No quiet renders (peak<3000: 0), no
  long leading silences. → The 🌟 "TTS 没读出来" hypothesis is FALSE at
  the file level; whatever TTS contributes is pronunciation-clarity,
  not broken audio.
- **The "corrupted transcript" bucket (8m's 64%) is heterogeneous** —
  eyeballing the 132 escalated ids ranked by query↔transcript difflib
  sim, at least four species: (a) rare-entity substitution (Mustafa
  Adebayo Balogun→Mustapha Arabo Balogun; Taurek→Turek); (b) spoken-math
  loss (999 − 103 heard as "nine hundred ninety nine hundred and three"
  — the operator vanishes; plus the known TTS-of-LaTeX artifact,
  `\Omega`/`\muF` read raw); (c) **not-transcription behavior: the
  talker sometimes ANSWERS instead of transcribing** (q0213 transcript =
  "D) Mongolia"; q0237 = a full answer to a 96 s question) — this
  corrupts the uplink but is an instruction-following failure, not
  hearing. [CORRECTION same day: an earlier draft listed source-text
  mojibake as species (d) based on `��pleasure��` in console output —
  the actual characters are U+201C/201D curly quotes (normal
  typography); the `��` was this machine's GBK console failing to print
  them. No mojibake exists in the pool; q0233's garbling is ordinary
  option-content mishearing.] Also: the difflib sim metric
  overstates corruption on rows that merely spell out digits (q0169
  content-intact at sim .054) — don't use it as a corruption *rate*.
- Meeting's proposed "orange line" (uplink transcript → expert, no
  relay) needs no new runs: 8d's five-arm table IS that arm on the
  escalated subset (self-1best .58 vs gold .82, expert answers scored
  directly), and 8m bounds the relay at 5/108 fails — orange ≈ blue
  ≈ 2 pts above it; the gap is uplink. Documented here so the ask
  doesn't resurface as an experiment.
- Next actions recorded in todo.tex: human listening verdict on the
  pack (does alloy enunciate entities/operators clearly?); re-render
  the escalated-fail wavs with a newer TTS (gpt-4o-mini-tts,
  instructed enunciation) and re-run the escalated subset — if heard-acc
  moves, the dual-view gap narrows and the figure gets redrawn;
  separate (c)-species rate from true mishearing (cheap: classify 132
  transcripts).

### 8q — input-side fairness audit: the "speakable subset" curve ($0, 2026-08-12)

User challenge: the bad cases make the live acc unfair — how to remove
the drop caused by "unfair" questions? Methodological rule enforced:
**exclusion must be decidable from the INPUT alone (query text + wav
bytes), never from outcomes** — anything else is cherry-picking.
Script `figures/fair_subset.py` → `figures/fair_subset_audit.json`.

Flags (pre-registered, input-side): `latex` = formula-symbolic content
(backslash commands, `^{`/`_{`/`^x` exponents, `$..$` only when the
inside contains math operators — plain dollar amounts like "$815.50"
deliberately NOT flagged, first draft over-flagged them);
`broken_wav` = digital silence from the 8p audit. Rare entities and
hard names are deliberately NOT flags — mishearing them is the
phenomenon. Result: **22/240 test ids unfair** (12 hard-knowledge with
embedded LaTeX, 10 hard-math formula problems; q0208 among them).
[Also corrected in the same pass: no mojibake exists in the pool —
8p's species (d) was a GBK console display artifact.]

**Dual-view on the fair subset (n=218, paired bootstrap 10k, seed 42):**

| tier | esc | heard (full→fair) | channel gap (full→fair) [CI fair] |
|---|---:|---|---|
| never | 0% | .400→.440 | — |
| conservative | 12% | .446→.486 | .054→.032 [+.014,+.060] |
| balanced | 32% | .529→.569 | .108→.069 [+.037,+.106] |
| aggressive | 52% | .633→.688 | .133→.083 [+.041,+.124] |
| always (synth) | 100% | .917→.922 | — |

- **~38% of the measured channel cost was the unspeakable-content
  artifact** (aggressive gap .133→.083); the remaining .083 is still
  significant and is the honest speech-channel price (entity
  substitutions, spoken-number slips) — it must NOT be excluded.
- Clean asymmetry as the sanity check: the gold-inject ceiling barely
  moves (.917→.922) — LaTeX questions specifically destroy the SPEECH
  channel, they are not intrinsically harder for the expert.
- Flagged rows account for 17/50 aggressive escalated heard-fails
  (34%), 12/39 balanced, 7/19 conservative.
- This formalizes 8d's existing scope note ("formula-symbolic math out
  of scope for the audio arm — nobody speaks LaTeX at a voice
  assistant") into a per-id filter applied uniformly to all arms.
- Borderline case NOT excluded: q0271 ("Estimate 999 − 103") is
  speakable arithmetic; whether the TTS actually voiced the minus is a
  render-quality question → if the listening pass confirms the minus is
  unvoiced, a third input-side flag (`render_defect`, verified by ear)
  becomes legitimate. Pending the user's listening verdict.
- Paper decision PENDING (user to ratify): make the speakable-subset
  curve the headline dual-view figure with the full-pool curve in the
  appendix, criteria stated as pre-registered input-side filters.

### 8r — audio-direct-to-expert: TTS exonerated, the leak is the talker's ears (~$5, 2026-08-12)

User-approved follow-up to 8q. For all 132 unique escalated test ids,
the ORIGINAL pool wav went straight to an audio-capable expert
(`modal_expert_audio.py::audio_expert`; auto-picked **gpt-audio** — no
gpt-5.5-class audio model exists in the API today) — no MiniCPM
self-transcription anywhere in the uplink. Judged against gold query +
reference by the standard judge (identical protocol to
`expert_adequate`). Control arm `::text_control`: SAME model, SAME
questions as gold text (gpt-audio rejects text-only requests, so the
text rides with a 0.25 s silent placeholder wav the model is told to
ignore). The two arms split "gpt-audio is a weaker brain than gpt-5.5"
from "the audio channel loses content". Anti-flag measures inherited
(per-id volume cache, user= attribution, concurrency 3, region-pinned).
132/132 + 132/132 collected, zero errors. Artifacts:
`audio_expert{,_text}.parquet` on gate-data + repo `data/`; analysis
`figures/audio_uplink.py`.

**Four-arm decomposition (same 132 escalated ids, same judge):**

| pool (fair subset, n=113) | gpt-5.5 text | gpt-audio text | gpt-audio audio | brain / channel |
|---|---:|---:|---:|---|
| easy-chat (24) | .88 | .83 | .71 | +.04 / +.12 |
| easy-fact (28) | .93 | .86 | .86 | +.07 / **.00** |
| hard-knowledge (34) | .91 | .59 | .41 | +.32 / +.18 |
| hard-math (9) | 1.00 | .89 | .56 | +.11 / +.33 |
| trap (18) | .61 | .50 | .50 | +.11 / **.00** |
| ALL | .87 | .72 | .61 | +.15 / +.11 |

(Full 132-id set: .86 / .71 / .54, brain +.15 / channel +.17 — the
extra channel loss vs the fair subset is the 8q LaTeX rows.)

- **⭐ TTS EXONERATED for speakable content.** On the two pools whose
  failures drove the "TTS 没读清楚" hypothesis, a good ear loses ZERO
  crossing the audio channel: trap .50→.50, easy-fact .86→.86. Star
  cases from the same wavs MiniCPM garbled: gpt-audio heard "Estimate
  999 − 103" (the minus IS voiced — q0271 adequate) and "Mustafa
  Adebayo Balogun" verbatim (q0552 adequate). **The meeting's 🌟
  root-cause question is answered: the alloy renders carry the content;
  the deployment loss is MiniCPM's own ASR.** Re-rendering with a
  better TTS is now PREDICTED NEGATIVE for entity/operator errors.
- **The remaining .083 fair gap attribution chain is CLOSED:** relay
  ≤5% (8m) → TTS 0 on speakable pools (8r) → expert fine on gold (.87)
  → the leak is the talker's transcription step, and it is not
  prompt-fixable (8d). Fixes are model-side (better duplex ASR) or
  architecture-side (forward audio, below).
- **Audio-direct as a deployment direction: REJECTED by user (same
  day).** The backend thinker stays TEXT-BASED — reasoning frontier is
  text-first, and binding the expert to an audio-native model accepts
  a permanent brain discount (measured −.15 here: gpt-audio gold-TEXT
  .72 vs gpt-5.5 .87; net swap negative, aggressive .638 vs deployed
  .688, `audio_uplink.py`). 8r stands as a DIAGNOSTIC control only.
  The one text-backend-compatible variant (audio uplink → cloud ASR →
  gpt-5.5 text) is already lower-bounded by the 8d Whisper arm (+4pp
  overall; fixes trap entities to gold-low, cannot fix MCQ option
  walls) — judged not worth further spend.
- **Long MCQ blocks are inherently audio-hostile even for the best
  ears** (hard-knowledge channel +.18 on speakable rows): holding 10
  spoken options is the failure, not entity perception. Strengthens the
  8q scope position and the public-benchmark arm (short open
  questions, no option walls).
- Caveats: easy-chat channel +.12 = 3 rows of open-ended judge
  variance (n=24); hard-math fair n=9; silent-placeholder control is
  mildly unnatural (declared in the prompt).

Spend ≈ $5 (264 gpt-audio calls + 264 judge). Project total ≈ **$380**.

**Addendum (earlier same day, 8q) — what the remaining fair gap is made of.** The 72
fair-subset escalated heard-fail rows split by transcript cleanliness
(sim .85): **35 clean-transcript** (trap 20, easy 12 — the expert is
wrong on gold too; these fail in BOTH views so they do NOT contribute
to the gap) vs **37 dirty-transcript** = genuine uplink loss,
concentrated in hard-knowledge 23 + hard-math 9 (long MCQ option
blocks). Of the 37, the gold expert answers **34/37 correctly** → the
remaining .083 gap is almost entirely recoverable-in-principle: deliver
the question faithfully and it converts. Species check: answered-
instead-of-transcribed contributes ~nothing to the LOSS — when the
talker answers instead of transcribing it usually already knows the
answer (q0213 "D) Mongolia": heard_ok=1 in both arms); only q0237
(96 s query) is a not-transcription fail. Next lever ranked: (1)
audio-direct-to-expert (send the wav, skip self-transcription — bounds
channel-inherent loss with the best available ears, ~250 calls, and
directly separates "TTS pronounces entities lossily" from "MiniCPM's
ears are weak"); (2) better-TTS re-render; (3) already measured
NEGATIVE: robust prompt, k-best GER; Whisper ears +4pp only.

---

### 8s — external-benchmark arm: the 8-figure deliverable (~$40, 2026-08-12/13)

User request: 8 figures — {our pool fair-subset, Speech TriviaQA, Speech
Web Questions, SD-QA} × {latency↔acc, escalation↔acc}. Infra:
`modal_bench.py` (build/transcribe/bench_live/ceiling/report — the sdqa
pipeline generalized to OpenAudioBench pools; 250 q each, seed 42,
official pre-rendered audio, NOT our TTS), SD-QA got its missing tiers
via the existing `run_sdqa_live`. All sweeps ran the FROZEN L22 gate +
frozen eot thresholds — zero recalibration; expert gpt-5.5 low, cached,
concurrency ≤3 throughout. Figures `figures/{bench}_{dualview,pareto}.*`
+ `fair_{dualview,pareto_latency}.*`, numbers in `bench_figures.json` /
`fair_figures.json`; all copied to paper/figures/. Gallery:
https://rhe9527--figures-gallery-web.modal.run/62dc5cd9

**Heard-acc per arm (never/cons/bal/agg), realized esc, ceiling:**

| pool | floor | cons | bal | agg | esc (c/b/a) | gpt-5.5 gold | official |
|---|---:|---:|---:|---:|---|---:|---:|
| ours-fair (218) | .440 | .486 | .569 | .688 | 12/32/52% | .922 | — |
| striviaqa (250) | .620 | .632 | .688 | .784 | 2/19/46% | .968 | 75.5 |
| swebq (250) | .412 | .436 | .456 | .604 | 6/23/47% | .736 | 70.2 |
| sdqa (200) | .495 | .535 | .595 | .720 | 3/16/44% | .930 | — |

**Gold-inject (green) added to the external figures (user question
2026-08-13 — the ceiling parquets already carry per-id gold-text expert
verdicts, so the counterfactual costs $0):** agg-arm gold-inject
striviaqa .824 / swebq .628 / sdqa .745. Two readouts: (1) **external
channel cost is small** — agg blue↔green gap .040/.024/.025 vs our
pool's .083, exactly the 8q prediction (short open questions transcribe
cleanly; no option walls); (2) **in the channel-controlled view the
gate DOES clear random on striviaqa** (+.044 at 46% vs the gold-paired
random line) — part of the deployed-view "gate ≈ random" is channel
cost eating the selection margin, same structure as the 600-pool
dual-view.

- **The gate TRANSFERS mechanically everywhere**: frozen thresholds
  fire at sane (compressed) rates on easier pools, curves rise
  monotonically, aggressive lifts +16/+19/+22 pts over the floors.
  P50 latency stays cheap (striviaqa 1.2→2.1 s, sdqa 1.4→2.3 s).
- **⚠️ Honest headline: gate ≈ random on Speech TriviaQA** (agg .784 vs
  random-at-46% .780); above random on swebq (+.04 at 47%) and sdqa
  (+.034 at 44%), both within per-arm CI width (n=250/200 — likely
  n.s. individually). Consistent with the 5b/8f audit: on our mixed
  pool a large share of the gate's random-margin is the pool/type
  shortcut; single-type external pools remove that shortcut and expose
  the thin per-query residual. The transferability figure honestly
  shows: mechanism ports, selectivity margin is small on easy
  homogeneous pools.
- Floors sit below the official chat-mode numbers (striviaqa .620 vs
  .755) — live streaming loop + judge severity, same offline→live tax
  as the 600-pool (8h: .588→.400). WebQ ceiling is only .736 (strict
  Freebase refs cap even gpt-5.5), so the .412 floor reads relative to
  that, not to 1.0.
- SD-QA escalated-subset acc .82-.84 at bal/agg — on REAL human speech
  the escalation payload works (expert latency P50 3.1-3.4 s).
- Judge protocol identical throughout (gpt-5.4-mini, ref-anchored;
  TriviaQA refs carry accepted-alias lists).

**Router-quality decomposition (user question 2026-08-13, $0):** eot
score's AUC against the never-arm local-fail label, plus
aggressive-arm escalation precision/recall:

| pool | base-fail | eot AUC | esc | precision (lift) | recall |
|---|---:|---:|---:|---|---:|
| ours-fair | .56 | .751 | 52% | .74 (×1.33) | .69 |
| striviaqa | .38 | .676 | 46% | .51 (×1.34) | .62 |
| swebq | .59 | .759 | 47% | .78 (×1.33) | .63 |
| sdqa | .51 | .719 | 44% | .68 (×1.35) | .59 |

Per-query discrimination is real and transfers (.68-.76 OOD vs .751
in-mix; in-calibration OOF was .843) with an eerily uniform ×1.33-1.35
precision lift — the transferring component looks like a generic
difficulty/confidence signal, not pool structure. The acc margin this
buys over random (+4-6 pts, gold view) is what AUC ≈ .7 mathematically
yields at these rates. Threshold quantiles transfer conservatively
(conservative tier fires 2-6% vs 14% in-mix — score distributions
shift on easier pools). Verdict logged: mechanism trained "enough to
transfer", not "enough to be decisive"; levers = calib-pool expansion
(todo, 500-1000/pool + more families) and unsupervised per-domain
threshold rescaling; the zero-training pitch survives as-is.

Spend ≈ $40 (experts for ~600 escalated + 700 ceiling calls, low
effort + judges + ~12 H100-hours). Project total ≈ **$420**.

**SD-QA dualview redraw ($0, 2026-08-13, user call):** on the SD-QA
escalation-vs-acc figure only, the grey random-escalation reference is
dropped and a **local-only floor line at 0.495** is drawn instead, so the
curve is read between the two bounds that matter (floor it starts from,
gpt-5.5 ceiling 0.930 it walks toward). `bench_figures.py` now takes
per-bench `random_line` / `floor_line` flags; the other three pools keep
the random reference (their gate-vs-random margin is the point there).
Regenerated + copied to paper/figures/; numbers unchanged.

---

### 8t — probe v2: retrain on the deployed signal + expanded pool ⭐ (~$25, 2026-08-13)

User: "你继续训练吧" — act on the 8s router diagnosis. `modal_train.py`
fixes BOTH diagnosed gaps at once: (1) **train/deploy signal mismatch**
— v1 was fit on chat-style full-prefill `h_last` but deployed on the
streaming end-of-turn read; v2 trains directly on the eot-read hidden;
(2) **calib coverage** — +800 public-benchmark queries (TriviaQA 150,
dolly 150, SimpleQA 100, GSM8K 150, **NQ-open 150**, **ARC-Challenge
100** — two new families), 8q speakable-filtered, deduped vs the frozen
pool, seed 43 (decorrelated from the eval seed), tts-1/alloy rendered,
answered from audio by MiniCPM and judged with the standard judge.
Expansion fail-rate spectrum: trap .98 / know-open .65 / easy-fact .38
/ easy-chat .32 / hard-math .09 / ARC .09 (pooled .41 — better balanced
than the original .56). **External eval pools stayed strictly out of
training; frozen 600 untouched; artifact = `midlayer_gate_audio_v2.json`
(v1 intact).** EOT hiddens captured once for all five pools (2100
streaming replays → `eoth_{tag}.shard*.npz`), so any future refit is
CPU-only.

**Transfer (AUC vs never-arm local-fail, identical eot hiddens):**

| pool | v1 | **v2** | Δ |
|---|---:|---:|---:|
| striviaqa | .676 | **.761** | +.085 |
| swebq | .759 | **.779** | +.020 |
| sdqa (real speech) | .719 | **.775** | +.056 |
| frozen-test (in-mix) | .811 | **.860** | +.049 |

In-mix does NOT regress (the pre-registered guard) — it improves. v2
OOF on its own train set = .878 (C=.0003, n=1160).

**Ablation — what bought the gain (external mean AUC):**

| fit | mean |
|---|---:|
| v1 (full-prefill signal, 360) | .718 |
| frozen-360 only, eot signal | .741 |
| expansion-800 only, eot signal | .765 |
| **both = v2 (1160)** | **.772** |

**Both levers real, roughly equal**: signal-match +.023, data +.031.
Notably expansion-alone already beats frozen-alone by +.024 — the new
families (open-domain NQ + ARC) carry transferable difficulty signal
the original five pools did not.

**Deployment readout at a matched 30% budget** (per-domain quantile
thresholds — label-free, needs only unlabeled score history):

| pool | base fail | v1 prec/rec | **v2 prec/rec** |
|---|---:|---|---|
| striviaqa | .38 | .53 / .42 | **.63 / .49** |
| swebq | .59 | .83 / .42 | **.84 / .43** |
| sdqa | .51 | .75 / .45 | **.80 / .48** |

striviaqa precision lift ×1.39→**×1.66** over base — the pool where 8s
found "gate ≈ random" is exactly where v2 helps most.

- **⚠️ Threshold-quantile transfer is STILL broken, and v2 flips its
  sign**: at the frozen global thresholds v1 under-fires (sdqa balanced
  16% vs 30% target) while v2 over-fires (sdqa balanced 47%,
  aggressive 80%). Score distributions shift per domain; a global
  quantile cannot follow. **Deployment recommendation: per-domain
  quantile thresholds** (used in the table above — label-free, exactly
  hits the budget by construction, and the AUC gain is
  threshold-independent so it survives). The zero-training claim is
  unaffected: still no gradient through the backbone, still a linear
  read on frozen activations; the calibration set simply grew.
- Live re-run DONE — see 8u below (this bullet's "not yet done" is
  resolved).

Spend ≈ $25 (800 TTS + 800 judge + ~10 H100-hours). Project ≈ **$445**.

### 8u — v2 live re-run: all 8 figures refreshed ⭐ (~$35, 2026-08-13)

User: "重新跑曲线". 12 live sweeps (4 pools × 3 escalating tiers, 2760
sessions) with the v2 probe + **per-domain quantile thresholds**;
never-arm rows reused from the v1 runs (thr=1e9 → probe never fires →
rows are probe-independent by construction; documented in
`report(never_glob=...)`). `modal_bench.py` generalized: POOLS registry
(frozen/striviaqa/swebq/sdqa), `art_path` + `suffix` params, so v1
artifacts and traces stay untouched. Figures regenerated from
`{pool}_v2_traces.parquet`; v1 versions archived as `*_v1.{png,pdf}`.

**Fix #1 confirmed live — thresholds now hit their budgets** (the 8t
diagnosis): realized rates 15/35/55% (frozen), 15/30/50%, 15/30/50%,
15/31/50% — versus v1's 2/19/46% (striviaqa) and 3/16/44% (sdqa).

**Curves (heard-acc, v1 → v2):**

| pool | never | conservative | balanced | aggressive |
|---|---|---|---|---|
| ours-fair | .440→.436 | .486→**.500** | .569→**.573** | .688→.670 |
| striviaqa | .620→.624 | .632→**.684** | .688→**.728** | .784→**.840** |
| swebq | .412→.404 | .436→.436 | .456→**.532** | .604→.560 |
| sdqa | .495→.510 | .535→**.610** | .595→**.740** | .720→**.785** |

Raw acc is rate-confounded (v2 escalates more at conservative/balanced
because its thresholds are now correct). **Rate-normalised selectivity
(lift over the random line, channel-controlled view) is the honest
comparison:**

| pool | cons v1→v2 | bal v1→v2 | agg v1→v2 |
|---|---|---|---|
| ours-fair | +.021→+.027 | +.045→+.048 | +.081→+.083 |
| striviaqa | +.004→**+.028** | +.013→**+.053** | +.043→**+.092** |
| swebq | +.002→−.002 | −.026→**+.044** | +.063→**+.010** |
| sdqa | +.027→**+.052** | +.043→**+.125** | +.059→**+.100** |

- **8s's headline finding is overturned on striviaqa**: the pool where
  v1 was indistinguishable from random (+.004/+.013/+.043) now clears
  it at every tier (+.028/+.053/+.092) — selectivity roughly doubled
  to tripled. sdqa likewise (balanced +.043→+.125). In-mix (ours-fair)
  is unchanged, as designed.
- **⚠️ swebq is the exception and it is not clean**: balanced improves
  a lot (−.026→+.044) but aggressive DROPS (+.063→+.010) and heard-acc
  falls .604→.560. Its ceiling is only .736 (strict Freebase refs), so
  at 50% escalation the headroom left to select from is thin and the
  arm is noisy at n=250; also the only pool where v2's conservative
  lift is ≈0. Reported as-is; a rerun at larger n is the honest way to
  settle it, not a re-pick of the tier.
- Latency essentially unchanged (P50 within ±0.5 s of v1 at matched
  tiers); the gate read itself stays ~30 ms.
- Figures: `fair_{dualview,pareto_latency}` + `{bench}_{dualview,pareto}`
  ×3, all v2, in figures/ + paper/figures/; gallery redeployed
  (same URL).

Spend ≈ $35. Project total ≈ **$480**.

### 8v — VoiceBench AlpacaEval: the official-matrix row ⭐ (~$20, 2026-08-13)

User challenge: "官方 benchmark 表里没有你用的那两行". **Verified by
fetching the raw README/model-card HTML (no summarizer in the loop):
`Speech TriviaQA` 75.5, `Speech Web Questions` 70.2 and `Speech CMMLU`
59.2 DO exist — in the full Audio-Understanding table; the screenshot is
the condensed matrix, which lists only `VoiceBench AlpacaEval` 4.8 as
its speech-QA row.** So 8s/8u's anchors were right, but sourced through
a WebFetch summary (which had garbled an earlier fetch) — the raw-text
verification is now on record. Everything else in the condensed matrix
is out of scope by construction: vision rows (we run
`init_vision=False`; meeting scoped audio-only), ASR rows (measure WER,
not answering — routing cannot fix ears, and 8r scoped ASR out), speech
GENERATION rows (our loop emits text), Omni rows (audio+video).

Fourth external pool added: **VoiceBench AlpacaEval, all 199 items**,
same v2 probe + per-domain quantile thresholds, four arms.
**Scoring uses VoiceBench's own judge**: gpt-4o-mini + their
`meta_prompt_open` copied verbatim from `MatthewCYM/VoiceBench`
`api_judge.py` (1-5, bare number out) — our first pass with a
home-grown rubric gave 2.78/4.86, i.e. ~2 points below the official
scale, confirming the rubric (not the model) drives the absolute level.

| arm | esc | judge score (1-5) | gold-inject |
|---|---:|---:|---:|
| never | 0% | 3.94 | 3.94 |
| conservative | 15% | 4.08 | 4.12 |
| balanced | 30% | 4.09 | 4.16 |
| aggressive | 50% | **4.26** | 4.37 |
| always (gpt-5.5, gold text) | 100% | — | **4.96** |

- **⭐ Chat-mode control settles the fairness question (user challenge,
  same day): the SAME 199 wavs answered offline with `model.chat`
  (1024-token budget, no streaming loop, no chunked prefill, no EOT
  read), judged identically, score **4.86** — i.e. we REPRODUCE the
  official 4.8 (slightly above it).** `valpaca_chatmode.parquet`,
  `modal_bench.py::valpaca_chatmode`. Therefore the entire 3.94→4.86
  gap is OUR LIVE LOOP, not model capability and not the judge:
  paired, the live loop is worse on 115/199 queries, better on 3.
  Mechanism = answer length: chat-mode median 2186 chars vs live 820.
  The duplex/omni system prompt puts the model in *spoken-reply* mode
  (short, conversational) while AlpacaEval's rubric rewards complete
  written answers; the live 512-token cap adds truncation on top
  (27% of live answers end mid-sentence, and the worst paired case —
  chat 5 vs live 1 — is a 3030-char answer cut off mid-clause).
  Consequence for the paper: **the official 4.8 line must be labelled
  as an offline-chat-mode number, and our own chat-mode 4.86 is the
  honest capability reference; only the four live arms are comparable
  to each other.** Ruled out as explanations along the way: degenerate
  repetition (4/199 rows, removing them moves 3.94→3.96) and the
  scoring rubric (already VoiceBench's own).
- **⚠️ On this pool the gate does NOT beat random** (aggressive 4.26 vs
  random-at-50% ≈ 4.45), and the mechanism is now measured, not
  inferred: **the queries the gate selected score 3.90 on the never
  arm vs 3.98 for those it skipped — no discrimination at all**, while
  escalated rows still gain (3.90→4.71) purely because gpt-5.5 writes
  better long-form answers. So the gain is expert quality, not
  selection; random picks would harvest the same. Root cause:
  open-ended instruction following has no "the model doesn't know this
  fact" event for the probe to read — every answer is partially
  creditable and the score range is compressed (3.94→4.96 = 1.0 point
  vs 40 accuracy points on our pool). Textbook species-3 of the
  three-failure taxonomy. Reported as a NEGATIVE result.
- Latency here is higher (P50 4.6 → 9.7 s) because AlpacaEval answers
  are long-form essays; the local decode dominates, not the expert.
- Figures `valpaca_{dualview,pareto}.{png,pdf}` (figures/ +
  paper/figures/), gallery now shows 10.
- Judge-infra gotcha: gpt-4o-mini 429s under batch load silently
  produced None scores concentrated at the tail of the batch (never arm
  lost 131/199 in the first pass). Fixed with concurrency 3 + 6-step
  backoff + persisted `score_err`; all four arms now n=199.

Spend ≈ $20. Project total ≈ **$500**.

---

### 8w — judge-protocol alignment: the official numbers ARE reproducible ⭐ (~$15, 2026-08-13)

User asked for comparison models on the figures; the chat-mode control
(8v's machinery, generalized to every pool) exposed a prerequisite
problem first. Offline chat mode under OUR judge: striviaqa .684 vs
official .755, **swebq .464 vs official .702** — a 24-point hole that
no loop tax explains. Root cause: **the judge**. We now copy
OpenAudioBench's own judging verbatim from
`tasks/trivia_qa_audio.py` (gpt-4o-2024-08-06, JSON
`analysis`+`judgment`, "correct if it matches **at least one** of the
reference aliases"; `_oab_judge` / `oab_rejudge_live` in
modal_bench.py) and re-scored every arm plus both ceilings.

**Chat-mode control under each judge (n=250):**

| pool | our judge | **OAB judge** | official |
|---|---:|---:|---:|
| striviaqa | .684 | **.712** | .755 |
| swebq | .464 | **.716** | .702 |

**We reproduce the official numbers** (swebq slightly above; striviaqa
4 pts below = subsample + protocol). Our reference-anchored judge is
simply much stricter on WebQ's Freebase alias lists. **Consequence: the
"swebq ceiling is only .736 / headroom is thin" story in 8s/8u was a
judge artifact — under the official judge the ceiling is .844.**

**All arms re-scored on the official scale (v2 probe):**

| pool | never | cons | bal | agg | ceiling | official MiniCPM | Qwen3-Omni-30B | Kimi-Audio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| striviaqa | .664 | .720 | .764 | **.860** | .972 | .755 | .629 | .419 |
| swebq | .572 | .648 | .680 | **.736** | .844 | .702 | .749 | .464 |

- **⭐ The headline comparison the paper needs**: on Speech TriviaQA a
  9B duplex model + our gate at 50% escalation scores **.860 live**,
  vs **.629** for Qwen3-Omni-30B-A3B (3.3× the parameters, offline) and
  .755 for MiniCPM-o's own official offline number. Routing beats
  scaling the speech model here. On swebq we clear the official
  MiniCPM number (.736 vs .702) and land just under Qwen3-Omni (.749).
- **Loop tax is now fully accounted**: striviaqa live floor .664 vs
  our chat-mode .712 (−.048 = the streaming loop) vs official .755
  (−.043 = 250-item subsample + protocol). No unexplained gap remains.
- **v1 vs v2 on the correct scale (lift over random, gold view)**:
  striviaqa +.024/+.031/+.052 → **+.033/+.062/+.080** (v2 wins at every
  tier, confirming 8u). swebq conservative −.001 → **+.045** and
  balanced +.045 → +.035, but **aggressive +.092 → +.066** — so the
  8u "swebq aggressive regression" SURVIVES the judge correction
  (smaller: −.026 lift instead of −.044 raw acc). It remains the one
  arm where v1 selected better; n=250, not re-picked.
- Comparison-model numbers extracted from the official table by
  position-mapping the raw HTML (verified against three values in the
  user's screenshot): Kimi-Audio and Qwen3-Omni-30B-A3B-Instruct, both
  offline chat mode — figure captions must say so.

### 8x — two more external pools: Llama Questions + Reasoning QA (~$25, 2026-08-13)

Added to cover the failure species the external arms still missed.
`sllama` = OpenAudioBench Llama Questions (250 of 300, English short
factoid), `sreason` = OpenAudioBench Reasoning QA (all 202,
**Chinese**, execution-type reasoning — no official MiniCPM number
exists for it, so no official line on its figure; it doubles as a
cross-lingual transfer test since the probe is calibrated mostly on
English). Four arms each, v2 probe, per-domain thresholds — realized
rates 15/30/49% on both.

**Data-integrity bug found and fixed while building these:**
`reasoning_qa`'s CSV keys audio by `.mp3` filenames while the builder
stripped only `.wav`, so keying failed and silently **fell back to
row-order pairing** — questions would have been matched to the wrong
audio. Now: strip any extension, and the row-order fallback raises
instead of guessing. The first sreason build was discarded. (Also
added `参考答案` to the reference-column detector.)

Per-domain thresholds again show why a global quantile cannot work:
the aggressive threshold is .111 on sllama vs .603 on sdqa — the same
probe's score distribution shifts by 5× across pools.

**Results (sllama on the OAB judge, sreason on ours):**

| pool | never | cons | bal | agg | always (gpt-5.5) |
|---|---:|---:|---:|---:|---:|
| sllama | .840 | .884 | .912 | **.944** | .924 |
| sreason (zh) | .584 | .624 | .683 | **.762** | .871 |

**⭐⭐ sllama: selective escalation BEATS always-escalate (.944 > .924)
— the strongest positive result in the project so far.** Decomposed on
the aggressive arm (n=250, official judge):

| gate decision | local model alone | gpt-5.5 alone |
|---|---:|---:|
| kept local (125) | **.976** | .960 |
| escalated (125) | .704 | **.888** |

The probe cleanly split a .976 subset from a .704 subset — genuine
per-query discrimination, not a pool/type shortcut (single-type pool).
**[SUPERSEDED 2026-08-20 by §8ad: paired McNemar gives p=1.00 on the easy half — "matches", not "beats"; the robust claims are the .976/.696 split (z=6.46) and the .696→.888 lift on the hard half (p<.0001).]** And because the small model *beats* the expert on the easy half
(.976 vs .960 — gpt-5.5 over-elaborates short factoids), **"escalate
everything" is NOT an accuracy upper bound**; only a selective router
collects the max of both. This is the cleanest external evidence for
the system's premise, measured with the benchmark's own judge. Judge
noise checked: 4 rows where the relay is right and gpt-5.5 wrong vs 2
the other way — an order of magnitude below the structural effect.

sreason adds the missing failure species (execution-type reasoning) and
a cross-lingual test: the probe was calibrated almost entirely on
English yet still lifts Chinese reasoning .584→.762 at 50%. Its
official-rubric scoring (per-item 打分prompt) was NOT replicated —
numbers are on our judge, no official line drawn.

**Why the P50 latency curves zigzag (verified 2026-08-14).** sllama's
per-arm P50 is non-monotonic — never 1.52s → cons **1.19s** → bal
1.73s → agg 1.64s — and this is a median-of-a-mixture effect, not a
measurement or gate bug. The probe escalates exactly the queries whose
*local* decode is longest (escalation@balanced by never-arm answer-length
quintile: 2/12/34/46/56%; conservative's 38 escalated ids had local P50
2.52s vs 1.27s for the 212 that stayed). At 15% the local pool loses its
slowest members (local-only P50 1.52→0.98s) while only 38/250 rows pay
the ~4.3s expert path, so the overall median *falls below never*; at 30%
the escalated mass reaches the median and pushes it back up. Mean
latency is monotonic (1.65/1.75/2.37/2.38s) — the fold-back lives only
in the quantile. On sreason the P50 is monotonic (3.17→3.84s) but the
**tail improves with escalation: P90 13.25→10.94s** (mean 5.08/5.20/
4.73/5.16) — long local CoT decodes (tail >13s) get replaced by 3–4s
expert round-trips, so on reasoning pools escalation buys accuracy AND
truncates the latency tail. Same frozen probe in all arms — nothing is
trained, and no latency model exists to "train"; the x-positions are
measured wall-clock medians.

### 8y — figures refreshed with comparison models (2026-08-13)

All 14 figures regenerated: 5 external pools × {escalation↔acc,
latency↔acc} + our pool × 2 + AlpacaEval × 2. OpenAudioBench pools now
carry four reference lines on one scale (their own judge): official
MiniCPM-o 4.5, **Qwen3-Omni-30B-A3B**, **Kimi-Audio**, and our own
offline chat-mode control. Gallery (14 figures):
https://rhe9527--figures-gallery-web.modal.run/62dc5cd9

Headline for the paper: on Speech TriviaQA, **9B + gate @50% = .860
live**, vs Qwen3-Omni-30B **.629** and MiniCPM-o's own official
**.755** — routing beats scaling the speech backbone, and on
Llama Questions it also beats escalating everything.

### 8z — probe v3: RL/SFT rejected; data + multi-position features executed ⭐ (~$45, 2026-08-16)

User asked whether RL (or SFT) should train the probe. **Decision: NO
to both** (rationale recorded in `todo.tex`): (1) the gate is a
single-step decision whose BOTH counterfactuals are observable offline
(never/always arms) — that is cost-sensitive supervised classification,
and policy-gradient RL would re-derive the same Bayes classifier with
orders-of-magnitude worse sample efficiency at n≈1k; (2) SFT on the
backbone breaks the zero-training frozen-checkpoint claim, shifts the
talker's answer distribution (invalidating every measured curve), and
small-n training is already falsified in-house (8f/8s router .669 <
pool-oracle .715); (3) the binding constraint is domain shift + judge
label noise (OOF .878 vs external .76–.78), which neither touches.
Executed the two supervised levers instead (`modal_train2.py`):

**1. expansion2** — 1150 new queries, 7 families NONE in the v1 mix
(dedup vs frozen + expansion + all 6 external pools, seed 44, tts-1
alloy, MiniCPM audio answers, standard judge). Fail-rate spectrum:
easy-mathword(SVAMP) .10 / know-openbook .17 / know-mmlu .31 /
know-commonsense .32 / trap-truthful .45 / hard-multihop(HotpotQA) .79
/ know-longtail(PopQA) .84 — pooled .50, adds the high-difficulty mass
the .41 expansion1 mix lacked. Train pool now 360+800+1150 = **2310**.

**2. multi-layer capture (`eoth2_*.npz`)** — ONE streaming replay per
query over all 9 pools (3901 replays, zero missing) storing
L{14,18,22,26,30} × (eot rolling last-8-token window + user-audio-mean)
in float16 → every future probe refit is CPU-only forever. Engineering
note (caught in smoke): the streaming assistant prefill runs 1-token
forwards, so "last-8 of the final forward" degenerates to a single
token — the tail window must roll ACROSS forwards.

**Refit sweep (19 configs, 5-fold OOF on train only):** L22 is still
the best single layer (eot_last .858; matches 5d), **multi-layer
concat HURTS** (L18+22+26 .837 — regularization cost exceeds the
information), **position diversity helps**: winner =
`eot_last + eot_mean8 + user_mean @ L22` (12288-d), C=1e-4, OOF
**.864**. All three reads are online-computable at zero eot latency
(running audio mean + rolling tail + last token).

**Transfer (AUC vs never-arm local fail, externals read once):**

| fit | striviaqa | swebq | sdqa | sllama | sreason | frozen-test | ext-mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2 stored artifact (sanity) | .761 | .779 | .775 | .815 | .621 | .860 | .750 |
| A v2-recipe refit (L22 last, frozen+x) | .761 | .779 | .775 | .815 | .621 | .860 | .750 |
| B data lever (A + expansion2) | .762 | .804 | .780 | .817 | .682 | .872 | .769 |
| C feature lever (winner cfg, frozen+x) | .791 | .758 | .788 | .811 | .628 | .873 | .755 |
| **D v3 = winner cfg + all data** | **.789** | **.785** | **.792** | .806 | **.683** | **.879** | **.771** |

- **Sanity anchor exact**: the re-captured L22 eot hidden reproduces
  8t's v2 numbers to 3 decimals, and refit A equals the stored
  artifact — capture + fit fully deterministic.
- **Data is the bigger lever again** (+.019 ext-mean vs +.005), same
  structure as 8t; combined +.021 (.750→.771).
- **⭐ Cross-lingual transfer jump: sreason .621→.683 (+.062)** — the
  new ENGLISH multihop/long-tail families improve CHINESE reasoning
  transfer; the difficulty signal the probe reads is language-general.
- striviaqa .761→.789 (+.028), sdqa .775→.792 (+.017); the one
  regression is sllama .815→.806 (−.009, was the best pool).
- **Guard passed with headroom**: frozen-test .860→.879 (pre-registered
  "must not regress"; selection never saw the externals).
- Honest note: B ≈ D on ext-mean (.769/.771) — if deployment ever
  wants a single-vector probe, B (plain L22 eot_last on 2310) keeps
  ~all the external gain; D wins in-mix and on striviaqa/sdqa.

Deployment artifacts: `midlayer_gate_audio_v3.json` +
`gate_v3_{pool}.json` per-domain quantile thresholds (label-free,
8t recipe). **Live 4-arm re-run with v3 NOT launched** — separate
spend decision (~$35, as 8u); AUC gains are threshold-independent.

Spend ≈ $45 (1150 TTS + 1150 answers + 1150 judge + 3901 replays ≈ 14
H100-h). Project ≈ **$490**.

**8z-live — v3 live re-run, all 14 figures refreshed (~$55, 2026-08-16).**
User: "图刷新一下". `modal_bench.py::bench_live` got a v3 path (rolling
last-8 tail across forwards + running user-audio mean at L22, concat →
same `Probe.score`; v1/v2 path untouched); `make_thresholds3` scale bug
fixed (raw logit → sigmoid, Probe.score convention) and regenerated.
21 sequential sweeps (7 pools × 3 escalating tiers, 4773 sessions,
expert concurrency ≤3 throughout), never arms + ceilings reused
(probe-independent); reports re-judged everything fresh, OAB pools
re-scored with the official judge. Realized rates on budget externally
(15/30/50); frozen overshot to 16/35/61 (calib-quantile → test drift).

**Live v2 → v3 per arm** (OAB pools = official judge; others = ours):

| pool | judge | never | cons v2→v3 | bal v2→v3 | agg v2→v3 |
|---|---|---:|---|---|---|
| striviaqa | official | .664 | .720→.732 | .764→**.800** | .860→.860 |
| swebq | official | .57 | .648→.628 | .680→.680 | .736→.732 |
| sllama | official | .84 | .884→**.904** | .912→.904 | .944→**.948** |
| sdqa | ours | .51 | .610→.600 | .740→.720 | .785→**.795** |
| sreason (zh) | ours | .59 | .624→**.653** | .683→**.713** | .762→**.772** |
| frozen (full) | ours | .39 | .467→.483 | .533→.554 | .621→.596 |
| valpaca (1-5) | VoiceBench | 3.99 | 3.96→3.96 | 4.23→4.23 | 4.26→**4.35** |

(never-arm deltas across versions are ±.005–.013 = pure judge re-run
variance; same trace rows.)

- **Gains land exactly where the offline AUC gained**: striviaqa
  balanced +.036 (the probe's biggest AUC jump pool among OAB),
  **sreason uniform +.010–.030 — the cross-lingual offline finding
  (+.062 AUC) survives live**; sllama conservative +.020.
- **sllama headline strengthens: selective @50% = .948 > always-escalate
  .928** (both re-judged same run) — the project's strongest positive
  result confirmed with the better probe.
- swebq/sdqa ≈ flat (within judge noise; both were probe-flat in the
  offline table too: swebq +.006, sdqa +.017 AUC).
- ⚠️ **[SUPERSEDED 2026-08-20 by §8ad — re-mixing the same measured outcomes at exactly 50% gives .600, i.e. +.004: the overshoot is a COST bug, not an accuracy bug, and the .621→.596 delta is 1 paired SE = noise.]** Original text: frozen aggressive .621→.596 at esc .61 (vs .50) — the
  overshoot pushes ~26 extra math/LaTeX-heavy queries through the
  transcript-tax channel (8d: ASR-distill −.31 on knowledge entities);
  balanced/conservative arms improved (+.021/+.016). Fair-subset curve
  (fair_dualview): heard .422→.633 @ 57%, gold-inject .761.
- valpaca stays a negative result: agg 4.35 < random@50% ≈4.45
  (species-3 open-ended pool — no retrieval-failure event to read).

All 14 figures regenerated from `{pool}_v3_traces.parquet` /
`valpaca_v3_scored.parquet` (v2 versions archived as `*_v2.{png,pdf}`,
jsons refreshed), copied into `paper/figures/`, gallery redeployed
(same URL). Live-loop deployment artifacts now v3 end-to-end
(`gate_v3_{pool}.json`, sigmoid-scale thresholds).

Spend ≈ $55 (4773 live sessions ≈ 11 H100-h + ~1.6k expert calls
mostly cache-hits + ~7k judge calls). Project ≈ **$545**.

### 8ab — Jisen review pass: kink mechanism, latency decomposition, matched-rate precision, random-pareto lines ($0 local, 2026-08-18)

Jisen's figure numbers = the 14-figure gallery order (fair ×2,
striviaqa ×2, swebq ×2, sllama ×2, sreason ×2, sdqa ×2, valpaca ×2).
Everything below computed locally from the existing v3 traces — no
model calls, no re-runs.

**图4 (striviaqa_pareto) "small kink" is mechanism, not noise — and a
seed-rerun would NOT remove it.** The conservative arm's P50 (1.186 s)
sits LEFT of never (1.227 s). Decomposition: the 38 queries the probe
escalated at 15% had never-arm local P50 **1.711 s** vs **1.198 s**
for the 212 it kept — the probe preferentially escalates the slowest
local decodes (same median-of-a-mixture effect as sllama, §8x), so the
kept-local median falls (0.937 s) faster than the 3.8 s expert path
can pull the mixture back up. Statistically the −41 ms dip is far
inside noise (query-bootstrap CI on the P50 difference: [−285, +254]
ms) — but because the mechanism is deterministic, 3–5 seeds would
average toward the same left-fold, not away from it. If a definitive
check is ever wanted: rerun never+conservative only, ~$6/seed.

**Addendum (2026-08-24, user asked for the interpretation, not the
assumption): the kink's behavioral story, from the traces.** Local
answer length and decode time are the same variable (r = .97), and the
small model's signature failure mode on striviaqa is
**uncertainty-as-verbosity**: when it doesn't know, it hedges at
length. The 38 conservative-escalated queries vs the 212 kept, all
measured in the NEVER arm (no escalation anywhere): local P50 1.69 s
vs 1.18 s, median answer 267 vs 172 chars, local accuracy **.32 vs
.73**. Canonical example striviaqa0192 ("Where does Dame Edna Everage
come from"): locally the talker rambles 921 chars for 5.8 s about the
character not being real and gets it WRONG; the escalated path returns
"Moonee Ponds, Melbourne" in 3.1 s total — faster AND correct,
because the expert round-trip is flat ~3 s while the local ramble is
5-9 s. (Two more of the same shape: striviaqa0074 Aung San Suu Kyi
5.1→3.9 s, striviaqa0221 Rastafari 4.5→3.6 s.) The probe reads this
BEFORE the answer exists — the hidden state carries the hesitation the
verbosity would later express. Honest counterpart: the kept-pool's 27%
errors are FAST confident wrongs (striviaqa: "Joe Gargery → Our Mutual
Friend", 0.6 s, no hedging signature) — the species a 15%-budget gate
misses and the reason the curve keeps rising toward 50%. Also
clarified for the record: figure error bars are query-resampling
bootstrap from ONE live run per arm, not repeated runs; the fold's
SIGN replicates deterministically (identical probe scores → identical
escalated set), only its depth is within the noise band.

**Addendum 2 (same day, user challenge: "text models don't hedge? prove
it's not MiniCPM-specific"). The wrong-answers-are-longer signature is
cross-model.** Effect size = P(len_wrong > len_right), .50 = null,
never/local answers only:

| model | striviaqa | swebq | sllama |
|---|---:|---:|---:|
| MiniCPM-o 9B (speech) | .62 | .53 | .77 |
| NVDA VoiceChat 11B (other family) | .56 | .48 | .64 |
| gpt-5.5 (frontier TEXT, gold input) | .56 (n=8) | .47 | .67 (n=18) |

Three readouts: (1) NOT MiniCPM-specific — an architecturally
unrelated duplex model replicates the direction, attenuated because
its trained voice style is terse (median 52-82 chars vs MiniCPM's
156-400). Quantified as a pre-registered prediction (user follow-up
2026-08-24): fold depth is fueled by the answer-length TAIL the probe
can remove — MiniCPM striviaqa p90-p10 spread 550 chars ≈ the slow
decile costing +2.7 s over the median (at its measured 6.2 ms/char),
NVDA only 97 chars ≈ +0.4 s — so if the live loop is ever ported to
NVDA we predict NOT merely a shallower fold but most likely NONE:
with every local decode under ~1 s and the expert RTT ~3 s,
escalation is a net time-add on every query and the latency curve
should be plainly monotonic. (Caveat: NVDA per-query decode time is
batch-contaminated; the 0.4 s assumes same-order decode speed.)
**VERIFIED 2026-08-24 (user: "那你跑呗", ~$0.2):** the batch-timing
problem dissolves for a speech-native duplex model — its deployed
answer latency is the FRAME CLOCK (1 text token = 1 LM frame = 80 ms),
so per-query local latency = token count x 0.08 s, immune to batching.
`modal_nvda.py::dump_scores` refit the winner probe (calib=frozen 600)
for per-query NVDA scores + exact Nemotron token counts; expert path =
per-query measured gpt-5.5 RTT + relay at frame rate. Same top-r
re-mix arithmetic on both models (`figures/nvda_fold_test.py`):
**NVDA strictly monotonic on both pools** (sllama 0.96→2.79 s over
0-60%, striviaqa 1.36→3.65 s; no dip anywhere), while the SAME
arithmetic on MiniCPM reproduces its dip (sllama 1.50→1.45) — a
same-math positive control. Mechanism as predicted: NVDA local P90 =
2.0 s < expert ~4 s, so escalation is a net time-add on every query.
The prediction is now an observation (under the stated frame-clock
convention; a full live port would add turn-take offsets, which are
constant and cannot create a fold). (2) TEXT models hedge too — gpt-5.5's own wrong answers on
sllama run ~2x longer; the claim is capability-relative (each model
hedges at ITS boundary), and the kink only needs the asymmetry that
TriviaQA sits inside the 9B's boundary (84 wrongs) but barely
intersects gpt-5.5's (8 wrongs), which is why the expert path is flat
~3 s. (3) swebq is null for ALL three models — its enumerative
Freebase-style answers pin length to question format and its strict
judge decouples "wrong" from "uncertain": neither evidence nor
counterexample. The kink itself does not require hedging universality
— only decode-time ∝ answer-length (an autoregressive identity,
r=.97) plus the probe selecting long-answer queries; a non-hedging
model just folds less.

**Addendum 3 (2026-08-24, user: "用实验说服我,不要 claim" — four
questions, all answered from the traces, $0).**

*Causation, alternatives killed (striviaqa n=250).* (A) "probe keys on
verbosity/style": corr(score, len) = +.05, and in len ~ wrong + score
+ audio the score coefficient conditioned on wrongness is −.08 ≈ 0 —
the probe selects WRONGNESS (corr .48); length rides along only
through wrong (β +.26). (B) "longer audio → longer answers":
corr(audio, len) = .03, β = +.04 — dead. (C) noise — 8ad. (D)
"hedging is just 'long' renamed": explicit hedge phrasings appear in
32% of wrong vs 7% of right answers, and WITHIN length bands hedge
still predicts wrong (short band 1.00 vs .27) — a real textual
behavior. Strongest piece is the NEGATIVE CONTROL: the probe selects
wrong equally everywhere (corr(esc, wrong) = .26-.32 in every pool),
but sdqa's wrongs are short and confident (P(wrong longer) = .48,
corr(esc, len) = −.12) and sdqa has NO fold (+0.20 s). Same probe,
same selection behavior, fold only where wrong happens to be slow —
so fold = (probe catches wrong) × (pool's wrongs are slow), and
"probe directly picks slow queries" is excluded. Self-correction: the
fold has TWO fuel lines — hedging on retrieval pools, and LONG CoT on
execution pools (sreason: char-length signature masked by CJK density,
but decode time directly: escalated 4.38 s vs kept 3.04 s).

*Router effectiveness on the phenomenon + why exactly one fold.*
Escalated-set local-wrong fraction (striviaqa, base .34): cons 68% /
bal 64% / agg 54% (lift x2.0/1.9/1.6); coverage of the pool's 84
wrongs: 31% / 57% / 80%. The single fold is quantile arithmetic:
local P50 falls monotonically (1.23→0.94→0.87→1.02 s) while the
~3.3-3.8 s expert mass share grows; at 15% the median still sits in
the (faster) local mass (1.19), by 30% it must cross expert mass
(1.37), 50% → 1.96. One left-fold then monotonic rise is the necessary
shape of a mixture median.

*Which benchmarks show it.* Fold requires (i) probe catches wrong
(all pools) AND (ii) wrong/hard is slow. Fuel present: sllama (.77
hedging, −.35 s), sreason (CoT, −.41 s), striviaqa (.62, −.04 s).
Fuel absent: swebq (enumerative format pins length, +.39 s), sdqa
(short confident wrongs, +.20 s), valpaca (everyone long). Predicts
NVDA (terse style) folds nowhere.

*Anti-hedge prompting (user Q4).* Testable for ~$10 (250-query never
arm, system-prompt line "If unsure, say 'I don't know' in five words
or fewer"). Expected honestly: hedging is symptom not cause — it
converts slow-wrong to fast-wrong (latency win, accuracy ~neutral,
trust possibly worse); AND it shifts the hidden-state distribution
under the frozen probe (8t/8z saw 5x score-scale shifts across
domains), so thresholds need re-quantiling — a new deployment
configuration, not a free patch. 8d's robust-prompt (transcription)
was negative but is a different behavior. Three readouts if run:
Δwrong-answer length, Δacc, Δprobe AUC/score drift. Awaiting go/no-go.

**Addendum 4 (2026-08-24, user: "答错但不 hedging 的情况存在吗?怎么验证
不确定→hedging?") — the verification FAILED and found something better.**
Design: the probe's end-of-turn read happens BEFORE generation, so it
is a temporally-prior proxy for the internal state; if internal
uncertainty causes hedging, the pre-answer score should predict which
failures will hedge. It does not: within wrong answers, AUC(score →
subsequent hedging) = **.488**, chance. And confident errors are the
MAJORITY: 57/84 wrongs (68%) carry no hedge phrasing and decode fast
(1.13 s). So the 8ab narrative link "uncertainty → hedging" is
RETRACTED at the individual level. What replaces it is stronger:
**the probe catches both error species identically** — hedged wrongs
score .634 / AUC .786 / 81% escalated @50%; confident wrongs score
.663 / AUC .800 / 81% — i.e. the L22 state carries the
failure-is-coming signal even when the surface text is confident:
verbal confidence ≠ internal state, and the gate does NOT depend on
hedging at all. Hedging's true correlate is SLOWNESS regardless of
correctness (hedged answers 3.3-3.4 s whether right or wrong; 12
right-but-hedged exist). Revised causal graph: internal
will-fail state → wrongness (probe reads this, both species); hedging
= an independent surface style bound to verbosity/latency; the two
correlate at the group level (32% vs 7%) without sharing the
probe-readable state. The kink story survives (the escalated set is
wrong-enriched, and wrongs contain the slow hedged subset) but the
anthropomorphic "the probe reads the hesitation" must be written as
"the probe reads the coming failure".

**Addendum 5 (2026-08-24, user go: "token 级熵轨迹做一下", ~$6, two
H100 passes).** `modal_bench.py::entropy_replay`: 93 striviaqa queries
in four behavior groups (hedged-wrong 27 / confident-wrong 27 / right
27 / hedged-right 12), replayed through the exact bench streaming path,
capturing per-step full-vocab entropy + P(terminator) via an lm_head
hook. Figure `entropy_traj.{png,pdf}`. The user's hypothesized chain
(retrieval failure → entropy up → EOS suppressed → hedging) is now
MEASURED at token level:

| group | first-5 entropy | traj median | steps | P(stop) at sentence boundaries |
|---|---:|---:|---:|---:|
| hedged-wrong | **.57** | .48 | 89 | **.0024** |
| confident-wrong | .50 | .32 | 34 | .0147 |
| right | .31 | .14 | 45 | **.083** |
| hedged-right | .36 | **.52** | 96 | .0001 |

(1) Trigger confirmed: hedged errors open at ~1.8x the entropy of
correct answers (P=.70) and wander high all the way. (2) EOS
suppression confirmed and large: at mid-answer sentence boundaries the
terminator carries **35x less probability** in hedged errors than in
correct answers — "keeps talking" is literally visible in the stop
token. (3) The two error species split at token level: confident
wrongs are SHORT (34 steps), low-trajectory-entropy (.32) — the output
distribution is deceived by the false fact, which is precisely why
entropy-based signals cap at AUC ~.70 while the L22 probe reaches .80
on both species. (4) Bonus: hedged-RIGHT answers open LOW (.36, like
correct) but wander highest and longest — early entropy tracks the
retrieval state, late entropy tracks the rambling style; the two
formerly-confounded quantities separate. Engineering footnote:
MiniCPM's actual terminator is token id **151704 and it is absent from
generation_config** (first run measured the wrong stop set and read
~1e-7 everywhere; recovered from the argmax tails).

**Case-study figure (user request: "从错题簿找一个典型,用 trace 说明拐弯
怎么出现"):** `kink_case_study.{png,pdf}` (gallery 图21) walks one real
sllama query through both worlds with measured milliseconds only —
sllama0164 "How many gurus are there in Sikhism?" (ref: Ten). Probe
OFF: 487-char ramble ("only one Guru… However… to avoid confusion"),
3.10 s, wrong. Probe ON (cons arm): running chunk scores .71/.67/.66,
21 ms end-of-turn read .631 ≥ .513 → escalate before a single answer
token exists; gpt-5.5 1.68 s + relay 0.62 s = 2.32 s, correct —
0.78 s faster AND right. Pool inset: the 38 such queries (local P50
2.38 s) leave the local queue, the remaining 212 drop to 0.94 s, arm
median 1.52 → 1.17 s = the fig-8 left-fold. **Web version (user: "更偏 demo
的"): `/cases` on the demo app —
https://rhe9527--gate-demo-web.modal.run/62dc5cd9/cases — six curated
real queries side-by-side, probe-OFF (never arm) vs probe-ON (the tier
that actually escalated/kept each), verbatim answers with hedge
phrasings highlighted, judge verdicts, probe-score-vs-threshold line,
latency segment bars. Covers the four fates: 2x escalate-faster-AND-
fix (sllama0164 3.10s wrong->2.32s right; Dame Edna 5.77->3.13), 1x
both-right-but-faster (5.07->3.87), 1x pay-latency-for-accuracy (Shema
2.49 wrong->6.86 right), 1x correctly-kept (score .049), 1x
confident-wrong missed at 15%/caught at 50% (Joe Gargery). Linked from
the main demo header.**

**图8 (sllama_pareto) latency zigzag decomposed → new figure
`sllama_latency_decomp.{png,pdf}`.** Two panels: (1) the fold lives
only in the median — the MEAN is monotonic (1.65/1.81/1.96/2.06 s vs
P50 1.52/1.17/1.60/1.42 s); (2) kept-local P50 falls 1.52→0.96→0.86→
0.43 s as the probe strips slow decodes while escalated rows pay a
flat ~3 s expert round-trip. The "weird" latency is a *positive*
property: on this pool escalation buys accuracy AND (at the median)
speed; the honest way to "remove the latency artifact" in a paper
figure is to show the mean alongside the median, not to re-measure.

**Escalation precision, v2→v3 at matched 50% rate** (rank never-arm
fail labels by the aggressive arm's live eot_score, take top half —
avoids the realized-rate confound; note the never-arm rows in the v3
parquets carry reused v2-era scores, so aggressive-arm scores are the
only valid v3 read):

| pool | v2 prec@50 | v3 prec@50 | base-fail | cap = base/rate |
|---|---:|---:|---:|---:|
| frozen | .858 | **.867** | .62 | 1.00 |
| striviaqa | .520 | **.544** | .34 | .68 |
| swebq | .576 | **.608** | .43 | .86 |
| sllama | .296 | **.304** | .16 | **.32** |
| sdqa | .720 | .720 | .48 | .96 |
| sreason | .525 | **.535** | .41 | .82 |

Key reframe for "can you push precision above 74%": precision at a
fixed escalation rate is CAPPED at base-fail/rate. The .74 Jisen
remembers was the v1 ours-fair receipt (base .56, esc 52% → cap ≈1);
on sllama the cap is .32 and v3's .304 is **95% of the theoretical
maximum** (recall .93). The honest dial is (a) AUC (v2→v3: OOF
.860→.879, ext-mean .750→.771) and (b) escalating at ≈ the base-fail
rate instead of a fixed 50%. Remaining levers unchanged from 8z:
calibration-pool width (the bigger lever, public families only),
judge-label denoising, per-domain threshold drift (the 8z-live
overshoot), asymmetric cost-sensitive thresholds.

**Random-escalation reference added to ALL pareto figures** (Jisen:
"所有图都可以加上 random escalation，不用重测"). Acc = the dualview
random line (pairs with the gold view); x = simulated P50 of a random
mixture at rate r — per-id local latency from the never arm, per-id
escalated latency where any arm escalated that id, pool-draw
otherwise (400 sims × 21 rates, seed-42 rng continuation). Patched
into bench_figures.py, fair_figures.py, valpaca_figures.py; all 14+1
figures regenerated and copied to paper/figures/. This supersedes the
2026-08-13 "no random line on sdqa" call for the PARETO view only
(dualview keeps the floor+ceiling design). Headline: on striviaqa and
sllama the gated curve now visibly dominates random in BOTH axes —
random needs ~3.4 s at the median to reach the ceiling striviaqa
region the gate reaches at 2.0 s, and on sllama random@50% sits at
~2.2 s/.88 vs the gate's 1.42 s/.948. Label bug fixed in the same
pass: figure subtitles said "probe v2" while VER="_v3" — now derived
from VER. `pareto_latency.py` (the frozen-pool paper figure) reads
pre-aggregated JSONs with no per-query rows, so its random line needs
a small rebuild from frozen_v3_traces — left as a todo.tex note.

**NVIDIA NemotronLabs-VoiceChat-11B scoped (Jisen #3).** HF card
verified live: 11B end-to-end **full-duplex** speech model — Fast
Conformer speech encoder + Nemotron Nano v2 9B (hybrid
Mamba/Transformer) + NVIDIA TTS decoder; in/out = user audio 16 kHz +
text → agent text + 22.05 kHz audio + user transcription; OpenMDW
1.1 license; vLLM + NeMo offline scripts + streaming WebSocket
container; claims VoiceBench #2 and Full-Duplex-Bench 1.0 #2 among
open FD models (0.82 smooth turn-taking, 448 ms). This is exactly the
§9 pre-registered prediction test ("a new open-weight full-duplex
model should show the late-layer text-input cliff"). Caveats before
committing spend: hybrid Mamba backbone means the L22-style layer
sweep must be redone from scratch (layer semantics differ; SSM states
vs attention residuals), and hidden-state hooks need the NeMo/HF
path, not the vLLM container. Plan (pre-registered order): (1) cliff
replication — text-vs-duplex layer×position sweep, predict a
late-layer cliff; (2) probe calibration on the same public calib
pool + frozen-methodology 4-arm live curve on the same 5 external
pools/judges → the transferability figure Jisen wants; (3) Anthony
trains/fine-tunes the NVDA model, we run the identical gate harness
on his checkpoints as the training-vs-routing ablation.

---

### 8ac — NVDA NemotronLabs-VoiceChat-11B: the second-duplex-family test executed (~$40, 2026-08-18/19)

User: "开始训练 nvda 的模型，结果放到 modal 界面" — interpreted (and
stated up front) as: train OUR probe on the new model's hidden states;
fine-tuning the model itself stays Anthony's ablation. Infra:
`modal_nvda.py` (download / smoke / run_answers / judge / fit), NeMo
Speech branch `nemotron-labs-voicechat`, 44 GB combined safetensors on
a new `nvda-weights` volume; same wavs, same judge (`escalate.
judge_many`), same recipe as eoth2/v3.

**Engineering receipts.** (1) Backbone = Nemotron Nano v2 9B, 56
NemotronHBlocks (27 Mamba2 / 4 attn / 25 MLP), reached at
`stt_model.llm.layers`; turn-taking is the agent text channel emitting
BOS/EOS (no separate head). (2) NeMo forces **cacheless** inference
for Nemotron (full prefix re-run per 80 ms frame) — so the FINAL
frame's forward contains every position, and one hook capture yields
the whole eot window + user-audio mean for free. (3) fp32 B=1 was
99.5 s/query; bf16 on the stt stack + length-bucketed batch-8 →
**4.3 s/query** with answers intact (23×). Frozen-pool math audio
(up to 3 min) OOMed fixed batches; adaptive batching (B × longest-wav
budget) recovered all 80. (4) Default system prompt makes the model
greet first — replaced with a QA prompt; answers carry `<$..$>/<|..|>`
timing markers — stripped before judging. Loading is 17.7 min/container
(fp32 key-by-key safetensors read).

**Floors (never-arm fail rate, our judge, offline replay):** frozen
.798, striviaqa .676, swebq .720, sllama .332, sdqa .690 — the 9B
Nemotron backbone is much weaker on knowledge retrieval than MiniCPM-o
(striviaqa local acc .32 vs .62 same judge). **Boundary finding:
sreason (Chinese) fail = 1.000 — VoiceChat is English-only** (Chinese
audio → fluent unrelated English hallucinations). The cross-lingual
transfer result has no analog on this model; pool skipped (zero label
variance).

**⭐ The §9 pre-registered test passes on first execution:**

- Layer sweep (eot_last, calib = frozen 600 only): mid-band peak
  L30-34 = **.714**, endpoints .693/.682 — the mid-band-readable
  structure replicates on a hybrid Mamba architecture.
- Same three reads @ L34: OOF .714 → .761 → **.790** — the same
  feature-stacking gains as MiniCPM.
- External transfer AUC: striviaqa **.781**, swebq **.793**, sdqa
  .754, sllama .701 — the MiniCPM v3 band (.79-.81) reached with a
  quarter of the calibration data.

Figures `nvda_layer_sweep` + `nvda_transfer` (图15/16) added to the
gallery and paper/figures. NOT yet done (next spend decisions): live
streaming 4-arm curve (needs the duplex loop ported to NeMo), calib
expansion to the full 2310, Anthony's fine-tuned checkpoints as the
training-vs-routing ablation.

### 8ad — noise audit + two superseded attributions + the stale live figures ($0 local, 2026-08-20)

Started as "fix the 8z-live threshold overshoot". The fix worked and
then falsified its own premise, which cascaded into an audit.

**Reconstruction method (new, $0).** Three properties of the live
sweeps make a CONTINUOUS rate-accuracy curve recoverable from the
existing traces: (1) the three gated arms carry bit-identical probe
scores (max spread 0.00000) so "top-r by score" is unambiguous; (2)
the tiers are perfectly NESTED (cons subset bal subset agg, 0
violations in 6 pools); (3) one query per session, so a query's
outcome does not depend on the arm's rate. Therefore for any r <=
agg-rate every selected id has a measured escalated outcome and every
other id has a measured local outcome (never arm). `figures/
rate_curve.py` -> `data/rate_curves.json`. Self-check: reconstruction
vs the measured arms deviates by .011 mean absolute (max .029),
consistent with the replication noise measured below.

**⚠️ SUPERSEDED #1 — the 8z-live overshoot attribution.** RESULTS 8z-live
attributed frozen aggressive .621→.596 to the calib-quantile threshold
firing at .613 instead of .50 ("pushes ~26 extra math/LaTeX queries
through the transcript-tax channel"). Re-mixing the same measured
outcomes at exactly 50% gives **.600 — i.e. +.004, nothing**. The rate
error is a COST bug, not an accuracy bug: correcting it removes 11.3%
of the expert calls at equal accuracy. `data/
gate_v3_thresholds_corrected.json` carries label-free corrected
quantiles for all 6 pools (only frozen drifts; the externals already
hit budget at 15/30/50).

**⚠️ SUPERSEDED #2 — "the small model BEATS the expert on the easy
half" (8x/8y).** Paired McNemar on the sllama aggressive arm: kept-local
half n=125, local .976 vs expert .968, discordant 2 vs 1, **p = 1.00 —
statistically indistinguishable, not "beats"**. The robust form of the
claim is (a) the probe's split is real and large — local acc .976
(kept) vs .696 (escalated), **z = 6.46**; (b) the expert's advantage is
concentrated entirely in the escalated half (.696→.888, discordant 27
vs 3, **p < .0001**); (c) therefore always-escalate spends 2× the
expert calls to buy nothing measurable over selective. The headline
"selective .948 > always .928" itself is 6-vs-1 discordant, **p =
.125** — directionally right, underpowered because both sit near
ceiling. Paper wording must move from "beats" to "matches at half the
cost".

**The replication-noise floor (why both corrections were needed).**
Same query, same audio, kept LOCAL in two different arms — the judge
verdict flips **2.3-18.8%** of the time (frozen .155, sdqa .188,
sreason .169, swebq .160, striviaqa .074, sllama .023). Repeat
ESCALATION flips 0.7-10.6%. Implied paired SE on an arm-vs-arm
accuracy delta: **.009-.028**. Against that floor, **all 18 live v2→v3
deltas are non-significant** (McNemar p = .16-1.00; the largest,
striviaqa balanced +.036, gives p = .16). This does not touch the
offline AUC gains (a much tighter statistic on the full score
distribution) — but the live-curve "confirmations" of them, including
sreason's cross-lingual +.010-.030, were over-read. Figure:
`noise_audit.{png,pdf}`.

**⚠️ Bug found while fixing the figures: the paper's two main LIVE
figures were two probe generations stale.** `live_dualview.json` /
`latency_profile.json` are written by `modal_stream.py::live_dualview`
off `gated_traces_v2.parquet` — that file is the streaming-loop-v2 /
probe-**v1** sweep (json mtime 2026-07-30). The v2 (8u) and v3
(8z-live) re-runs wrote `frozen_v{2,3}_traces.parquet` and only the
external-bench figures were refreshed, so fig:live and fig:pareto
still showed v1 arms. Rebuilt locally at $0 by `figures/
live_v3_figures.py` (v1 archived as `*_v1.{json,png,pdf}`):

| view | v1 (shown until today) | v3 (correct) |
|---|---|---|
| rates | 0/14/35/55% | 0/16/35/61% |
| heard | .400/.446/.529/.633 | .383/.483/.554/.596 |
| gold-inject | .400/.500/.637/.767 | .383/.525/.654/.771 |
| P50 latency | 2.02/2.76/4.00/4.69 s | 2.02/2.61/3.53/4.44 s |
| channel cost @agg | −.133 | **−.175** |

The v3 deployed curve is FLATTER and the channel cost LARGER: at
bal/agg the heard curve now sits below the gold-paired random line —
i.e. on this mixed pool the speech-channel tax (.175) exceeds the
selection margin, and only the channel-controlled (gold) view clears
random (+.061 at 61%). That is the honest headline for the frozen
pool and it strengthens the case for the audio-direct-to-expert lever.
`pareto_latency.py` also got the random-escalation curve it was
missing (8ab todo) and its hard-coded P99 text is now read from the
json (17.8→32.7 s, was 30.4).

**Consequence for "can we improve further".** Our measurement
precision on a 200-250-query pool (±.02-.03 per arm) is now the
binding constraint: any lever worth less than ~3 points is
undetectable in a single live sweep. Ranked next steps: (1)
audio-direct-to-expert — gold-inject says .175 sits in that channel on
this pool, far above the noise floor; (2) evaluate on paired/
variance-reduced statistics (AUC, matched-rate precision, the
reconstruction curve) rather than arm accuracy; (3) only then spend on
bigger n.

### 8ae — cloud-ASR uplink: the first lever in weeks that clears the noise floor (~$8, 2026-08-20)

User said "go" on the channel lever. Checking first stopped a bad
spend: **audio-direct-to-expert was already run (8r) and REJECTED by
the user the same day** — an audio-native expert costs -.15 of brain,
and a model-list check today confirms the premise still holds (audio
family = gpt-audio / gpt-audio-1.5 / gpt-audio-mini; **still no
gpt-5.5-class audio model**). So that arm was NOT re-run.

What had only been LOWER-bounded is the variant that keeps the frontier
TEXT brain: audio uplink -> hosted ASR -> gpt-5.5. 8d bounded it with
`openai/whisper-large-v3` (open weights, run locally) at +4pp. The
hosted frontier ASRs did not exist then. `modal_uplink2.py`, all 147
escalated frozen-pool ids, `gpt-transcribe` (auto-picked), same expert
protocol (`escalate.ask_expert`) and same judge (`escalate.judge_many`):

| arm | what the expert reads | acc (n=147) |
|---|---|---:|
| A deployed | MiniCPM's own self-transcript | .585 |
| **B cloud-ASR** | **gpt-transcribe of the same wav** | **.694** |
| C ceiling | the gold question text | .871 |

**B-A = +.109, McNemar p = .007** (24 rescued vs 8 broken) — ~5x the
paired SE (8ad), i.e. the first change in weeks that is unambiguously
real rather than noise, and **2.7x the whisper-large-v3 bound** the
lever was previously written off with. It recovers **38%** of the
gold gap; C-B = +.177 (p<.0001) remains.

Per-pool (n): easy-chat 28 **.679 -> .964**, hard-knowledge 50
.440 -> .560, trap 20 .550 -> .600, easy-fact 32 .906 -> .938,
**hard-math 17 .294 -> .294 (+.000, gold 1.000)**. The math wall is
exactly the 8q prediction: spoken LaTeX is lossy at the SOURCE, so no
ASR can recover it — the fix there is input-side (don't speak formulas)
not uplink-side. (easy-chat's B .964 > C .893 is open-ended judge
variance at n=28, not a real ASR-beats-gold effect.)

Deployment consequence: this is architecture-compatible with the
2026-08-12 text-backend decision — one extra ASR call per ESCALATED
turn only (~15-50% of turns depending on tier), the talker's own
transcript stops being load-bearing, and the expert keeps its frontier
brain. Not yet measured: the ASR call's added latency (the arm was run
offline; the call is a single short-audio request, but it belongs on
the critical path between EOT and the expert call, partially maskable
by the stall). Next: re-run one live 4-arm sweep with the uplink in the
loop to get the end-to-end curve + latency, and check whether the
+.109 survives on the external pools.

### 8af — interactive demo app (2026-08-20)

`demo_app.py` -> **https://rhe9527--gate-demo-web.modal.run/62dc5cd9**
Two modes, one probe ON/OFF switch, live metric tiles and an event log.

- **Replay ($0, scales to zero).** All 4773 measured sessions across 6
  pools. Flipping the probe OFF is not a simulation: 8ad established
  the tiers are nested with bit-identical scores, so the OFF view is
  the never-arm's MEASURED outcome for the same query. Shows the real
  per-chunk probe trace against the real per-domain threshold
  (`gate_v3_{pool}.json`), both answers, the judge verdict and the
  measured timings.
- **Live (opt-in H100, ~5 s warm / ~1 min cold).** Type anything ->
  tts-1/alloy (same voice as the frozen pool) -> MiniCPM streams it in
  1 s chunks -> v3 probe reads L22 at end-of-turn (rolling last-8 +
  running user-audio mean, byte-identical to `bench_live`) -> frozen
  threshold decides -> talker answers, or gpt-5.5 does and the talker
  relays it under a stall.

Verified end-to-end on the user's own test question. "What is NVDA
trading at right now?", balanced tier: 2.05 s of audio, 3 chunks,
running P(fail) .776/.676/.693, **end-of-turn read 54 ms -> P(fail)
.807 >= .680 -> ESCALATE**; expert 2.68 s, stall 37 ms, relay 2.33 s,
total 5.07 s. Same question probe OFF: local answer in 1.53 s. (Both
refuse honestly here — the talker also knows it lacks real-time data —
which makes it a good latency-cost illustration but a poor accuracy
one; the app ships three example questions including a long-tail fact
the talker got wrong in the sweep and the gate rescued.)

**Microphone input (user: "这个 demo 要让我能说话的").** The talker is a
speech model, so typing + TTS was a stand-in. The page now records from
the browser (MediaRecorder -> webm/opus), the container transcodes with
ffmpeg to 16 kHz mono and streams THAT into the duplex loop — no TTS
anywhere on this path. One consequence is scientifically useful: with
real speech there is no gold text, so the escalation uplink MUST be a
transcript, and the demo uses the 8ae hosted-ASR uplink (the arm
measured +.109 the same day) and shows the reader exactly what the
expert was told.

Both branches verified with real payloads built off the volume's own
audio (SD-QA human speech and a frozen-pool wav, re-encoded to
webm/opus so the request is byte-shaped like the browser's):

| | local branch (SD-QA, real human voice) | escalation branch (q0225, 48.7 s spoken MCQ) |
|---|---|---|
| end-of-turn read | 40 ms, P(fail) **.105** | 21 ms, P(fail) **.867** |
| gate | < .680 -> keep local | >= .680 -> **escalate** |
| outcome | correct answer in 1.6 s | ASR heard the full question -> gpt-5.5 "B. +7.3 J/mol" -> talker relayed it |
| total | 1.6 s | 13.2 s |

**This also closes 8ae's open latency question with a first datapoint:
the hosted-ASR call cost 4.81 s on the critical path** for a 48.7 s
clip (expert total 12.8 s, of which the talker's stall covered only
0.1 s). Short queries will pay far less, but the uplink is not free and
the stall phrase does not hide it — a live 4-arm sweep with the uplink
in the loop is still the number that matters.

Engineering notes for whoever redeploys: a live turn far exceeds the
web proxy's synchronous window, so the endpoint `spawn`s and the page
polls; `demo_app.py` imports `modal_app` at module level, so BOTH
images must mount `modal_app.py` or the web container dies before
serving (cost one confusing hang); and the mic needs HTTPS, which the
Modal URL already provides.

### 8ag — demo v2: continuous voice + GPU readiness gating (2026-08-21)

User feedback on 8af, both points valid: (1) a record-button is not a
voice conversation — they want to just TALK; (2) the mic must not be
clickable while the GPU is cold. Rebuilt the live path:

- **Resident GPU class** (`Voice`, modal.cls): the model loads once in
  `@enter` (~12-30 s off the warm volume), the browser's WebSocket
  lands on the same container, `scaledown_window=420`. `/ready` cannot
  return before `@enter` finishes, so the mic button being enabled IS
  the readiness proof — the page polls it with a visible elapsed
  counter and keeps the button disabled until then.
- **Continuous voice**: the page streams 16 kHz int16 PCM continuously
  (ScriptProcessor; no start/stop per turn). Server-side energy VAD
  (speech ≥0.2 s, then 1.25 s silence) ends the turn; then the same
  primitives as bench_live run: per-1s-chunk probe scores (streamed to
  the page live as you speak), the end-of-turn L22 read, the frozen
  threshold, local answer or 8ae-uplink escalation — then it resets
  and listens again. Multi-turn on one socket.
- Typed questions now go through the same warm container (`/say`) —
  no more per-turn cold model load.

Verified end-to-end with browser-shaped PCM streams built from the
volume's own audio: turn 0 (SD-QA human speech) P(fail) .126 -> local,
correct, 3.5 s of speech; turn 1 (spoken thermodynamics MCQ) P(fail)
.755 >= .680 -> escalated, hosted ASR transcript -> gpt-5.5 -> relay;
both turns on one socket, session survives into a third listen state.

Debug receipts (each cost a failed round): (1) the GPU image must
carry fastapi — the container crash-looped importing the in-container
ASGI app while /ready timed out silently for 8 min; the image now
replicates modal_app's proven MiniCPM spec verbatim + fastapi, because
Modal forbids stacking layers on an image that ends in add_local_dir.
(2) During cold start Modal serves /ready as a 303 long-poll redirect
chain — both the page and any client must POLL with short timeouts,
not follow one long request. (3) A WS upgrade against a cold container
times out at the proxy — always /ready first, then connect (the page
already did; the first test didn't). (4) VAD at 0.9 s cut a long
spoken question at a thinking pause -> 1.25 s. (5) A client that stops
streaming after its audio can strand the VAD one frame short of EOT
forever — a real mic never stops sending, and the test now mimics
that (background silence frames until the turn lands).

### 8ah — the "it just keeps listening" bug: two real defects, both invisible to clean-audio tests (2026-08-21)

User tried the voice demo: mic streams, model never answers, probe
on/off irrelevant. Root-caused to TWO independent defects, each of
which alone produces exactly that symptom, and NEITHER of which the
8ag test could catch because it streamed clean TTS/SD-QA audio and
stopped sending after each clip:

1. **Fixed VAD threshold vs real microphones.** The 8ag VAD used an
   absolute RMS threshold (0.010) tuned on clean wavs. Real mics have
   a noise floor and browser AGC pumps quiet passages, so silence
   never accumulates (or quiet speech never triggers) and the turn
   never ends. Fix: adaptive noise floor — EMA down 0.10 / up 0.02,
   up-adaptation only outside speech so long utterances don't erode
   their own threshold; speech = rms > max(.005, floor x 3.5). A
   first fix used instant-min tracking down and a single digitally
   silent frame (TTS inter-sentence zeros) collapsed the floor,
   making steady noise read as speech forever — hence the EMA.
2. **The post-answer drain ate the stream.** After each answer the
   server discarded backlogged frames "until a 0.05 s receive gap".
   A real mic never pauses, so the drain never exited and swallowed
   every subsequent utterance. Deleted outright: backlogged frames
   just flow through the VAD (quiet settles the floor, speech starts
   the next turn).

Also added, so this class of bug is diagnosable from the page instead
of by proxy: a live VAD readout (level / adaptive threshold / speech
state / silence progress, streamed every ~0.5 s) and an "I'm done
talking" button that forces end-of-turn if the detector misjudges —
the demo can no longer dead-end silently.

Regression test rebuilt to be mic-shaped (`_ws_test.py`): steady
rms .008 noise over EVERYTHING including the speech, continuous
frames with no gaps, immediate next utterance after an answer, plus a
manual-eot scenario. All three turns pass on one socket; turn 1's
eot read came out .671 vs the .680 threshold (fired last run at
.755) — the boundary sensitivity is the 8ad noise band doing exactly
what it says, and the local answer it kept was on track anyway.

Meta-lesson for the paper's demo section: every one of 8ag's five
receipts came from testing with idealized inputs; both 8ah defects
were only reachable with mic-shaped input. Test the transducer you
ship, not the files you have.

### 8ai — anti-hedge prompt arm: suppresses the behavior, costs 9.6 points, probe unmoved (~$5, 2026-08-24)

User: "先测4". `bench_live` gained a `sys_suffix` param (default empty
= byte-identical); `run_nohedge` appends to the stock omni persona:
"If you are not sure of the answer, say only 'I am not sure.' in five
words or fewer. Never explain your uncertainty or give background;
answer in one short sentence." Never arm, striviaqa 250, v3 probe
artifact, judged both scales; all comparisons paired on the same ids.

**1. The behavior is fully suppressible by prompt.** Median answer
181 -> 42 chars; WRONG answers 258 -> 38; local decode P50 1.20 ->
0.36 s and P90 3.31 -> **0.71 s** — the latency tail (the kink's
fuel) is annihilated, exactly as the 8ab mechanism predicts.

**2. It costs real accuracy: .664 -> .568 official judge (−.096,
McNemar 10 vs 34, p < .001)** (ours judge −.044, p = .15 — the OAB
judge rewards the context the short answers dropped). Decomposition of
the 34 right->wrong flips: **13 are explicit abstentions** ("I am not
sure") on questions the model previously got RIGHT while rambling —
its verbal self-assessment false-abstains at 43% (13 of 30
abstentions were on known items); **21 are information loss** (terse
answer drops the alias/context the judge needed, or a different
answer surfaces).

**3. The probe does not care about the persona.** First read looked
like drift (paired r = .45) — that was the 8ad scale artifact (stored
never-arm scores are v2-era). Same-generation comparison (v3
aggressive-arm scores vs v3 nohedge scores, same 250 ids): paired
**r = .95**, medians .468 vs .470, AUC .796 vs .813, virtual
escalation at the frozen corrected thresholds 15/30/50 -> 19/30/50.
**The L22 read is style-invariant: suppressing the hedging TEXT does
not touch the internal uncertainty STATE** — the strongest evidence
yet that the probe reads the state, not the style (and it makes the
probe strictly better calibrated than the model's own verbal
abstention: AUC .81 vs a 43% false-abstain rate at n=30).

**Verdict on Q4:** prompting can kill the symptom but it is a bad
trade — −9.6 points bought back latency the router already recovers
surgically (escalation FIXES the uncertain queries instead of
shortening them; the gate needs no persona change, no threshold
re-quantiling, no new deployment claim). The gate IS the correct
anti-hedge. Spend ≈ $5.

---






### 2.1 public pools ✅ (2026-07-07)

`build_public_queries` → **400 queries**: `hard-math` 150 (GSM8K test tail 100 +
MATH-500 50), `hard-knowledge` 150 (MMLU-Pro; **GPQA 401-gated** with no HF token
→ gracefully topped up from MMLU-Pro), `easy-fact` 100 (TriviaQA
unfiltered.nocontext). Dataset-loading + formatting code validated end-to-end.
Remaining 200 (easy-chat 150 + trap 50) are Claude-generated → need the secret.

Pure helpers (`src/queries.py`) unit-tested locally: MCQ formatting, GSM8K
reference extraction, and stratified 60/40 split (deterministic, seed 42) all pass.
