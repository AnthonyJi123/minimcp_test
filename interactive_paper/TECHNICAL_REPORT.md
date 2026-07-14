# When Does a Small Model Know to Hand Off?
## Zero-Training Escalation Gates for MiniCPM-o 4.5

**Technical Report — v2** (v1: 2026-07-13; v2 adds Phase 5b: audit deep-dive,
p(True) baseline, 3-backbone replication)
Date: 2026-07-14 · Seed: 42 · Status: Phases 0–5 + 5b complete; Phase 6 (duplex) not started

---

## Abstract

This is **step 1 of a larger plan**: teach a small conversational model to know
*when to hand a query off* to a large model (step 2 — how the big model's result
comes back — is future work). We attach zero-training escalation gates to
MiniCPM-o 4.5 (9B omni) and evaluate them against small-model failure labels on
600 public-benchmark queries.

**v1 finding (probe).** A linear probe on the prompt's last hidden state predicts
failure at **AUC 0.822** [0.777, 0.863] and, as a gate, beats random escalation at
every operating point (area **+0.054** [+0.040, +0.068]); escalating 33% of
traffic recovers ~58% of the small→big accuracy gap.

**v1 audit.** The probe's number is largely a **query-type shortcut**: h_prompt
classifies the source pool at 95.8%, a pool-identity oracle alone reaches AUC
0.715, and appending true pool labels to the probe adds nothing (0.822 → 0.821).
Leave-one-pool-out, the probe *inverts* on math (0.372) and scores 100%-fail trap
questions as low-risk (0.23) — surface familiarity, not self-knowledge.

**v2 finding (the revision).** Simply **asking the model** — reading P(Yes) off
the first token of "Would you answer this correctly?" (pre) or "Is this answer
correct?" (post) — with zero training and zero calibration:

- **ptrue_post beats the trained probe on all three backbones tested**
  (0.899 / 0.897 / 0.814 vs probe 0.822 / 0.838 / 0.758), and yields the best
  test-set tradeoff curve (area +0.068 vs probe's +0.054; 0.821 accuracy at 38%
  escalation).
- **Pre-answer trap introspection is universal (3/3)**: on 100%-fail SimpleQA
  long-tail questions the model says "No, I won't get this right" (MiniCPM mean
  score 0.945 vs 0.309 elsewhere) — *if asked before it answers*. Asked after,
  it believes its own confident-wrong answer (0.945 → 0.360).
- **Probe transferability is a property of the backbone, not the method**: LOPO
  transfer is fine on raw Qwen3-8B (math 0.961), partial on Mistral-7B, inverted
  on MiniCPM-o; a label-coverage-matched control (`lopo_matched`) rules out
  labeling differences — the omni fine-tune changed the representation.
- **Entropy is the least portable signal**: ≈ chance on 2 of 3 backbones.

**Step-1 answer, revised:** genuine zero-training self-knowledge exists and is
*verbalizable*; it is not (reliably) linearly readable from hidden states. The
hand-off gate should be built on verbalized self-eval — pre-answer for
pre-decode firing, post-answer check where a draft is affordable — with the
probe at best a free in-distribution auxiliary.

---

## 1. Setup

| Component | Value |
|---|---|
| Small model | MiniCPM-o 4.5 (9B, Qwen3 backbone), bf16, SDPA, text mode |
| Replication backbones | Qwen/Qwen3-8B (raw, no-think), mistralai/Mistral-7B-Instruct-v0.3 |
| Infra | Modal, H100s; torch 2.8.0, transformers 4.51.0 (pinned) |
| Judge | `gpt-5.4-mini` (blind, structured outputs); escalation target `gpt-5.5` |
| Signals | forward hooks on last decoder layer + lm_head (hooked text ≡ plain text) |
| Determinism | greedy, seed 42; 600 queries frozen, 360 calib / 240 test |

**Query pool (600, public benchmarks only):** trap = SimpleQA 50; hard-knowledge
= MMLU-Pro 150; easy-fact = TriviaQA 100; easy-chat = dolly-15k + alpaca-zh 150;
hard-math = GSM8K tail + MATH-500 150. MiniCPM per-pool fail rates: trap 1.00,
knowledge .48, fact .34, chat .21, math .19 (overall .358).

**Candidate gate signals (all zero-training):**

1. *Scalars*: token entropy / logit margin over the first K decode steps.
2. *Probe*: logistic regression on `h_prompt` (4096-d, captured at prefill) —
   fit on calib, thresholds from 5-fold OOF scores (in-sample is separable/1.000).
3. *p(True), added in v2*: P(Yes) from first-token logits of a self-eval prompt —
   **pre** ("Would you answer this correctly?" before answering) and **post**
   ("Is this proposed answer correct?" given its own stored answer). No fitting
   of any kind; Yes/No token mass median ≈ 1.0, so the read is clean.

---

## 2. RQ1 — which signal knows the model will fail?

### 2.1 Headline AUCs (MiniCPM, calib rows, probe = 5-fold OOF)

| signal | AUC | training needed |
|---|---:|---|
| **ptrue_post** | **0.899** | none |
| pre+post mean | 0.877 | none |
| probe on h_prompt | 0.821 [0.777, 0.863] | LR on calib |
| **ptrue_pre** | 0.807 | none |
| probe on h_mean8 | 0.776 | LR on calib |
| max_entropy@4 (best scalar) | 0.696 | none |

### 2.2 What the probe actually reads (audit + audit2)

- **Type shortcut, evidence closed:** h_prompt → 5-way pool classifier = 95.8%;
  pool-identity oracle = 0.715; probe + true pool dummies = 0.821 vs probe alone
  0.822 (**zero added information** — h_prompt already contains the full type
  signal). Within-pool (shortcut controlled) macro AUC = 0.742.
- **No cross-type transfer:** LOPO math 0.372 (inverts), LOPO trap mean score
  0.232 on 100%-fail questions (would not escalate).
- **Math inversion root-caused (audit2):** math failure ≈ "is it MATH-500"
  (corr +0.59; MATH-500 fail .55 vs GSM8K .04), and the LOPO probe scores terse
  symbolic MATH-500 *safer* than verbose GSM8K stories (corr −0.246, length ≈
  irrelevant). The probe reads *style familiarity*; competition-terse text looks
  "clean/safe" to a probe trained on knowledge/chat/trap pools.

### 2.3 p(True): the signal the probe was missing

Per-pool AUC — for p(True) this **is** the transfer condition (it has no
calibration pool):

| pool | probe OOF | probe LOPO | ptrue_pre | ptrue_post |
|---|---:|---:|---:|---:|
| easy-chat | .696 | .679 | .708 | .812 |
| easy-fact | .792 | .660 | .764 | .855 |
| hard-knowledge | .638 | .563 | .688 | .797 |
| hard-math | .851 | **.377** | .809 | **.944** |
| trap (100% fail) | n/a | score .33 (miss) | **score .945 — all caught** | score .360 |

Two mechanistic reads:

- **Retrieval-type failures are introspectable pre-answer**: the model knows at
  read time whether it knows a long-tail entity — but only *says* so if asked;
  the information didn't surface in a linear h_prompt direction on this model.
- **Ask before it answers.** Post-hoc self-check collapses exactly on traps
  (0.945 → 0.360): once a confident-wrong answer exists, the model endorses it.
  Conversely post beats pre everywhere *else* (seeing a draft helps on math:
  0.944). Pre and post are complementary, not redundant.

---

## 3. RQ2 — end-to-end tradeoff (frozen test, n=240)

Conditions: small-only **0.588** [.525, .650]; big-only (gpt-5.5) **0.917**
[.879, .950]; paraphrase-relayed big **0.879** [.833, .917] (relay tax 1–4 pts).

| gate signal | area vs random | acc @ ~30–38% escalation | thr transfer calib→test |
|---|---:|---:|---|
| probe (Phase-5 headline) | +0.0541 [+.040, +.068] | 0.779 @ 33% | poor (rates drift) |
| ptrue_pre | +0.0594 | 0.779 @ 36% | good |
| **ptrue_post** | **+0.0682** | **0.821 @ 38%** | good |

(`figures/tradeoff_ptrue.png`; thresholds = calib score quantiles, no fitting.)
All gates beat random escalation everywhere; the better AUC does translate into
a better curve. Latency/cost: expert P50 3.0 s / P95 24.4 s; big-only $1.12/100q,
hybrid ∝ escalation rate. Deployment shape: ptrue_pre = one extra short prefill
*before* decoding (fits streaming); ptrue_post = draft-then-check (the draft is
what step-2 injection consumes anyway).

---

## 4. Generality — 3-backbone replication

Same 600 queries, same judge; each backbone gets its own labels (its own
failures), signals, probe, and p(True).

| check | MiniCPM-o 4.5 | Qwen3-8B (raw) | Mistral-7B-v0.3 |
|---|---:|---:|---:|
| overall fail rate | .358 | .394 | .519 |
| probe OOF AUC | 0.822 | 0.838 | 0.758 |
| pool-oracle AUC | 0.715 | 0.704 | 0.730 |
| probe − oracle edge | +0.107 | +0.134 | **+0.028 (≈ nothing)** |
| pool-classifier acc | 95.8% | 95.8% | 95.3% |
| max_entropy@4 | 0.696 | **0.468** | **0.501** |
| LOPO math | **0.372** | **0.961** | 0.817 |
| LOPO fact | 0.717 | 0.596 | **0.445** |
| **ptrue_post** | **0.899** | **0.897** | **0.814** |
| ptrue_pre | 0.807 | 0.736 | 0.723 |
| trap introspection (pre) | score .945 | AUC .939 | AUC .837 (post .901) |

Findings:

1. **ptrue_post > trained probe on all three backbones.** The most portable
   signal in the study.
2. **The type shortcut is universal** (~95% pool classification everywhere); on
   Mistral the probe is *barely more* than a type classifier (+0.028 over oracle).
3. **Probe transfer is backbone-dependent, and the confound is resolved**:
   `lopo_matched` subsamples Qwen's training pools to MiniCPM's exact fail rates
   — LOPO-math stays 0.962–0.968 (5 seeds). Not label coverage: the
   **representation** differs. Raw Qwen3-8B encodes transferable difficulty
   linearly; its omni-fine-tuned sibling does not. (Since MiniCPM-o *is*
   fine-tuned Qwen3-8B, this is an unusually controlled comparison.)
4. **Entropy ≈ chance on 2 of 3 backbones** — the classic uncertainty scalar is
   the least reliable component of the folk wisdom.
5. **Pre-answer trap introspection is 3-for-3** — the "knows it doesn't know
   long-tail facts" effect is not a MiniCPM quirk.

---

## 5. Revised step-1 conclusion

> A deployable hand-off gate does not need training — but it should **ask the
> model, not probe it**. Verbalized self-evaluation (p(True)) beats a trained
> hidden-state probe in aggregate, transfers across query types without any
> calibration distribution, catches the deadliest case (confidently-wrong
> long-tail facts) that the probe misses entirely, and replicates across three
> backbones. The linear probe's headline AUC is mostly query-type recognition;
> its transferability depends on which model you probe. Entropy should not be
> trusted at all without per-model validation.

Design consequence for the eventual duplex system (step 2 / Phase 6):

- **Stage 1 (pre-decode):** ptrue_pre — one extra short prefill; fires before
  the first token; catches traps.
- **Stage 2 (draft-check):** ptrue_post on the small model's draft — strongest
  overall signal; the draft doubles as the distilled query for escalation and as
  fallback output. Both stages remain zero-training.

## 6. Remaining gaps (honest list)

- **Judge variance unbounded**: single judge (gpt-5.4-mini), single greedy run.
  Bootstrap CIs cover sampling noise, not judge bias. (A second judge model +
  human spot-check is the cheapest remaining credibility win.)
- **p(True) prompt sensitivity untested**: one phrasing of pre/post each; MCQ
  queries occasionally leak option letters into the first token (mass ≈ 0 rows,
  rare — median mass 1.0).
- **ptrue thresholds vs judge**: p(True) scores cluster near 0/1; calibration of
  *probability* (not ranking) unexamined — fine for thresholding, not for cost
  modeling.
- **Test split reused** for the v2 signal comparison (protocol identical to
  Phase 5, signals pre-specified, but a fresh test set would be cleaner for a
  paper).
- **Combining signals** (ptrue_pre + ptrue_post + probe) unexplored beyond a
  naive mean (0.877 — *below* ptrue_post alone).
- **Phase 6 (duplex/streaming, RQ3)** untouched — the original omni motivation.
- Total project spend ≈ **$50** (GPU + API) of the $2000 budget.

## 7. Artifacts

`RESULTS.md` (full log incl. Phase 5b), `gate_config.json`, `src/` (gate.py,
decode.py, hf_decode.py, escalate.py, distill.py, inject.py, queries.py,
signals.py, test_gate.py), `modal_app.py` (all entry points: …, audit2,
collect_ptrue/run_ptrue/ptrue_analyze, ptrue_gate_eval, run_signals_hf/label_hf/
run_ptrue_hf/xmodel_report, lopo_matched), figures/roc.png, figures/tradeoff.png,
figures/tradeoff_ptrue.png. Volumes: `gate-data` (queries, signals, features,
ptrue shards, per-backbone features), `minicpm-o45-weights` (+ qwen3-8b,
mistral-7b snapshots).
