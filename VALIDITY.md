# Validity Review — MiniCPM-o 4.5 Full-Duplex Probes

This document records (1) the adversarial validity review of the first version of
the suite, (2) the fixes applied in response, and (3) the final, multi-run results
with causality controls. Methodology: each probe drives the duplex loop at 1 Hz,
injects a controlled event at a known chunk **T**, and scores the per-second
`is_listen` / `text` / `end_of_turn` / `kv_cache_length` stream. Behaviors are
sampled (temperature 0.7), so the headline numbers are **rates over 5 runs** plus
**no-injection control arms** to establish causality. (p8 is a single 300-chunk run.)

## 1. Overall verdict

The first pass produced plausible-looking results that an adversarial review showed
were partly **measurement artifacts** (off-by-one latency, truncated scoring
windows, precondition not met, keyword detectors). After fixing the metrics, adding
preconditions, control arms, and 5× repetition, the suite gives a defensible and
more interesting picture: MiniCPM-o 4.5 has **genuinely strong** visual-interruption,
cross-modal-conflict, proactive-alert, and online-revision behaviors; a **moderate**
verbal-stop behavior; and **real weaknesses** in backchanneling, long-horizon
latency stability, and memory under distractor load. Crucially, the corrected suite
also caught a confound the v1 "success" hid: the model **spontaneously code-switches
to English**, so the v1 "acts on a switch-to-English instruction" result is not
causally attributable.

## 2. What the adversarial review caught, and the fix

| Probe | v1 artifact | Fix applied |
|-------|-------------|-------------|
| p2 interruption | `stop_latency=0` — off-by-one (`first_eot_after(T)` inclusive) + a coincidental mid-word EOT at the injection chunk | search **sustained** listen strictly after T; add stop-acknowledgement signal; add no-stop control |
| p3 visual interruption | precondition unmet — model was *listening* (not speaking) at the swap, so it measured next-turn re-description | keep model speaking continuously; **assert speaking@T**; add no-swap control |
| p5 backchannel | verdict inverted to "good_listener" — scoring window truncated at monologue end + per-chunk char count | score the **full contiguous speaking run**; flag `interrupted_user` (onset ≤ last user-audio chunk) |
| p6 proactive | prime-acknowledgement miscounted as idle false-positive; no causal control | count idle FPs only after the ack turn ends **and** with alert keywords; add no-event control |
| p7 online correction | `used_correction_language=true` fired on the temporal word "现在" | tightened to genuine self-correction markers; split **narrated-new-count** vs **explicitly-corrected**; add no-change control |
| p8 long-horizon | not actually long (60 chunks); `crashed` compared run to its own length; drift mixed speak/listen regimes | check vs **intended N**; **within-listen-regime** drift; force periodic speaking; report KV growth rate; run 300 chunks |
| p9 memory | recall inflated — silent filler + model self-restatement (effective distance ~6, not 15) | real **distractor speech**; report **effective distance** from last restatement |
| p1 AV duplex | compliance via brittle substrings ("the","a ") | language-ID (CJK vs Latin) transition; **no-inject control** (which then exposed spontaneous code-switching) |
| harness core | `frame_track` pre-seeded frame 0 (latent swap corruption); `sess.stop()` skipped on exception; `stop_on_eot` truncation | `cur=None`; `stop()`/`finalize()` in `finally`; gate stop on "after last event"; `follows_which` negation-aware; peak-VRAM sampling |

## 3. Final results (5 runs each, with controls)

| # | Capability | Result | Verdict |
|---|-----------|--------|---------|
| 1 | AV full-duplex (act while speaking) | precondition (speaking, no EOT) **100%**; switches in 4/5 at ~2.5 chunks — **but no-inject control also switches 100%** | **confounded**: input-while-speaking is real (see p3/p7), but the switch-to-English test can't isolate causality (model code-switches spontaneously) |
| 2 | User interruption ("停") | sustained stop **causal 80%**, latency ~3.5 chunks (2–5); verbal acknowledgement 40% | **moderate**: usually stops, not always; the v1 "instant stop" was an artifact |
| 3 | Visual interruption (scene swap) | revises in-progress speech **causal 100%**, ~3.0 chunks; control never mentions B | **strong** |
| 4 | Cross-modal conflict (vision vs audio) | **trusts vision in both directions 100%** (red-frame/green-claim and mirror) | **strong** |
| 5 | Backchannel during monologue | **barges in 100%** (full sentence, ~33 chars), short-backchannel 0% | **weakness**: does not backchannel; interrupts the user |
| 6 | Proactive timing | silent when idle **100%**; proactive alert **causal 80%**, ~0.5 chunks; control never alerts | **strong (when primed)** |
| 7 | Online correction | narrates new count **causal 100%**, ~2.6 chunks; explicit-correction language **0%** | **strong behaviorally**, but it re-narrates rather than saying "I was wrong" |
| 8 | Long-horizon (1 run, 300 chunks ≈ 5 min) | completes, no crash; **speak-latency mean 942 ms, max 1659 ms, 72 chunks > 1 s**; listen-latency drift +167 ms; **KV 169→25 557 (84.9 tok/chunk, unbounded); VRAM +4.3 GB** | **weakness over time**: sustained generation breaks the 1 s real-time budget; KV/VRAM grow unbounded |
| 9 | Memory under distractors (5 runs) | recall **40%** at both queries with real distractor speech (vs 100% with silent filler); effective distance ~9–14 chunks | **weakness**: distractor speech derails retrieval |

## 4. What is credibly shown vs. what still needs a harder test

**Credibly shown** (causal, multi-run): the model **revises in-progress speech when
the scene changes** (p3), **trusts vision over a conflicting verbal claim both ways**
(p4), **stays silent when idle and proactively alerts on a salient event** (p6), and
**updates a factual claim on new visual evidence** (p7). These are exactly the
audio+vision+speech behaviors that the vision-only LiveSports-3K-CC benchmark cannot
measure, supporting the thesis that they are under-evaluated.

**Real weaknesses surfaced**: it **does not backchannel** — it barges in over a
monologue (p5); **latency degrades and breaks real-time under sustained generation**
with unbounded KV/VRAM growth (p8); and **memory recall collapses to ~40% under
distractor speech** (p9). The verbal-stop response (p2) is **only ~80% reliable**.

**Still needs a harder test**: p1's instruction-following-while-speaking is confounded
by spontaneous code-switching — it should use an instruction the model would not do
on its own (e.g. "stop mentioning colors", "count down from ten"). All probes are
n=5 on synthetic stimuli at 1 s granularity; publishable numbers would want larger N,
real video/audio, an LLM judge for the semantic probes (p4/p7), and — for p8/p9 — a
genuinely **tens-of-minutes** run and a **shrunken KV budget** to create real context
pressure.

## 5. Reproducing

```bash
cd harness
REPEAT=5 python run_eval.py p1 p2 p3 p4 p5 p6 p7 p9   # rates + controls
LONGHORIZON_CHUNKS=300 python run_eval.py p8          # stability run
python run_eval.py --report-only                      # rebuild report.md
```
Per-run traces: `results/<probe>.run{i}.jsonl`; controls: `results/<probe>.control.<name>.jsonl`.
