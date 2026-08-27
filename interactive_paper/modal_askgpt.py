# -*- coding: utf-8 -*-
"""One-shot: ask gpt-5.5 for small-experiment suggestions on the
last-layer cliff (8-27 advisor request). Run:
  modal run modal_askgpt.py
"""
import modal

app = modal.App("askgpt-interp")
IMG = modal.Image.debian_slim().pip_install("openai")
OPENAI = modal.Secret.from_name("openai")

PROMPT = """You are advising on a NeurIPS interactive-track paper about a
full-duplex speech LLM (MiniCPM-o 4.5, Qwen3-8B backbone, 36 layers).

Core finding: a linear probe on the FINAL-layer last-token hidden state,
predicting whether the model will fail a spoken query, INVERTS under
leave-one-pool-out transfer (AUC 0.372 on held-out math), while the same
probe at mid-network L22 holds 0.93. The raw (non-duplex) backbone shows
no cliff at any layer. Mean-pooled reads survive at the final layer but
0.13-0.15 below the mid-layer last-token peak. The cliff is specific to
(late layer x last token) and appears only after duplex fine-tuning.
Our hypothesis: duplex training repurposes the late-layer last-token
residual stream for turn-taking control (listen/speak/barge-in), a
Listen-or-Speak state machine, crowding out instance-level competence.

Available artifacts (no new GPU captures unless tiny): float16 dumps of
EVERY layer's hidden state (last-token + mean-pooled) for ~600 queries,
for: duplex model text-input, duplex model audio-input, raw backbone
text-input, a second duplex/backbone pair (MiniCPM-o 2.6 / Qwen2.5-7B),
an omni-streaming control (Qwen2.5-Omni). Labels: per-query fail/success,
source pool (5 pools), modality. Model weights accessible for logit-lens
/ small forward passes on a single GPU if strictly necessary.

List the 5-8 most decisive SMALL experiments (hours, not days; one
figure) to (1) support or falsify the turn-control-repurposing story and
(2) explain WHY the final layer inverts rather than merely degrades.
For each: name, exact computation, expected result if our story is true,
and what result would falsify it. Rank by evidence-per-effort. Be
concrete and terse."""


@app.function(image=IMG, secrets=[OPENAI], timeout=600)
def ask() -> str:
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": PROMPT}],
    )
    return r.choices[0].message.content


@app.local_entrypoint()
def main():
    out = ask.remote()
    with open("askgpt_interp.md", "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
