"""Phase 4: distill the (possibly multi-turn) user context into a standalone
question for the expert model.

Design note (single-turn caveat): in this text eval every query is ALREADY a
self-contained question, so distillation is near-identity and can only lose
information — its real payoff is multi-turn / duplex, where you must NOT ship the
whole conversation to the expert (Phase 6). Implemented per PLAN §Phase-4 and
exercised in e2e_demo; the Phase-5 eval escalates the ORIGINAL query so the small
model can't distort an already-standalone question.
"""
from __future__ import annotations

import decode  # flat import: src/ is on sys.path in the container

DISTILL_INSTRUCTION = (
    "You are helping a user and need to consult an expert. Rewrite the user's "
    "request below as a single, self-contained question for the expert — include "
    "any essential background, keep all specifics, and do not answer it yourself. "
    "At most 200 words. Output only the question."
)


def distill_query(model, tok, query: str, max_new_tokens: int = 256) -> str:
    prompt = f"{DISTILL_INSTRUCTION}\n\nUser's request:\n{query}"
    return decode.generate(model, tok, prompt, max_new_tokens=max_new_tokens)
