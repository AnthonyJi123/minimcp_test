"""Phase 4: inject the expert answer back into the small model, which relays it
to the user in its own words (colloquial paraphrase; the expert wins conflicts).

Design note: for a text ACCURACY eval this step can only match or hurt vs. the
raw expert answer (the small model may corrupt it) — it exists for the
conversational / duplex UX (speak the result naturally). Phase 5 therefore
reports hybrid accuracy BOTH ways (raw-expert-inject vs. small-model paraphrase);
the gap is the faithfulness cost of relaying through the small model.
"""
from __future__ import annotations

import decode  # flat import: src/ is on sys.path in the container

INJECT_INSTRUCTION = (
    "An expert was consulted. Using ONLY the expert result below, answer the "
    "user's question in your own words — clearly, completely, and faithfully. "
    "Do not add facts beyond the expert result. If it conflicts with anything you "
    "might have said before, defer to the expert."
)


def paraphrase(model, tok, query: str, expert_answer: str,
               max_new_tokens: int = 512) -> str:
    prompt = (f"{INJECT_INSTRUCTION}\n\nUser's question:\n{query}\n\n"
              f"<result>\n{expert_answer}\n</result>")
    return decode.generate(model, tok, prompt, max_new_tokens=max_new_tokens)
