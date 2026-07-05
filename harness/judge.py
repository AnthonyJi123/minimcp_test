"""Lightweight judging for the semantic probes (cross-modal conflict, visual
interruption, online correction).

Default is rule-based keyword matching so the harness runs fully offline. If an
Anthropic API key is present, `llm_judge` can be used for fuzzier classification.
"""
from __future__ import annotations

import os
from typing import List, Optional


def _counts(text: str):
    cjk = sum(1 for c in (text or "") if "一" <= c <= "鿿")
    latin = sum(1 for c in (text or "") if ("a" <= c.lower() <= "z"))
    return cjk, latin


def is_mostly_english(text: str) -> bool:
    cjk, latin = _counts(text)
    return latin >= 3 and latin > cjk


def is_mostly_chinese(text: str) -> bool:
    cjk, latin = _counts(text)
    return cjk >= 2 and cjk >= latin


def mentions(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


def first_chunk_mentioning(records, keywords: List[str], start: int = 0) -> Optional[int]:
    """Index of the first record at/after `start` whose text mentions any keyword."""
    for r in records:
        if r.chunk_idx >= start and mentions(r.text, keywords):
            return r.chunk_idx
    return None


def joined_text(records, start: int = 0, end: Optional[int] = None) -> str:
    return " ".join(r.text for r in records
                    if r.chunk_idx >= start and (end is None or r.chunk_idx <= end) and r.text)


NEG_PREFIXES = ["不是", "不", "并非", "而不是", "not ", "no "]


def _affirmed(text: str, kw: str) -> bool:
    """True if `kw` appears affirmatively (not immediately negated like 不是绿)."""
    t = (text or "")
    i = 0
    while True:
        j = t.find(kw, i)
        if j < 0:
            return False
        pre = t[max(0, j - 3):j]
        if not any(pre.endswith(n.strip()) or n.strip() in pre for n in NEG_PREFIXES):
            return True
        i = j + len(kw)


def follows_which(text: str, vision_kw: List[str], audio_kw: List[str]) -> str:
    """Did the model follow the vision content or the (conflicting) audio claim?
    Negation-aware: '不是绿色，是红色' counts as vision, not 'both'."""
    v = any(_affirmed(text, k) for k in vision_kw)
    a = any(_affirmed(text, k) for k in audio_kw)
    if v and not a:
        return "vision"
    if a and not v:
        return "audio"
    if v and a:
        return "both"
    return "neither"


# -- optional LLM judge (Anthropic) ------------------------------------------

def llm_judge(question: str, transcript: str) -> Optional[str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user",
                       "content": f"{question}\n\nMODEL TRANSCRIPT:\n{transcript}\n\n"
                                  f"Answer concisely."}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"[llm_judge error: {e}]"
