"""Big Bench Audio: prompt + rule-based answer extraction + accuracy.

Dataset `ArtificialAnalysis/big_bench_audio` (1000 spoken BIG-Bench-Hard
questions). Four categories, each with a constrained gold `official_answer`:

    formal_fallacies -> "valid" / "invalid"
    navigate         -> "Yes" / "No"
    web_of_lies      -> "Yes" / "No"
    object_counting  -> an integer count

The question is entirely inside the audio, so the text prompt only tells the
model to answer concisely; we then extract the answer from its free text.
Artificial Analysis's official pipeline uses an LLM judge — rule-based
extraction is the pilot's cheap stand-in. We keep every raw output so a judge
pass can be bolted on if extraction turns out to miss real correct answers.
"""
from __future__ import annotations

import re

PROMPT = ("You will hear a reasoning question. Answer it. "
          "State your final answer clearly and concisely at the end.")

_WORD2NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}


def _extract_int(text: str):
    t = text.lower()
    m = re.findall(r"-?\d+", t)
    if m:
        return int(m[-1])  # last number = usually the concluded count
    for w, n in _WORD2NUM.items():
        if re.search(rf"\b{w}\b", t):
            return n
    return None


def _has(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text.lower()) is not None


def is_correct(category: str, official_answer: str, output: str) -> bool:
    gold = (official_answer or "").strip().lower()
    out = (output or "").lower()
    if gold in ("valid", "invalid"):
        # check the longer word first so "invalid" isn't read as "valid"
        if _has(out, "invalid"):
            return gold == "invalid"
        if _has(out, "valid"):
            return gold == "valid"
        return False
    if gold in ("yes", "no"):
        yes, no = _has(out, "yes"), _has(out, "no")
        if yes and not no:
            return gold == "yes"
        if no and not yes:
            return gold == "no"
        # both/neither present: fall back to the last one mentioned
        iy = out.rfind("yes")
        ino = out.rfind("no")
        if iy == ino == -1:
            return False
        return gold == ("yes" if iy > ino else "no")
    # object counting: compare integers
    g = _extract_int(gold)
    o = _extract_int(out)
    return g is not None and g == o


def score(rows: list) -> dict:
    """rows: [{category, official_answer, output, ok?}]. Returns overall + per-cat."""
    from collections import defaultdict

    per = defaultdict(lambda: [0, 0])
    tot = [0, 0]
    for r in rows:
        ok = r.get("ok")
        if ok is None:
            ok = is_correct(r["category"], r["official_answer"], r["output"])
        per[r["category"]][0] += int(ok)
        per[r["category"]][1] += 1
        tot[0] += int(ok)
        tot[1] += 1
    return {
        "n": tot[1],
        "accuracy": tot[0] / tot[1] if tot[1] else None,
        "per_category": {c: {"acc": v[0] / v[1], "n": v[1]} for c, v in per.items()},
    }
