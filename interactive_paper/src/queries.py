"""Pure helpers for building the calibration/eval query pool (no I/O, testable).

Record schema (one per query):
    {"id", "pool", "query", "reference_answer" (str|None), "split"}
Pools: easy-chat, easy-fact, hard-math, hard-knowledge, trap.
"""
from __future__ import annotations

import string


def mcq_prompt(question: str, options: list[str]) -> str:
    """Format a multiple-choice question, asking for the answer letter + content."""
    letters = string.ascii_uppercase
    lines = [question, ""]
    for i, opt in enumerate(options):
        lines.append(f"{letters[i]}. {opt}")
    lines.append("")
    lines.append("Answer with the correct option letter and its content.")
    return "\n".join(lines)


def mcq_reference(options: list[str], answer_index: int) -> str:
    letters = string.ascii_uppercase
    return f"{letters[answer_index]}. {options[answer_index]}"


def gsm8k_reference(answer_field: str) -> str:
    """GSM8K answer fields end with '#### <final>' — return the final answer."""
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip()
    return answer_field.strip()


def make_split(records: list[dict], calib_frac: float = 0.6, seed: int = 42) -> None:
    """Assign a 'split' field per record, stratified within pool, seeded shuffle.

    Mutates records in place. calib = first calib_frac after a per-pool shuffle.
    """
    import random
    by_pool: dict[str, list[dict]] = {}
    for r in records:
        by_pool.setdefault(r["pool"], []).append(r)
    for pool, rs in by_pool.items():
        rng = random.Random(f"{seed}:{pool}")
        idx = list(range(len(rs)))
        rng.shuffle(idx)
        n_calib = round(len(rs) * calib_frac)
        calib_ids = set(idx[:n_calib])
        for j, r in enumerate(rs):
            r["split"] = "calib" if j in calib_ids else "test"
