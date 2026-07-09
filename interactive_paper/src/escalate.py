"""OpenAI API roles for the gate: the judge (Phase 2 labels).

  * judge — gpt-5.4-mini grades the small model's answer as adequate / not
            (produces the Phase-2 escalate labels).

(Query GENERATION was removed 2026-07-08: all five pools now come from public
datasets — the GPT-generated trap pool failed its design and generator==judge
is a circularity. See modal_app.build_public_queries.)

The escalation target itself (gpt-5.5) lives in Phase 4 (src/inject etc.).

Provider switched from Anthropic → OpenAI on 2026-07-08 (user has OpenAI
credit, not Anthropic). Model ids from the OpenAI pricing/models pages
(July 2026): judge/gen = gpt-5.4-mini, escalation target = gpt-5.5. These are
GPT-5.x *reasoning* models — hidden reasoning tokens bill as output, so calls
use reasoning_effort="low" (schema adherence without heavy spend) and a
generous max_completion_tokens (reasoning eats the budget before the visible
JSON). All calls use Structured Outputs (response_format json_schema, strict)
for parse-free reliability. The client reads OPENAI_API_KEY from the env
(injected by the Modal `openai` secret). If an id 404s, pin a dated snapshot.
"""
from __future__ import annotations

JUDGE_MODEL = "gpt-5.4-mini"
JUDGE_EFFORT = "low"        # reasoning_effort for GPT-5.x judge calls
JUDGE_MAX_TOKENS = 4096     # room for reasoning tokens + the small JSON verdict

# Fixed judge prompt — stored in repo for reproducibility (PLAN Phase 2.2).
JUDGE_SYSTEM = (
    "You are a strict, fair grader of a small language model's answers. "
    "You are given a QUESTION, an optional REFERENCE answer, and a CANDIDATE "
    "answer produced by a small model. Decide whether the CANDIDATE is ADEQUATE, "
    "meaning ALL of: (1) factually correct; (2) any reasoning shown is valid; "
    "(3) it actually and completely answers the question that was asked. "
    "For multiple-choice questions the candidate must land on the correct option "
    "(by letter or by unambiguously stating the option's content). A partially "
    "correct, hedged, refused, off-topic, or truncated answer is NOT adequate. "
    "If a REFERENCE is provided, grade against it. Be strict but do not penalize "
    "harmless differences in format or extra correct detail."
)

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string",
                   "description": "one sentence justifying the verdict"},
        "adequate": {"type": "boolean"},
    },
    "required": ["reason", "adequate"],
    "additionalProperties": False,
}


def _resp_format(name: str, schema: dict) -> dict:
    """OpenAI Structured Outputs wrapper (strict json_schema)."""
    return {"type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": True}}


def _client():
    from openai import OpenAI
    return OpenAI()


def _async_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI()


def _content(resp) -> str:
    """Pull the JSON string from a chat completion, or raise on refusal/truncation."""
    msg = resp.choices[0].message
    if getattr(msg, "refusal", None):
        raise RuntimeError(f"model refused: {msg.refusal}")
    if not msg.content:
        raise RuntimeError(
            f"empty content (finish_reason={resp.choices[0].finish_reason}); "
            "reasoning likely consumed max_completion_tokens")
    return msg.content


def _judge_user(query: str, reference: str | None, answer: str) -> str:
    ref = f"\n\nREFERENCE answer:\n{reference}" if reference else ""
    return (f"QUESTION:\n{query}{ref}\n\nCANDIDATE answer (small model):\n{answer}\n\n"
            "Grade the candidate.")


def judge(query: str, reference: str | None, answer: str) -> dict:
    """Synchronous single judge call → {'adequate': bool, 'reason': str}."""
    import json
    resp = _client().chat.completions.create(
        model=JUDGE_MODEL, reasoning_effort=JUDGE_EFFORT,
        max_completion_tokens=JUDGE_MAX_TOKENS,
        response_format=_resp_format("verdict", _JUDGE_SCHEMA),
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user",
                   "content": _judge_user(query, reference, answer)}],
    )
    return json.loads(_content(resp))


async def judge_many(rows: list[dict], concurrency: int = 8) -> list[dict]:
    """Label a list of {query, reference_answer, answer} rows concurrently.

    Returns the same rows with 'adequate', 'judge_reason', 'escalate_label' added.
    """
    import asyncio
    import json

    client = _async_client()
    sem = asyncio.Semaphore(concurrency)

    async def one(row):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=JUDGE_MODEL, reasoning_effort=JUDGE_EFFORT,
                    max_completion_tokens=JUDGE_MAX_TOKENS,
                    response_format=_resp_format("verdict", _JUDGE_SCHEMA),
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": _judge_user(
                                  row["query"], row.get("reference_answer"),
                                  row["answer"])}],
                )
                v = json.loads(_content(resp))
                row["adequate"] = bool(v["adequate"])
                row["judge_reason"] = v.get("reason", "")
            except Exception as e:  # keep the row; mark unjudged
                row["adequate"] = None
                row["judge_reason"] = f"ERROR: {e}"
            row["escalate_label"] = (None if row["adequate"] is None
                                     else int(not row["adequate"]))
            return row

    return await asyncio.gather(*(one(r) for r in rows))
