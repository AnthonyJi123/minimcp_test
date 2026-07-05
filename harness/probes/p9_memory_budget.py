"""P9 — Memory Budget: present a passphrase early, then query it later.

Corrected reporting: the model tends to repeat the passphrase back when first
told, which reinforces it in its own KV cache. So in addition to raw recall we
compute the EFFECTIVE distance = chunks from the LAST occurrence of the
passphrase in the stream (including the model's own restatements) to each query.
We also replace silent filler with unrelated distractor speech so the fact is
genuinely buried, and report KV length at query time. (A true fixed-budget /
eviction test would require shrinking the model's KV cap — noted as future work.)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timeline import Chunk, Scenario
from stimuli import card, utterance_chunks, silence_chunk, frame_track
from probe_base import Probe

CODE_NUM = "7392"
CODE_WORD = "河马"
FACT = "请牢牢记住，今天的暗号是 蓝色河马 七三九二。"
DISTRACTOR = "另外随便聊聊，今天天气不错，我中午吃了牛肉面，下午还要开个会。"
Q1, Q2 = "请问今天的暗号是什么？", "再确认一次，暗号是什么？"
q1_at, q2_at = 15, 32


def _place(audio, text, start):
    for k, ch in enumerate(utterance_chunks(text)):
        if start + k < len(audio):
            audio[start + k] = ch


def build() -> Scenario:
    total = 40
    scene = card("ROOM", color=(48, 48, 60))
    frames = frame_track([(0, scene)], total)
    audio = [silence_chunk() for _ in range(total)]
    _place(audio, FACT, 0)
    _place(audio, DISTRACTOR, 7)        # real distractor speech between fact and query
    _place(audio, DISTRACTOR, 22)
    _place(audio, Q1, q1_at)
    _place(audio, Q2, q2_at)
    return Scenario(
        [Chunk(audio=audio[i], frame=frames[i],
               event=("QUERY1" if i == q1_at else ("QUERY2" if i == q2_at else "")))
         for i in range(total)],
        name="p9_memory_budget",
    )


def _has_code(text):
    return (CODE_NUM in (text or "")) or (CODE_WORD in (text or ""))


def _recall(records, q_idx, window=6):
    txt = " ".join(r.text for r in records if q_idx <= r.chunk_idx <= q_idx + window)
    return _has_code(txt), txt


def _last_code_before(records, q_idx):
    """Last chunk < q_idx where the passphrase appears in the model's OWN output."""
    last = None
    for r in records:
        if r.chunk_idx < q_idx and r.speaking and _has_code(r.text):
            last = r.chunk_idx
    return last


def score(records, scenario, controls) -> dict:
    q1 = scenario.event_index("QUERY1")
    q2 = scenario.event_index("QUERY2")
    r1, t1 = _recall(records, q1)
    r2, t2 = _recall(records, q2)
    kv_at = {r.chunk_idx: r.kv_cache_length for r in records}
    restate1 = _last_code_before(records, q1)
    restate2 = _last_code_before(records, q2)
    return {
        "query1_chunk": q1, "recall_at_q1": r1, "kv_at_q1": kv_at.get(q1),
        "effective_distance_q1": (q1 - restate1) if restate1 is not None else q1,
        "query2_chunk": q2, "recall_at_q2": r2, "kv_at_q2": kv_at.get(q2),
        "effective_distance_q2": (q2 - restate2) if restate2 is not None else q2,
        "model_restated_passphrase": restate1 is not None,
        "answer_q1": t1[:100], "answer_q2": t2[:100],
        "caveat": "no fixed KV budget reached; effective distance reflects model self-restatement",
    }


PROBE = Probe(
    name="p9_memory_budget",
    capability="Memory Budget (retain key fact across distance)",
    system_prompt=("你是一个有记忆的语音助手。请牢牢记住用户告诉你的关键信息，"
                   "在之后被问到时准确复述。"),
    build=build,
    score=score,
    stop_on_eot=False,
)
