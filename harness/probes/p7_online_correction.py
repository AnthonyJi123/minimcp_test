"""P7 — Online Correction: model counts 2 dots, then the scene changes to 5 dots.
Does it revise, and does it explicitly correct itself?

Corrected: separates "narrated new count" (said 5 after change) from "explicitly
corrected" (used genuine self-correction language — dropped the temporal word
现在 that v1 falsely matched). A no-change control (dots stay 2) confirms the new
count is evidence-driven, not per-frame re-narration.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timeline import Chunk, Scenario
from stimuli import dots, utterance_chunks, silence_chunk, frame_track
from probe_base import Probe, latency_chunks
from judge import first_chunk_mentioning, joined_text, mentions

SYS = ("你是一个实时视频助手。请数一数画面里有几个圆点并说出来，"
       "如果数量发生变化，请立即纠正并说出新的数量。")
TWO_KW = ["2", "二", "两"]
FIVE_KW = ["5", "五"]
CORRECTION_KW = ["其实", "更正", "抱歉", "不是", "变成", "变为", "改为"]   # genuine self-correction (no 现在)
T = 7


def _scenario(change: bool) -> Scenario:
    total = 16
    spec = [(0, dots(2))] + ([(T, dots(5))] if change else [])
    frames = frame_track(spec, total)
    audio = [silence_chunk() for _ in range(total)]
    for k, ch in enumerate(utterance_chunks("请数一数画面里有几个圆点。")):
        if k < total:
            audio[k] = ch
    return Scenario(
        [Chunk(audio=audio[i], frame=frames[i],
               event="COUNT_CHANGE" if (change and i == T) else "") for i in range(total)],
        name="p7_online_correction" + ("" if change else "_control"),
    )


def build():
    return _scenario(change=True)


def score(records, scenario, controls) -> dict:
    two_before = first_chunk_mentioning(records, TWO_KW, 0)
    stated_two = two_before is not None and two_before < T
    five_after = first_chunk_mentioning(records, FIVE_KW, start=T)
    after_txt = joined_text(records, T)
    ctrl = controls.get("no_change", [])
    ctrl_five = first_chunk_mentioning(ctrl, FIVE_KW, 0) if ctrl else None
    return {
        "change_chunk_T": T,
        "stated_2_before_change": bool(stated_two),
        "narrated_new_count_5_chunk": five_after,
        "narration_latency_chunks": latency_chunks(T, five_after),
        "explicitly_corrected": mentions(after_txt, CORRECTION_KW),
        "control_said_5_without_change": ctrl_five is not None,
        "revision_is_causal": bool(stated_two and five_after is not None and ctrl_five is None),
        "transcript": joined_text(records)[:240],
    }


PROBE = Probe(
    name="p7_online_correction",
    capability="Online Correction (revise prior claim on new evidence)",
    system_prompt=SYS,
    build=build,
    score=score,
    stop_on_eot=False,
    controls={"no_change": lambda: _scenario(change=False)},
)
