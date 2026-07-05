"""P3 — Visual Interruption: the scene changes while the model is *mid-utterance*.

Corrected: v1 swapped during a listen gap, so it only measured next-turn
re-description. Here we keep the model speaking continuously (detailed-commentary
prompt) and assert the precondition that it was speaking at T; if not, the run is
marked invalid. A no-swap control (scene stays A) confirms the model wouldn't
mention B spontaneously.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timeline import Chunk, Scenario
from stimuli import card, utterance_chunks, silence_chunk, frame_track
from probe_base import Probe, latency_chunks, was_speaking
from judge import first_chunk_mentioning, joined_text

SYS = ("你是一个实时视频解说助手。请用中文不停地、详细地描述你当前看到的画面，"
       "画面变化时立刻改为描述新画面。")
A_KW = ["红", "圆", "circle", "red"]
B_KW = ["绿", "三角", "triangle", "green"]
T = 6


def _scenario(swap: bool) -> Scenario:
    a = card("A", color=(10, 10, 30), shape="circle", shape_color=(230, 40, 40))
    b = card("B", color=(10, 40, 10), shape="triangle", shape_color=(40, 220, 60))
    total = 18
    spec = [(0, a)] + ([(T, b)] if swap else [])
    frames = frame_track(spec, total)
    audio = [silence_chunk() for _ in range(total)]
    for k, ch in enumerate(utterance_chunks("请开始不停地、详细地描述你看到的画面。")):
        if k < total:
            audio[k] = ch
    return Scenario(
        [Chunk(audio=audio[i], frame=frames[i],
               event="SCENE_SWAP" if (swap and i == T) else "") for i in range(total)],
        name="p3_visual_interruption" + ("" if swap else "_control"),
    )


def build():
    return _scenario(swap=True)


def score(records, scenario, controls) -> dict:
    speaking_at_T = was_speaking(records, T - 1) or was_speaking(records, T)
    b_idx = first_chunk_mentioning(records, B_KW, start=T)
    # contamination: still describing A after the swap?
    a_after = first_chunk_mentioning([r for r in records if r.chunk_idx >= T], A_KW, start=T)
    ctrl = controls.get("no_swap", [])
    ctrl_b = first_chunk_mentioning(ctrl, B_KW, start=T) if ctrl else None
    return {
        "precondition_speaking_at_T": bool(speaking_at_T),
        "swap_chunk_T": T,
        "described_B_after_swap_chunk": b_idx,
        "revise_latency_chunks": latency_chunks(T, b_idx),
        "still_described_A_after_swap": a_after,
        "control_mentioned_B_without_swap": ctrl_b is not None,
        "valid_causal": bool(speaking_at_T and b_idx is not None and ctrl_b is None),
        "transcript_before": joined_text(records, 0, T - 1)[:160],
        "transcript_after": joined_text(records, T)[:160],
    }


PROBE = Probe(
    name="p3_visual_interruption",
    capability="Visual Interruption (revise in-progress speech on scene change)",
    system_prompt=SYS,
    build=build,
    score=score,
    stop_on_eot=False,
    controls={"no_swap": lambda: _scenario(swap=False)},
)
