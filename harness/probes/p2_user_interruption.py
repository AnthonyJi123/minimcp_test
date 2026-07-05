"""P2 — User Interruption: while the model is speaking, the user says "停".

Corrected metric (the v1 reported a spurious 0): we search for a *sustained*
listen state strictly AFTER T (>=2 consecutive silent-listen chunks), and also
look for an explicit stop-acknowledgement ("停下来"/"好的"). A no-injection
control measures when the model would naturally end its turn, so we can tell a
real stop from the model simply finishing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timeline import Chunk, Scenario
from stimuli import card, utterance_chunks, silence_chunk, frame_track
from probe_base import (Probe, latency_chunks, was_speaking,
                        sustained_listen_after, first_eot_after)
from judge import joined_text, mentions

SYS = ("你是一个实时视频解说助手。请用中文持续、详细地解说你看到的画面，"
       "不要停顿，除非用户要求你停止。")
STOP_UTTERANCE = "停，请停下来。"
ACK_KW = ["停下来", "停下", "好的，我停", "我停", "不说了", "停止"]
T = 9


def _scenario(inject: bool) -> Scenario:
    scene = card("MATCH", color=(20, 110, 40), shape="circle", shape_color=(240, 240, 240))
    total = 20
    frames = frame_track([(0, scene)], total)
    audio = [silence_chunk() for _ in range(total)]
    for k, ch in enumerate(utterance_chunks("请开始非常详细地解说这个画面，持续不断地说。")):
        if k < total:
            audio[k] = ch
    if inject:
        for k, ch in enumerate(utterance_chunks(STOP_UTTERANCE)):
            if T + k < total:
                audio[T + k] = ch
    return Scenario(
        [Chunk(audio=audio[i], frame=frames[i],
               event="INJECT_STOP" if (inject and i == T) else "") for i in range(total)],
        name="p2_user_interruption" + ("" if inject else "_control"),
    )


def build():
    return _scenario(inject=True)


def score(records, scenario, controls) -> dict:
    was_speaking_at_T = was_speaking(records, T - 1) or was_speaking(records, T)
    stop_idx = sustained_listen_after(records, T, k=2)
    ack = any(r.chunk_idx >= T and mentions(r.text, ACK_KW) for r in records)
    # control: natural sustained-listen onset with no "停" injected
    ctrl = controls.get("no_stop", [])
    ctrl_stop_idx = sustained_listen_after(ctrl, T, k=2) if ctrl else None
    lat = latency_chunks(T, stop_idx)
    ctrl_lat = latency_chunks(T, ctrl_stop_idx)
    return {
        "precondition_speaking_before_stop": bool(was_speaking_at_T),
        "inject_chunk_T": T,
        "sustained_stop_chunk": stop_idx,
        "stop_latency_chunks": lat,
        "acknowledged_stop": bool(ack),
        "control_natural_stop_latency_chunks": ctrl_lat,
        "stop_is_causal": bool(was_speaking_at_T and lat is not None
                               and (ctrl_lat is None or lat < ctrl_lat)),
        "transcript_around_stop": joined_text(records, T, T + 6)[:200],
    }


PROBE = Probe(
    name="p2_user_interruption",
    capability="User Interruption (stop after 停)",
    system_prompt=SYS,
    build=build,
    score=score,
    stop_on_eot=False,
    controls={"no_stop": lambda: _scenario(inject=False)},
)
