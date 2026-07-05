"""P1 — Audio-Visual Full Duplex (infra validation + core claim).

Claim: the model receives user speech *while it is speaking* and acts on it.
Scenario: model commentates continuously; mid-utterance we inject a spoken
"switch to English" instruction at T. Compliance = a Chinese->English transition
detected by character-script ratio (not brittle substrings). Precondition
(model speaking with no end_of_turn around T) is asserted; if unmet the result is
marked invalid rather than reporting a bogus latency.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timeline import Chunk, Scenario
from stimuli import card, utterance_chunks, silence_chunk, frame_track
from probe_base import Probe, latency_chunks, was_speaking
from judge import joined_text, is_mostly_english, is_mostly_chinese

SYS = ("你是一个实时视频助手。请用中文持续、详细地解说你看到的画面，"
       "不要停顿，并随时听取并执行用户的语音指令。")
INSTRUCTION = "请改用英文继续解说。"


def _scenario(inject: bool) -> Scenario:
    scene = card("STREET", color=(60, 70, 90), shape="square", shape_color=(200, 200, 60))
    total = 20
    frames = frame_track([(0, scene)], total)
    audio = [silence_chunk() for _ in range(total)]
    for k, ch in enumerate(utterance_chunks("请开始持续、详细地解说你看到的画面。")):
        if k < total:
            audio[k] = ch
    T = 8
    if inject:
        for k, ch in enumerate(utterance_chunks(INSTRUCTION)):
            if T + k < total:
                audio[T + k] = ch
    return Scenario(
        [Chunk(audio=audio[i], frame=frames[i],
               event="INJECT_INSTRUCTION" if (inject and i == T) else "") for i in range(total)],
        name="p1_av_fullduplex" + ("" if inject else "_control"),
    )


def build():
    return _scenario(inject=True)


def score(records, scenario, controls) -> dict:
    T = scenario.event_index("INJECT_INSTRUCTION")
    # precondition: speaking just before T, and Chinese (so a switch is observable)
    precond = was_speaking(records, T - 1) and is_mostly_chinese(
        joined_text(records, max(0, T - 3), T - 1))
    eng_idx = next((r.chunk_idx for r in records
                    if r.chunk_idx >= T and is_mostly_english(r.text)), None)
    # control: with no instruction, the model should NOT switch to English
    ctrl = controls.get("no_inject", [])
    ctrl_switched = any(is_mostly_english(r.text) for r in ctrl)
    return {
        "precondition_met_speaking_cn_before_T": bool(precond),
        "inject_chunk_T": T,
        "switched_to_english_chunk": eng_idx,
        "comply_latency_chunks": latency_chunks(T, eng_idx),
        "control_switched_spontaneously": ctrl_switched,
        "valid_causal": bool(precond and eng_idx is not None and not ctrl_switched),
        "transcript": joined_text(records)[:300],
    }


PROBE = Probe(
    name="p1_av_fullduplex",
    capability="Audio-Visual Full Duplex (act on speech while speaking)",
    system_prompt=SYS,
    build=build,
    score=score,
    stop_on_eot=False,
    controls={"no_inject": lambda: _scenario(inject=False)},
)
