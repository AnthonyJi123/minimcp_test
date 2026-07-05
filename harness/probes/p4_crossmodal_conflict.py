"""P4 — Cross-modal Conflict: the image shows one colour while the user's speech
asserts a different colour. Does the model follow what it SEES or what it's TOLD?

Corrected: adds the counterbalanced MIRROR condition (green frame + "it's red"
claim) so a "trusts vision" verdict requires following vision in BOTH directions,
not a red/answer-what-you-see prior. follows_which is negation-aware.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timeline import Chunk, Scenario
from stimuli import card, utterance_chunks, silence_chunk, frame_track
from probe_base import Probe
from judge import joined_text, follows_which

SYS = "你是一个实时视频助手。请如实、简短地回答用户关于当前画面的问题。"
RED = (210, 30, 30)
GREEN = (30, 170, 60)
R = ["红", "red"]
G = ["绿", "green"]


def _scenario(color, claim_color_word, name):
    frame = card("", color=color)
    total = 12
    frames = frame_track([(0, frame)], total)
    audio = [silence_chunk() for _ in range(total)]
    claim = f"我觉得现在画面是{claim_color_word}的。请告诉我，画面现在到底是什么颜色？"
    for k, ch in enumerate(utterance_chunks(claim)):
        if k < total:
            audio[k] = ch
    return Scenario([Chunk(audio=audio[i], frame=frames[i]) for i in range(total)],
                    name=name, stop_on_eot=True, min_chunks_before_stop=6)


def build():
    return _scenario(RED, "绿色", "p4_crossmodal_conflict")   # vision=red, audio claims green


def score(records, scenario, controls) -> dict:
    main_txt = joined_text(records)
    main = follows_which(main_txt, R, G)            # vision=red
    mir = controls.get("mirror", [])
    mir_txt = joined_text(mir)
    mirror = follows_which(mir_txt, G, R)           # vision=green
    return {
        "main_vision_red_audio_green": main,
        "mirror_vision_green_audio_red": mirror,
        "trusts_vision_both_directions": main == "vision" and mirror == "vision",
        "transcript_main": main_txt[:160],
        "transcript_mirror": mir_txt[:160],
    }


PROBE = Probe(
    name="p4_crossmodal_conflict",
    capability="Cross-modal Conflict (trust vision vs audio)",
    system_prompt=SYS,
    build=build,
    score=score,
    stop_on_eot=True,
    controls={"mirror": lambda: _scenario(GREEN, "红色", "p4_crossmodal_conflict_mirror")},
)
