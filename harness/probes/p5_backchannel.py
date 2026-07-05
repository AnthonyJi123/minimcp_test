"""P5 — Backchannel: during a long user monologue, a well-behaved duplex model
should keep listening (maybe short "嗯"/"明白"), not barge in with full sentences.

Corrected: v1 truncated the scoring window at MONOLOGUE_END and used per-chunk
char counts, which mislabelled a full barge-in as "good_listener". Now we (a)
detect interruption = first speaking onset *before the user's last audio chunk*,
and (b) classify the whole contiguous speaking segment by its concatenated length,
not per-chunk fragments.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timeline import Chunk, Scenario
from stimuli import card, utterance_chunks, silence_chunk, frame_track
from probe_base import Probe, speaking_onset_after, speaking_run_from
from judge import mentions

SYS = ("你是一个耐心的语音助手。用户正在讲述一件事，请在用户说话时保持倾听，"
       "必要时只用简短的回应（如“嗩”“明白”），不要打断用户。")
MONOLOGUE = ("我跟你说，我今天早上出门的时候，发现地铁居然停运了，"
             "然后我只好打车，结果路上又特别堵，司机还走错了路，"
             "我迟到了快半个小时，到公司之后老板的脸色特别难看，"
             "我解释了半天他才勉强相信我，真的是非常倒霉的一天。")
BACKCHANNEL_KW = ["嗯", "明白", "好的", "哦", "嗯嗯", "是的"]


def build() -> Scenario:
    scene = card("", color=(40, 40, 50))
    mono = utterance_chunks(MONOLOGUE)
    total = len(mono) + 6                      # generous tail so a barge-in is fully captured
    frames = frame_track([(0, scene)], total)
    audio = [silence_chunk() for _ in range(total)]
    for k, ch in enumerate(mono):
        audio[k] = ch
    last_audio = len(mono) - 1
    return Scenario(
        [Chunk(audio=audio[i], frame=frames[i],
               label="monologue" if i <= last_audio else "after",
               event="MONOLOGUE_END" if i == last_audio else "") for i in range(total)],
        name="p5_backchannel",
    )


def score(records, scenario, controls) -> dict:
    last_audio = scenario.event_index("MONOLOGUE_END")
    onset = speaking_onset_after(records, 0)
    _, run_end, run_text = speaking_run_from(records, onset)
    seg_len = len(run_text.strip())
    interrupted = onset is not None and onset <= last_audio
    is_backchannel = seg_len <= 6 and mentions(run_text, BACKCHANNEL_KW)
    if onset is None:
        verdict = "stayed_silent"
    elif interrupted and not is_backchannel:
        verdict = "barges_in"
    elif is_backchannel:
        verdict = "backchannels"
    else:
        verdict = "replied_after_user_finished"
    return {
        "last_user_audio_chunk": last_audio,
        "first_speaking_onset_chunk": onset,
        "interrupted_user": bool(interrupted),
        "speaking_segment_chars": seg_len,
        "is_short_backchannel": bool(is_backchannel),
        "verdict": verdict,
        "segment_text": run_text[:160],
    }


PROBE = Probe(
    name="p5_backchannel",
    capability="Backchannel (listen vs. barge in during monologue)",
    system_prompt=SYS,
    build=build,
    score=score,
    stop_on_eot=False,
)
