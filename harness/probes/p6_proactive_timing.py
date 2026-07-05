"""P6 — Proactive Timing: stay silent during an idle scene, proactively speak
when something noteworthy appears.

Corrected: v1 mislabelled the model's acknowledgement of the priming instruction
as an "idle false positive", and had no control for causality. Now: idle
false-positives are only counted AFTER the prime-acknowledgement turn ends (first
end_of_turn) and must contain alert keywords; proactive speech must reference the
alert; and a NO-EVENT control (scene stays idle) establishes that the speech was
caused by the event, not the model's natural cadence.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timeline import Chunk, Scenario
from stimuli import card, silence_chunk, utterance_chunks, frame_track
from probe_base import Probe, latency_chunks, first_eot_after
from judge import joined_text, mentions

SYS = ("你是一个实时监控助手。请安静观察画面，平时不要说话；"
       "一旦画面出现火焰、警告或危险标志，立刻主动用中文提醒我。")
PRIME = "看到画面里出现火焰或者警告标志时，请第一时间提醒我。"
ALERT_KW = ["火", "警", "注意", "危险", "警告", "提醒"]
T = 10


def _scenario(event: bool) -> Scenario:
    boring = card("", color=(45, 45, 45))
    alert = card("FIRE 火警 !!!", color=(210, 30, 0), shape="triangle", shape_color=(255, 220, 0))
    total = 20
    spec = [(0, boring)] + ([(T, alert)] if event else [])
    frames = frame_track(spec, total)
    audio = [silence_chunk() for _ in range(total)]
    for k, ch in enumerate(utterance_chunks(PRIME)):
        if k < total:
            audio[k] = ch
    return Scenario(
        [Chunk(audio=audio[i], frame=frames[i],
               label="idle" if (not event or i < T) else "alert",
               event="SALIENT_EVENT" if (event and i == T) else "") for i in range(total)],
        name="p6_proactive_timing" + ("" if event else "_control"),
    )


def build():
    return _scenario(event=True)


def score(records, scenario, controls) -> dict:
    T_ = scenario.event_index("SALIENT_EVENT")
    ack_end = first_eot_after(records, 0)          # end of the prime-acknowledgement turn
    idle_start = (ack_end + 1) if ack_end is not None else 5
    # genuine idle false positive: alert-like speech in the truly-idle window
    idle_fp = [r for r in records if idle_start <= r.chunk_idx < T_
               and r.speaking and mentions(r.text, ALERT_KW)]
    # proactive speech must reference the alert
    proactive_idx = next((r.chunk_idx for r in records
                          if r.chunk_idx >= T_ and r.speaking and mentions(r.text, ALERT_KW)), None)
    # control: no event -> should NOT produce alert speech around T
    ctrl = controls.get("no_event", [])
    ctrl_alert = any(r.chunk_idx >= T_ and r.speaking and mentions(r.text, ALERT_KW) for r in ctrl)
    return {
        "event_chunk_T": T_,
        "idle_false_positive_alerts": len(idle_fp),
        "stayed_silent_when_idle": len(idle_fp) == 0,
        "proactive_alert_chunk": proactive_idx,
        "proactive_latency_chunks": latency_chunks(T_, proactive_idx),
        "control_alerted_without_event": ctrl_alert,
        "proactive_is_causal": bool(proactive_idx is not None and not ctrl_alert),
        "transcript_after_event": joined_text(records, T_)[:200],
    }


PROBE = Probe(
    name="p6_proactive_timing",
    capability="Proactive Timing (silent when idle, alert on salient event)",
    system_prompt=SYS,
    build=build,
    score=score,
    stop_on_eot=False,
    controls={"no_event": lambda: _scenario(event=False)},
)
