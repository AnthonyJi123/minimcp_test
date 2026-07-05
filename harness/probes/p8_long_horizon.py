"""P8 — Long-Horizon Streaming: run the duplex loop for many chunks and check
stability — no crash, no per-chunk latency drift, no VRAM creep, bounded KV.

Corrected: `crashed` is checked against the INTENDED N (env), not the scenario's
own length; latency drift is computed WITHIN the listen regime (comparing all-speak
head to all-listen tail was meaningless); we report KV growth rate and force
periodic speaking so the expensive generation path is stressed across the whole
run. Set LONGHORIZON_CHUNKS to push to tens of minutes (e.g. 1200 ~ 20 min).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timeline import Chunk, Scenario
from stimuli import card, utterance_chunks, silence_chunk, frame_track
from probe_base import Probe

N = int(os.environ.get("LONGHORIZON_CHUNKS", "120"))
NUDGE_EVERY = 40
SYS = ("你是一个实时监控助手。请安静观察画面，"
       "当被询问时用一句话简短描述你看到的内容。")


def build() -> Scenario:
    scene = card("LOBBY", color=(50, 55, 70), shape="square", shape_color=(150, 150, 170))
    frames = frame_track([(0, scene)], N)
    audio = [silence_chunk() for _ in range(N)]
    nudge = utterance_chunks("请简短描述一下现在的画面。")
    # re-nudge periodically so the model speaks across the whole horizon
    for start in range(0, N, NUDGE_EVERY):
        for k, ch in enumerate(nudge):
            if start + k < N:
                audio[start + k] = ch
    return Scenario([Chunk(audio=audio[i], frame=frames[i], label=f"t{i}") for i in range(N)],
                    name="p8_long_horizon")


def _mean(xs):
    return round(sum(xs) / len(xs), 1) if xs else None


def score(records, scenario, controls) -> dict:
    n = len(records)
    listen = [r for r in records if r.is_listen]
    speak = [r for r in records if r.speaking]
    ll = [r.gen_latency_ms for r in listen]
    sl = [r.gen_latency_ms for r in speak]
    kv = [r.kv_cache_length for r in records if r.kv_cache_length is not None]
    vram = [r.vram_mb for r in records if r.vram_mb is not None]
    # within-regime drift (listen): first 10 vs last 10 listen chunks
    lh, lt = ll[:10], ll[-10:]
    return {
        "intended_chunks": N,
        "completed_chunks": n,
        "crashed": n < N,
        "speaking_chunks": len(speak),
        "listen_latency_head_mean_ms": _mean(lh),
        "listen_latency_tail_mean_ms": _mean(lt),
        "listen_latency_drift_ms": (round(_mean(lt) - _mean(lh), 1)
                                    if lh and lt else None),
        "speak_latency_mean_ms": _mean(sl),
        "speak_latency_max_ms": round(max(sl), 1) if sl else None,
        "real_time_violation_chunks": sum(1 for r in records if r.gen_latency_ms > 1000),
        "kv_start": kv[0] if kv else None,
        "kv_end": kv[-1] if kv else None,
        "kv_tokens_per_chunk": round((kv[-1] - kv[0]) / max(n - 1, 1), 1) if len(kv) > 1 else None,
        "kv_bounded": None,   # informational: monotonic growth expected (no pruning in this model)
        "vram_mb_start": vram[0] if vram else None,
        "vram_mb_end": vram[-1] if vram else None,
        "vram_growth_mb": round(vram[-1] - vram[0], 1) if len(vram) > 1 else None,
    }


PROBE = Probe(
    name="p8_long_horizon",
    capability="Long-Horizon Streaming (stability over time)",
    system_prompt=SYS,
    build=build,
    score=score,
    stop_on_eot=False,
)
