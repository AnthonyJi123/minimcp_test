"""Probe contract shared by all 9 capability probes.

A probe declares a system prompt, builds a Scenario (timeline of stimuli with a
named event marking the reference point T), and scores the recorded chunk stream
into a metric dict. Probes may also declare `controls`: named alternate scenarios
(e.g. a no-injection baseline) used to establish causality. run_eval loads the
model once and drives the main scenario + every control, passing the control
traces to score().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from timeline import Scenario


@dataclass
class Probe:
    name: str                       # e.g. "p2_user_interruption"
    capability: str                 # human-readable capability name
    system_prompt: str
    build: Callable[[], Scenario]   # () -> Scenario  (main arm)
    score: Callable                 # (records, scenario, controls) -> metric dict
    stop_on_eot: bool = False
    # name -> () -> Scenario ; each run with the same system_prompt, traces passed to score
    controls: Dict[str, Callable[[], Scenario]] = field(default_factory=dict)


# ---- reference-point latency helpers ---------------------------------------

def latency_chunks(event_idx, hit_idx):
    """Chunks from event T to first satisfying chunk (None if never/invalid)."""
    if event_idx is None or hit_idx is None:
        return None
    d = hit_idx - event_idx
    return d if d >= 0 else None


def was_speaking(records, idx) -> bool:
    for r in records:
        if r.chunk_idx == idx:
            return r.speaking and bool(r.text.strip())
    return False


def speaking_onset_after(records, start):
    """First chunk index >= start where the model is speaking with text."""
    for r in records:
        if r.chunk_idx >= start and r.speaking and r.text.strip():
            return r.chunk_idx
    return None


def speaking_run_from(records, onset):
    """Extend a contiguous speaking segment from `onset`; return (onset, last, text).

    'Contiguous' tolerates the model's per-chunk emission but stops at the first
    listen chunk after speech has begun.
    """
    if onset is None:
        return (None, None, "")
    text, last = [], onset
    started = False
    for r in records:
        if r.chunk_idx < onset:
            continue
        if r.speaking:
            started = True
            last = r.chunk_idx
            if r.text:
                text.append(r.text)
        elif started:
            break
    return (onset, last, "".join(text))


def first_listen_after(records, start):
    for r in records:
        if r.chunk_idx >= start and r.is_listen:
            return r.chunk_idx
    return None


def first_eot_after(records, start):
    for r in records:
        if r.chunk_idx >= start and r.end_of_turn:
            return r.chunk_idx
    return None


def sustained_listen_after(records, T, k=2):
    """First index i (> T) such that chunks i..i+k-1 are all listen with no text —
    i.e. the model has genuinely stopped, not just hit a momentary turn boundary."""
    rec = {r.chunk_idx: r for r in records}
    idxs = sorted(rec)
    for i in idxs:
        if i <= T:
            continue
        win = [rec.get(j) for j in range(i, i + k)]
        if all(w is not None and w.is_listen and not w.text.strip() for w in win):
            return i
    return None
