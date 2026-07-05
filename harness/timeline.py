"""Scenario timeline: an ordered list of 1-second chunks, each carrying a
(audio_waveform, frame, label, event) tuple. This is the single mechanism every
probe configures — scheduled events (scene swaps, injected utterances) are just
chunks whose audio/frame change at a known index, tagged with an `event` marker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from stimuli import silence_chunk  # noqa: E402

SR = 16000
CHUNK = SR


@dataclass
class Chunk:
    audio: np.ndarray            # float32, length CHUNK (16000)
    frame: object = None         # PIL.Image or None
    label: str = ""              # human annotation
    event: str = ""              # event marker fired at this chunk (e.g. "INJECT_STOP")


class Scenario:
    """A list of Chunks plus a couple of run-control flags.

    Build directly (`Scenario([...])`) or via the builder helpers below.
    `event_index(name)` returns the chunk index where a named event fires — probes
    use it as the reference point T for latency metrics.
    """

    def __init__(self, chunks: List[Chunk], name: str = "",
                 stop_on_eot: bool = False, min_chunks_before_stop: int = 6):
        self._chunks = chunks
        self.name = name
        self.stop_on_eot = stop_on_eot
        self.min_chunks_before_stop = min_chunks_before_stop

    def __len__(self):
        return len(self._chunks)

    def last_event_index(self) -> int:
        last = -1
        for i, c in enumerate(self._chunks):
            if c.event:
                last = i
        return last

    def allow_stop(self, idx: int) -> bool:
        # Only honor end-of-turn after the last scheduled event AND the floor, so
        # an early natural EOT can't truncate the post-event observation window.
        return idx >= max(self.min_chunks_before_stop, self.last_event_index() + 1)

    def chunks(self):
        for c in self._chunks:
            a = c.audio
            if a is None:
                a = silence_chunk()
            if len(a) != CHUNK:
                a = _fit(a)
            yield a.astype(np.float32), c.frame, c.label, c.event

    def event_index(self, name: str) -> Optional[int]:
        for i, c in enumerate(self._chunks):
            if c.event == name:
                return i
        return None


def _fit(a: np.ndarray) -> np.ndarray:
    if len(a) >= CHUNK:
        return a[:CHUNK]
    return np.pad(a, (0, CHUNK - len(a)))


# --------- builders ----------------------------------------------------------

def from_tracks(audio_track: List[np.ndarray],
                frame_track: List[object],
                labels: Optional[List[str]] = None,
                events: Optional[dict] = None,
                name: str = "") -> Scenario:
    """Zip an audio track and a frame track (same length) into a Scenario.

    events: {chunk_idx: "EVENT_NAME"} marks named events for metric reference.
    """
    n = max(len(audio_track), len(frame_track))
    events = events or {}
    labels = labels or [""] * n
    chunks = []
    for i in range(n):
        a = audio_track[i] if i < len(audio_track) else silence_chunk()
        f = frame_track[i] if i < len(frame_track) else (frame_track[-1] if frame_track else None)
        chunks.append(Chunk(audio=a, frame=f,
                            label=labels[i] if i < len(labels) else "",
                            event=events.get(i, "")))
    return Scenario(chunks, name=name)
