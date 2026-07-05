"""Reproducible audio/video stimuli for the probes.

Video: synthetic frames via PIL (text/shape cards, scene swaps) — deterministic.
Audio: user utterances synthesized with edge-tts (Chinese/English), resampled to
16 kHz mono, sliced into 1-second chunks aligned to the timeline. Silence = zeros.

All audio chunks are float32 length 16000. All frames are PIL.Image (RGB).
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from typing import List, Optional, Tuple

import numpy as np

SR = 16000
CHUNK = SR
CACHE = "/workspace/minicpm-o-eval/fixtures"
os.makedirs(CACHE, exist_ok=True)


# ---------- audio ------------------------------------------------------------

def silence_chunk(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def chunkify(wave: np.ndarray, chunk: int = CHUNK) -> List[np.ndarray]:
    """Split a waveform into fixed-size chunks; pad the tail with zeros."""
    out = []
    if len(wave) == 0:
        return [silence_chunk()]
    for i in range(0, len(wave), chunk):
        c = wave[i:i + chunk]
        if len(c) < chunk:
            c = np.pad(c, (0, chunk - len(c)))
        out.append(c.astype(np.float32))
    return out


def load_wav(path: str) -> np.ndarray:
    import librosa
    w, _ = librosa.load(path, sr=SR, mono=True)
    return w.astype(np.float32)


def tts(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> np.ndarray:
    """Synthesize `text` to a 16 kHz mono waveform via edge-tts (cached on disk).

    Falls back to a short silence if edge-tts/network is unavailable, so the
    harness still runs offline (the probe logs the fallback).
    """
    key = hashlib.md5(f"{voice}:{text}".encode("utf-8")).hexdigest()[:16]
    wav_path = os.path.join(CACHE, f"tts_{key}.wav")
    if not os.path.exists(wav_path):
        try:
            _edge_tts_to_file(text, voice, wav_path)
        except Exception as e:  # offline / not installed
            print(f"[stimuli.tts] edge-tts failed ({e}); using silence for {text!r}")
            return silence_chunk(1.5)
    return load_wav(wav_path)


def _edge_tts_to_file(text: str, voice: str, out_path: str) -> None:
    import edge_tts  # type: ignore
    mp3 = out_path.replace(".wav", ".mp3")

    async def _run():
        await edge_tts.Communicate(text, voice).save(mp3)

    asyncio.run(_run())
    # transcode mp3 -> 16k wav via librosa/soundfile
    import librosa, soundfile as sf
    w, _ = librosa.load(mp3, sr=SR, mono=True)
    sf.write(out_path, w, SR)


def utterance_chunks(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> List[np.ndarray]:
    return chunkify(tts(text, voice))


# ---------- video ------------------------------------------------------------

def card(text: str, color=(30, 60, 120), fg=(255, 255, 255),
         shape: Optional[str] = None, shape_color=(220, 60, 60),
         size=(448, 448)):
    """A simple labelled scene card. `shape` in {None,'square','circle','triangle'}."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, color)
    d = ImageDraw.Draw(img)
    w, h = size
    if shape == "square":
        d.rectangle([w * 0.3, h * 0.3, w * 0.7, h * 0.7], fill=shape_color)
    elif shape == "circle":
        d.ellipse([w * 0.3, h * 0.3, w * 0.7, h * 0.7], fill=shape_color)
    elif shape == "triangle":
        d.polygon([(w * 0.5, h * 0.28), (w * 0.28, h * 0.72), (w * 0.72, h * 0.72)],
                  fill=shape_color)
    # label
    for dx, dy in ((1, 1), (0, 0)):
        d.text((w * 0.08 + dx, h * 0.06 + dy), text,
               fill=(0, 0, 0) if (dx, dy) == (1, 1) else fg)
    return img


def dots(n: int, color=(20, 20, 30), dot_color=(240, 230, 60), size=(448, 448)):
    """A scene with `n` clearly separated filled circles (for counting tasks)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, color)
    d = ImageDraw.Draw(img)
    w, h = size
    r = 34
    cols = 3
    for i in range(n):
        cx = w * (0.22 + 0.28 * (i % cols))
        cy = h * (0.25 + 0.25 * (i // cols))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=dot_color)
    return img


def frame_track(spec: List[Tuple[int, object]], total: int) -> List[object]:
    """Expand a sparse spec [(start_idx, frame), ...] into a per-chunk frame list
    of length `total`, holding each frame until the next change (scene-swap)."""
    spec = sorted(spec)
    out = []
    cur = None          # blank until the first spec entry's start_idx (was: pre-seeded
    j = 0               # to spec[0][1], which silently corrupted swaps not starting at 0)
    for i in range(total):
        while j < len(spec) and spec[j][0] == i:
            cur = spec[j][1]
            j += 1
        out.append(cur)
    return out
