"""Daily-Omni: audio-visual multiple-choice QA prompt, parsing, and accuracy.

Dataset `liarliar/Daily-Omni` (Videos.tar + qa.json). qa.json is a dict keyed
"Entry_N"; each entry:

    {"Question": str, "Choice": ["A. ...", "B. ...", "C. ...", "D. ..."],
     "Answer": "A"|"B"|"C"|"D", "video_id": <youtube id>, "Type": <task>,
     "video_duration": "30s"|"60s", ...}

Videos live at Videos/{video_id}.mp4. The model sees interleaved frames+audio
(omni_mode) plus this text prompt and must reply with a single letter; we parse
it and score overall + split by duration and task Type.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import tempfile

PROMPT_TMPL = (
    "Watch the video and listen to its audio, then answer the multiple-choice "
    "question. Both the visuals and the sound may be needed.\n\n"
    "Question: {q}\n{choices}\n\n"
    "Respond with only the single letter (A, B, C, or D) of the correct option."
)


def load_qa(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    # qa.json is a list of entries in the released dataset; tolerate a dict too
    items = data.items() if isinstance(data, dict) else enumerate(data)
    rows = []
    for key, e in items:
        rows.append({
            "id": str(key),
            "question": e["Question"],
            "choices": e["Choice"],
            "answer": e["Answer"].strip().upper()[:1],
            "video_id": e["video_id"],
            "type": e.get("Type", ""),
            "duration": e.get("video_duration", ""),
        })
    return rows


def build_prompt(row: dict) -> str:
    return PROMPT_TMPL.format(q=row["question"], choices="\n".join(row["choices"]))


def _has_audio_stream(path: str) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    return bool(r.stdout.strip())


def prepare_media(videos_dir: str, video_id: str):
    """Return a path playable by get_video_frame_audio_segments.

    Layout: Videos/{id}/{id}_video.mp4 + a separate {id}_audio.wav. If the mp4
    already carries an audio track we use it as-is; otherwise we mux the wav in
    (the model needs the audio for these audio-visual questions). Returns
    (media_path, is_temp) so the caller can clean up a muxed temp file.
    """
    d = os.path.join(videos_dir, video_id)
    vids = glob.glob(f"{d}/*_video.mp4") or glob.glob(f"{d}/*.mp4")
    if not vids:
        return None, False
    video = vids[0]
    if _has_audio_stream(video):
        return video, False
    wavs = glob.glob(f"{d}/*_audio.wav") or glob.glob(f"{d}/*.wav")
    if not wavs:
        return video, False  # no audio available anywhere; go video-only
    out = tempfile.mktemp(suffix=".mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-i", wavs[0], "-c:v", "copy",
         "-c:a", "aac", "-shortest", out],
        capture_output=True)
    return (out, True) if os.path.exists(out) else (video, False)


def parse_letter(output: str):
    t = (output or "").upper()
    # "answer is B", "option (C)", "final answer: D"
    m = re.search(r"ANSWER\s*(?:IS|:)?\s*\(?\s*([ABCD])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCD])\b", t)  # first standalone letter
    return m.group(1) if m else None


def _norm_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if len(w) > 2}


def lexical_letter(output: str, choices: list):
    """Recover a letter when the model answers in prose: pick the option whose
    words are most covered by the output, requiring a clear, non-trivial win."""
    ow = _norm_words(output or "")
    if not ow:
        return None
    best = second = -1.0
    letter = None
    for i, ch in enumerate(choices):
        cw = _norm_words(re.sub(r"^\s*[A-Da-d][.)]\s*", "", ch))
        if not cw:
            continue
        ov = len(ow & cw) / len(cw)
        if ov > best:
            second, best, letter = best, ov, "ABCD"[i]
        elif ov > second:
            second = ov
    return letter if best >= 0.5 and best - second >= 0.15 else None


def resolve_pred(output: str, choices: list):
    """A clean letter if present, else a lexical-overlap fallback."""
    return parse_letter(output) or lexical_letter(output, choices)


def score(rows: list) -> dict:
    """rows: [{answer, pred, duration, type}]. Overall + by duration + by type."""
    from collections import defaultdict

    def acc(items):
        ok = sum(1 for r in items if r.get("pred") and r["pred"] == r["answer"])
        return {"acc": ok / len(items) if items else None, "n": len(items)}

    by_dur = defaultdict(list)
    by_type = defaultdict(list)
    for r in rows:
        by_dur[r.get("duration", "")].append(r)
        by_type[r.get("type", "")].append(r)
    return {
        "overall": acc(rows),
        "by_duration": {k: acc(v) for k, v in sorted(by_dur.items())},
        "by_type": {k: acc(v) for k, v in sorted(by_type.items())},
        "unparsed": sum(1 for r in rows if not r.get("pred")),
    }
