"""Full-Duplex-Bench v1.0 inference driver for MiniCPM-o 4.5 duplex.

Streams each sample's input.wav through the model's 1 Hz duplex loop and writes
a time-aligned output.wav (24 kHz mono, same duration as the input) in the
layout the benchmark's ASR + evaluation scripts expect, plus a trace.jsonl of
per-chunk decisions. All task annotation JSONs are copied alongside so each
output dir is a self-contained eval root entry.

Timeline semantics (tick = 1 s of stream time):
  - input chunk i  = input samples  [i*16000, (i+1)*16000)
  - output slot i  = output samples [i*24000, (i+1)*24000)
  - audio generated from chunk i is queued and drained from slot i+1 onward
    (drain-before-process), the earliest causally possible playback. Latency is
    therefore quantized to the model's native 1 s decision ticks.
  - the loop runs exactly ceil(duration) ticks; output is trimmed to the input
    duration, matching the reference implementation (freeze-omni) and the
    benchmark's example data (output.wav duration == input.wav duration).
"""
import argparse
import base64
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import soundfile as sf

HARNESS_DIR = "/workspace/minicpm-o-eval/harness"
if HARNESS_DIR not in sys.path:
    sys.path.insert(0, HARNESS_DIR)

IN_SR = 16000
OUT_SR = 24000
MAX_TICKS = 180  # safety cap (3 min); benchmark clips are well under this

# Official audio-duplex "English Call" preset from MiniCPM-o-Demo
# (assets/presets/audio_duplex/english_call.yaml) — the vendor-tuned prompt and
# English reference voice for audio-only duplex conversation.
SYSTEM_PROMPT = (
    "Replicate the tone and style from the input audio. Your task is to be a "
    "helpful assistant using this voice pattern. Please answer the user's "
    "questions seriously and in a high quality. Please chat with the user in "
    "a high naturalness style. You are in duplex mode, where you can listen "
    "and speak at the same time."
)
REF_AUDIO = "/workspace/MiniCPM-o-Demo/assets/ref_audio/ref_en_dlc_1.wav"


def load_input(path: str) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=IN_SR, mono=True)
    return np.asarray(wav, dtype=np.float32)


def run_sample(sess, sample_dir: str, out_dir: str) -> dict:
    wav = load_input(os.path.join(sample_dir, "input.wav"))
    dur = len(wav) / IN_SR
    n = min(math.ceil(len(wav) / IN_SR), MAX_TICKS)
    padded = np.zeros(n * IN_SR, dtype=np.float32)
    padded[: min(len(wav), n * IN_SR)] = wav[: n * IN_SR]

    out = np.zeros(n * OUT_SR, dtype=np.float32)
    pending = np.zeros(0, dtype=np.float32)
    trace = []

    sess.prepare(SYSTEM_PROMPT)
    t_start = time.time()
    try:
        d = sess.duplex
        for i in range(n):
            # drain up to 1 s of already-generated speech into this tick's slot
            k = min(pending.size, OUT_SR)
            if k:
                out[i * OUT_SR : i * OUT_SR + k] = pending[:k]
                pending = pending[k:]

            t0 = time.time()
            d.prefill(audio_waveform=padded[i * IN_SR : (i + 1) * IN_SR],
                      frame_list=None)
            try:
                r = d.generate()
            finally:
                d.finalize()

            seg_n = 0
            if not r.is_listen:
                ad = getattr(r, "audio_data", None)
                if isinstance(ad, str) and ad:
                    seg = np.frombuffer(base64.b64decode(ad), dtype="<f4")
                    pending = np.concatenate([pending, seg.astype(np.float32)])
                    seg_n = seg.size
            trace.append({
                "chunk": i,
                "is_listen": bool(r.is_listen),
                "eot": bool(r.end_of_turn),
                "text": r.text or "",
                "audio_samples": seg_n,
                "ms": round((time.time() - t0) * 1000.0, 1),
            })
    finally:
        sess.stop()

    out = out[: int(round(dur * OUT_SR))]
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, "output.wav"), out, OUT_SR, subtype="PCM_16")
    for f in os.listdir(sample_dir):
        if f.endswith(".json"):
            shutil.copy2(os.path.join(sample_dir, f), os.path.join(out_dir, f))
    with open(os.path.join(out_dir, "trace.jsonl"), "w", encoding="utf-8") as fh:
        for row in trace:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    spoke = sum(1 for r in trace if not r["is_listen"])
    return {
        "ticks": n,
        "spoke_ticks": spoke,
        "out_rms": float(np.sqrt(np.mean(out**2))) if out.size else 0.0,
        "wall_s": round(time.time() - t_start, 1),
        "text": "".join(r["text"] for r in trace).strip(),
    }


def run_shard(pairs, data_root: str, out_root: str, overwrite: bool = False):
    """pairs: list of (subset, sample_id). Loads the model once, runs each sample."""
    from session import DuplexSession

    sess = DuplexSession(ref_audio_path=REF_AUDIO)
    print(f">>> model loaded in {sess.load_s:.1f}s | {len(pairs)} samples", flush=True)

    done, skipped = 0, 0
    for k, (subset, sid) in enumerate(pairs):
        sample_dir = os.path.join(data_root, subset, sid)
        out_dir = os.path.join(out_root, subset, sid)
        if not overwrite and os.path.exists(os.path.join(out_dir, "output.wav")):
            skipped += 1
            continue
        try:
            s = run_sample(sess, sample_dir, out_dir)
        except Exception as e:
            print(f"[{k + 1}/{len(pairs)}] {subset}/{sid} FAILED: {e}", flush=True)
            import traceback

            traceback.print_exc()
            continue
        done += 1
        print(f"[{k + 1}/{len(pairs)}] {subset}/{sid} ticks={s['ticks']} "
              f"spoke={s['spoke_ticks']} rms={s['out_rms']:.4f} "
              f"{s['wall_s']}s :: {s['text'][:80]!r}", flush=True)
    print(f">>> shard done: {done} ran, {skipped} skipped", flush=True)
    return {"done": done, "skipped": skipped}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data/v1.0")
    ap.add_argument("--out-root", default="/data/exp/minicpm-o-4_5")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--ids", default="", help="comma-separated sample ids (default: all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    subset_dir = os.path.join(args.data_root, args.subset)
    ids = ([i for i in args.ids.split(",") if i] if args.ids
           else sorted(d for d in os.listdir(subset_dir)
                       if os.path.exists(os.path.join(subset_dir, d, "input.wav"))))
    if args.limit:
        ids = ids[: args.limit]
    run_shard([(args.subset, i) for i in ids], args.data_root, args.out_root,
              overwrite=args.overwrite)
