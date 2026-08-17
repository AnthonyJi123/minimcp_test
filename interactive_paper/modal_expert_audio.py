"""Audio-direct-to-expert arm (RESULTS 8q follow-up, user-approved 2026-08-12).

For every unique escalated test id in gated_traces_v2, send the ORIGINAL
pool wav straight to an audio-capable OpenAI model — no MiniCPM
self-transcription in the uplink. Judged against the gold query +
reference with the standard judge (same protocol as expert_adequate),
so the result is directly comparable to the gold-inject view.

Reads: gold .82 / self-transcript .58 / Whisper-ears .62 (8d). This arm
bounds the channel: audio-expert ~ gold means the wav carries the
content and MiniCPM's ears were the loss; audio-expert ~ Whisper means
the spoken audio itself is lossy (TTS pronunciation / inherent).

Anti-flag measures inherited: per-id cache on the volume (never resend),
user=USER_ID, EXPERT_SYSTEM purpose declaration, concurrency 3,
region-pinned. Smoke first: modal run modal_expert_audio.py::audio_expert --limit 6
Full:                       modal run modal_expert_audio.py::audio_expert
"""
import base64
import json
import os
import struct
import sys
import time

import modal

from modal_app import (gen_app, util_image, gate_data, DATA, OPENAI,
                       API_REGION)

HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(HERE, "modal_app.py")
util_st = util_image.add_local_file(_APP_PY, "/root/modal_app.py")

AUDIO_DIR = f"{DATA}/audio_pool"
CACHE = f"{DATA}/audio_expert_cache"
OUT = f"{DATA}/audio_expert.parquet"

# preference order; first available id wins (override with --model)
PREFERRED = ("gpt-5.5-audio", "gpt-5.4-audio", "gpt-audio",
             "gpt-4o-audio-preview")

AUDIO_NOTE = (
    " The user's question arrives as an audio recording. Listen to it and "
    "answer the question it asks. Answer directly and completely; if the "
    "question is multiple-choice, name the correct option."
)


def _wav_b64(path):
    raw = bytearray(open(path, "rb").read())
    n = len(raw)
    struct.pack_into("<I", raw, 4, n - 8)      # streaming writes left RIFF/
    struct.pack_into("<I", raw, 40, n - 44)    # data sizes at 0xFFFFFFFF
    return base64.b64encode(bytes(raw)).decode()


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 50, region=API_REGION)
def audio_expert(limit: int = 0, concurrency: int = 3, model: str = ""):
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    df = pd.read_parquet(f"{DATA}/gated_traces_v2.parquet")
    esc = (df[df["mode"] == "escalated"]
           .drop_duplicates("id")[["id", "pool", "query", "reference_answer"]]
           .sort_values("id").to_dict("records"))
    if limit:
        esc = esc[:limit]
    os.makedirs(CACHE, exist_ok=True)

    client = escalate._async_client()

    if not model:
        avail = {m.id for m in escalate._client().models.list().data}
        model = next((m for m in PREFERRED if m in avail), "")
        if not model:
            auds = sorted(m for m in avail if "audio" in m)
            raise RuntimeError(f"no preferred audio model; available: {auds}")
    print(f">>> audio_expert: {len(esc)} escalated ids -> {model} "
          f"(concurrency {concurrency})", flush=True)

    sem = asyncio.Semaphore(concurrency)

    async def one(q):
        cpath = f"{CACHE}/{q['id']}.{model}.json"
        if os.path.exists(cpath):
            return json.load(open(cpath))
        wav = f"{AUDIO_DIR}/{q['id']}.wav"
        if not os.path.exists(wav):
            return {"id": q["id"], "answer": None, "latency_s": None,
                    "error": "missing wav"}
        b64 = _wav_b64(wav)
        async with sem:
            t0 = time.monotonic()
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    modalities=["text"],
                    max_completion_tokens=2048,
                    messages=[
                        {"role": "system",
                         "content": escalate.EXPERT_SYSTEM + AUDIO_NOTE},
                        {"role": "user", "content": [
                            {"type": "input_audio",
                             "input_audio": {"data": b64, "format": "wav"}}]}],
                    user=escalate.USER_ID,
                )
                out = {"id": q["id"], "answer": resp.choices[0].message.content,
                       "latency_s": round(time.monotonic() - t0, 2),
                       "error": None, **escalate._usage(resp)}
            except Exception as e:
                out = {"id": q["id"], "answer": None,
                       "latency_s": round(time.monotonic() - t0, 2),
                       "error": str(e)[:300]}
        if out.get("answer"):
            json.dump(out, open(cpath, "w"))
        return out

    async def run_all():
        return await asyncio.gather(*(one(q) for q in esc))

    results = asyncio.run(run_all())
    gate_data.commit()
    errs = [r for r in results if r.get("error")]
    print(f">>> collected {len(results) - len(errs)} ok, {len(errs)} errors",
          flush=True)
    for r in errs[:5]:
        print(f"    {r['id']}: {r['error']}", flush=True)

    # judge against the GOLD query text + reference (same protocol as
    # expert_adequate -> directly comparable to the gold-inject view)
    byid = {q["id"]: q for q in esc}
    rows = [{"id": r["id"], "query": byid[r["id"]]["query"],
             "reference_answer": byid[r["id"]]["reference_answer"],
             "answer": r["answer"]}
            for r in results if r.get("answer")]
    judged = asyncio.run(escalate.judge_many(rows, concurrency=8))
    adq = {r["id"]: r["adequate"] for r in judged}

    out = pd.DataFrame(results)
    out["pool"] = [byid[i]["pool"] for i in out["id"]]
    out["adequate"] = [adq.get(i) for i in out["id"]]
    out["model"] = model
    out.to_parquet(OUT)
    gate_data.commit()

    ok = out[out["adequate"].notna()]
    print(f"\n=== audio-direct expert ({model}), n={len(ok)} ===", flush=True)
    for p, g in ok.groupby("pool"):
        print(f"  {p:15s} n={len(g):3d} acc={g['adequate'].mean():.2f}",
              flush=True)
    print(f"  {'ALL':15s} n={len(ok):3d} acc={ok['adequate'].mean():.2f}",
          flush=True)
    lat = ok["latency_s"].astype(float)
    print(f"  latency P50={lat.quantile(.5):.1f}s P95={lat.quantile(.95):.1f}s",
          flush=True)
    print(f">>> wrote {OUT}", flush=True)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 50, region=API_REGION)
def text_control(limit: int = 0, concurrency: int = 3, model: str = ""):
    """Same model, same questions, GOLD TEXT input — isolates the audio
    channel from gpt-audio's capability gap vs gpt-5.5. Judged identically.
    gpt-audio rejects text-only requests ("requires that either input
    content or output modality contain audio"), so the text question is
    accompanied by a 0.25 s silent placeholder wav the model is told to
    ignore — everything else matches the audio arm exactly.
    Writes audio_expert_text.parquet."""
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    df = pd.read_parquet(f"{DATA}/gated_traces_v2.parquet")
    esc = (df[df["mode"] == "escalated"]
           .drop_duplicates("id")[["id", "pool", "query", "reference_answer"]]
           .sort_values("id").to_dict("records"))
    if limit:
        esc = esc[:limit]
    os.makedirs(CACHE, exist_ok=True)
    client = escalate._async_client()

    if not model:
        avail = {m.id for m in escalate._client().models.list().data}
        model = next((m for m in PREFERRED if m in avail), "")
    print(f">>> text_control: {len(esc)} ids -> {model} (gold text input)",
          flush=True)

    # 0.25 s of 24 kHz 16-bit mono silence — satisfies gpt-audio's
    # audio-in-request requirement without carrying any content
    sr, dur = 24000, 0.25
    nbytes = int(sr * dur) * 2
    hdr = (b"RIFF" + struct.pack("<I", 36 + nbytes) + b"WAVEfmt "
           + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
           + b"data" + struct.pack("<I", nbytes))
    silence_b64 = base64.b64encode(hdr + b"\x00" * nbytes).decode()

    sem = asyncio.Semaphore(concurrency)

    async def one(q):
        cpath = f"{CACHE}/{q['id']}.{model}.text.json"
        if os.path.exists(cpath):
            return json.load(open(cpath))
        async with sem:
            t0 = time.monotonic()
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    modalities=["text"],
                    max_completion_tokens=2048,
                    messages=[
                        {"role": "system",
                         "content": escalate.EXPERT_SYSTEM
                         + " The attached audio clip is a silent placeholder"
                           " required by the API — ignore it and answer the"
                           " text question."},
                        {"role": "user", "content": [
                            {"type": "text", "text": q["query"]},
                            {"type": "input_audio",
                             "input_audio": {"data": silence_b64,
                                             "format": "wav"}}]}],
                    user=escalate.USER_ID,
                )
                out = {"id": q["id"], "answer": resp.choices[0].message.content,
                       "latency_s": round(time.monotonic() - t0, 2),
                       "error": None, **escalate._usage(resp)}
            except Exception as e:
                out = {"id": q["id"], "answer": None,
                       "latency_s": round(time.monotonic() - t0, 2),
                       "error": str(e)[:300]}
        if out.get("answer"):
            json.dump(out, open(cpath, "w"))
        return out

    async def run_all():
        return await asyncio.gather(*(one(q) for q in esc))

    results = asyncio.run(run_all())
    gate_data.commit()
    errs = [r for r in results if r.get("error")]
    print(f">>> collected {len(results) - len(errs)} ok, {len(errs)} errors",
          flush=True)
    for r in errs[:5]:
        print(f"    {r['id']}: {r['error']}", flush=True)

    byid = {q["id"]: q for q in esc}
    rows = [{"id": r["id"], "query": byid[r["id"]]["query"],
             "reference_answer": byid[r["id"]]["reference_answer"],
             "answer": r["answer"]}
            for r in results if r.get("answer")]
    judged = asyncio.run(escalate.judge_many(rows, concurrency=8))
    adq = {r["id"]: r["adequate"] for r in judged}

    out = pd.DataFrame(results)
    out["pool"] = [byid[i]["pool"] for i in out["id"]]
    out["adequate"] = [adq.get(i) for i in out["id"]]
    out["model"] = model
    out.to_parquet(f"{DATA}/audio_expert_text.parquet")
    gate_data.commit()

    ok = out[out["adequate"].notna()]
    print(f"\n=== {model} on GOLD TEXT (control), n={len(ok)} ===", flush=True)
    for p, g in ok.groupby("pool"):
        print(f"  {p:15s} n={len(g):3d} acc={g['adequate'].mean():.2f}",
              flush=True)
    print(f"  {'ALL':15s} n={len(ok):3d} acc={ok['adequate'].mean():.2f}",
          flush=True)
