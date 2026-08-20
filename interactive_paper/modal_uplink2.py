"""Cloud-ASR uplink arm (2026-08-20) — the ONE channel lever that is
compatible with the text-backend decision.

Context. The v3 rebuild put the frozen pool's speech-channel cost at
**-.175** (gold-inject .771 vs deployed heard .596 at the aggressive
arm) — an order above the .009-.028 replication-noise floor (8ad), so
it is the only lever a single sweep can resolve. Two ways to attack it
were already settled and are NOT re-run here:
  * audio-direct-to-expert (8r): measured, then REJECTED by the user
    the same day — binding the expert to an audio-native model costs
    -.15 of brain, and the model list today still has no gpt-5.5-class
    audio model (checked 2026-08-20: gpt-audio / gpt-audio-1.5 only).
  * MiniCPM-side prompt fixes / k-best GER (8d): measured NEGATIVE.
What was only LOWER-bounded is the third path: keep the frontier TEXT
brain, but stop feeding it the talker's own transcription — send the
audio to a hosted ASR and give gpt-5.5 that text instead. 8d bounded it
with **openai/whisper-large-v3** (open weights, local) at +4pp. The
hosted frontier ASRs (`gpt-transcribe`, `gpt-4o-transcribe`) did not
exist in that arm; this re-runs the bound with them.

Three arms on the SAME escalated ids, one judge (escalate.judge_many):
  A deployed   : MiniCPM self-transcript -> gpt-5.5   (from v3 traces)
  B cloud-ASR  : wav -> gpt-transcribe   -> gpt-5.5   (this file)
  C gold       : gold query text         -> gpt-5.5   (eval_expert)
B-A = what the uplink swap buys; C-B = what no ASR can recover.

  modal run modal_uplink2.py::uplink --limit 6      # smoke
  modal run modal_uplink2.py::uplink                # full
  modal run modal_uplink2.py::report
"""
import json
import os
import sys

import modal

from modal_app import (gen_app, app, util_image, gate_data, DATA, OPENAI,
                       API_REGION)

HERE = os.path.dirname(os.path.abspath(__file__))
util_st = util_image.add_local_file(os.path.join(HERE, "modal_app.py"),
                                    "/root/modal_app.py")

AUDIO_DIR = f"{DATA}/audio_pool"
CACHE = f"{DATA}/uplink2_cache"
OUT = f"{DATA}/uplink2.parquet"
ASR_PREFERRED = ("gpt-transcribe", "gpt-4o-transcribe", "whisper-1")


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 50, region=API_REGION)
def uplink(limit: int = 0, concurrency: int = 3, asr_model: str = ""):
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    df = pd.read_parquet(f"{DATA}/frozen_v3_traces.parquet")
    esc = (df[df["mode"] == "escalated"]
           .drop_duplicates("id")[["id", "pool", "query",
                                   "reference_answer", "transcript"]]
           .sort_values("id").to_dict("records"))
    if limit:
        esc = esc[:limit]

    if not asr_model:
        avail = {m.id for m in escalate._client().models.list().data}
        asr_model = next((m for m in ASR_PREFERRED if m in avail), "")
    if not asr_model:
        raise RuntimeError("no hosted ASR model available")
    os.makedirs(CACHE, exist_ok=True)
    print(f">>> uplink2: {len(esc)} escalated ids | ASR={asr_model} "
          f"-> expert={escalate.EXPERT_MODEL}", flush=True)

    client = escalate._async_client()
    sem = asyncio.Semaphore(concurrency)

    async def transcribe(q):
        cpath = f"{CACHE}/{q['id']}.{asr_model}.json"
        if os.path.exists(cpath):
            return json.load(open(cpath))
        wav = f"{AUDIO_DIR}/{q['id']}.wav"
        rec = {"id": q["id"], "asr": None, "err": None}
        if not os.path.exists(wav):
            rec["err"] = "missing wav"
            return rec
        for attempt in range(5):
            async with sem:
                try:
                    with open(wav, "rb") as fh:
                        r = await client.audio.transcriptions.create(
                            model=asr_model, file=fh,
                            response_format="text")
                    rec["asr"] = r if isinstance(r, str) else getattr(
                        r, "text", str(r))
                    break
                except Exception as e:
                    rec["err"] = f"{type(e).__name__}: {str(e)[:120]}"
            await asyncio.sleep(min(60, 3 * 2 ** attempt))
        json.dump(rec, open(cpath, "w"))
        return rec

    async def transcribe_all():
        return await asyncio.gather(*(transcribe(q) for q in esc))

    asr = asyncio.run(transcribe_all())
    ok = [a for a in asr if a["asr"]]
    print(f">>> transcribed {len(ok)}/{len(esc)}", flush=True)

    # same expert protocol as the live loop (cached per query text)
    amap = {a["id"]: a["asr"] for a in ok}
    rows = [q for q in esc if q["id"] in amap]
    answers = asyncio.run(escalate.ask_expert_many(
        [amap[q["id"]] for q in rows], concurrency=concurrency,
        cache_dir=f"{CACHE}/expert"))

    jr = [{"id": q["id"], "query": q["query"],
           "reference_answer": q["reference_answer"],
           "answer": a.get("answer")} for q, a in zip(rows, answers)]
    labeled = asyncio.run(escalate.judge_many(
        [dict(r) for r in jr], concurrency=8))

    out = pd.DataFrame([{
        "id": q["id"], "pool": q["pool"], "asr_model": asr_model,
        "asr_text": amap[q["id"]],
        "self_transcript": q.get("transcript"),
        "expert_answer": a.get("answer"),
        "expert_latency_s": a.get("latency_s"),
        "uplink_ok": l["adequate"],
    } for q, a, l in zip(rows, answers, labeled)])
    out.to_parquet(OUT)
    gate_data.commit()
    n_ok = out["uplink_ok"].notna().sum()
    print(f">>> uplink2: n={len(out)} judged={n_ok} "
          f"acc={out['uplink_ok'].dropna().mean():.3f}", flush=True)


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 10)
def report():
    import numpy as np
    import pandas as pd
    up = pd.read_parquet(OUT).set_index("id")
    df = pd.read_parquet(f"{DATA}/frozen_v3_traces.parquet")
    exp = pd.read_parquet(f"{DATA}/eval_expert.parquet").set_index("id")

    esc = df[df["mode"] == "escalated"].drop_duplicates("id").set_index("id")
    ids = up.index.intersection(esc.index).intersection(exp.index)
    b = lambda s: s.reindex(ids).apply(
        lambda x: 1 if x is True or x == 1 else 0)
    A = b(esc["heard_ok"])                       # deployed
    B = b(up["uplink_ok"])                       # cloud ASR -> gpt-5.5
    C = b(exp["expert_adequate"])                # gold text -> gpt-5.5

    def mc(x, y):
        n01 = int(((x == 0) & (y == 1)).sum())
        n10 = int(((x == 1) & (y == 0)).sum())
        from math import comb
        n = n01 + n10
        p = 1.0 if n == 0 else min(1.0, sum(
            comb(n, i) * .5 ** n for i in range(n + 1)
            if comb(n, i) * .5 ** n <= comb(n, n01) * .5 ** n + 1e-12))
        return n01, n10, p

    print(f"escalated ids n={len(ids)} | ASR={up['asr_model'].iloc[0]}")
    print(f"  A deployed  (self-transcript -> gpt-5.5): {A.mean():.3f}")
    print(f"  B cloud-ASR (hosted ASR      -> gpt-5.5): {B.mean():.3f}")
    print(f"  C gold text (gold            -> gpt-5.5): {C.mean():.3f}")
    n01, n10, p = mc(A, B)
    print(f"  B-A = {B.mean() - A.mean():+.3f}  (B-only right {n01}, "
          f"A-only right {n10}, McNemar p={p:.4f})")
    n01, n10, p = mc(B, C)
    print(f"  C-B = {C.mean() - B.mean():+.3f}  (C-only right {n01}, "
          f"B-only right {n10}, McNemar p={p:.4f})")
    rec = ((B.mean() - A.mean()) / max(C.mean() - A.mean(), 1e-9))
    print(f"  channel recovery: {rec:.1%} of the gold gap")
    by = pd.DataFrame({"pool": esc["pool"].reindex(ids), "A": A, "B": B,
                       "C": C}).groupby("pool").mean().round(3)
    print(by.to_string())
    json.dump({"n": int(len(ids)), "asr": str(up["asr_model"].iloc[0]),
               "deployed": float(A.mean()), "cloud_asr": float(B.mean()),
               "gold": float(C.mean()), "recovery": float(rec),
               "by_pool": by.to_dict()},
              open(f"{DATA}/uplink2_report.json", "w"), indent=1)
    gate_data.commit()
