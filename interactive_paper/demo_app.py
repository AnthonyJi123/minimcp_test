"""Interactive demo of the escalation gate — probe ON/OFF, live metrics,
event log.  `modal deploy demo_app.py`

Two modes, both faithful to the deployed pipeline:

REPLAY ($0, always on).  Every one of the 4773 measured live sessions is
on the volume, and because the three gated tiers are perfectly NESTED
and carry bit-identical probe scores (RESULTS 8ad), flipping the probe
OFF for a query is not a simulation: it is the never-arm's MEASURED
outcome for that same query. So the toggle shows two real recordings of
the same question, with the real probe score, the real threshold
comparison, the real answers, the real judge verdicts and the real
wall-clock timings.

LIVE (opt-in, spins one H100).  Type any question -> OpenAI tts-1
renders it with the same voice as the frozen pool -> MiniCPM-o 4.5
streams it in 1 s chunks -> the v3 probe reads L22 at end-of-turn
(rolling last-8 tail + running user-audio mean, exactly
modal_bench.py::bench_live) -> the frozen per-tier threshold decides ->
either the talker answers or gpt-5.5 does and the talker relays it.
Ask "what is NVDA trading at" with the probe on and off.
"""
import json
import os
import time

import modal

TOKEN = "62dc5cd9"

app = modal.App("gate-demo")
gate_data = modal.Volume.from_name("gate-data")
weights = modal.Volume.from_name("minicpm-o45-weights")
DATA = "/data"
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"
LAYER = 22
POOLS = ("frozen", "striviaqa", "swebq", "sllama", "sreason", "sdqa")
TIERS = ("conservative", "balanced", "aggressive")

# reuse the PROVEN MiniCPM image (torch 2.8 / transformers 4.51.0 pin +
# minicpmo-utils) rather than rebuilding it — 4.52+ breaks the Resampler.
# demo_app.py imports modal_app at module level, so BOTH images must
# carry that file or the container dies before serving anything.
from modal_app import image as _mini_image, OPENAI  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(_HERE, "modal_app.py")

web_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("fastapi[standard]", "pandas", "pyarrow")
             .add_local_file(_APP_PY, "/root/modal_app.py"))
gpu_image = _mini_image.add_local_file(_APP_PY, "/root/modal_app.py")
STALL = "Let me check that for you."
RELAY_TMPL = ("A verified answer came back: {ans}\n"
              "Relay it to the user in one or two spoken sentences.")


# ----------------------------------------------------------------- live ---
@app.function(image=gpu_image, gpu="H100",
              volumes={"/workspace/models": weights, DATA: gate_data},
              secrets=[OPENAI], timeout=60 * 20, scaledown_window=300)
def live_once(question: str = "", tier: str = "balanced",
              probe_on: bool = True, audio_b64: str = "",
              audio_ext: str = "webm"):
    """One real end-to-end turn.

    audio_b64 set  -> YOUR VOICE goes into the duplex talker unchanged
                      (no TTS anywhere) and, on escalation, the wav is
                      transcribed by the hosted ASR and that text is
                      what the expert reads — the 8ae uplink, which is
                      also the only option once there is no gold text.
    question set   -> tts-1/alloy renders it first (frozen-pool path).
    """
    import base64
    import glob as _glob
    import inspect
    import shutil
    import subprocess
    import sys
    import threading
    import numpy as np
    import librosa
    import torch
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import escalate
    import gate as gate_mod

    ev, t00 = [], time.time()

    def log(msg, **kw):
        ev.append({"t_ms": int((time.time() - t00) * 1000),
                   "msg": msg, **kw})

    cache = os.path.expanduser("~/.cache/huggingface/modules/"
                               "transformers_modules/"
                               + os.path.basename(MODEL_DIR))
    os.makedirs(cache, exist_ok=True)
    for f in _glob.glob(f"{MODEL_DIR}/*.py"):
        shutil.copy(f, cache)
    log("loading MiniCPM-o 4.5 (duplex, audio in / text out)")
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=False).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    log("talker ready")

    art = json.load(open(f"{DATA}/midlayer_gate_audio_v3.json"))
    probe = gate_mod.Probe(art["w"], art["b"])
    thr = art["eot_thresholds"][tier]
    K3, modes = art.get("k_eot", 8), art["modes"]
    log(f"probe v{art.get('version')} loaded: L{LAYER}, reads {modes}, "
        f"tier={tier}, threshold={thr:.3f}")

    t0 = time.time()
    wav_path = "/tmp/q.wav"
    if audio_b64:
        src = f"/tmp/in.{audio_ext}"
        open(src, "wb").write(base64.b64decode(audio_b64))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-ar", "16000", "-ac", "1", wav_path], check=True)
        au, _ = librosa.load(wav_path, sr=16000, mono=True)
        log(f"your microphone: {len(au) / 16000:.1f} s of real speech "
            f"(no TTS in this path)", ms=int((time.time() - t0) * 1000))
    else:
        r = escalate._client().audio.speech.create(
            model="tts-1", voice="alloy", input=question,
            response_format="wav")
        open(wav_path, "wb").write(r.content)
        au, _ = librosa.load(wav_path, sr=16000, mono=True)
        log(f"TTS rendered the question ({len(au) / 16000:.1f} s of "
            f"audio, voice=alloy)", ms=int((time.time() - t0) * 1000))

    def call_def(fn, /, **kw):
        p = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in p})

    def gen_text(**kw):
        kw.setdefault("max_new_tokens", 512)
        res = call_def(model.streaming_generate, tokenizer=tok,
                       temperature=0.1, generate_audio=False, **kw)
        parts = []
        if inspect.isgenerator(res) or hasattr(res, "__next__"):
            for x in res:
                t = getattr(x, "text", None)
                if t is None and isinstance(x, dict):
                    t = x.get("text")
                if t is None and isinstance(x, (tuple, list)) and x:
                    t = x[0]
                if isinstance(t, str):
                    parts.append(t)
        else:
            parts.append(str(res))
        return "".join(parts).strip()

    st3 = {"accum": False, "tail": None, "sum": None, "cnt": 0}

    def hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        h = hs[0].detach().float()
        t = h[-K3:].cpu()
        st3["tail"] = (t if st3["tail"] is None
                       else torch.cat([st3["tail"], t])[-K3:])
        if st3["accum"]:
            s = h.sum(0).cpu()
            st3["sum"] = s if st3["sum"] is None else st3["sum"] + s
            st3["cnt"] += h.shape[0]

    def score_now():
        parts = []
        for m in modes:
            if m == "eot_last":
                parts.append(st3["tail"][-1])
            elif m == "eot_mean":
                parts.append(st3["tail"].mean(0))
            elif m == "user_mean":
                parts.append(st3["sum"] / max(1, st3["cnt"]))
        return float(probe.score(torch.cat(parts).numpy()))

    chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
    model.reset_session()
    sys_msg = call_def(model.get_sys_prompt, mode="omni", language="en")
    call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
             tokenizer=tok)
    h = model.llm.model.layers[LAYER].register_forward_hook(hook)
    st3.update(tail=None, sum=None, cnt=0, accum=True)
    scores = []
    try:
        for i, ch in enumerate(chunks):
            if len(ch) < 16000:
                ch = np.pad(ch, (0, 16000 - len(ch)))
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "user",
                            "content": [ch.astype(np.float32)]}],
                     tokenizer=tok, is_last_chunk=(i == len(chunks) - 1))
            s = round(score_now(), 4)
            scores.append(s)
            log(f"chunk {i + 1}/{len(chunks)} streamed — running "
                f"P(fail)={s:.3f}", score=s)
        st3["accum"] = False
        t_eot0 = time.time()
        call_def(model.streaming_prefill, session_id="s1",
                 msgs=[{"role": "assistant", "content": [" "]}],
                 tokenizer=tok, is_last_chunk=True)
        eot = score_now()
        eot_ms = int((time.time() - t_eot0) * 1000)
    finally:
        h.remove()
    log(f"END OF TURN — probe read L{LAYER} in {eot_ms} ms: "
        f"P(fail)={eot:.3f}", score=round(eot, 4))

    fired = bool(probe_on and eot >= thr)
    if not probe_on:
        log("PROBE OFF — gate bypassed, always answer locally")
    else:
        log(f"gate: {eot:.3f} {'>=' if fired else '<'} {thr:.3f} → "
            f"{'ESCALATE' if fired else 'keep local'}")

    t_eot = time.time()
    out = {"question": question, "tier": tier, "probe_on": probe_on,
           "eot_score": round(eot, 4), "threshold": round(thr, 4),
           "scores": scores, "eot_read_ms": eot_ms,
           "audio_s": round(len(au) / 16000, 2), "fired": fired}
    if not fired:
        ans = gen_text(session_id="s1")
        out.update(mode="local", answer=ans,
                   answer_ms=int((time.time() - t_eot) * 1000))
        log(f"talker answered locally in {out['answer_ms']} ms")
    else:
        exp = {}

        def expert_call():
            t0 = time.time()
            uplink = question
            if audio_b64:
                # no gold text exists for real speech — this is the 8ae
                # cloud-ASR uplink (.585 -> .694 vs the talker's own
                # transcript on the frozen pool, McNemar p=.007)
                with open(wav_path, "rb") as fh:
                    tr = escalate._client().audio.transcriptions.create(
                        model="gpt-transcribe", file=fh,
                        response_format="text")
                uplink = tr if isinstance(tr, str) else getattr(
                    tr, "text", str(tr))
                exp["uplink_text"] = uplink
                exp["asr_s"] = time.time() - t0
            r = escalate.ask_expert(uplink, effort="low")
            exp["answer"] = r.get("answer") or f"[error: {r.get('error')}]"
            exp["wall_s"] = time.time() - t0

        th = threading.Thread(target=expert_call, daemon=True)
        th.start()
        log(f"escalating to {escalate.EXPERT_MODEL}; talker stalls: "
            f"“{STALL}”")
        call_def(model.streaming_prefill, session_id="s1",
                 msgs=[{"role": "assistant", "content": [STALL]}],
                 tokenizer=tok, is_last_chunk=True)
        t_stall = time.time()
        th.join(timeout=120)
        t_expert = time.time()
        if exp.get("uplink_text"):
            log(f"uplink: hosted ASR heard “{exp['uplink_text'][:120]}” "
                f"({exp.get('asr_s', 0):.1f} s) — that text is what the "
                f"expert reads (RESULTS 8ae)")
        log(f"expert answered in {exp.get('wall_s', -1):.1f} s "
            f"(talker's stall covered "
            f"{(t_stall - t_eot):.1f} s of it)")
        call_def(model.streaming_prefill, session_id="s1",
                 msgs=[{"role": "user",
                        "content": [RELAY_TMPL.format(
                            ans=exp.get("answer", ""))]}],
                 tokenizer=tok, is_last_chunk=True)
        relay = gen_text(session_id="s1")
        out.update(mode="escalated", answer=relay,
                   expert_answer=exp.get("answer", ""),
                   expert_latency_s=round(exp.get("wall_s", -1), 2),
                   stall_ms=int((t_stall - t_eot) * 1000),
                   uplink_text=exp.get("uplink_text"),
                   asr_s=round(exp.get("asr_s", 0), 2),
                   relay_ms=int((time.time() - t_expert) * 1000))
        log(f"talker relayed the verified answer "
            f"({out['relay_ms']} ms)")
    out["total_ms"] = int((time.time() - t_eot) * 1000) + eot_ms
    out["events"] = ev
    return out


# ---------------------------------------------------------------- replay --
def _load():
    import pandas as pd
    store, agg, thr = {}, {}, {}
    for p in POOLS:
        f = f"{DATA}/gate_v3_{p}.json"
        if os.path.exists(f):
            a = json.load(open(f))
            t = a.get("eot_thresholds") or a.get("thresholds") or a
            thr[p] = {k: float(v) for k, v in t.items()
                      if k in TIERS and isinstance(v, (int, float))}
    for p in POOLS:
        f = f"{DATA}/{p}_v3_traces.parquet"
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        col = "oab_ok" if ("oab_ok" in df.columns
                           and df["oab_ok"].notna().any()) else "heard_ok"
        df["_ok"] = df[col]
        df["expert_ms"] = df["expert_latency_s"].fillna(0) * 1000
        e = df["mode"] == "escalated"
        df["total_ms"] = (
            df["eot_read_ms"]
            + e * (df[["stall_ms", "expert_ms"]].max(axis=1).fillna(0)
                   + df["relay_ms"].fillna(0))
            + (~e) * df["answer_ms"].fillna(0))
        keep = [c for c in ("id", "pool", "tier", "mode", "query",
                            "transcript", "reference_answer", "eot_score",
                            "eot_read_ms", "scores", "answer", "relay",
                            "expert_answer", "expert_latency_s", "stall_ms",
                            "relay_ms", "answer_ms", "audio_s", "_ok",
                            "total_ms") if c in df.columns]
        d = df[keep].copy()
        store[p] = {t: g.set_index("id").to_dict("index")
                    for t, g in d.groupby("tier")}
        arms = {}
        for t in ("never",) + TIERS:
            g = d[d["tier"] == t]
            if len(g):
                arms[t] = {"n": int(len(g)), "acc": float(g["_ok"].mean()),
                           "esc": float((g["mode"] == "escalated").mean()),
                           "p50_ms": float(g["total_ms"].median()),
                           "judge": col}
        agg[p] = arms
    return store, agg, thr


@app.function(image=web_image, volumes={DATA: gate_data}, timeout=60 * 10,
              min_containers=0)
@modal.asgi_app()
def web():
    import random
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel

    STORE, AGG, THR = _load()
    api = FastAPI()

    @api.get(f"/{TOKEN}", response_class=HTMLResponse)
    def index():
        return PAGE.replace("__TOKEN__", TOKEN).replace(
            "__AGG__", json.dumps(AGG)).replace("__THR__", json.dumps(THR))

    @api.get(f"/{TOKEN}/api/pick")
    def pick(pool: str = "frozen", tier: str = "balanced", qid: str = ""):
        if pool not in STORE:
            raise HTTPException(404, "unknown pool")
        tiers = STORE[pool]
        if tier not in tiers or "never" not in tiers:
            raise HTTPException(404, "arm missing")
        ids = sorted(set(tiers[tier]) & set(tiers["never"]))
        if not ids:
            raise HTTPException(404, "no shared ids")
        qid = qid if qid in ids else random.choice(ids)
        on, off = tiers[tier][qid], tiers["never"][qid]

        def clean(d):
            out = {}
            for k, v in d.items():
                if hasattr(v, "tolist"):
                    v = v.tolist()
                if v is not None and not isinstance(
                        v, (str, int, float, bool, list, dict)):
                    v = str(v)
                if isinstance(v, float) and v != v:
                    v = None
                out[k] = v
            return out

        return JSONResponse({"id": qid, "pool": pool, "tier": tier,
                             "n_ids": len(ids), "on": clean(on),
                             "off": clean(off)})

    @api.get(f"/{TOKEN}/api/ids")
    def ids(pool: str = "frozen", tier: str = "balanced", q: str = ""):
        t = STORE.get(pool, {})
        if tier not in t:
            return JSONResponse([])
        rows = [{"id": k, "query": (v.get("query") or "")[:110]}
                for k, v in sorted(t[tier].items())]
        if q:
            ql = q.lower()
            rows = [r for r in rows
                    if ql in r["id"].lower() or ql in r["query"].lower()]
        return JSONResponse(rows[:400])

    class LiveReq(BaseModel):
        question: str = ""
        tier: str = "balanced"
        probe_on: bool = True
        audio_b64: str = ""
        audio_ext: str = "webm"

    # a live turn is a cold H100 + model load + streaming + expert — far
    # past the proxy's synchronous response window, so spawn and poll
    @api.post(f"/{TOKEN}/api/live")
    def live(req: LiveReq):
        if not (req.question.strip() or req.audio_b64):
            raise HTTPException(400, "need a question or a recording")
        fc = live_once.spawn(req.question.strip(), req.tier, req.probe_on,
                             req.audio_b64, req.audio_ext)
        return JSONResponse({"call_id": fc.object_id})

    @api.get(f"/{TOKEN}/api/live_result")
    def live_result(call_id: str):
        fc = modal.FunctionCall.from_id(call_id)
        try:
            return JSONResponse({"status": "done", "result": fc.get(0)})
        except TimeoutError:
            return JSONResponse({"status": "running"})
        except Exception as e:
            return JSONResponse({"status": "error",
                                 "error": f"{type(e).__name__}: {e}"})

    return api


PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Escalation gate — live demo</title><style>
:root{--blue:#2a78d6;--green:#1e9e50;--red:#b00;--ink:#0b0b0b;
--mut:#52514e;--grid:#e6e4de;--bg:#faf9f7;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
header{background:var(--card);border-bottom:1px solid var(--grid);
padding:.7rem 1rem;display:flex;gap:1rem;align-items:center;flex-wrap:wrap;
position:sticky;top:0;z-index:9}
h1{font-size:1rem;margin:0;font-weight:640}
.sub{color:var(--mut);font-size:.78rem}
.wrap{display:grid;grid-template-columns:270px 1fr 340px;gap:1rem;
padding:1rem;align-items:start}
@media(max-width:1100px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;
padding:.85rem}
.card h2{font-size:.8rem;margin:0 0 .6rem;text-transform:uppercase;
letter-spacing:.04em;color:var(--mut);font-weight:640}
button{font:inherit;border:1px solid var(--grid);background:#fff;
border-radius:7px;padding:.4rem .7rem;cursor:pointer}
button:hover{border-color:var(--blue)}
button.primary{background:var(--blue);color:#fff;border-color:var(--blue)}
button.primary:disabled{opacity:.5;cursor:not-allowed}
select,input{font:inherit;border:1px solid var(--grid);border-radius:7px;
padding:.38rem .5rem;background:#fff;width:100%}
.toggle{display:flex;align-items:center;gap:.55rem;padding:.35rem .6rem;
border:1px solid var(--grid);border-radius:99px;background:#fff}
.sw{width:44px;height:24px;border-radius:99px;background:#cfcdc8;
position:relative;transition:.16s;cursor:pointer;flex:none}
.sw.on{background:var(--blue)}
.sw::after{content:"";position:absolute;top:3px;left:3px;width:18px;
height:18px;border-radius:50%;background:#fff;transition:.16s}
.sw.on::after{left:23px}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}
.tile{border:1px solid var(--grid);border-radius:8px;padding:.5rem .6rem}
.tile .k{font-size:.68rem;color:var(--mut);text-transform:uppercase;
letter-spacing:.03em}
.tile .v{font-size:1.25rem;font-weight:650;font-variant-numeric:tabular-nums}
.tile .d{font-size:.7rem;color:var(--mut)}
.log{font:11.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
background:#fbfbfa;border:1px solid var(--grid);border-radius:8px;
padding:.55rem;height:330px;overflow:auto;white-space:pre-wrap}
.log b{color:var(--blue)}.log .esc{color:var(--green);font-weight:700}
.log .off{color:var(--mut)}
.ans{border-left:3px solid var(--grid);padding:.15rem 0 .15rem .7rem;
margin:.45rem 0;white-space:pre-wrap}
.ans.local{border-color:var(--blue)}.ans.esc{border-color:var(--green)}
.pill{display:inline-block;font-size:.7rem;padding:.1rem .5rem;
border-radius:99px;color:#fff;vertical-align:middle}
.pill.ok{background:var(--green)}.pill.no{background:var(--red)}
.pill.g{background:var(--blue)}.pill.m{background:var(--mut)}
.row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
.q{font-size:.95rem;margin:.2rem 0 .6rem}
.qlist{max-height:340px;overflow:auto;margin-top:.5rem}
.qlist div{padding:.3rem .4rem;border-radius:6px;cursor:pointer;
font-size:.78rem;border:1px solid transparent}
.qlist div:hover{background:#f3f6fb;border-color:var(--grid)}
.qlist div.sel{background:#eaf1fb;border-color:var(--blue)}
.spark{width:100%;height:54px;display:block}
.muted{color:var(--mut);font-size:.76rem}
table{width:100%;border-collapse:collapse;font-size:.76rem}
td,th{text-align:right;padding:.22rem .3rem;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
.warn{background:#fff8e6;border:1px solid #f0e0b0;border-radius:8px;
padding:.5rem .6rem;font-size:.76rem;color:#6b5a1e;margin-top:.5rem}
</style></head><body>
<header>
  <h1>Escalation gate</h1>
  <span class=sub>MiniCPM-o 4.5 duplex talker · L22 probe v3 · gpt-5.5 expert</span>
  <div class=toggle><div id=sw class="sw on"></div>
    <b id=swlab>PROBE ON</b></div>
  <select id=tier style="width:auto">
    <option>conservative</option><option selected>balanced</option>
    <option>aggressive</option></select>
  <select id=mode style="width:auto">
    <option value=replay>replay — measured sessions ($0)</option>
    <option value=live>live — real GPU turn</option></select>
  <span id=state class=sub></span>
</header>
<div class=wrap>
  <div>
    <div class=card id=picker>
      <h2>Query</h2>
      <select id=pool></select>
      <input id=search placeholder="filter by id or text" style="margin-top:.45rem">
      <div class=qlist id=qlist></div>
      <div class=row style="margin-top:.5rem">
        <button id=rand>Random</button><button id=run class=primary>Run</button>
      </div>
    </div>
    <div class=card id=livebox style="display:none;margin-top:1rem">
      <h2>Talk to it</h2>
      <button id=mic class=primary style="width:100%;font-size:.95rem">
        Hold to speak — or click to start</button>
      <div class=row style="margin-top:.4rem">
        <span id=rec class=muted>mic idle</span>
        <audio id=play controls style="height:28px;display:none;flex:1"></audio>
      </div>
      <div class=muted style="margin-top:.35rem">Your real voice goes
        straight into the duplex talker — no TTS. If the gate fires, the
        recording is transcribed by the hosted ASR and <i>that</i> is what
        the expert reads (the +.109 uplink, RESULTS 8ae).</div>
      <h2 style="margin-top:.9rem">…or type it</h2>
      <input id=q placeholder="e.g. what is NVDA trading at right now?">
      <div class=row style="margin-top:.4rem" id=chips></div>
      <div class=muted style="margin-top:.4rem">Rendered with tts-1/alloy,
        streamed into the duplex talker in 1 s chunks — the same path the
        benchmarks used.</div>
      <button id=runlive class=primary style="margin-top:.5rem;width:100%">
        Run one live turn</button>
      <div class=warn>Spins an H100 (~1 min cold). Costs a few cents per turn.</div>
    </div>
  </div>
  <div class=card>
    <h2>Turn</h2>
    <div id=turn class=muted>Pick a query and press Run.</div>
  </div>
  <div>
    <div class=card>
      <h2>Session metrics</h2>
      <div class=tiles id=tiles></div>
      <div class=muted style="margin-top:.5rem">Accumulated over the turns
        you ran in this browser session.</div>
      <h2 style="margin-top:.9rem">Reference (full sweep)</h2>
      <table id=ref></table>
    </div>
    <div class=card style="margin-top:1rem">
      <h2>Event log</h2><div class=log id=log></div>
    </div>
  </div>
</div>
<script>
const T="__TOKEN__", AGG=__AGG__, THR=__THR__;
const $=s=>document.querySelector(s);
let probeOn=true, sel=null, S={n:0,ok:0,esc:0,ms:0,onN:0,onOk:0,offN:0,offOk:0};

const pool=$("#pool");
Object.keys(AGG).forEach(p=>{const o=document.createElement("option");
  o.value=p;o.textContent=p;pool.appendChild(o)});

$("#sw").onclick=()=>{probeOn=!probeOn;
  $("#sw").classList.toggle("on",probeOn);
  $("#swlab").textContent=probeOn?"PROBE ON":"PROBE OFF";
  log(probeOn?"probe switched ON — the gate decides per query"
             :"probe switched OFF — every query stays local (never arm)","off");};
$("#mode").onchange=()=>{const live=$("#mode").value==="live";
  $("#livebox").style.display=live?"":"none";
  $("#picker").style.display=live?"none":"";};
$("#tier").onchange=()=>{loadIds();refTable()};
pool.onchange=()=>{loadIds();refTable()};
$("#search").oninput=()=>loadIds();
$("#rand").onclick=()=>{sel=null;run()};
$("#run").onclick=()=>run();
$("#runlive").onclick=()=>runLive();

function log(msg,cls){const l=$("#log");const t=new Date().toLocaleTimeString();
  l.innerHTML+=`<div class="${cls||''}"><b>${t}</b> ${msg}</div>`;
  l.scrollTop=l.scrollHeight;}
function fmt(x,d=3){return x==null?"—":(+x).toFixed(d)}
function esc(s){return (s==null?"":String(s)).replace(/[<>&]/g,
  c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]))}

async function loadIds(){
  if($("#mode").value==="live")return;
  const r=await fetch(`/${T}/api/ids?pool=${pool.value}&tier=${$("#tier").value}`
    +`&q=${encodeURIComponent($("#search").value)}`);
  const rows=await r.json();
  $("#qlist").innerHTML=rows.map(x=>
    `<div data-id="${x.id}" class="${x.id===sel?'sel':''}">
      <b>${x.id}</b> ${esc(x.query)}</div>`).join("");
  [...document.querySelectorAll("#qlist div")].forEach(d=>
    d.onclick=()=>{sel=d.dataset.id;loadIds();run()});
}
function refTable(){
  const a=AGG[pool.value]||{};
  const rows=Object.entries(a).map(([k,v])=>
    `<tr><td>${k}</td><td>${(v.esc*100).toFixed(0)}%</td>
     <td>${fmt(v.acc,3)}</td><td>${(v.p50_ms/1000).toFixed(2)}s</td></tr>`);
  $("#ref").innerHTML=`<tr><th>arm</th><th>esc</th><th>acc</th><th>P50</th></tr>`
    +rows.join("");
}
function tiles(){
  const t=[["turns",S.n,""],
    ["accuracy",S.n?fmt(S.ok/S.n,3):"—","judge verdict"],
    ["escalated",S.n?Math.round(100*S.esc/S.n)+"%":"—","of your turns"],
    ["P50-ish",S.n?(S.ms/S.n/1000).toFixed(2)+"s":"—","mean total"],
    ["probe ON",S.onN?fmt(S.onOk/S.onN,3):"—",`${S.onN} turns`],
    ["probe OFF",S.offN?fmt(S.offOk/S.offN,3):"—",`${S.offN} turns`]];
  $("#tiles").innerHTML=t.map(([k,v,d])=>
    `<div class=tile><div class=k>${k}</div><div class=v>${v}</div>
     <div class=d>${d}</div></div>`).join("");
}
function spark(scores,thr,eot){
  if(!scores||!scores.length)return"";
  const w=300,h=54,n=scores.length,pad=4;
  const X=i=>pad+i*(w-2*pad)/Math.max(n-1,1), Y=v=>h-pad-v*(h-2*pad);
  const d=scores.map((v,i)=>`${i?"L":"M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  return `<svg class=spark viewBox="0 0 ${w} ${h}">
   <line x1=0 y1=${Y(thr).toFixed(1)} x2=${w} y2=${Y(thr).toFixed(1)}
    stroke="#b00" stroke-dasharray="3 3" stroke-width=1/>
   <path d="${d}" fill=none stroke="#2a78d6" stroke-width=2/>
   <circle cx=${X(n-1).toFixed(1)} cy=${Y(scores[n-1]).toFixed(1)} r=3
    fill="#2a78d6"/>
   <text x=${w-2} y=${(Y(thr)-3).toFixed(1)} font-size=8 fill="#b00"
    text-anchor=end>threshold ${thr.toFixed(2)}</text></svg>`;
}
function verdict(ok){return ok===1||ok===true
  ?'<span class="pill ok">judge: correct</span>'
  :(ok===0||ok===false?'<span class="pill no">judge: wrong</span>'
   :'<span class="pill m">unjudged</span>')}

async function run(){
  if($("#mode").value==="live")return runLive();
  $("#state").textContent="loading…";
  const r=await fetch(`/${T}/api/pick?pool=${pool.value}`
    +`&tier=${$("#tier").value}&qid=${sel||""}`);
  if(!r.ok){$("#state").textContent="error";return}
  const d=await r.json(); sel=d.id; $("#state").textContent=
    `${d.n_ids} measured queries in ${d.pool}`;
  const row=probeOn?d.on:d.off;
  const fired=probeOn&&row.mode==="escalated";
  const thr=probeOn?impliedThr(d):null;
  log(`— replay ${d.id} (${d.pool}), probe ${probeOn?"ON":"OFF"}, `
    +`tier ${d.tier} —`);
  (row.scores||[]).forEach((s,i)=>log(
    `chunk ${i+1}/${row.scores.length} — running P(fail)=${fmt(s)}`));
  log(`END OF TURN — probe read L22 in ${row.eot_read_ms??"?"} ms: `
    +`P(fail)=<b>${fmt(row.eot_score)}</b>`);
  if(!probeOn) log("PROBE OFF — gate bypassed, answering locally","off");
  else log(`gate → <span class="${fired?'esc':''}">`
    +`${fired?"ESCALATE":"keep local"}</span>`);
  if(fired){log(`expert answered in ${fmt(row.expert_latency_s,1)}s; `
    +`talker stalled ${row.stall_ms??"?"} ms, relayed in ${row.relay_ms??"?"} ms`);}
  else log(`talker answered locally in ${row.answer_ms??"?"} ms`);

  const ans=fired?(row.relay||row.expert_answer):row.answer;
  $("#turn").innerHTML=`
   <div class=row><span class="pill g">${d.pool}</span>
     <span class="pill m">${d.id}</span>
     <span class="pill ${fired?'ok':'m'}">${fired?"ESCALATED":"LOCAL"}</span>
     ${verdict(row._ok)}</div>
   <div class=q><b>Q.</b> ${esc(row.query)}</div>
   <div class=muted>what the talker actually heard (its own transcript):
     <i>${esc(row.transcript||"—")}</i></div>
   ${spark(row.scores,thr??1.01,row.eot_score)}
   <div class=muted>P(fail) as the question streams in; red = the
     ${d.tier} threshold${probeOn?"":" (probe is OFF — not applied)"}</div>
   <div class="ans ${fired?'esc':'local'}"><b>${fired?"Relay (talker voicing the expert)":"Talker, alone"}:</b>
     ${esc(ans)}</div>
   ${fired?`<div class=muted><b>expert (gpt-5.5) said:</b>
     ${esc(row.expert_answer)}</div>`:""}
   <div class=muted style="margin-top:.5rem">reference answer:
     ${esc(row.reference_answer)}</div>
   <table style="margin-top:.6rem">
    <tr><th>metric</th><th>value</th></tr>
    <tr><td>probe score (P fail)</td><td>${fmt(row.eot_score)}</td></tr>
    <tr><td>probe read latency</td><td>${row.eot_read_ms??"—"} ms</td></tr>
    <tr><td>total response</td><td>${(row.total_ms/1000).toFixed(2)} s</td></tr>
    <tr><td>audio length</td><td>${fmt(row.audio_s,1)} s</td></tr>
   </table>`;
  S.n++;S.ok+=(row._ok===1||row._ok===true)?1:0;S.esc+=fired?1:0;
  S.ms+=row.total_ms||0;
  if(probeOn){S.onN++;S.onOk+=(row._ok===1||row._ok===true)?1:0}
  else{S.offN++;S.offOk+=(row._ok===1||row._ok===true)?1:0}
  tiles();
}
function impliedThr(d){
  // the real per-domain quantile threshold the sweep deployed
  return ((THR[d.pool]||{})[d.tier]);
}

let MR=null,CH=[],BLOB=null,T0=0,TIMER=null;
async function startRec(){
  try{
    const st=await navigator.mediaDevices.getUserMedia({audio:{
      channelCount:1,echoCancellation:true,noiseSuppression:true}});
    MR=new MediaRecorder(st);CH=[];
    MR.ondataavailable=e=>{if(e.data.size)CH.push(e.data)};
    MR.onstop=()=>{
      BLOB=new Blob(CH,{type:MR.mimeType||"audio/webm"});
      $("#play").src=URL.createObjectURL(BLOB);
      $("#play").style.display="";
      st.getTracks().forEach(t=>t.stop());
      clearInterval(TIMER);
      const s=((Date.now()-T0)/1000).toFixed(1);
      $("#rec").textContent=`recorded ${s}s — sending…`;
      log(`recorded ${s}s of speech`);
      runLive();
    };
    MR.start();T0=Date.now();
    $("#mic").textContent="● recording — click to stop";
    $("#mic").style.background="#b00";$("#mic").style.borderColor="#b00";
    TIMER=setInterval(()=>{$("#rec").textContent=
      `recording ${((Date.now()-T0)/1000).toFixed(1)}s`},100);
  }catch(e){log("microphone blocked: "+e,"off");
    $("#rec").textContent="mic permission denied"}
}
function stopRec(){
  if(MR&&MR.state!=="inactive")MR.stop();
  MR=null;$("#mic").textContent="Hold to speak — or click to start";
  $("#mic").style.background="";$("#mic").style.borderColor="";
}
$("#mic").onclick=()=>{ if(MR&&MR.state==="recording") stopRec();
                        else startRec(); };
const b64=b=>new Promise(r=>{const f=new FileReader();
  f.onloadend=()=>r(f.result.split(",")[1]);f.readAsDataURL(b)});

async function runLive(){
  const q=$("#q").value.trim();
  if(!q&&!BLOB){log("record something or type a question first","off");return}
  $("#runlive").disabled=true;$("#state").textContent=
    "live turn running (cold start ~1 min)…";
  log(`— live turn, probe ${probeOn?"ON":"OFF"}, tier ${$("#tier").value} —`);
  try{
    const r=await fetch(`/${T}/api/live`,{method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({question:BLOB?"":q,tier:$("#tier").value,
        probe_on:probeOn,
        audio_b64:BLOB?await b64(BLOB):"",
        audio_ext:BLOB&&/ogg/.test(BLOB.type)?"ogg":"webm"})});
    if(!r.ok){log("live error: "+await r.text(),"off");return}
    const {call_id}=await r.json();
    log(`queued on an H100 (call ${call_id.slice(0,10)}…) — polling`);
    let d=null;
    for(let i=0;i<200;i++){
      await new Promise(s=>setTimeout(s,3000));
      const p=await fetch(`/${T}/api/live_result?call_id=${call_id}`);
      const j=await p.json();
      if(j.status==="done"){d=j.result;break}
      if(j.status==="error"){log("live error: "+j.error,"off");return}
      $("#state").textContent=`live turn running… ${(i+1)*3}s`;
    }
    if(!d){log("live turn timed out after 10 min","off");return}
    (d.events||[]).forEach(e=>log(`+${e.t_ms}ms ${esc(e.msg)}`));
    const fired=d.fired;
    $("#turn").innerHTML=`
     <div class=row><span class="pill g">live</span>
       <span class="pill ${fired?'ok':'m'}">${fired?"ESCALATED":"LOCAL"}</span></div>
     <div class=q><b>Q.</b> ${esc(d.question)||"<i>(your voice)</i>"}</div>
     ${d.uplink_text?`<div class=muted>hosted ASR heard (this is what the
       expert read): <i>${esc(d.uplink_text)}</i></div>`:""}
     ${spark(d.scores,d.threshold,d.eot_score)}
     <div class=muted>P(fail) as the question streams in; red = the
       ${d.tier} threshold</div>
     <div class="ans ${fired?'esc':'local'}">
       <b>${fired?"Relay (talker voicing the expert)":"Talker, alone"}:</b>
       ${esc(d.answer)}</div>
     ${fired?`<div class=muted><b>expert (gpt-5.5) said:</b>
       ${esc(d.expert_answer)}</div>`:""}
     <table style="margin-top:.6rem">
      <tr><th>metric</th><th>value</th></tr>
      <tr><td>probe score (P fail)</td><td>${fmt(d.eot_score)}</td></tr>
      <tr><td>threshold (${d.tier})</td><td>${fmt(d.threshold)}</td></tr>
      <tr><td>probe read latency</td><td>${d.eot_read_ms} ms</td></tr>
      <tr><td>total response</td><td>${(d.total_ms/1000).toFixed(2)} s</td></tr>
      <tr><td>audio length</td><td>${fmt(d.audio_s,1)} s</td></tr>
     </table>`;
    S.n++;S.esc+=fired?1:0;S.ms+=d.total_ms||0;
    if(probeOn)S.onN++;else S.offN++;
    tiles();BLOB=null;$("#rec").textContent="mic idle";
  }catch(e){log("live error: "+e,"off")}
  finally{$("#runlive").disabled=false;$("#state").textContent=""}
}
const EX=[
 ["What is NVDA trading at right now?","no model can know this — watch it escalate"],
 ["Who, more famous for his opera Faust, wrote the music used for the anthem of Vatican City?","a long-tail fact the talker got wrong in the sweep; the gate rescued it"],
 ["John made 4 wooden tables at 20 dollars each and 2 roof frames at 15 dollars each. How much did he earn?","easy arithmetic — the gate should stay local"],
];
$("#chips").innerHTML=EX.map((e,i)=>
  `<button data-i="${i}" title="${esc(e[1])}" style="font-size:.72rem">
    ${esc(e[0].slice(0,34))}…</button>`).join("");
[...document.querySelectorAll("#chips button")].forEach(b=>
  b.onclick=()=>{$("#q").value=EX[b.dataset.i][0];
    log("loaded example: "+EX[b.dataset.i][1],"off")});
loadIds();refTable();tiles();
log("ready — replay mode reads the 4773 measured sessions; "
   +"flipping the probe shows the same query's other measured arm.");
</script></body></html>"""
