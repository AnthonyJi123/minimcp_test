"""Native-regime LIVE escalation run (8bg): the conclive protocol on
MiniCPMODuplex — real ASR uplink, real gpt-5.5, real wait pacing.

Validates the 8be offline remix end-to-end: per query, a fresh duplex
session streams the question; at the head's listen->speak commit the
gate reads; fired turns inject the stall note, uplink the RAW question
audio through hosted ASR to the expert (web tool, low effort), pace
the wait at 1 chunk/second wall clock (deployment-faithful silence
prefill), then inject the relay unit (nudge retry as deployed) and
collect the relayed text. Delivered-channel outcome = relay text on
fired turns, local answer otherwise. Latency components logged per
fired turn: eot->stall note, ASR wall, expert wall, relay chunks.

Outputs: /data/native_live/{tier}.jsonl.shard{i} (restartable by id).

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run modal_native_live.py::run_live --tier balanced --limit 6
  modal run modal_native_live.py::run_live --tier balanced --workers 4
  modal run modal_native_live.py::run_live --tier aggressive --workers 4
"""
import json
import os
import time

import modal

app = modal.App("native-live")
gate_data = modal.Volume.from_name("gate-data")
weights = modal.Volume.from_name("minicpm-o45-weights")
DATA = "/data"
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"
PROMPT_WAV = f"{MODEL_DIR}/assets/system_ref_audio.wav"
LAYER = 22
K3 = 8
ART = f"{DATA}/gate_native.json"
OUT_DIR = f"{DATA}/native_live"
MAX_WAIT_S = 150
MAX_ANS = 60

RELAY_TMPL = ("A verified answer came back: {ans}\n"
              "Relay it to the user in one or two spoken sentences.")
RELAY_NUDGE = "Say the verified answer aloud to the user now."
STALL = "Hmm, let me double-check that — one moment."
STALL_NOTE = ("[SYSTEM NOTE] Your answer so far is likely wrong. You "
              "just told the user: \"" + STALL + "\" A verified answer "
              "will arrive in a moment.")

from modal_app import OPENAI  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(_HERE, "modal_app.py")

util_img = (modal.Image.debian_slim(python_version="3.11")
            .pip_install("pandas", "pyarrow")
            .add_local_file(_APP_PY, "/root/modal_app.py"))
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsndfile1")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install(
        "minicpmo-utils[all]",
        "transformers==4.51.0",
        "accelerate==1.12.0",
        "setuptools<81",
        "pydantic>=2.11",
        "PyYAML",
        "soundfile",
        "opencv-python-headless",
        "huggingface_hub[hf_transfer]",
        "scikit-learn",
        "pandas",
        "pyarrow",
        "openai",
        "sentencepiece",
        "fastapi[standard]",   # layer-hash parity
    )
    .add_local_dir(os.path.join(_HERE, "src"), "/workspace/gate")
    .add_local_file(_APP_PY, "/root/modal_app.py"))


@app.function(image=gpu_image, gpu="H100",
              volumes={"/workspace/models": weights, DATA: gate_data},
              secrets=[OPENAI], timeout=60 * 60 * 5)
def live_shard(shard: list, tier: str, shard_id: int = -1) -> list:
    import glob as _glob
    import shutil
    import sys
    import threading

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import escalate

    cache = os.path.expanduser("~/.cache/huggingface/modules/"
                               "transformers_modules/"
                               + os.path.basename(MODEL_DIR))
    os.makedirs(cache, exist_ok=True)
    for f in _glob.glob(f"{MODEL_DIR}/*.py"):
        shutil.copy(f, cache)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=True,
    ).eval().cuda()
    _ = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    duplex = model.as_duplex(generate_audio=False)
    ref, _sr = librosa.load(PROMPT_WAV, sr=16000, mono=True)

    art = json.load(open(ART))
    w = np.array(art["w"], dtype=np.float32)
    b = float(art["b"])
    thr = {"never": 1e9, "always": -1e9}.get(
        tier, art["eot_thresholds"].get(tier, 1e9))

    st3 = {"accum": False, "tail": None, "sum": None, "cnt": 0}

    def hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        h = hs[0].detach().float()
        t = h[-K3:].cpu()
        st3["tail"] = (t if st3["tail"] is None
                       else torch.cat([st3["tail"], t])[-K3:])
        if st3["accum"]:
            sm = h.sum(0).cpu()
            st3["sum"] = sm if st3["sum"] is None else st3["sum"] + sm
            st3["cnt"] += h.shape[0]
    hh = model.llm.model.layers[LAYER].register_forward_hook(hook)

    def score_now():
        parts = [st3["tail"][-1], st3["tail"].mean(0),
                 st3["sum"] / max(1, st3["cnt"])]
        v = torch.cat(parts).numpy()
        return float(1.0 / (1.0 + np.exp(-(float(v @ w) + b))))

    rng = np.random.default_rng(9)

    def sil():
        return rng.normal(0, 0.003, 16000).astype(np.float32)

    results = []
    for qi, q in enumerate(shard):
        wav_p = f"{DATA}/audio_pool/{q['id']}.wav"
        if not os.path.exists(wav_p):
            continue
        au, _sr = librosa.load(wav_p, sr=16000, mono=True)
        chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        chunks = [np.pad(c, (0, 16000 - len(c)))
                  if len(c) < 16000 else c for c in chunks]

        duplex.prepare(
            prefix_system_prompt="Streaming Omni Conversation.",
            ref_audio=ref, prompt_wav_path=None)
        st3.update(tail=None, sum=None, cnt=0, accum=False)

        rec = {"id": q["id"], "tier": tier, "score": None,
               "fired": False, "mode": "local", "answer": "",
               "relay": "", "expert_answer": "", "uplink_text": "",
               "asr_s": None, "expert_s": None, "stall_note_ms": None,
               "wait_chunks": 0, "relay_nudged": False,
               "n_q_chunks": len(chunks), "onset_chunk": None,
               "eot_seen": False}
        exp = {}
        exp_done = threading.Event()

        def expert_call(snapshot):
            try:
                t0 = time.time()
                sf.write(f"/tmp/up{shard_id}.wav", snapshot, 16000)
                with open(f"/tmp/up{shard_id}.wav", "rb") as fh:
                    tr = (escalate._client().audio.transcriptions
                          .create(model="gpt-transcribe", file=fh,
                                  response_format="text"))
                up = (tr if isinstance(tr, str)
                      else getattr(tr, "text", str(tr)))
                exp["uplink"] = str(up)
                exp["asr_s"] = round(time.time() - t0, 2)
                t1 = time.time()
                r = escalate.ask_expert_web(up, effort="low")
                if r.get("error"):
                    r = escalate.ask_expert(up, effort="low")
                exp["answer"] = (r.get("answer")
                                 or f"[error: {r.get('error')}]")
                exp["expert_s"] = round(time.time() - t1, 2)
            except Exception as e:
                exp["answer"] = f"[thinker failed: {str(e)[:100]}]"
            finally:
                exp_done.set()

        texts, n_ans = [], 0
        prev_listen = True
        feed = list(chunks)
        waiting, relay_started = False, False
        t_wait0 = None
        ci = -1
        try:
            while ci < 400:
                ci += 1
                if waiting and not relay_started:
                    # deployment-faithful pacing: 1 silence chunk per
                    # wall-clock second while the thinker runs
                    time.sleep(max(0.0, 1.0 - 0.15))
                    if exp_done.is_set() or \
                            time.time() - t_wait0 > MAX_WAIT_S:
                        relay_started = True
                        duplex.streaming_prefill(text_list=[
                            RELAY_TMPL.format(
                                ans=exp.get("answer", "[no answer]"))])
                        r = duplex.streaming_generate()
                        if not r.get("text"):
                            rec["relay_nudged"] = True
                            duplex.streaming_prefill(
                                text_list=[RELAY_NUDGE])
                            r = duplex.streaming_generate()
                        if r.get("text"):
                            texts.append(r["text"])
                        prev_listen = r["is_listen"]
                        if r.get("end_of_turn"):
                            rec["eot_seen"] = True
                            break
                        continue
                    rec["wait_chunks"] += 1
                ch = feed.pop(0) if feed else sil()
                st3["accum"] = True
                ok = duplex.streaming_prefill(audio_waveform=ch)
                st3["accum"] = False
                if not ok.get("success"):
                    continue
                r = duplex.streaming_generate()
                if r.get("text"):
                    texts.append(r["text"])

                if prev_listen and not r["is_listen"] \
                        and rec["onset_chunk"] is None:
                    rec["onset_chunk"] = ci
                    sc = score_now()
                    rec["score"] = round(sc, 4)
                    rec["fired"] = bool(sc >= thr)
                    if rec["fired"]:
                        rec["mode"] = "escalated"
                        snap = au[-30 * 16000:]
                        threading.Thread(target=expert_call,
                                         args=(snap,),
                                         daemon=True).start()
                        t0n = time.time()
                        if not r.get("end_of_turn"):
                            duplex.streaming_prefill(
                                text_list=[STALL_NOTE])
                            r2 = duplex.streaming_generate()
                            if r2.get("text"):
                                texts.append(r2["text"])
                            rec["stall_note_ms"] = int(
                                (time.time() - t0n) * 1000)
                            prev_listen = r2["is_listen"]
                            if r2.get("end_of_turn"):
                                waiting = True
                                t_wait0 = time.time()
                            continue
                        waiting = True
                        t_wait0 = time.time()
                        continue

                if not r["is_listen"]:
                    n_ans += 1
                if r.get("end_of_turn"):
                    if rec["fired"] and not relay_started:
                        waiting = True
                        t_wait0 = time.time()
                        prev_listen = True
                        continue
                    rec["eot_seen"] = True
                    break
                if not rec["fired"] and n_ans >= MAX_ANS:
                    break
                if relay_started and n_ans >= MAX_ANS:
                    break
                prev_listen = r["is_listen"]
        except Exception as e:
            rec["answer"] = f"[error: {str(e)[:120]}]"

        full = "".join(texts).strip()
        if rec["fired"]:
            rec["relay"] = full
            rec["expert_answer"] = exp.get("answer", "")
            rec["uplink_text"] = exp.get("uplink", "")[:300]
            rec["asr_s"] = exp.get("asr_s")
            rec["expert_s"] = exp.get("expert_s")
        else:
            rec["answer"] = full
        results.append(rec)
        print(f"  [{qi}] {q['id']} score={rec['score']} "
              f"fired={rec['fired']} wait={rec['wait_chunks']} "
              f"txt={full[:60]!r}", flush=True)

    hh.remove()
    os.makedirs(OUT_DIR, exist_ok=True)
    sfx = "smoke" if shard_id < 0 else f"shard{max(shard_id, 0)}"
    with open(f"{OUT_DIR}/{tier}.jsonl.{sfx}", "a",
              encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    return [r["id"] for r in results]


judge_img = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("openai", "pandas", "pyarrow")
             .add_local_dir(os.path.join(_HERE, "src"), "/workspace/gate")
             .add_local_file(_APP_PY, "/root/modal_app.py"))


@app.function(image=judge_img, volumes={DATA: gate_data},
              secrets=[OPENAI], timeout=60 * 60)
def judge_live(tier: str):
    """gpt-5.4-mini judge over the DELIVERED content of a live tier run
    (relay text on fired turns, local answer otherwise) ->
    /data/native_live/{tier}_judged.parquet."""
    import asyncio
    import glob as _glob
    import sys

    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    qs = {q["id"]: q for q in
          (json.loads(x) for x in
           open(f"{DATA}/queries.jsonl", encoding="utf-8") if x.strip())}
    rows = {}
    for p in sorted(_glob.glob(f"{OUT_DIR}/{tier}.jsonl.shard*")):
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                rows[r["id"]] = r
    out_p = f"{OUT_DIR}/{tier}_judged.parquet"
    old = (pd.read_parquet(out_p)
           if os.path.exists(out_p) else pd.DataFrame(columns=["id"]))
    have = set(old["id"])
    todo = []
    for r in rows.values():
        if r["id"] in have or r["id"] not in qs:
            continue
        delivered = r["relay"] if r["fired"] else r["answer"]
        todo.append({"id": r["id"], "query": qs[r["id"]]["query"],
                     "reference_answer":
                     qs[r["id"]].get("reference_answer"),
                     "answer": delivered or "", "fired": r["fired"],
                     "score": r["score"]})
    print(f">>> judge_live[{tier}]: {len(todo)} to judge")
    if todo:
        judged = asyncio.run(escalate.judge_many(todo, concurrency=8))
        new = pd.concat([old, pd.DataFrame(judged)], ignore_index=True)
        new.to_parquet(out_p)
        gate_data.commit()
        ok = [r for r in judged if r["adequate"] is not None]
        acc = sum(r["adequate"] for r in ok) / max(1, len(ok))
        fr = sum(1 for r in ok if r["fired"]) / max(1, len(ok))
        print(f">>> live[{tier}]: delivered acc {acc:.3f} "
              f"(fire {fr:.2f}, n={len(ok)})")
    return len(todo)


@app.function(image=util_img, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_test(tier: str) -> list:
    import glob as _glob
    qs = [json.loads(x) for x in
          open(f"{DATA}/queries.jsonl", encoding="utf-8") if x.strip()]
    qs = [q for q in qs if q.get("split") == "test"]
    done = set()
    for p in _glob.glob(f"{OUT_DIR}/{tier}.jsonl.shard*"):
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                done.add(json.loads(ln)["id"])
    return [q for q in qs if q["id"] not in done]


@app.local_entrypoint()
def run_live(tier: str = "balanced", workers: int = 4, limit: int = 0):
    qs = _read_test.remote(tier)
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> native live [{tier}]: {len(qs)} queries, "
          f"{workers} workers")
    done = list(live_shard.starmap(
        [(shards[i], tier, i if not limit else -1)
         for i in range(workers)]))
    print(f">>> complete: {sum(len(d) for d in done)}")
