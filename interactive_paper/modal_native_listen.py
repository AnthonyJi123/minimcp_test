"""Mid-turn listen suppression probe (8bk): is the head reacting to
"stop" mid-answer while the stock serving code suppresses it?

The duplex wrapper force-rewrites a mid-turn sampled <|listen|> to
tts_bos ("not allowed to listen"), so the only permitted stop is a
full <|turn_eos|>. Hypothesis: on overlap the head DOES sample listen
mid-turn (its yield signal) and the rewrite is why answers run to
completion (user report: said "stop" at count three, model counted to
ten).

Cells (n per cell): carrier = "Please count from one to ten." (TTS'd
once, long enumerable answer, no natural boundary), stim injected 2
chunks after speak-onset.
  mode=instrument  : stock behavior + listen_attempts counter
  mode=yield       : allow_midturn_yield (>=2 attempts in a chunk ->
                     honored as turn_eos)
  stim=none | stop (floor_sweep stim) | bq (sustained pool question)

Outputs per trial: per-chunk (listen/speak, listen_attempts, text),
stim_at, turn_end, post-stim chunks. -> /data/native_listen/trials.jsonl

Run: modal run modal_native_listen.py::run_listen --limit 1   # smoke
     modal run modal_native_listen.py::run_listen             # full
"""
import json
import os
import time

import modal

app = modal.App("native-listen")
gate_data = modal.Volume.from_name("gate-data")
weights = modal.Volume.from_name("minicpm-o45-weights")
DATA = "/data"
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"
PROMPT_WAV = f"{MODEL_DIR}/assets/system_ref_audio.wav"
OUT_DIR = f"{DATA}/native_listen"
CARRIER_TEXT = "Please count slowly from one to thirty."
N_PER_CELL = 6

from modal_app import OPENAI  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(_HERE, "modal_app.py")

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
    .add_local_dir(os.path.join(_HERE, "_model_src"),
                   "/workspace/model_src")
    .add_local_file(_APP_PY, "/root/modal_app.py"))


@app.function(image=gpu_image, gpu="H100",
              volumes={"/workspace/models": weights, DATA: gate_data},
              secrets=[OPENAI], timeout=60 * 60 * 2)
def listen_shard(trials: list, shard_id: int = -1) -> list:
    import glob as _glob
    import shutil
    import sys

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
    for f in _glob.glob("/workspace/model_src/*.py"):
        shutil.copy(f, cache)      # 8bk instrumented sources win
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=True,
    ).eval().cuda()
    _ = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    # TTS ON (audio discarded): the head paces itself to speech
    # rhythm only with the TTS template — without it the whole
    # answer fits 2-3 chunks and there is no mid-turn to probe
    duplex = model.as_duplex(generate_audio=True)
    ref, _sr = librosa.load(PROMPT_WAV, sr=16000, mono=True)

    # carrier wav (TTS once per container)
    cw = "/tmp/carrier.wav"
    r = escalate._client().audio.speech.create(
        model="tts-1", voice="alloy", input=CARRIER_TEXT,
        response_format="wav")
    open("/tmp/c0.wav", "wb").write(r.content)
    cau, _ = librosa.load("/tmp/c0.wav", sr=16000, mono=True)
    sf.write(cw, cau.astype(np.float32), 16000)

    def to_chunks(au):
        cs = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        return [np.pad(c, (0, 16000 - len(c)))
                if len(c) < 16000 else c for c in cs]

    carrier = to_chunks(cau.astype(np.float32))
    rng = np.random.default_rng(11)

    def sil():
        return rng.normal(0, 0.003, 16000).astype(np.float32)

    def load_stim(kind, i):
        if kind == "stop":
            au, _s = librosa.load(f"{DATA}/floor_sweep/stim/stop{i % 3}.wav",
                                  sr=16000, mono=True)
        else:
            au, _s = librosa.load(f"{DATA}/audio_pool/q0461.wav",
                                  sr=16000, mono=True)
        return to_chunks(au.astype(np.float32))

    results = []
    for ti, t in enumerate(trials):
        duplex.allow_midturn_yield = (t["mode"] == "yield")
        duplex.midturn_yield_k = 2
        duplex.prepare(
            prefix_system_prompt="Streaming Omni Conversation.",
            ref_audio=ref, prompt_wav_path=PROMPT_WAV)

        rec = dict(t, pattern="", attempts=[], stim_at=None,
                   onset=None, end_at=None, texts=[])
        feed = list(carrier)
        prev_listen = True
        speak_seen = 0
        ci = -1
        while ci < 60:
            ci += 1
            ch = feed.pop(0) if feed else sil()
            ok = duplex.streaming_prefill(audio_waveform=ch)
            if not ok.get("success"):
                rec["pattern"] += "x"
                continue
            r = duplex.streaming_generate()
            rec["pattern"] += "L" if r["is_listen"] else "S"
            rec["attempts"].append(int(r.get("listen_attempts", 0)))
            if r.get("text"):
                rec["texts"].append(r["text"])
            if prev_listen and not r["is_listen"] and rec["onset"] is None:
                rec["onset"] = ci
            if not r["is_listen"]:
                speak_seen += 1
            if (t["stim"] != "none" and rec["stim_at"] is None
                    and speak_seen == 2):
                feed = load_stim(t["stim"], ti) + feed
                rec["stim_at"] = ci + 1
            if r.get("end_of_turn") and rec["onset"] is not None:
                rec["end_at"] = ci
                break
            if rec["onset"] is not None and ci - rec["onset"] > 40:
                break
            prev_listen = r["is_listen"]
        rec["texts"] = "".join(rec["texts"])[-200:]
        post = (rec["end_at"] - rec["stim_at"]
                if rec["end_at"] is not None and rec["stim_at"] is not None
                else None)
        print(f"  [{ti}] {t['mode']}:{t['stim']} onset={rec['onset']} "
              f"stim@{rec['stim_at']} end@{rec['end_at']} post={post} "
              f"attempts={rec['attempts']} "
              f"txt={rec['texts'][:70]!r}", flush=True)
        results.append(rec)

    os.makedirs(OUT_DIR, exist_ok=True)
    sfx = "smoke" if shard_id < 0 else f"shard{max(shard_id, 0)}"
    with open(f"{OUT_DIR}/trials.jsonl.{sfx}", "a",
              encoding="utf-8") as fh:
        for rr in results:
            fh.write(json.dumps(rr, ensure_ascii=False) + "\n")
    gate_data.commit()
    return [1] * len(results)


@app.local_entrypoint()
def run_listen(workers: int = 2, limit: int = 0):
    trials = []
    for mode in ("instrument", "yield"):
        for stim in ("none", "stop", "bq"):
            for j in range(limit or N_PER_CELL):
                trials.append({"mode": mode, "stim": stim, "j": j})
    shards = [trials[i::workers] for i in range(workers)]
    print(f">>> listen probe: {len(trials)} trials, {workers} workers")
    done = list(listen_shard.starmap(
        [(shards[i], i if not limit else -1) for i in range(workers)]))
    print(f">>> complete: {sum(len(d) for d in done)}")
