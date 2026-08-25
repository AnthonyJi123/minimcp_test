"""TTS-output smoke on the demo stack (audio-out milestone, RESULTS gap #3).

First-ever init_tts=True run in this project: verifies MiniCPM-o 4.5's talker
head + stepaudio2 Token2wav vocoder work under the pinned image
(torch 2.8 / transformers 4.51) and measures what audio-out costs.

Replicates demo_app.py's Voice session flow EXACTLY (omni sys prompt, 1 s
audio chunks, assistant-" " eot prefill) so the result transfers 1:1 —
the only change is init_tts=True at load and generate_audio=True at answer
time (the <|tts_bos|> is inserted AFTER the probe's end-of-turn read, so
frozen thresholds are untouched). Two turns: local-answer path, then the
stall+relay path. Wavs land on the gate-data volume as tts_smoke_*.wav.

Run:  modal run modal_tts_smoke.py::smoke
"""
import os
import time

import modal

from demo_app import (gpu_image, weights, gate_data, DATA, MODEL_DIR,
                      STALL, RELAY_TMPL, _call_def)

app = modal.App("gate-tts-smoke")

# Modal auto-mounts only the entry module; the container re-imports
# demo_app, so mount it explicitly (same gotcha as modal_audio.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
image_smoke = gpu_image.add_local_file(
    os.path.join(_HERE, "demo_app.py"), "/root/demo_app.py")


@app.function(image=image_smoke, gpu="H100", timeout=60 * 30,
              volumes={"/workspace/models": weights, DATA: gate_data})
def smoke(wav: str = "q0593.wav"):
    import glob
    import shutil

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModel, AutoTokenizer

    r = {}
    cache = os.path.expanduser("~/.cache/huggingface/modules/"
                               "transformers_modules/"
                               + os.path.basename(MODEL_DIR))
    os.makedirs(cache, exist_ok=True)
    for f in glob.glob(f"{MODEL_DIR}/*.py"):
        shutil.copy(f, cache)

    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True,
        init_tts=True).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    r["load_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    model.init_tts(model_dir=f"{MODEL_DIR}/assets/token2wav")
    # one-time voice prompt for the vocoder (official default timbre);
    # vocoder-side only — never enters the LLM context, probe untouched
    ref, _ = librosa.load(f"{MODEL_DIR}/assets/system_ref_audio.wav",
                          sr=16000, mono=True)
    model.init_token2wav_cache(ref)
    r["vocoder_s"] = round(time.time() - t0, 1)
    r["vram_gb"] = round(torch.cuda.memory_allocated() / 1e9, 1)

    # ---- session flow, verbatim demo_app.Voice ---------------------------
    model.reset_session(reset_token2wav_cache=False)
    sys_msg = _call_def(model.get_sys_prompt, mode="omni", language="en")
    _call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
              tokenizer=tok)

    au, _ = librosa.load(f"{DATA}/audio_pool/{wav}", sr=16000, mono=True)
    n = max(1, (len(au) + 15999) // 16000)
    for i in range(n):
        ch = au[i * 16000:(i + 1) * 16000]
        if len(ch) < 16000:
            ch = np.pad(ch, (0, 16000 - len(ch)))
        _call_def(model.streaming_prefill, session_id="s1",
                  msgs=[{"role": "user", "content": [ch.astype("float32")]}],
                  tokenizer=tok, is_last_chunk=(i == n - 1))
    _call_def(model.streaming_prefill, session_id="s1",
              msgs=[{"role": "assistant", "content": [" "]}],
              tokenizer=tok, is_last_chunk=True)

    def gen_audio(tag):
        t0 = time.time()
        res = _call_def(model.streaming_generate, session_id="s1",
                        tokenizer=tok, temperature=0.1,
                        generate_audio=True, use_tts_template=True,
                        max_new_tokens=256)
        chunks, texts, t_first = [], [], None
        for item in res:
            wf, txt = (item if isinstance(item, tuple)
                       else (getattr(item, "audio_wav", None),
                             getattr(item, "text", None)))
            if wf is not None:
                if t_first is None:
                    t_first = time.time() - t0
                chunks.append(wf.float().cpu().numpy().reshape(-1))
            if txt:
                texts.append(txt)
        full = (np.concatenate(chunks) if chunks
                else np.zeros(1, dtype="float32"))
        sf.write(f"{DATA}/tts_smoke_{tag}.wav", full, 24000)
        r[f"{tag}_first_audio_s"] = round(t_first, 2) if t_first else None
        r[f"{tag}_total_s"] = round(time.time() - t0, 2)
        r[f"{tag}_audio_s"] = round(len(full) / 24000, 2)
        r[f"{tag}_text"] = "".join(texts).strip()[:300]
        r[f"{tag}_n_chunks"] = len(chunks)
        return r[f"{tag}_text"]

    gen_audio("local")                       # turn 1: local-answer path

    # turn 2: escalated path — stall prefill, then spoken relay
    model.reset_session(reset_token2wav_cache=False)
    _call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
              tokenizer=tok)
    for i in range(n):
        ch = au[i * 16000:(i + 1) * 16000]
        if len(ch) < 16000:
            ch = np.pad(ch, (0, 16000 - len(ch)))
        _call_def(model.streaming_prefill, session_id="s1",
                  msgs=[{"role": "user", "content": [ch.astype("float32")]}],
                  tokenizer=tok, is_last_chunk=(i == n - 1))
    _call_def(model.streaming_prefill, session_id="s1",
              msgs=[{"role": "assistant", "content": [STALL]}],
              tokenizer=tok, is_last_chunk=True)
    _call_def(model.streaming_prefill, session_id="s1",
              msgs=[{"role": "user", "content": [RELAY_TMPL.format(
                  ans="NVIDIA closed at $178.42 today, up 1.2 percent.")]}],
              tokenizer=tok, is_last_chunk=True)
    gen_audio("relay")

    r["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    gate_data.commit()
    print("SMOKE_RESULT", r, flush=True)
    return r
