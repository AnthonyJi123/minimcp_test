"""Phase 6a — audio-input replication on MiniCPM-o 4.5 (arm A: TTS matched pairs).

Does the mid-layer self-knowledge signal (5d) and p(True) survive when the SAME
frozen 600-query pool enters through the audio channel instead of text?

Design: TTS-render the frozen pool (OpenAI tts-1; query CONTENT stays the
public-benchmark pool — only the modality is synthetic, matching field practice
in Spoken-SQuAD / VoiceBench), then rerun the collection pipeline with audio
input. Outputs use tag `minicpm-o45-audio` in the SAME file formats as the text
pipeline, so modal_app.py's label_hf / layer_sweep_report / xmodel_report work
unchanged on the audio data. Only the cross-modal transfer analysis is new here.

Kept in a separate file so the Phase-5d/5e working tree (modal_app.py) is not
touched. Run everything with cwd=interactive_paper and PYTHONUTF8=1:

  modal run modal_audio.py::tts_pool --limit 6          # TTS smoke
  modal run modal_audio.py::audio_smoke                 # audio-path smoke (3 q)
  modal run modal_audio.py::tts_pool                    # full 600
  modal run modal_audio.py::run_audio_signals           # 4x H100
  modal run modal_app.py::label_hf --tag minicpm-o45-audio
  modal run modal_audio.py::run_audio_ptrue
  modal run modal_app.py::layer_sweep_report --tag minicpm-o45-audio
  modal run modal_audio.py::audio_report                # cross-modal transfer
"""
import os
import sys

import modal

from modal_app import (app, gen_app, image, util_image, GPU_VOL, gate_data,
                       DATA, MODEL_DIR, OPENAI, QUERIES, FEATURES,
                       PTRUE_PRE, API_REGION, _read_jsonl, _load_queries,
                       _load_ptrue_rows_hf, _load_model)

TAG = "minicpm-o45-audio"
AUDIO_DIR = f"{DATA}/audio_pool"
TTS_VOICE = "alloy"            # fixed voice, en+zh; pinned in RESULTS.md

# Modal only auto-mounts the entry module; functions defined HERE import
# modal_app at container runtime, so mount it into their images explicitly.
HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(HERE, "modal_app.py")
image_au = image.add_local_file(_APP_PY, "/root/modal_app.py")
util_image_au = util_image.add_local_file(_APP_PY, "/root/modal_app.py")

# Audio variants of the p(True) prompts: the question arrives as audio, the
# meta-question stays text (mirrors the deployment where the user speaks).
PTRUE_PRE_AU = ("You just heard a question in the audio. Do NOT answer it. "
                "Judge honestly whether you yourself would answer it "
                "correctly.\n\n"
                "Would you answer this question correctly? "
                "Reply with exactly one word: Yes or No.")
PTRUE_POST_AU = ("You just heard a question in the audio. Here is a proposed "
                 "answer.\n\nProposed answer:\n{a}\n\n"
                 "Is the proposed answer correct? "
                 "Reply with exactly one word: Yes or No.")


# ============================ TTS rendering ================================
@gen_app.function(image=util_image_au, volumes={DATA: gate_data},
                  secrets=[OPENAI], timeout=60 * 60, region=API_REGION)
def tts_pool(limit: int = 0, concurrency: int = 8):
    """Render the frozen query pool to wav (idempotent — skips existing)."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    qs = _read_jsonl(QUERIES)
    if limit:
        qs = qs[:limit]
    os.makedirs(AUDIO_DIR, exist_ok=True)
    client = OpenAI()
    n_trunc = 0

    def render(q):
        nonlocal n_trunc
        out = f"{AUDIO_DIR}/{q['id']}.wav"
        if os.path.exists(out):
            return "skip"
        text = q["query"]
        if len(text) > 4000:                      # tts-1 input cap 4096 chars
            text = text[:4000]
            n_trunc += 1
        resp = client.audio.speech.create(model="tts-1", voice=TTS_VOICE,
                                          input=text, response_format="wav")
        with open(out, "wb") as fh:
            fh.write(resp.content)
        return "done"

    with ThreadPoolExecutor(concurrency) as ex:
        results = list(ex.map(render, qs))
    gate_data.commit()
    print(f">>> tts_pool: {results.count('done')} rendered, "
          f"{results.count('skip')} skipped, {n_trunc} truncated at 4000 chars",
          flush=True)


# ============================ model loading ================================
def _load_model_audio(model_dir: str = MODEL_DIR):
    """_load_model with the audio encoder enabled (apff/Whisper branch)."""
    import glob as _glob
    import shutil
    import torch
    from transformers import AutoModel, AutoTokenizer
    cache = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/"
                               + os.path.basename(model_dir))
    os.makedirs(cache, exist_ok=True)
    for f in _glob.glob(f"{model_dir}/*.py"):
        shutil.copy(f, cache)
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=False,
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tok


def _load_wav(qid: str):
    import librosa
    au, _ = librosa.load(f"{AUDIO_DIR}/{qid}.wav", sr=16000, mono=True)
    return au


def _audio_content(au, instr: str):
    return [au, instr] if instr else [au]


# ============================ smoke ========================================
@app.function(image=image_au, gpu="H100", volumes=GPU_VOL, timeout=60 * 30)
def audio_smoke(mtag: str = "minicpm-o45"):
    """Validate the audio input path on 3 queries: (a) does chat accept a raw
    16k numpy array, (b) does the model ANSWER (vs transcribe) with/without a
    text instruction, (c) do the all-layer hooks fire on audio prefill."""
    sys.path.insert(0, "/workspace/gate")
    import decode
    import layers as layers_mod

    qs = [q for q in _read_jsonl(QUERIES)
          if os.path.exists(f"{AUDIO_DIR}/{q['id']}.wav")][:3]
    if not qs:
        raise RuntimeError("no wavs on volume — run tts_pool --limit 6 first")

    model, tok = _load_model_audio(_mtag_dir(mtag))
    kw = decode._chat_kwargs(model, tok)
    import inspect
    print(">>> chat params:", sorted(inspect.signature(model.chat).parameters),
          flush=True)

    for q in qs:
        au = _load_wav(q["id"])
        print(f"\n=== {q['id']} ({q['pool']}) dur={len(au)/16000:.1f}s ===",
              flush=True)
        print(f"  Q(text): {q['query'][:120]!r}", flush=True)
        for name, instr in (("audio-only", ""),
                            ("audio+instr", "Answer the question you just heard.")):
            try:
                out = model.chat(
                    msgs=[{"role": "user", "content": _audio_content(au, instr)}],
                    max_new_tokens=128, **kw)
                print(f"  [{name}] {str(out)[:200]!r}", flush=True)
            except Exception as e:
                print(f"  [{name}] FAILED: {type(e).__name__}: {e}", flush=True)

    # hook check: all 36 decoder layers must fire during audio prefill
    au = _load_wav(qs[0]["id"])
    hl, hm = layers_mod.capture_prefill_layers(
        list(model.llm.model.layers),
        lambda: model.chat(msgs=[{"role": "user", "content": [au]}],
                           max_new_tokens=1, **kw))
    print(f"\n>>> layer capture OK: h_last {tuple(hl.shape)} "
          f"h_mean {tuple(hm.shape)}", flush=True)
    print(">>> audio_smoke OK", flush=True)


# ============================ collection ===================================
def _mtag_dir(mtag: str) -> str:
    return MODEL_DIR if mtag == "minicpm-o45" else f"/workspace/models/{mtag}"


@app.function(image=image_au, gpu="H100", volumes=GPU_VOL, timeout=60 * 60 * 2)
def collect_audio(shard: list, shard_id: int, instr: str,
                  mtag: str = "minicpm-o45") -> int:
    """One chat call per query with BOTH hook families: decode._Collector
    (entropy/margin/h_prompt/h_mean8 -> signals parquet, same schema as
    collect_signals_mo) and capture_prefill_layers (all-layer h_last/h_mean ->
    layers npz, same schema as collect_layers_mo)."""
    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0, "/workspace/gate")
    import decode
    import layers as layers_mod

    dtag = f"{mtag}-audio"
    model, tok = _load_model_audio(_mtag_dir(mtag))
    kw = decode._chat_kwargs(model, tok)
    mods = list(model.llm.model.layers)
    print(f">>> {dtag} shard {shard_id}: {len(shard)} queries", flush=True)

    sig_rows, ids, lasts, means = [], [], [], []
    for k, q in enumerate(shard):
        au = _load_wav(q["id"])
        coll = decode._Collector(16)
        handles = decode._register(model, coll)
        res = {}
        try:
            hl, hm = layers_mod.capture_prefill_layers(
                mods,
                lambda: res.update(text=model.chat(
                    msgs=[{"role": "user", "content": _audio_content(au, instr)}],
                    max_new_tokens=512, **kw)))
        finally:
            for h in handles:
                h.remove()
        text = res["text"]
        text = text.strip() if isinstance(text, str) else str(text)
        h_prompt = coll.hiddens[0] if coll.hiddens else None
        steps = coll.hiddens[1:9]
        h_mean8 = torch.stack(steps).mean(0) if steps else h_prompt
        sig_rows.append({**{f: q[f] for f in
                            ("id", "pool", "source", "query", "reference_answer",
                             "split")},
                         "answer": text, "n_forward": len(coll.hiddens),
                         "entropy": coll.entropy, "margin": coll.margin,
                         "h_prompt": h_prompt.tolist() if h_prompt is not None else None,
                         "h_mean8": h_mean8.tolist() if h_mean8 is not None else None})
        ids.append(q["id"])
        lasts.append(hl.numpy().astype(np.float16))
        means.append(hm.numpy().astype(np.float16))
        if k < 2 or k % 50 == 0:
            print(f"  [{k}] {q['pool']} :: {text[:60]!r}", flush=True)

    pd.DataFrame(sig_rows).to_parquet(f"{DATA}/signals_{dtag}.shard{shard_id}.parquet")
    np.savez_compressed(f"{DATA}/layers_{dtag}.shard{shard_id}.npz",
                        ids=np.array(ids), h_last=np.stack(lasts),
                        h_mean=np.stack(means))
    gate_data.commit()
    print(f">>> wrote signals+layers shard {shard_id} ({len(ids)} rows)",
          flush=True)
    return len(ids)


@app.local_entrypoint()
def run_audio_signals(workers: int = 4, limit: int = 0, instr: str = "",
                      mtag: str = "minicpm-o45"):
    qs = _load_queries.remote()
    if limit:
        qs = qs[:limit]
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {mtag}-audio: {len(qs)} queries / {workers} H100 workers")
    total = sum(collect_audio.starmap(
        [(shards[i], i, instr, mtag) for i in range(workers)]))
    print(f">>> collected audio signals for {total} queries")


# ============================ p(True), audio ===============================
@app.function(image=image_au, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_ptrue_audio(shard: list, shard_id: int,
                        mtag: str = "minicpm-o45") -> int:
    """p(True) with the question as audio: first-token P(Yes) via lm_head hook
    (decode.first_token_logits generalized to multimodal content)."""
    import pandas as pd
    import torch
    sys.path.insert(0, "/workspace/gate")
    import decode

    dtag = f"{mtag}-audio"
    model, tok = _load_model_audio(_mtag_dir(mtag))
    kw = decode._chat_kwargs(model, tok)

    def tok_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    YES = tok_ids(["Yes", "yes", "YES", " Yes", " yes", "是", "能", "对", "会"])
    NO = tok_ids(["No", "no", "NO", " No", " no", "否", "不", "错"])
    print(f">>> {dtag} shard {shard_id}: {len(shard)} | |YES|={len(YES)} "
          f"|NO|={len(NO)}", flush=True)

    def p_yes(content):
        store = {}

        def head_hook(_m, _inp, out):
            if "logits" not in store:
                store["logits"] = out[0, -1, :].detach().float().cpu()

        h = model.llm.lm_head.register_forward_hook(head_hook)
        try:
            model.chat(msgs=[{"role": "user", "content": content}],
                       max_new_tokens=1, **kw)
        finally:
            h.remove()
        logits = store["logits"]
        p = torch.softmax(logits, dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        arg = tok.decode([int(logits.argmax())])
        return py / (py + pn) if (py + pn) > 0 else 0.5, py + pn, arg

    rows = []
    for k, q in enumerate(shard):
        au = _load_wav(q["id"])
        pre, pre_mass, pre_arg = p_yes([au, PTRUE_PRE_AU])
        post, post_mass, post_arg = p_yes(
            [au, PTRUE_POST_AU.format(a=q["answer"][:4000])])
        rows.append({"id": q["id"], "p_yes_pre": pre, "p_yes_post": post,
                     "mass_pre": pre_mass, "mass_post": post_mass})
        if k < 3 or k % 50 == 0:
            print(f"  [{k}] pre={pre:.3f} (mass {pre_mass:.2f}, {pre_arg!r}) "
                  f"post={post:.3f} (mass {post_mass:.2f}, {post_arg!r})",
                  flush=True)
    pd.DataFrame(rows).to_parquet(f"{DATA}/ptrue_{dtag}.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote ptrue shard {shard_id}", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_audio_ptrue(workers: int = 4, mtag: str = "minicpm-o45"):
    qs = _load_ptrue_rows_hf.remote(f"{mtag}-audio")   # audio-conditioned answers
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {mtag}-audio: {len(qs)} queries / {workers} workers")
    total = sum(collect_ptrue_audio.starmap(
        [(shards[i], i, mtag) for i in range(workers)]))
    print(f">>> collected audio p(True) for {total} queries")


# ============================ TTS-template control =========================
TTS_TAG = "minicpm-o45-ttstpl"


@app.function(image=image_au, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_layers_ttstpl(shard: list, shard_id: int) -> int:
    """Finding-1 mechanism probe: TEXT input under the speak-mode template
    (use_tts_template=True, generate_audio=False). Does the 5d late-layer
    cliff track the MODE (speak-prep) or the text modality itself?
    Prefill-only all-layer capture, same npz format as collect_layers_mo."""
    import numpy as np
    sys.path.insert(0, "/workspace/gate")
    import decode
    import layers as layers_mod

    model, tok = _load_model()
    kw = decode._chat_kwargs(model, tok)
    kw["use_tts_template"] = True          # smoke confirmed chat accepts it
    mods = list(model.llm.model.layers)
    print(f">>> {TTS_TAG} shard {shard_id}: {len(shard)} queries", flush=True)

    ids, lasts, means = [], [], []
    for k, q in enumerate(shard):
        hl, hm = layers_mod.capture_prefill_layers(
            mods,
            lambda: model.chat(msgs=[{"role": "user", "content": [q["query"]]}],
                               max_new_tokens=1, **kw))
        ids.append(q["id"])
        lasts.append(hl.numpy().astype(np.float16))
        means.append(hm.numpy().astype(np.float16))
        if k < 2 or k % 50 == 0:
            print(f"  [{k}] {q['pool']} layers={hl.shape[0]}", flush=True)
    np.savez_compressed(f"{DATA}/layers_{TTS_TAG}.shard{shard_id}.npz",
                        ids=np.array(ids), h_last=np.stack(lasts),
                        h_mean=np.stack(means))
    gate_data.commit()
    print(f">>> wrote layers shard {shard_id} ({len(ids)} rows)", flush=True)
    return len(ids)


@app.function(image=util_image_au, volumes={DATA: gate_data}, timeout=60 * 5)
def ttstpl_link_labels():
    """Labels are unchanged (same model, same text questions — only the
    template differs), so reuse the text-run features for layer_sweep_report."""
    import shutil
    shutil.copy(FEATURES, f"{DATA}/features_{TTS_TAG}.parquet")
    gate_data.commit()
    print(f">>> linked {FEATURES} -> features_{TTS_TAG}.parquet", flush=True)


@app.local_entrypoint()
def run_layers_ttstpl(workers: int = 4, limit: int = 0):
    qs = _load_queries.remote()
    if limit:
        qs = qs[:limit]
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {TTS_TAG}: {len(qs)} queries / {workers} H100 workers")
    total = sum(collect_layers_ttstpl.starmap(
        [(shards[i], i) for i in range(workers)]))
    print(f">>> layer-swept {total} queries under TTS template")


# ============================ qwen2.5-omni audio path ======================
# Cross-family control (user: o2.6 is same-family, weak generalization).
# The Thinker class carries the audio tower; input goes through the
# Qwen2_5OmniProcessor, not model.chat. Fresh image: omni stack + librosa
# (modal_app's omni_image can't be extended — add_local_dir must stay last).
OMNI_TAG = "qwen2.5-omni-7b"
omni_image_au = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch==2.8.0", "torchvision==0.23.0")
    .pip_install("transformers==4.57.6", "accelerate==1.12.0", "pandas",
                 "pyarrow", "huggingface_hub[hf_transfer]", "librosa",
                 "soundfile", "pillow")
    .add_local_dir(os.path.join(HERE, "src"), "/workspace/gate")
    .add_local_file(_APP_PY, "/root/modal_app.py")
)


def _load_omni_audio():
    import torch
    from transformers import (Qwen2_5OmniThinkerForConditionalGeneration,
                              Qwen2_5OmniProcessor)
    model_dir = f"/workspace/models/{OMNI_TAG}"
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").eval().cuda()
    proc = Qwen2_5OmniProcessor.from_pretrained(model_dir)
    return model, proc


def _omni_inputs(proc, au, extra_text: str = ""):
    """Processor inputs for one user turn: audio (+ optional trailing text)."""
    content = [{"type": "audio", "audio": au}]
    if extra_text:
        content.append({"type": "text", "text": extra_text})
    text = proc.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
    try:
        inputs = proc(text=text, audio=[au], sampling_rate=16000,
                      return_tensors="pt")
    except TypeError:
        inputs = proc(text=text, audios=[au], sampling_rate=16000,
                      return_tensors="pt")
    return {k: (v.cuda() if hasattr(v, "cuda") else v)
            for k, v in inputs.items()}


@app.function(image=omni_image_au, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 30)
def omni_audio_smoke():
    """3 queries through the omni audio path: answer sanity + hook shapes."""
    import torch
    sys.path.insert(0, "/workspace/gate")
    import layers as layers_mod

    qs = [q for q in _read_jsonl(QUERIES)
          if os.path.exists(f"{AUDIO_DIR}/{q['id']}.wav")][:3]
    model, proc = _load_omni_audio()
    for q in qs:
        au = _load_wav(q["id"])
        inputs = _omni_inputs(proc, au)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0]
        print(f"\n=== {q['id']} ({q['pool']}) ===", flush=True)
        print(f"  Q(text): {q['query'][:100]!r}", flush=True)
        print(f"  A(audio): {ans[:200]!r}", flush=True)

    au = _load_wav(qs[0]["id"])
    inputs = _omni_inputs(proc, au)
    hl, hm = layers_mod.capture_prefill_layers(
        list(model.model.layers), lambda: model(**inputs))
    print(f"\n>>> layer capture OK: h_last {tuple(hl.shape)}", flush=True)
    print(">>> omni_audio_smoke OK", flush=True)


@app.function(image=omni_image_au, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 2)
def collect_audio_omni(shard: list, shard_id: int) -> int:
    """Audio-input collection for qwen2.5-omni: answer + all-layer prefill
    hiddens in one generate call. Signals schema matches label_hf's needs."""
    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0, "/workspace/gate")
    import layers as layers_mod

    dtag = f"{OMNI_TAG}-audio"
    model, proc = _load_omni_audio()
    mods = list(model.model.layers)
    print(f">>> {dtag} shard {shard_id}: {len(shard)} queries", flush=True)

    sig_rows, ids, lasts, means = [], [], [], []
    for k, q in enumerate(shard):
        au = _load_wav(q["id"])
        inputs = _omni_inputs(proc, au)
        res = {}
        hl, hm = layers_mod.capture_prefill_layers(
            mods,
            lambda: res.update(out=model.generate(
                **inputs, max_new_tokens=512, do_sample=False)))
        ans = proc.batch_decode(res["out"][:, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0].strip()
        sig_rows.append({**{f: q[f] for f in
                            ("id", "pool", "source", "query", "reference_answer",
                             "split")},
                         "answer": ans, "n_forward": 0,
                         "entropy": [], "margin": [],
                         "h_prompt": hl[-1].tolist(), "h_mean8": None})
        ids.append(q["id"])
        lasts.append(hl.numpy().astype(np.float16))
        means.append(hm.numpy().astype(np.float16))
        if k < 2 or k % 50 == 0:
            print(f"  [{k}] {q['pool']} :: {ans[:60]!r}", flush=True)

    pd.DataFrame(sig_rows).to_parquet(
        f"{DATA}/signals_{dtag}.shard{shard_id}.parquet")
    np.savez_compressed(f"{DATA}/layers_{dtag}.shard{shard_id}.npz",
                        ids=np.array(ids), h_last=np.stack(lasts),
                        h_mean=np.stack(means))
    gate_data.commit()
    print(f">>> wrote signals+layers shard {shard_id} ({len(ids)} rows)",
          flush=True)
    return len(ids)


@app.local_entrypoint()
def run_audio_omni(workers: int = 4, limit: int = 0):
    qs = _load_queries.remote()
    if limit:
        qs = qs[:limit]
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {OMNI_TAG}-audio: {len(qs)} queries / {workers} H100 workers")
    total = sum(collect_audio_omni.starmap(
        [(shards[i], i) for i in range(workers)]))
    print(f">>> collected omni audio signals for {total} queries")


@app.function(image=omni_image_au, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60)
def collect_ptrue_audio_omni(shard: list, shard_id: int) -> int:
    """Audio-question p(True) for qwen2.5-omni via a single prefill forward."""
    import pandas as pd
    import torch

    dtag = f"{OMNI_TAG}-audio"
    model, proc = _load_omni_audio()
    tok = proc.tokenizer

    def tok_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    YES = tok_ids(["Yes", "yes", "YES", " Yes", " yes", "是", "能", "对", "会"])
    NO = tok_ids(["No", "no", "NO", " No", " no", "否", "不", "错"])
    print(f">>> {dtag} shard {shard_id}: {len(shard)} | |YES|={len(YES)} "
          f"|NO|={len(NO)}", flush=True)

    def p_yes(au, meta_text):
        inputs = _omni_inputs(proc, au, meta_text)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :].float().cpu()
        p = torch.softmax(logits, dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        arg = tok.decode([int(logits.argmax())])
        return py / (py + pn) if (py + pn) > 0 else 0.5, py + pn, arg

    rows = []
    for k, q in enumerate(shard):
        au = _load_wav(q["id"])
        pre, pre_mass, pre_arg = p_yes(au, PTRUE_PRE_AU)
        post, post_mass, post_arg = p_yes(
            au, PTRUE_POST_AU.format(a=q["answer"][:4000]))
        rows.append({"id": q["id"], "p_yes_pre": pre, "p_yes_post": post,
                     "mass_pre": pre_mass, "mass_post": post_mass})
        if k < 3 or k % 50 == 0:
            print(f"  [{k}] pre={pre:.3f} (mass {pre_mass:.2f}, {pre_arg!r}) "
                  f"post={post:.3f} (mass {post_mass:.2f}, {post_arg!r})",
                  flush=True)
    pd.DataFrame(rows).to_parquet(f"{DATA}/ptrue_{dtag}.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote ptrue shard {shard_id}", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_ptrue_omni_audio(workers: int = 4):
    qs = _load_ptrue_rows_hf.remote(f"{OMNI_TAG}-audio")
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {OMNI_TAG}-audio ptrue: {len(qs)} queries / {workers} workers")
    total = sum(collect_ptrue_audio_omni.starmap(
        [(shards[i], i) for i in range(workers)]))
    print(f">>> collected omni audio p(True) for {total} queries")


# ============================ ASR audit (6a follow-up) =====================
ASR_INSTR = ("Transcribe the speech in the audio verbatim. Output ONLY the "
             "transcription. Do not answer any question it contains.")


@app.function(image=image_au, gpu="H100", volumes=GPU_VOL, timeout=60 * 60 * 2)
def collect_asr(shard: list, shard_id: int) -> int:
    """Perception-vs-introspection disambiguation for the trap collapse:
    (1) let the model transcribe each wav (what did it actually hear?), then
    (2) run TEXT ptrue_pre on its OWN transcript (does knows-it-doesn't-know
    come back once the same content is presented as text?)."""
    import pandas as pd
    import torch
    sys.path.insert(0, "/workspace/gate")
    import decode

    model, tok = _load_model_audio()
    kw = decode._chat_kwargs(model, tok)

    def tok_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    YES = tok_ids(["Yes", "yes", "YES", " Yes", " yes", "是", "能", "对", "会"])
    NO = tok_ids(["No", "no", "NO", " No", " no", "否", "不", "错"])
    print(f">>> asr shard {shard_id}: {len(shard)} queries", flush=True)

    rows = []
    for k, q in enumerate(shard):
        au = _load_wav(q["id"])
        asr = model.chat(msgs=[{"role": "user", "content": [au, ASR_INSTR]}],
                         max_new_tokens=512, **kw)
        asr = asr.strip() if isinstance(asr, str) else str(asr)
        logits = decode.first_token_logits(model, tok,
                                           PTRUE_PRE.format(q=asr))
        p = torch.softmax(logits, dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        p_tr = py / (py + pn) if (py + pn) > 0 else 0.5
        rows.append({"id": q["id"], "transcript": asr,
                     "p_yes_transcript": p_tr, "mass": py + pn})
        if k < 3 or k % 50 == 0:
            print(f"  [{k}] p_tr={p_tr:.3f} :: {asr[:70]!r}", flush=True)
    pd.DataFrame(rows).to_parquet(f"{DATA}/asr_{TAG}.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote asr shard {shard_id}", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_asr(workers: int = 4):
    qs = _load_queries.remote()
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> asr audit: {len(qs)} queries / {workers} workers")
    total = sum(collect_asr.starmap([(shards[i], i) for i in range(workers)]))
    print(f">>> transcribed {total} queries")


@app.function(image=util_image_au, volumes={DATA: gate_data}, timeout=60 * 30)
def asr_report():
    """Three-arm comparison per pool: p_yes(text original) vs
    p_yes(own transcript as text) vs p_yes(audio), conditioned on WER."""
    import glob
    import re
    import difflib
    import numpy as np
    import pandas as pd

    def norm_tokens(s: str):
        s = re.sub(r"[^\w一-鿿]+", " ", str(s).lower()).strip()
        if re.search(r"[一-鿿]", s):
            return list(s.replace(" ", ""))          # CJK: char-level
        return s.split()

    def wer(ref: str, hyp: str) -> float:
        r, h = norm_tokens(ref), norm_tokens(hyp)
        if not r:
            return 0.0
        sm = difflib.SequenceMatcher(a=r, b=h)
        matched = sum(bl.size for bl in sm.get_matching_blocks())
        return 1.0 - matched / len(r)

    asr = pd.concat([pd.read_parquet(s) for s in
                     sorted(glob.glob(f"{DATA}/asr_{TAG}.shard*.parquet"))],
                    ignore_index=True)
    au = pd.concat([pd.read_parquet(s) for s in
                    sorted(glob.glob(f"{DATA}/ptrue_{TAG}.shard*.parquet"))],
                   ignore_index=True)[["id", "p_yes_pre"]] \
        .rename(columns={"p_yes_pre": "p_yes_audio"})
    tx_files = sorted(glob.glob(f"{DATA}/ptrue.shard*.parquet"))
    feats = pd.read_parquet(f"{DATA}/features_{TAG}.parquet")[
        ["id", "pool", "query", "escalate_label"]]
    df = feats.merge(asr, on="id").merge(au, on="id")
    if tx_files:
        tx = pd.concat([pd.read_parquet(s) for s in tx_files],
                       ignore_index=True)[["id", "p_yes_pre"]] \
            .rename(columns={"p_yes_pre": "p_yes_text"})
        df = df.merge(tx, on="id", how="left")
    else:
        df["p_yes_text"] = np.nan
        print("!! no text ptrue shards found — text arm from RESULTS.md only",
              flush=True)
    df["wer"] = [wer(r["query"], r["transcript"]) for _, r in df.iterrows()]

    print(f"=== ASR audit (n={len(df)}) ===", flush=True)
    for pool, g in df.groupby("pool"):
        print(f"  {pool:16s} n={len(g):4d} WER mean={g['wer'].mean():.3f} "
              f"median={g['wer'].median():.3f} "
              f"p_yes text={g['p_yes_text'].mean():.3f} "
              f"transcript={g['p_yes_transcript'].mean():.3f} "
              f"audio={g['p_yes_audio'].mean():.3f}", flush=True)

    trap = df[df["pool"] == "trap"]
    lo, hi = trap[trap["wer"] <= 0.15], trap[trap["wer"] > 0.15]
    print(f"\n=== trap three-arm, split by transcription quality ===",
          flush=True)
    for name, g in (("heard ok (WER<=.15)", lo), ("misheard (WER>.15)", hi)):
        if len(g):
            print(f"  {name:22s} n={len(g):3d} "
                  f"p_yes text={g['p_yes_text'].mean():.3f} "
                  f"transcript={g['p_yes_transcript'].mean():.3f} "
                  f"audio={g['p_yes_audio'].mean():.3f}", flush=True)
    c = df[["wer", "p_yes_audio"]].corr().iloc[0, 1]
    print(f"\n  corr(WER, p_yes_audio) all pools = {c:+.3f}", flush=True)
    df.to_parquet(f"{DATA}/asr_audit_{TAG}.parquet")
    gate_data.commit()
    print(f">>> wrote {DATA}/asr_audit_{TAG}.parquet", flush=True)


# ============================ mechanism arms (6a follow-up 2) ==============
# Arm ctx: irrelevant filler audio in context + the question as TEXT — does
#   the mere presence of audio shift the verbal self-assessment?
# Arm dup: the question audio + the SAME question duplicated as text — does
#   making the instance available in text tokens restore the judgment?
FILLER_WAV = f"{AUDIO_DIR}/_filler.wav"
FILLER_TEXT = ("The weather was mild yesterday, and the streets were quiet "
               "in the early evening.")


@gen_app.function(image=util_image_au, volumes={DATA: gate_data},
                  secrets=[OPENAI], timeout=60 * 5, region=API_REGION)
def tts_filler():
    from openai import OpenAI
    os.makedirs(AUDIO_DIR, exist_ok=True)
    resp = OpenAI().audio.speech.create(model="tts-1", voice=TTS_VOICE,
                                        input=FILLER_TEXT,
                                        response_format="wav")
    with open(FILLER_WAV, "wb") as fh:
        fh.write(resp.content)
    gate_data.commit()
    print(">>> wrote filler wav", flush=True)


@app.function(image=image_au, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_ptrue_arms(shard: list, shard_id: int) -> int:
    import librosa
    import pandas as pd
    import torch
    sys.path.insert(0, "/workspace/gate")
    import decode

    model, tok = _load_model_audio()
    kw = decode._chat_kwargs(model, tok)
    filler, _ = librosa.load(FILLER_WAV, sr=16000, mono=True)

    def tok_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    YES = tok_ids(["Yes", "yes", "YES", " Yes", " yes", "是", "能", "对", "会"])
    NO = tok_ids(["No", "no", "NO", " No", " no", "否", "不", "错"])
    print(f">>> arms shard {shard_id}: {len(shard)} queries", flush=True)

    def p_yes(content):
        store = {}

        def head_hook(_m, _inp, out):
            if "logits" not in store:
                store["logits"] = out[0, -1, :].detach().float().cpu()

        h = model.llm.lm_head.register_forward_hook(head_hook)
        try:
            model.chat(msgs=[{"role": "user", "content": content}],
                       max_new_tokens=1, **kw)
        finally:
            h.remove()
        p = torch.softmax(store["logits"], dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        return py / (py + pn) if (py + pn) > 0 else 0.5

    rows = []
    for k, q in enumerate(shard):
        prompt = PTRUE_PRE.format(q=q["query"])
        ctx = p_yes([filler, prompt])
        dup = p_yes([_load_wav(q["id"]), prompt])
        rows.append({"id": q["id"], "p_yes_ctx": ctx, "p_yes_dup": dup})
        if k < 3 or k % 50 == 0:
            print(f"  [{k}] ctx={ctx:.3f} dup={dup:.3f}", flush=True)
    pd.DataFrame(rows).to_parquet(
        f"{DATA}/ptrue_arms_{TAG}.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote arms shard {shard_id}", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_ptrue_arms(workers: int = 4):
    qs = _load_queries.remote()
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> arms: {len(qs)} queries / {workers} workers")
    total = sum(collect_ptrue_arms.starmap(
        [(shards[i], i) for i in range(workers)]))
    print(f">>> collected arms for {total} queries")


@app.function(image=util_image_au, volumes={DATA: gate_data}, timeout=60 * 10)
def arms_report():
    """Four-way table per pool: text vs filler-audio+text (ctx) vs
    audio+text-dup (dup) vs audio-only, from the paired files."""
    import glob
    import numpy as np
    import pandas as pd

    arms = pd.concat([pd.read_parquet(s) for s in
                      sorted(glob.glob(f"{DATA}/ptrue_arms_{TAG}.shard*.parquet"))],
                     ignore_index=True)
    base = pd.read_parquet(f"{DATA}/asr_audit_{TAG}.parquet")[
        ["id", "pool", "p_yes_text", "p_yes_audio"]]
    df = base.merge(arms, on="id")

    def lo(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    print(f"=== mechanism arms (n={len(df)}) ===", flush=True)
    print("  pool              p_yes: text   ctx   dup   audio", flush=True)
    for pool, g in df.groupby("pool"):
        print(f"  {pool:16s} n={len(g):4d}  {g['p_yes_text'].mean():.3f} "
              f"{g['p_yes_ctx'].mean():.3f} {g['p_yes_dup'].mean():.3f} "
              f"{g['p_yes_audio'].mean():.3f}", flush=True)
    trap = df[df["pool"] == "trap"]
    print(f"\n  trap dlo medians vs text: "
          f"ctx {np.median(lo(trap['p_yes_ctx']) - lo(trap['p_yes_text'])):+.2f} "
          f"dup {np.median(lo(trap['p_yes_dup']) - lo(trap['p_yes_text'])):+.2f} "
          f"audio {np.median(lo(trap['p_yes_audio']) - lo(trap['p_yes_text'])):+.2f}",
          flush=True)
    df.to_parquet(f"{DATA}/ptrue_arms_{TAG}.parquet")
    gate_data.commit()
    print(">>> arms_report done", flush=True)


# ============================ gpt-5.5 re-judge =============================
@gen_app.function(image=util_image_au, volumes={DATA: gate_data},
                  secrets=[OPENAI], timeout=60 * 60 * 2, region=API_REGION)
def rejudge(tag: str, concurrency: int = 8):
    """Re-judge a features parquet with gpt-5.5 (JUDGE_MODEL monkeypatched);
    report agreement with the gpt-5.4-mini labels per pool."""
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate
    escalate.JUDGE_MODEL = "gpt-5.5"
    escalate.JUDGE_MAX_TOKENS = 8192

    path = FEATURES if tag == "minicpm-o45" else f"{DATA}/features_{tag}.parquet"
    df = pd.read_parquet(path)
    print(f">>> rejudge {tag}: {len(df)} rows with gpt-5.5...", flush=True)
    rows = [{"query": r["query"],
             "reference_answer": (None if pd.isna(r["reference_answer"])
                                  else r["reference_answer"]),
             "answer": r["answer"]} for _, r in df.iterrows()]
    labeled = asyncio.run(escalate.judge_many(rows, concurrency=concurrency))
    df["adequate55"] = [x["adequate"] for x in labeled]
    df["judge_reason55"] = [x["judge_reason"] for x in labeled]
    df["escalate_label55"] = [x["escalate_label"] for x in labeled]
    df.to_parquet(f"{DATA}/features_gpt55_{tag}.parquet")
    gate_data.commit()

    ok = df[df["escalate_label"].notna() & df["escalate_label55"].notna()]
    agree = (ok["escalate_label"] == ok["escalate_label55"]).mean()
    n_err = int(df["adequate55"].isna().sum())
    print(f">>> {tag}: agreement {agree:.3f} on {len(ok)} rows "
          f"({n_err} judge errors)", flush=True)
    for pool, g in ok.groupby("pool"):
        a = (g["escalate_label"] == g["escalate_label55"]).mean()
        flips01 = int(((g["escalate_label"] == 0)
                       & (g["escalate_label55"] == 1)).sum())
        flips10 = int(((g["escalate_label"] == 1)
                       & (g["escalate_label55"] == 0)).sum())
        print(f"  {pool:16s} n={len(g):4d} agree={a:.3f} "
              f"esc mini={g['escalate_label'].mean():.3f} "
              f"5.5={g['escalate_label55'].mean():.3f} "
              f"flips 0->1={flips01} 1->0={flips10}", flush=True)


@app.function(image=util_image_au, volumes={DATA: gate_data}, timeout=60 * 20,
              cpu=8)
def rescore55():
    """Do the headline numbers move under gpt-5.5 labels? Re-score audio
    ptrue pre/post AUC and the L22 audio probe OOF under both label sets."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    f55 = pd.read_parquet(f"{DATA}/features_gpt55_{TAG}.parquet")[
        ["id", "pool", "split", "escalate_label", "escalate_label55"]]
    pt = pd.concat([pd.read_parquet(s) for s in
                    sorted(glob.glob(f"{DATA}/ptrue_{TAG}.shard*.parquet"))],
                   ignore_index=True).merge(f55, on="id")
    for lab in ("escalate_label", "escalate_label55"):
        ok = pt[pt[lab].notna()]
        y = ok[lab].astype(int)
        print(f"  audio ptrue [{lab}]: "
              f"pre={roc_auc_score(y, -ok['p_yes_pre']):.3f} "
              f"post={roc_auc_score(y, -ok['p_yes_post']):.3f}", flush=True)

    ids, hl = [], []
    for s in sorted(glob.glob(f"{DATA}/layers_{TAG}.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    hl = np.concatenate(hl)
    m = pd.DataFrame({"id": ids, "row": range(len(ids))}).merge(f55, on="id")
    m = m[m["split"] == "calib"]
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for lab in ("escalate_label", "escalate_label55"):
        mm = m[m[lab].notna()]
        X = hl[mm["row"].to_numpy(), 22, :].astype(np.float32)
        y = mm[lab].astype(int).to_numpy()
        oof = cross_val_predict(LogisticRegression(max_iter=2000), X, y,
                                cv=cv, method="predict_proba")[:, 1]
        print(f"  audio L22 probe OOF [{lab}]: "
              f"{roc_auc_score(y, oof):.3f}", flush=True)
    print(">>> rescore55 done", flush=True)


# ============================ p(True) shift decomposition ==================
@app.function(image=util_image_au, volumes={DATA: gate_data}, timeout=60 * 10)
def ptrue_shift_report():
    """Is the audio p(True) collapse a GLOBAL prior shift (uniform log-odds
    bump from audio context) or trap-specific? Per-query paired
    delta = logit(p_yes_audio) - logit(p_yes_text), by pool."""
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(f"{DATA}/asr_audit_{TAG}.parquet")
    df = df[df["p_yes_text"].notna()]

    def lo(p):
        p = np.clip(p.astype(float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    df["dlo"] = lo(df["p_yes_audio"]) - lo(df["p_yes_text"])
    g_med = df["dlo"].median()
    print(f"=== paired log-odds shift audio-vs-text (n={len(df)}) ===",
          flush=True)
    print(f"  global median dlo = {g_med:+.2f}", flush=True)
    for pool, g in df.groupby("pool"):
        print(f"  {pool:16s} n={len(g):4d} dlo median={g['dlo'].median():+.2f} "
              f"mean={g['dlo'].mean():+.2f} IQR=[{g['dlo'].quantile(.25):+.2f},"
              f"{g['dlo'].quantile(.75):+.2f}]", flush=True)
    trap = df[df["pool"] == "trap"]
    excess = trap["dlo"].median() - g_med
    print(f"\n  trap excess over global median = {excess:+.2f} log-odds "
          f"(0 => pure global prior would explain trap)", flush=True)
    n_flip = int(((trap["p_yes_text"] < .5) & (trap["p_yes_audio"] > .5)).sum())
    print(f"  trap queries flipping No->Yes side: {n_flip}/{len(trap)}",
          flush=True)


# ============================ Phase-6 streaming smoke ======================
@app.function(image=image_au, gpu="H100", volumes=GPU_VOL, timeout=60 * 30)
def streaming_smoke():
    """Phase-6 feasibility: does the duplex streaming loop run headless (no
    demo framework)? Stages, each independently try/except'd:
      (1) do streaming methods exist on the model object, with what signatures
      (2) chunked 1s prefill of a real TTS wav — does the model 'listen'
      (3) end-of-turn generate — does it answer the heard question
      (4) mid-layer hidden capture at a chunk boundary (the gate's read point)
      (5) a follow-up TEXT turn in the same session (the <result> relay analog)
    """
    import inspect
    import traceback
    import numpy as np
    sys.path.insert(0, "/workspace/gate")

    model, tok = _load_model_audio()

    print("\n=== stage 1: streaming API surface ===", flush=True)
    api = {}
    for name in ("streaming_prefill", "streaming_generate", "get_sys_prompt",
                 "reset_session", "init_tts", "listen", "duplex_generate"):
        fn = getattr(model, name, None)
        api[name] = fn
        if fn is None:
            print(f"  {name}: MISSING", flush=True)
        elif callable(fn):
            try:
                print(f"  {name}{inspect.signature(fn)}", flush=True)
            except (TypeError, ValueError):
                print(f"  {name}: callable (signature unavailable)", flush=True)
        else:
            print(f"  {name}: attr of type {type(fn).__name__}", flush=True)
    if not callable(api.get("streaming_prefill")):
        print("!! no streaming_prefill — headless duplex NOT available via "
              "remote code; stopping here", flush=True)
        return

    def call_defensive(fn, /, **kw):
        params = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in params})

    q = next(q for q in _read_jsonl(QUERIES)
             if os.path.exists(f"{AUDIO_DIR}/{q['id']}.wav"))
    au = _load_wav(q["id"])
    chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
    print(f"\n=== using {q['id']} ({q['pool']}), {len(au)/16000:.1f}s -> "
          f"{len(chunks)} x 1s chunks ===", flush=True)
    print(f"  Q(text): {q['query'][:100]!r}", flush=True)

    try:
        if callable(api.get("reset_session")):
            model.reset_session()
        sys_msg = None
        if callable(api.get("get_sys_prompt")):
            sys_msg = call_defensive(api["get_sys_prompt"],
                                     mode="omni", language="en")
            print(f"\n=== stage 2a: sys prompt ok (keys: "
                  f"{list(sys_msg)[:5] if isinstance(sys_msg, dict) else type(sys_msg).__name__}) ===",
                  flush=True)
            call_defensive(api["streaming_prefill"], session_id="s1",
                           msgs=[sys_msg], tokenizer=tok)
        print("=== stage 2b: feeding chunks ===", flush=True)
        for i, ch in enumerate(chunks):
            if len(ch) < 16000:        # pad tail chunk — apm conv needs full frames
                ch = np.pad(ch, (0, 16000 - len(ch)))
            msg = {"role": "user", "content": [ch.astype(np.float32)]}
            call_defensive(api["streaming_prefill"], session_id="s1",
                           msgs=[msg], tokenizer=tok,
                           is_last_chunk=(i == len(chunks) - 1))
            if i < 2 or i == len(chunks) - 1:
                print(f"  chunk {i}: prefilled ok", flush=True)
    except Exception:
        print("!! stage 2 FAILED:", flush=True)
        traceback.print_exc()
        return

    print("\n=== stage 3: end-of-turn generate ===", flush=True)
    try:
        res = call_defensive(api["streaming_generate"], session_id="s1",
                             tokenizer=tok, temperature=0.1,
                             generate_audio=False)
        texts = []
        if inspect.isgenerator(res) or hasattr(res, "__next__"):
            for j, r in enumerate(res):
                t = getattr(r, "text", None)
                if t is None and isinstance(r, dict):
                    t = r.get("text")
                texts.append(t if t is not None else repr(r)[:60])
                if j > 200:
                    break
            print(f"  generator yielded {len(texts)} items", flush=True)
            print(f"  ANSWER: {''.join(x for x in texts if isinstance(x, str))[:300]!r}",
                  flush=True)
        else:
            print(f"  returned {type(res).__name__}: {str(res)[:300]!r}",
                  flush=True)
    except Exception:
        print("!! stage 3 FAILED:", flush=True)
        traceback.print_exc()
        return

    print("\n=== stage 4: mid-layer read at chunk boundary ===", flush=True)
    try:
        shapes = []
        h = model.llm.model.layers[22].register_forward_hook(
            lambda _m, _i, out: shapes.append(
                tuple((out[0] if isinstance(out, tuple) else out).shape)))
        try:
            call_defensive(api["streaming_prefill"], session_id="s1",
                           msgs=[{"role": "user",
                                  "content": [chunks[0].astype(np.float32)]}],
                           tokenizer=tok)
        finally:
            h.remove()
        print(f"  L22 fired {len(shapes)}x, shapes {shapes[:3]} -> gate can "
              f"read at chunk boundaries", flush=True)
    except Exception:
        print("!! stage 4 FAILED:", flush=True)
        traceback.print_exc()

    print("\n=== stage 5: follow-up TEXT turn (result-injection analog) ===",
          flush=True)
    try:
        call_defensive(api["streaming_prefill"], session_id="s1",
                       msgs=[{"role": "user",
                              "content": ["Expert result: the answer is 42. "
                                          "Please relay this to the user in "
                                          "one sentence."]}],
                       tokenizer=tok)
        res = call_defensive(api["streaming_generate"], session_id="s1",
                             tokenizer=tok, temperature=0.1,
                             generate_audio=False)
        if inspect.isgenerator(res) or hasattr(res, "__next__"):
            texts = [getattr(r, "text", r.get("text") if isinstance(r, dict)
                             else repr(r)[:40]) for r in res]
            print(f"  RELAY: {''.join(x for x in texts if isinstance(x, str))[:300]!r}",
                  flush=True)
        else:
            print(f"  RELAY: {str(res)[:300]!r}", flush=True)
    except Exception:
        print("!! stage 5 FAILED:", flush=True)
        traceback.print_exc()

    print("\n>>> streaming_smoke done", flush=True)


# ============================ thinking ablation (6c) =======================
# Does the gate+escalation beat o4.5's OWN slow path (enable_thinking)?
THINK_TAG = "minicpm-o45-think"


@app.function(image=image_au, gpu="H100", volumes=GPU_VOL, timeout=60 * 60 * 3)
def collect_think(shard: list, shard_id: int, max_new_tokens: int = 2048) -> int:
    import time
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import decode

    model, tok = _load_model()
    kw = decode._chat_kwargs(model, tok)
    if "enable_thinking" not in kw:
        raise RuntimeError("chat() has no enable_thinking param on this build")
    kw["enable_thinking"] = True
    print(f">>> {THINK_TAG} shard {shard_id}: {len(shard)} queries", flush=True)

    rows = []
    for k, q in enumerate(shard):
        t0 = time.perf_counter()
        out = model.chat(msgs=[{"role": "user", "content": [q["query"]]}],
                         max_new_tokens=max_new_tokens, **kw)
        dt = time.perf_counter() - t0
        raw = out if isinstance(out, str) else str(out)
        ans = raw.split("</think>")[-1].strip() if "</think>" in raw \
            else raw.strip()
        rows.append({**{f: q[f] for f in
                        ("id", "pool", "source", "query", "reference_answer",
                         "split")},
                     "answer": ans, "raw_len": len(raw),
                     "had_think": "</think>" in raw, "latency_s": dt})
        if k < 3 or k % 50 == 0:
            print(f"  [{k}] {q['pool']} {dt:.1f}s think={'</think>' in raw} "
                  f":: {ans[:60]!r}", flush=True)
    pd.DataFrame(rows).to_parquet(
        f"{DATA}/signals_{THINK_TAG}.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote think shard {shard_id}", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_think(workers: int = 4, limit: int = 0):
    qs = _load_queries.remote()
    if limit:
        qs = qs[:limit]
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {THINK_TAG}: {len(qs)} queries / {workers} H100 workers")
    total = sum(collect_think.starmap(
        [(shards[i], i) for i in range(workers)]))
    print(f">>> collected thinking answers for {total} queries")


@app.function(image=util_image_au, volumes={DATA: gate_data}, timeout=60 * 30,
              cpu=8)
def think_report():
    """Ablation: fast vs think vs gated tiers. Per-pool fail rates, test acc,
    thinking latency, and gated-think vs gated-cloud at matched escalation."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    fast = pd.read_parquet(FEATURES)[["id", "pool", "split", "escalate_label"]]
    thk = pd.read_parquet(f"{DATA}/features_{THINK_TAG}.parquet")
    lat = pd.concat([pd.read_parquet(s)[["id", "latency_s", "had_think"]]
                     for s in sorted(glob.glob(
                         f"{DATA}/signals_{THINK_TAG}.shard*.parquet"))],
                    ignore_index=True)
    df = fast.merge(thk[["id", "escalate_label"]], on="id",
                    suffixes=("_fast", "_think")).merge(lat, on="id")
    df = df[df["escalate_label_fast"].notna()
            & df["escalate_label_think"].notna()]

    print(f"=== fast vs think (n={len(df)}) ===", flush=True)
    for pool, g in df.groupby("pool"):
        print(f"  {pool:16s} n={len(g):4d} "
              f"fail fast={g['escalate_label_fast'].mean():.3f} "
              f"think={g['escalate_label_think'].mean():.3f} "
              f"lat P50={g['latency_s'].median():.1f}s "
              f"P95={g['latency_s'].quantile(.95):.1f}s "
              f"think_used={g['had_think'].mean():.2f}", flush=True)
    te = df[df["split"] == "test"]
    print(f"\n  TEST acc: fast={1 - te['escalate_label_fast'].mean():.3f} "
          f"think={1 - te['escalate_label_think'].mean():.3f} "
          f"(big-only gpt-5.5 = 0.917 from Phase 5)", flush=True)

    # gated tiers at matched escalation budget (L22 probe, calib-trained)
    ids, hl = [], []
    for s in sorted(glob.glob(f"{DATA}/layers_minicpm-o45.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    hl = np.concatenate(hl)
    m = pd.DataFrame({"id": ids, "row": range(len(ids))}).merge(df, on="id")
    ca, tt = m[m["split"] == "calib"], m[m["split"] == "test"]
    X = hl[ca["row"].to_numpy(), 22, :].astype(np.float32)
    probe = LogisticRegression(max_iter=2000).fit(
        X, ca["escalate_label_fast"].astype(int))
    sc = probe.predict_proba(
        hl[tt["row"].to_numpy(), 22, :].astype(np.float32))[:, 1]
    for esc in (0.15, 0.33, 0.50):
        thr = np.quantile(sc, 1 - esc)
        esc_mask = sc >= thr
        acc_gt = np.where(esc_mask, 1 - tt["escalate_label_think"],
                          1 - tt["escalate_label_fast"]).mean()
        print(f"  gated-THINK @esc={esc:.2f}: acc={acc_gt:.3f} "
              f"(escalated {esc_mask.mean():.2f} of test)", flush=True)

    ee = f"{DATA}/eval_expert.parquet"
    if os.path.exists(ee):
        big = pd.read_parquet(ee)
        print(f"\n  eval_expert columns: {list(big.columns)}", flush=True)
        bcol = next((c for c in ("adequate", "expert_adequate") if c in big),
                    None)
        if bcol:
            tt2 = tt.merge(big[["id", bcol]], on="id")
            sc2 = probe.predict_proba(
                hl[tt2["row"].to_numpy(), 22, :].astype(np.float32))[:, 1]
            for esc in (0.15, 0.33, 0.50):
                thr = np.quantile(sc2, 1 - esc)
                esc_mask = sc2 >= thr
                acc_gc = np.where(esc_mask, tt2[bcol].astype(float),
                                  1 - tt2["escalate_label_fast"]).mean()
                print(f"  gated-CLOUD @esc={esc:.2f}: acc={acc_gc:.3f}",
                      flush=True)
    print(">>> think_report done", flush=True)


# ============================ e2e policy latency (6c) ======================
@app.function(image=util_image_au, volumes={DATA: gate_data}, timeout=60 * 30,
              cpu=8)
def e2e_latency_report():
    """Per-REQUEST end-to-end latency x accuracy on the test split, per
    policy: fast / all-think / gated-think / gated-cloud. Components:
    per-query THINK latency (measured in collect_think), per-query FAST
    latency (n_forward x per-token cost fitted on latency_bench), gate
    decision = truncated-L22 median from the bench, cloud tier = Phase-5
    expert latency (per-query if eval_expert has it, else P50 3.0s const)."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    b = pd.read_parquet(f"{DATA}/latency_bench.parquet")
    b = b[~b["warmup"]]
    gate_s = float(b["text_l22_s"].median())
    # per-pool MEASURED fast-answer medians from the bench (n_forward in the
    # signals store is capped at k+1=17 by the Collector — unusable for
    # latency; bench numbers are wall-clock at max_new_tokens=256, a mild
    # underestimate for the longest answers)
    pool_fast = b.groupby("pool")["text_answer_s"].median().to_dict()
    print(f"gate decision (L22 truncated) = {gate_s * 1000:.0f} ms; "
          f"fast per-pool P50s: " +
          ", ".join(f"{p}={v:.1f}s" for p, v in sorted(pool_fast.items())),
          flush=True)

    fast = pd.read_parquet(FEATURES)[
        ["id", "pool", "split", "escalate_label", "n_forward"]]
    thk = pd.read_parquet(f"{DATA}/features_{THINK_TAG}.parquet")[
        ["id", "escalate_label"]].rename(
        columns={"escalate_label": "esc_think"})
    tlat = pd.concat([pd.read_parquet(s)[["id", "latency_s"]]
                      for s in sorted(glob.glob(
                          f"{DATA}/signals_{THINK_TAG}.shard*.parquet"))],
                     ignore_index=True).rename(
        columns={"latency_s": "lat_think"})
    df = fast.merge(thk, on="id").merge(tlat, on="id")
    df["lat_fast"] = df["pool"].map(pool_fast)
    df = df[df["escalate_label"].notna() & df["esc_think"].notna()]

    print("\n=== think overhead per pool (vs fast, seconds) ===", flush=True)
    for pool, g in df.groupby("pool"):
        print(f"  {pool:16s} fast P50={g['lat_fast'].median():.1f} "
              f"think P50={g['lat_think'].median():.1f} "
              f"overhead P50={(g['lat_think'] - g['lat_fast']).median():+.1f} "
              f"P95={(g['lat_think'] - g['lat_fast']).quantile(.95):+.1f}",
              flush=True)

    ee_lat = None
    ee = f"{DATA}/eval_expert.parquet"
    big = pd.read_parquet(ee) if os.path.exists(ee) else None
    if big is not None:
        lat_col = next((c for c in big.columns if "lat" in c.lower()
                        or c.endswith("_s")), None)
        print(f"\n  eval_expert columns: {list(big.columns)} "
              f"(latency col: {lat_col})", flush=True)
        if lat_col:
            ee_lat = big[["id", lat_col]].rename(columns={lat_col: "lat_big"})
    if ee_lat is not None:
        df = df.merge(ee_lat, on="id", how="left")
    if "lat_big" not in df or df["lat_big"].isna().all():
        df["lat_big"] = 3.0          # Phase-5 expert P50; P95 24.4 noted
        print("  (no per-query expert latency — using P50 3.0s constant)",
              flush=True)
    bcol = next((c for c in ("adequate", "expert_adequate")
                 if big is not None and c in big), None)
    if bcol:
        df = df.merge(big[["id", bcol]].rename(columns={bcol: "big_ok"}),
                      on="id", how="left")

    ids, hl = [], []
    for s in sorted(glob.glob(f"{DATA}/layers_minicpm-o45.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    hl = np.concatenate(hl)
    m = pd.DataFrame({"id": ids, "row": range(len(ids))}).merge(df, on="id")
    ca, tt = m[m["split"] == "calib"], m[m["split"] == "test"].copy()
    probe = LogisticRegression(max_iter=2000).fit(
        hl[ca["row"].to_numpy(), 22, :].astype(np.float32),
        ca["escalate_label"].astype(int))
    tt["score"] = probe.predict_proba(
        hl[tt["row"].to_numpy(), 22, :].astype(np.float32))[:, 1]

    def policy(esc_rate, tier):
        if esc_rate == 0:
            lat, acc = tt["lat_fast"], 1 - tt["escalate_label"]
        elif esc_rate == 1:
            lat = (tt["lat_think"] if tier == "think"
                   else gate_s + tt["lat_big"])
            acc = (1 - tt["esc_think"] if tier == "think"
                   else tt.get("big_ok", pd.Series(np.nan, index=tt.index)))
        else:
            thr = np.quantile(tt["score"], 1 - esc_rate)
            e = tt["score"] >= thr
            if tier == "think":
                lat = gate_s + np.where(e, tt["lat_think"], tt["lat_fast"])
                acc = np.where(e, 1 - tt["esc_think"],
                               1 - tt["escalate_label"])
            else:
                lat = gate_s + np.where(e, tt["lat_big"], tt["lat_fast"])
                ok = tt["big_ok"] if "big_ok" in tt else pd.Series(
                    np.nan, index=tt.index)
                acc = np.where(e, ok.astype(float),
                               1 - tt["escalate_label"])
        lat = pd.Series(np.asarray(lat, dtype=float))
        acc = pd.Series(np.asarray(acc, dtype=float))
        return (float(np.nanmean(acc)), float(lat.mean()),
                float(lat.median()), float(lat.quantile(.95)))

    print(f"\n=== policy table (test n={len(tt)}; latency s: "
          f"mean/P50/P95) ===", flush=True)
    print(f"  {'policy':24s} {'acc':>5s} {'mean':>6s} {'P50':>6s} {'P95':>6s}",
          flush=True)
    a, mn, p5, p9 = policy(0, "-")
    print(f"  {'fast-only':24s} {a:.3f} {mn:6.1f} {p5:6.1f} {p9:6.1f}",
          flush=True)
    a, mn, p5, p9 = policy(1, "think")
    print(f"  {'all-THINK (o4.5 own)':24s} {a:.3f} {mn:6.1f} {p5:6.1f} "
          f"{p9:6.1f}", flush=True)
    for esc in (0.15, 0.33, 0.50):
        for tier in ("think", "cloud"):
            a, mn, p5, p9 = policy(esc, tier)
            print(f"  {f'gated-{tier} @{esc:.2f}':24s} {a:.3f} {mn:6.1f} "
                  f"{p5:6.1f} {p9:6.1f}", flush=True)
    print("\n  (cloud tier P95 understated if constant 3.0s used — "
          "Phase-5 expert P95 was 24.4s)", flush=True)
    print(">>> e2e_latency_report done", flush=True)


# ============================ latency bench (6c) ===========================
@app.function(image=image_au, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def latency_bench(n_per_pool: int = 10, max_new_tokens: int = 256):
    """Per-query paired timings, text AND audio arms: TTFT, full-answer time,
    truncated-forward-to-L22 decision time, ptrue_pre time. CUDA-synced,
    3-query warmup excluded, all configs interleaved per query."""
    import time
    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0, "/workspace/gate")
    import decode

    model, tok = _load_model_audio()
    kw = decode._chat_kwargs(model, tok)
    L22 = model.llm.model.layers[22]
    head = model.llm.lm_head

    by_pool = {}
    for q in _read_jsonl(QUERIES):
        by_pool.setdefault(q["pool"], []).append(q)
    sample = [q for p in sorted(by_pool) for q in by_pool[p][:n_per_pool]]

    class _Stop(Exception):
        pass

    def timed_chat(content, mnt):
        t = {}

        def hook(_m, _i, _o):
            if "first" not in t:
                torch.cuda.synchronize()
                t["first"] = time.perf_counter()
            t["n"] = t.get("n", 0) + 1

        h = head.register_forward_hook(hook)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            model.chat(msgs=[{"role": "user", "content": content}],
                       max_new_tokens=mnt, **kw)
        finally:
            h.remove()
        torch.cuda.synchronize()
        return (t.get("first", t0) - t0, time.perf_counter() - t0,
                t.get("n", 0))

    def timed_l22(content):
        def hook(_m, _i, out):
            raise _Stop()

        h = L22.register_forward_hook(hook)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            model.chat(msgs=[{"role": "user", "content": content}],
                       max_new_tokens=1, **kw)
        except _Stop:
            pass
        finally:
            h.remove()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    print(f">>> latency bench: {len(sample)} queries x 2 arms", flush=True)
    rows = []
    for k, q in enumerate(sample):
        au = _load_wav(q["id"])
        rec = {"id": q["id"], "pool": q["pool"], "warmup": k < 3}
        for arm, content, meta in (
                ("text", [q["query"]], [PTRUE_PRE.format(q=q["query"])]),
                ("audio", [au], [au, PTRUE_PRE_AU])):
            ttft, total, n = timed_chat(content, max_new_tokens)
            rec[f"{arm}_ttft"] = ttft
            rec[f"{arm}_answer_s"] = total
            rec[f"{arm}_tokens"] = n
            rec[f"{arm}_l22_s"] = timed_l22(content)
            pre_ttft, pre_total, _ = timed_chat(meta, 1)
            rec[f"{arm}_ptrue_pre_s"] = pre_total
        rows.append(rec)
        if k < 5 or k % 10 == 0:
            print(f"  [{k}] {q['pool']}: text ttft={rec['text_ttft']*1000:.0f}ms "
                  f"l22={rec['text_l22_s']*1000:.0f}ms | audio "
                  f"ttft={rec['audio_ttft']*1000:.0f}ms "
                  f"l22={rec['audio_l22_s']*1000:.0f}ms", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(f"{DATA}/latency_bench.parquet")
    gate_data.commit()
    d = df[~df["warmup"]]
    print(f"\n=== latency summary (n={len(d)}, ms, P50/P95) ===", flush=True)
    for arm in ("text", "audio"):
        for cfg in ("l22_s", "ttft", "ptrue_pre_s", "answer_s"):
            v = d[f"{arm}_{cfg}"] * 1000
            print(f"  {arm:5s} {cfg:12s} {v.median():8.0f} / "
                  f"{v.quantile(.95):8.0f}", flush=True)
    print(f"  decode rate: "
          f"{(d['text_tokens'] / d['text_answer_s']).median():.1f} tok/s",
          flush=True)
    print(">>> latency_bench done", flush=True)


# ============================ cross-modal report ===========================
@app.function(image=util_image_au, volumes={DATA: gate_data}, timeout=60 * 60,
              cpu=8)
def audio_report(mtag: str = "minicpm-o45", dtag: str = ""):
    """The analyses the text-pipeline reports can't do: paired text-vs-audio
    fail rates, per-layer CROSS-MODAL probe transfer (text-trained -> audio
    hiddens and vice versa; the deployment question 'calibrate on text, deploy
    on speech'), and audio p(True) AUCs incl. the trap text-vs-audio contrast.
    dtag overrides the audio-side tag when it isn't {mtag}-audio (e.g.
    text=qwen2-7b vs audio=freeze-omni-audio, same frozen weights)."""
    import glob
    import json
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    dtag = dtag or f"{mtag}-audio"
    feats_tx = (FEATURES if mtag == "minicpm-o45"
                else f"{DATA}/features_{mtag}.parquet")
    ptx_glob = (f"{DATA}/ptrue.shard*.parquet" if mtag == "minicpm-o45"
                else f"{DATA}/ptrue_{mtag}.shard*.parquet")

    def load_layers(tag):
        ids, hl = [], []
        for s in sorted(glob.glob(f"{DATA}/layers_{tag}.shard*.npz")):
            z = np.load(s)
            ids += [str(x) for x in z["ids"]]
            hl.append(z["h_last"])
        return ids, np.concatenate(hl)

    ftx = pd.read_parquet(feats_tx)[["id", "pool", "split", "escalate_label"]]
    fau = pd.read_parquet(f"{DATA}/features_{dtag}.parquet")[
        ["id", "pool", "split", "escalate_label"]]

    # ---- 1. paired fail-rate shift ---------------------------------------
    pair = ftx.merge(fau, on="id", suffixes=("_tx", "_au"))
    pair = pair[pair["escalate_label_tx"].notna()
                & pair["escalate_label_au"].notna()]
    print(f"=== paired fail rates (n={len(pair)}) ===", flush=True)
    for pool, g in pair.groupby("pool_tx"):
        tx, au = g["escalate_label_tx"].mean(), g["escalate_label_au"].mean()
        both = ((g["escalate_label_tx"] == 1) & (g["escalate_label_au"] == 1)).mean()
        print(f"  {pool:16s} n={len(g):4d} fail_text={tx:.3f} "
              f"fail_audio={au:.3f} d={au - tx:+.3f} both={both:.3f}", flush=True)

    # ---- 2. per-layer cross-modal probe transfer -------------------------
    ids_tx, hl_tx = load_layers(mtag)
    ids_au, hl_au = load_layers(dtag)
    rtx = pd.DataFrame({"id": ids_tx, "row": range(len(ids_tx))}).merge(
        ftx, on="id")
    rau = pd.DataFrame({"id": ids_au, "row": range(len(ids_au))}).merge(
        fau, on="id")
    rtx = rtx[(rtx["split"] == "calib") & rtx["escalate_label"].notna()]
    rau = rau[(rau["split"] == "calib") & rau["escalate_label"].notna()]
    ytx = rtx["escalate_label"].astype(int).to_numpy()
    yau = rau["escalate_label"].astype(int).to_numpy()
    n_layers = hl_tx.shape[1]
    print(f"\n=== cross-modal transfer (calib: text n={len(rtx)}, "
          f"audio n={len(rau)}; layers={n_layers}) ===", flush=True)

    curves = []
    for li in range(n_layers):
        Xtx = hl_tx[rtx["row"].to_numpy(), li, :].astype(np.float32)
        Xau = hl_au[rau["row"].to_numpy(), li, :].astype(np.float32)
        t2a = LogisticRegression(max_iter=2000).fit(Xtx, ytx) \
            .predict_proba(Xau)[:, 1]
        a2t = LogisticRegression(max_iter=2000).fit(Xau, yau) \
            .predict_proba(Xtx)[:, 1]
        rec = {"layer": li,
               "text2audio": float(roc_auc_score(yau, t2a)),
               "audio2text": float(roc_auc_score(ytx, a2t))}
        curves.append(rec)
        print(f"  L{li:02d} text->audio={rec['text2audio']:.3f} "
              f"audio->text={rec['audio2text']:.3f}", flush=True)

    # ---- 3. audio p(True) + trap text-vs-audio contrast ------------------
    pt = pd.concat([pd.read_parquet(s) for s in
                    sorted(glob.glob(f"{DATA}/ptrue_{dtag}.shard*.parquet"))],
                   ignore_index=True).merge(fau, on="id")
    pt = pt[pt["escalate_label"].notna()]
    y = pt["escalate_label"].astype(int)
    print(f"\n=== audio p(True) (n={len(pt)}) ===", flush=True)
    for col in ("p_yes_pre", "p_yes_post"):
        auc = roc_auc_score(y, -pt[col])       # low P(Yes) -> escalate
        print(f"  {col}: AUC={auc:.3f}", flush=True)
    tx_files = sorted(glob.glob(ptx_glob))
    if tx_files:
        ptx = pd.concat([pd.read_parquet(s) for s in tx_files],
                        ignore_index=True)[["id", "p_yes_pre"]] \
            .rename(columns={"p_yes_pre": "p_yes_pre_tx"})
        both = pt.merge(ptx, on="id")
        for pool, g in both.groupby("pool"):
            print(f"  {pool:16s} p_yes_pre text={g['p_yes_pre_tx'].mean():.3f} "
                  f"audio={g['p_yes_pre'].mean():.3f}", flush=True)

    with open(f"{DATA}/audio_xmodal_{dtag}.json", "w") as f:
        json.dump({"curves": curves}, f)
    gate_data.commit()
    print(f"\n>>> wrote {DATA}/audio_xmodal_{dtag}.json", flush=True)


# ============================ fork timing (7a) =============================
@app.function(image=image_au, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def prefill_timing(n_per_pool: int = 5):
    """Per-LAYER truncated-forward wall-clock, text and audio arms: at what
    time during prefill is layer k's hidden state available, i.e. how early
    can a mid-layer probe fire?  Same truncation methodology as
    latency_bench's timed_l22, generalized to every decoder layer; the probe
    itself is one 4096-dim dot product (~us) so time-to-layer IS the decision
    time. layer=-1 rows = full prefill + 1 decoded token (TTFT reference).
    -> prefill_timing.parquet"""
    import time
    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0, "/workspace/gate")
    import decode

    model, tok = _load_model_audio()
    kw = decode._chat_kwargs(model, tok)
    layers = model.llm.model.layers
    n_layers = len(layers)

    by_pool = {}
    for q in _read_jsonl(QUERIES):
        by_pool.setdefault(q["pool"], []).append(q)
    sample = [q for p in sorted(by_pool) for q in by_pool[p][:n_per_pool]]

    class _Stop(Exception):
        pass

    def timed_layer(content, li):
        def hook(_m, _i, _o):
            raise _Stop()

        h = layers[li].register_forward_hook(hook)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            model.chat(msgs=[{"role": "user", "content": content}],
                       max_new_tokens=1, **kw)
        except _Stop:
            pass
        finally:
            h.remove()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def timed_full(content):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.chat(msgs=[{"role": "user", "content": content}],
                   max_new_tokens=1, **kw)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    print(f">>> prefill_timing: {len(sample)} queries x 2 arms x "
          f"{n_layers} layers", flush=True)
    rows = []
    for k, q in enumerate(sample):
        au = _load_wav(q["id"])
        warm = k < 2
        for arm, content in (("text", [q["query"]]), ("audio", [au])):
            for li in range(n_layers):
                rows.append({"id": q["id"], "pool": q["pool"], "arm": arm,
                             "warmup": warm, "layer": li,
                             "t_s": timed_layer(content, li)})
            rows.append({"id": q["id"], "pool": q["pool"], "arm": arm,
                         "warmup": warm, "layer": -1,
                         "t_s": timed_full(content)})
        if k < 5 or k % 10 == 0:
            d = [r for r in rows if r["id"] == q["id"]]
            t22 = next(r["t_s"] for r in d
                       if r["arm"] == "text" and r["layer"] == 22)
            tf = next(r["t_s"] for r in d
                      if r["arm"] == "text" and r["layer"] == -1)
            print(f"  [{k}] {q['pool']}: text L22={t22*1000:.0f}ms "
                  f"prefill+1tok={tf*1000:.0f}ms", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(f"{DATA}/prefill_timing.parquet")
    gate_data.commit()
    d = df[~df["warmup"]]
    for arm in ("text", "audio"):
        g = d[d["arm"] == arm].groupby("layer")["t_s"].median() * 1000
        print(f"  {arm}: L0={g[0]:.0f} L11={g[11]:.0f} L22={g[22]:.0f} "
              f"L{n_layers-1}={g[n_layers-1]:.0f} full+1tok={g[-1]:.0f} ms",
              flush=True)
    print(">>> prefill_timing done", flush=True)


@app.function(image=util_image_au, volumes={DATA: gate_data}, timeout=60 * 30,
              cpu=8)
def fork_report():
    """Joins prefill_timing (WHEN is layer k readable) with the 5d layer
    sweep (HOW GOOD is layer k) -> the fork Pareto: decision quality vs
    decision time, text and audio arms. -> fork_pareto.png"""
    import json
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = pd.read_parquet(f"{DATA}/prefill_timing.parquet")
    t = t[~t["warmup"]]
    curves = {}
    for arm, tag in (("text", "minicpm-o45"), ("audio", "minicpm-o45-audio")):
        with open(f"{DATA}/layer_sweep_{tag}.json") as f:
            sweep = {c["layer"]: c for c in json.load(f)["curves"]}
        g = t[t["arm"] == arm].groupby("layer")["t_s"].median() * 1000
        full = g.pop(-1)
        curves[arm] = pd.DataFrame({
            "layer": g.index, "ms": g.values,
            "pct_of_prefill": g.values / full * 100,
            "oof": [sweep[li]["last_oof"] for li in g.index]})
        print(f"\n=== {arm} (prefill+1tok = {full:.0f} ms) ===", flush=True)
        for _, r in curves[arm].iterrows():
            print(f"  L{int(r['layer']):02d} {r['ms']:6.1f} ms "
                  f"({r['pct_of_prefill']:5.1f}%) oof={r['oof']:.3f}",
                  flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, arm in zip(axes, ("text", "audio")):
        c = curves[arm]
        ax.plot(c["ms"], c["oof"], "o-", ms=3, label="probe OOF AUC")
        for li in (11, 16, 22):
            r = c[c["layer"] == li].iloc[0]
            ax.annotate(f"L{li}", (r["ms"], r["oof"]),
                        textcoords="offset points", xytext=(4, 4))
        ax.set_title(f"{arm}: decision quality vs decision time")
        ax.set_xlabel("time-to-layer during prefill (ms, median)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("calib OOF AUC (last-token probe)")
    fig.tight_layout()
    fig.savefig(f"{DATA}/fork_pareto.png", dpi=140)
    gate_data.commit()
    print(f"\n>>> wrote {DATA}/fork_pareto.png", flush=True)


# ============================ escalation overlap (7a) ======================
@app.function(image=util_image_au, volumes={DATA: gate_data}, timeout=60 * 30,
              cpu=8)
def overlap_report():
    """Jisen's question: when the gate escalates, is the cloud result ready
    before the talker finishes speaking the current turn?  Timeline per
    escalated query: gate fires at t=l22 (pre-decode) -> cloud call starts
    in parallel with local decode; expert result lands at l22 +
    expert_latency; the talker holds the floor for its local answer
    (latency_bench wall-clock, pool-matched).  overlap = P(expert ready
    before local speech ends + slack), slack = turn-taking gap. Uses
    Phase-5 per-query gpt-5.5 latencies (Modal us-east; includes API RTT).
    -> overlap.png"""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = pd.read_parquet(f"{DATA}/latency_bench.parquet")
    b = b[~b["warmup"]]
    gate = {a: float(b[f"{a}_l22_s"].median()) for a in ("text", "audio")}
    local = {a: {p: g[f"{a}_answer_s"].to_numpy()
                 for p, g in b.groupby("pool")} for a in ("text", "audio")}

    big = pd.read_parquet(f"{DATA}/eval_expert.parquet")
    big = big[big["expert_error"].isna() & (big["expert_latency"] > 0)]
    print(f">>> {len(big)} expert queries; expert latency "
          f"P50={big['expert_latency'].median():.1f}s "
          f"P95={big['expert_latency'].quantile(.95):.1f}s; gate "
          f"text={gate['text']*1000:.0f}ms audio={gate['audio']*1000:.0f}ms",
          flush=True)

    slacks = (0.0, 2.0, 5.0)

    def overlap_frac(rows, arm, slack):
        """Mean over expert rows of P(gate + big <= local + slack), local
        drawn from the same pool's bench durations (cross product)."""
        vals = [np.mean(gate[arm] + r.expert_latency
                        <= local[arm][r.pool] + slack)
                for r in rows.itertuples() if r.pool in local[arm]]
        return float(np.mean(vals))

    print("\n=== overlap: P(cloud result ready before talker finishes) ===",
          flush=True)
    print(f"  {'pool':16s} " + " ".join(f"{a}+{s:.0f}s" for a in
                                        ("text", "audio") for s in slacks),
          flush=True)
    for pool, g in big.groupby("pool"):
        print(f"  {pool:16s} " +
              " ".join(f"{overlap_frac(g, a, s):6.3f}"
                       for a in ("text", "audio") for s in slacks),
              flush=True)
    print(f"  {'ALL (test mix)':16s} " +
          " ".join(f"{overlap_frac(big, a, s):6.3f}"
                   for a in ("text", "audio") for s in slacks), flush=True)

    # --- escalated-only: the queries the L22 gate actually fires on --------
    feats = pd.read_parquet(FEATURES)[["id", "pool", "split",
                                       "escalate_label"]]
    ids, hl = [], []
    for s in sorted(glob.glob(f"{DATA}/layers_minicpm-o45.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    hl = np.concatenate(hl)
    m = pd.DataFrame({"id": ids, "row": range(len(ids))}).merge(feats, on="id")
    m = m[m["escalate_label"].notna()]
    ca, tt = m[m["split"] == "calib"], m[m["split"] == "test"].copy()
    probe = LogisticRegression(max_iter=2000).fit(
        hl[ca["row"].to_numpy(), 22, :].astype(np.float32),
        ca["escalate_label"].astype(int))
    tt["score"] = probe.predict_proba(
        hl[tt["row"].to_numpy(), 22, :].astype(np.float32))[:, 1]
    print("\n=== escalated-only overlap (L22 gate, test split) ===",
          flush=True)
    for esc in (0.15, 0.33, 0.50):
        sel = tt[tt["score"] >= np.quantile(tt["score"], 1 - esc)]
        e = big.merge(sel[["id"]], on="id")
        print(f"  esc@{esc:.2f} (n={len(e)}): " +
              " ".join(f"{a}+{s:.0f}s={overlap_frac(e, a, s):.3f}"
                       for a in ("text", "audio") for s in slacks),
              flush=True)
        for arm in ("text", "audio"):
            med = {p: float(np.median(v)) for p, v in local[arm].items()}
            wait = np.array([gate[arm] + r.expert_latency - med[r.pool]
                             for r in e.itertuples() if r.pool in med])
            wait = np.clip(wait, 0, None)     # already-ready => 0 stall
            print(f"           {arm}: stall needed after local answer "
                  f"P50={np.median(wait):.1f}s P90="
                  f"{np.percentile(wait, 90):.1f}s", flush=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, ls in (("text", "-"), ("audio", "--")):
        ready = np.sort(gate[arm] + big["expert_latency"].to_numpy())
        ax.plot(ready, np.linspace(0, 1, len(ready)), "C0" + ls,
                label=f"cloud result ready ({arm} gate)")
        dur = np.sort(np.concatenate(list(local[arm].values())))
        ax.plot(dur, np.linspace(0, 1, len(dur)), "C1" + ls,
                label=f"local answer duration ({arm})")
    ax.set_xlim(0, 15)
    ax.set_xlabel("seconds after query end")
    ax.set_ylabel("CDF")
    ax.set_title("Escalation overlap: cloud round-trip vs talker floor time")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{DATA}/overlap.png", dpi=140)
    gate_data.commit()
    print(f"\n>>> wrote {DATA}/overlap.png", flush=True)
