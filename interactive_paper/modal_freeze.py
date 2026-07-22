"""Phase 6d — Freeze-Omni: the frozen-backbone control.

Freeze-Omni (arXiv 2411.00774, ICML'25) bolts streaming speech encoder +
state-prediction duplex onto a FROZEN Qwen2-7B-Instruct. It separates
"duplex capability" from "backbone weight updates": if its readouts are
intact on audio input, the 6a/6b damage is attributable to duplex TRAINING
of the backbone, not duplex operation; if its audio p(True) also collapses,
the instance-binding gap exists even without any weight updates. Either
way the mechanism claim sharpens. Text side is analytically the raw
qwen2-7b (= Qwen/Qwen2-7B-Instruct, already tag `qwen2-7b` on the volume
with 5c features).

Data tag: `freeze-omni-audio` (same file schema; label_hf/layer_sweep_report
reuse). Env: their pins — torch 2.2.0 / transformers 4.45.2 (audioLLM
manipulates LEGACY TUPLE past_key_values; newer Cache classes break it).
Repo cloned into the image at /opt/freeze.

  modal run modal_freeze.py::download_freeze
  modal run modal_freeze.py::freeze_smoke
  modal run modal_freeze.py::run_freeze --workers 4
  modal run modal_app.py::label_hf --tag freeze-omni-audio
  modal run modal_app.py::layer_sweep_report --tag freeze-omni-audio
"""
import os
import sys

import modal

from modal_app import (app, gen_app, GPU_VOL, gate_data, weights, DATA,
                       QUERIES, _read_jsonl, _load_queries)
from modal_audio import AUDIO_DIR, PTRUE_PRE_AU, PTRUE_POST_AU

FZ_TAG = "freeze-omni-audio"
FZ_DIR = "/workspace/models/freeze-omni"
LLM_DIR = "/workspace/models/qwen2-7b"           # Qwen/Qwen2-7B-Instruct (5c)
HERE = os.path.dirname(os.path.abspath(__file__))

freeze_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libsndfile1", "ffmpeg")
    .pip_install("torch==2.2.0", "torchaudio==2.2.0")
    .pip_install("transformers==4.45.2", "accelerate", "PyYAML==6.0.2",
                 "librosa==0.10.2.post1", "soundfile==0.12.1",
                 "numpy==1.24.4", "pandas", "pyarrow",
                 "huggingface_hub[hf_transfer]")
    .run_commands("git clone --depth 1 "
                  "https://github.com/VITA-MLLM/Freeze-Omni /opt/freeze")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1"})
    .add_local_file(os.path.join(HERE, "modal_app.py"), "/root/modal_app.py")
    .add_local_file(os.path.join(HERE, "modal_audio.py"),
                    "/root/modal_audio.py")
)


@app.function(image=freeze_image, volumes={"/workspace/models": weights},
              timeout=60 * 60)
def download_freeze():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"   # flaky here; plain http
    from huggingface_hub import snapshot_download
    for attempt in range(3):
        try:
            snapshot_download("VITA-MLLM/Freeze-Omni", local_dir=FZ_DIR)
            break
        except Exception as e:
            print(f"  attempt {attempt}: {type(e).__name__}: {e}", flush=True)
            if attempt == 2:
                raise
    weights.commit()
    print(">>> freeze-omni checkpoints downloaded", flush=True)
    for root, _dirs, files in os.walk(FZ_DIR):
        for f in files[:50]:
            print("  ", os.path.join(root, f).replace(FZ_DIR, ""), flush=True)


class _Args:
    top_k, top_p, temperature = 1, 0.0, 1.0     # top_k=1 => greedy

    def __init__(self, model_path, llm_path):
        self.model_path, self.llm_path = model_path, llm_path


class _AudioProcessor:
    """Vendored from bin/inference.py::audioEncoderProcessor (importing that
    module drags in the flask web stack)."""

    def __init__(self, chunk_size=16):
        self.chunk_size = 16
        self.chunk_overlap = 3
        self.feat_dim = 80
        self.frame_size = 400
        self.frame_shift = 160
        self.frame_overlap = self.frame_size - self.frame_shift
        self.CHUNK = self.frame_shift * self.chunk_size
        self.reset()

    def get_chunk_size(self):
        return self.CHUNK

    def reset(self):
        import torch
        self.input_chunk = torch.zeros(
            [1, self.chunk_size + self.chunk_overlap, self.feat_dim])
        self.input_sample = torch.zeros(
            [1, self.CHUNK + self.frame_overlap, 1])

    def process(self, audio):
        import torch
        import torchaudio.compliance.kaldi as k
        with torch.no_grad():
            sample_data = torch.tensor(audio).reshape(1, -1, 1)[:, :, :1] * 32768
            self.input_sample[:, :self.frame_overlap, :] = \
                self.input_sample[:, -self.frame_overlap:, :].clone()
            self.input_sample[:, self.frame_overlap:, :] = sample_data
            xs = k.fbank(waveform=self.input_sample.squeeze(-1), dither=0,
                         frame_length=25, frame_shift=10,
                         num_mel_bins=self.feat_dim)
            self.input_chunk[:, :self.chunk_overlap, :] = \
                self.input_chunk[:, -self.chunk_overlap:, :].clone()
            self.input_chunk[:, self.chunk_overlap:, :] = xs.squeeze(0)
        return self.input_chunk.clone()


def _load_freeze():
    sys.path.insert(0, "/opt/freeze")
    from models.pipeline import inferencePipeline
    # audiollm dir may be nested one level in the HF snapshot
    mp = FZ_DIR
    if not os.path.exists(f"{mp}/audiollm") and \
            os.path.exists(f"{mp}/checkpoints/audiollm"):
        mp = f"{mp}/checkpoints"
    return inferencePipeline(_Args(mp, LLM_DIR))


def _audio_chunks(qid: str, processor):
    import math
    import librosa
    import torch
    au, _ = librosa.load(f"{AUDIO_DIR}/{qid}.wav", sr=16000, mono=True)
    wav = torch.tensor(au)
    cs = processor.get_chunk_size()
    pad = torch.zeros(math.ceil(wav.shape[0] / cs) * cs)
    pad[:wav.shape[0]] = wav
    return [pad[i:i + cs] for i in range(0, pad.shape[0], cs)]


def _run_dialogue(pipeline, processor, qid, max_tokens=512,
                  layer_capture=None):
    """Listen the wav chunk-by-chunk, then speak (text only, no TTS).
    Returns (text, kv_at_end_of_listen)."""
    import copy
    processor.reset()
    out = pipeline.speech_dialogue(None, stat='pre',
                                   role="You are a helpful assistant.")
    if layer_capture is not None:
        layer_capture["on"] = True
    for chunk in _audio_chunks(qid, processor):
        fbank = processor.process(chunk)
        out = pipeline.speech_dialogue(fbank, **out)
        out['stat'] = 'cl'
    if layer_capture is not None:
        layer_capture["on"] = False
    kv = copy.deepcopy(out['past_key_values'])
    out['adapter_cache'] = None
    out['encoder_cache'] = None
    out['pe_index'] = 0
    out['stat'] = 'ss'
    out = pipeline.speech_dialogue(None, **out)
    text = ''
    while True:
        if out['stat'] == 'sl':
            break
        if out['stat'] == 'cs':
            text = out['text']
            if len(out['past_tokens']) > max_tokens:
                break
        out.pop('text', None)
        out.pop('hidden_state', None)
        out = pipeline.speech_dialogue(None, **out)
    return text.strip(), kv


def _ptrue_from_kv(pipeline, kv, prompt_text, YES, NO):
    """First-token P(Yes) after appending text tokens to the audio KV —
    a plain HF forward on the frozen Qwen2 (mirrors set_system_role)."""
    import copy
    import torch
    from transformers import DynamicCache
    llm = pipeline.model.llm_decoder
    tok = pipeline.model.tokenizer
    # stay inside the user turn (after the audio), then close it with the
    # model's own chat suffix (<|im_end|> + assistant header) so the next
    # token is the assistant's first reply token
    ids = torch.cat([tok("\n" + prompt_text, return_tensors="pt").input_ids,
                     pipeline.model.chat_template['suffix']], dim=1).cuda()
    emb = llm.transformer.wte(ids)
    past_len = kv[0][0].size(2)
    cache = DynamicCache.from_legacy_cache(copy.deepcopy(kv))
    mask = torch.ones(1, past_len + ids.size(1), dtype=torch.long).cuda()
    with torch.no_grad(), torch.autocast(device_type="cuda",
                                         dtype=torch.bfloat16):
        logits = llm(inputs_embeds=emb, attention_mask=mask,
                     past_key_values=cache, use_cache=False
                     ).logits[0, -1, :].float().cpu()
    p = torch.softmax(logits, dim=-1)
    py, pn = float(p[YES].sum()), float(p[NO].sum())
    return py / (py + pn) if (py + pn) > 0 else 0.5, py + pn


@app.function(image=freeze_image, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 30)
def freeze_smoke():
    import torch
    pipeline = _load_freeze()
    processor = _AudioProcessor()

    llm = pipeline.model.llm_decoder
    layers = llm.model.layers
    print(f">>> llm type={type(llm).__name__} layers={len(layers)} "
          f"hidden={llm.config.hidden_size}", flush=True)

    tok = pipeline.model.tokenizer

    def tok_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    YES = tok_ids(["Yes", "yes", " Yes", " yes", "是", "能", "对", "会"])
    NO = tok_ids(["No", "no", " No", " no", "否", "不", "错"])

    qs = [q for q in _read_jsonl(QUERIES)
          if os.path.exists(f"{AUDIO_DIR}/{q['id']}.wav")][:3]
    for q in qs:
        text, kv = _run_dialogue(pipeline, processor, q["id"],
                                 max_tokens=128)
        pre, mass = _ptrue_from_kv(pipeline, kv, PTRUE_PRE_AU, YES, NO)
        print(f"\n=== {q['id']} ({q['pool']}) ===", flush=True)
        print(f"  Q: {q['query'][:100]!r}", flush=True)
        print(f"  A: {text[:200]!r}", flush=True)
        print(f"  ptrue_pre={pre:.3f} (mass {mass:.2f}) "
              f"kv_len={kv[0][0].size(2)}", flush=True)
    print(">>> freeze_smoke OK", flush=True)


@app.function(image=freeze_image, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 3)
def collect_freeze(shard: list, shard_id: int) -> int:
    import numpy as np
    import pandas as pd
    import torch
    pipeline = _load_freeze()
    processor = _AudioProcessor()

    llm = pipeline.model.llm_decoder
    layers = list(llm.model.layers)
    tok = pipeline.model.tokenizer

    def tok_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    YES = tok_ids(["Yes", "yes", " Yes", " yes", "是", "能", "对", "会"])
    NO = tok_ids(["No", "no", " No", " no", "否", "不", "错"])

    # capture per-layer: LAST listen-forward last position + running mean
    cap = {"on": False}
    n = len(layers)
    state = {}

    def mk(i):
        def hook(_m, _inp, out):
            if not cap["on"]:
                return
            hs = out[0] if isinstance(out, tuple) else out
            state["last"][i] = hs[0, -1, :].detach().float().cpu()
            state["sum"][i] += hs[0].detach().float().sum(dim=0).cpu()
            state["cnt"][i] += hs.shape[1]
        return hook

    handles = [m.register_forward_hook(mk(i)) for i, m in enumerate(layers)]
    d = llm.config.hidden_size
    print(f">>> {FZ_TAG} shard {shard_id}: {len(shard)} queries "
          f"({n} layers, d={d})", flush=True)

    sig_rows, pt_rows, ids, lasts, means = [], [], [], [], []
    try:
        for k, q in enumerate(shard):
            state["last"] = [torch.zeros(d)] * n
            state["sum"] = [torch.zeros(d) for _ in range(n)]
            state["cnt"] = [0] * n
            text, kv = _run_dialogue(pipeline, processor, q["id"],
                                     max_tokens=512, layer_capture=cap)
            pre, m_pre = _ptrue_from_kv(pipeline, kv, PTRUE_PRE_AU, YES, NO)
            post, m_post = _ptrue_from_kv(
                pipeline, kv, PTRUE_POST_AU.format(a=text[:4000]), YES, NO)
            del kv
            hl = torch.stack(state["last"])
            hm = torch.stack([state["sum"][i] / max(state["cnt"][i], 1)
                              for i in range(n)])
            sig_rows.append({**{f: q[f] for f in
                                ("id", "pool", "source", "query",
                                 "reference_answer", "split")},
                             "answer": text, "n_forward": 0,
                             "entropy": [], "margin": [],
                             "h_prompt": hl[-1].tolist(), "h_mean8": None})
            pt_rows.append({"id": q["id"], "p_yes_pre": pre,
                            "p_yes_post": post, "mass_pre": m_pre,
                            "mass_post": m_post})
            ids.append(q["id"])
            lasts.append(hl.numpy().astype(np.float16))
            means.append(hm.numpy().astype(np.float16))
            if k < 3 or k % 25 == 0:
                print(f"  [{k}] {q['pool']} pre={pre:.3f} post={post:.3f} "
                      f":: {text[:60]!r}", flush=True)
    finally:
        for h in handles:
            h.remove()

    pd.DataFrame(sig_rows).to_parquet(
        f"{DATA}/signals_{FZ_TAG}.shard{shard_id}.parquet")
    pd.DataFrame(pt_rows).to_parquet(
        f"{DATA}/ptrue_{FZ_TAG}.shard{shard_id}.parquet")
    np.savez_compressed(f"{DATA}/layers_{FZ_TAG}.shard{shard_id}.npz",
                        ids=np.array(ids), h_last=np.stack(lasts),
                        h_mean=np.stack(means))
    gate_data.commit()
    print(f">>> wrote freeze shard {shard_id} ({len(ids)} rows)", flush=True)
    return len(ids)


@app.local_entrypoint()
def run_freeze(workers: int = 4, limit: int = 0):
    qs = _load_queries.remote()
    if limit:
        qs = qs[:limit]
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {FZ_TAG}: {len(qs)} queries / {workers} H100 workers")
    total = sum(collect_freeze.starmap(
        [(shards[i], i) for i in range(workers)]))
    print(f">>> collected freeze-omni audio for {total} queries")
