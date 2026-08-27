# -*- coding: utf-8 -*-
"""8ax: the two GPU experiments the 8aw CPU battery could not run.

exp3  control-token logit lens: apply each model's own final norm +
      lm_head to the STORED per-layer states (Phase-5d dumps, no new
      forwards) and measure how much of the vocabulary distribution
      sits on added/special control tokens by depth, plus the alignment
      between the (inverting) final-layer probe direction and the
      control tokens' unembedding rows.
exp6  listen/speak prompt intervention (causal): same calib queries
      with a neutral / still-listening / answer-now suffix, all-layer
      last-token capture on the duplex model AND its raw backbone; the
      cue direction's alignment with the probe is analyzed locally.

Run:
  modal run modal_interp.py::run_logitlens --tag minicpm-o45
  modal run modal_interp.py::run_logitlens --tag qwen3-8b
  modal run modal_interp.py::run_cues --tag minicpm-o45
  modal run modal_interp.py::run_cues --tag qwen3-8b
"""
import json
import os
import sys

import modal

# self-contained copies of modal_app's environment (the container mounts
# only this file, so importing modal_app there fails)
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"
DATA = "/data"
weights = modal.Volume.from_name("minicpm-o45-weights")
gate_data = modal.Volume.from_name("gate-data")
GPU_VOL = {"/workspace/models": weights, DATA: gate_data}

image = (
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
    )
    .add_local_dir(os.path.join(HERE, "src"), "/workspace/gate")
)

app = modal.App("think-gate-interp")


def _load_model(model_dir: str = MODEL_DIR):
    import glob as _glob
    import shutil
    import torch
    from transformers import AutoModel, AutoTokenizer
    cache = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/"
        + os.path.basename(model_dir))
    os.makedirs(cache, exist_ok=True)
    for f in _glob.glob(f"{model_dir}/*.py"):
        shutil.copy(f, cache)
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=False, init_tts=False,
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tok

CUES = {
    "neutral": "",
    "listen": "\n\nWait - I haven't finished my question yet. "
              "Do not answer; keep listening.",
    "speak": "\n\nThat's my whole question. Answer right now.",
}


def _control_ids(tok):
    """Added-vocab + special tokens = the trained-in control interface."""
    ctrl = dict(tok.get_added_vocab())
    for t, i in zip(tok.all_special_tokens, tok.all_special_ids):
        ctrl.setdefault(t, i)
    return ctrl


def _lens_one(tag, model_dir, get_parts):
    import glob

    import numpy as np
    import torch

    model, tok = get_parts(model_dir)
    if hasattr(model, "llm"):                      # MiniCPM-o remote code
        norm, head = model.llm.model.norm, model.llm.lm_head
    else:
        norm, head = model.model.norm, model.lm_head
    ctrl = _control_ids(tok)
    cids = torch.tensor(sorted(set(ctrl.values()))).cuda()
    print(f">>> {tag}: {len(cids)} control tokens:",
          sorted(ctrl)[:40], flush=True)

    ids, hs = [], []
    for s in sorted(glob.glob(f"{DATA}/layers_{tag}.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hs.append(z["h_last"])
    H = np.concatenate(hs)                          # (n, L, d) fp16
    n, L, d = H.shape
    print(f">>> states {H.shape}", flush=True)

    mass = np.zeros((n, L), np.float32)             # log P(control)
    argmax_ctrl = np.zeros((n, L), bool)
    top_tokens = []
    with torch.no_grad():
        for li in range(L):
            h = torch.from_numpy(H[:, li, :].astype(np.float32)) \
                .cuda().to(torch.bfloat16)
            logits = head(norm(h)).float()          # (n, V)
            lse_all = torch.logsumexp(logits, -1)
            lse_ctl = torch.logsumexp(logits[:, cids], -1)
            mass[:, li] = (lse_ctl - lse_all).cpu().numpy()
            am = logits.argmax(-1)
            argmax_ctrl[:, li] = torch.isin(am, cids).cpu().numpy()
            vals, cnts = am.unique(return_counts=True)
            order = cnts.argsort(descending=True)[:8]
            top_tokens.append([(tok.decode([int(vals[i])]),
                                int(cnts[i])) for i in order])
            if li % 8 == 0 or li == L - 1:
                print(f"  L{li:02d} mean log-mass {mass[:, li].mean():.2f} "
                      f"argmax-in-ctrl {argmax_ctrl[:, li].mean():.2f} "
                      f"top {top_tokens[-1][:3]}", flush=True)

    # unembedding rows of the control tokens + the norm's scale vector,
    # so probe-direction alignment can be computed locally in either space
    W_ctrl = head.weight[cids.cpu()].detach().float().cpu().numpy()
    g = norm.weight.detach().float().cpu().numpy()
    rng = np.random.default_rng(0)
    W_rand = head.weight[torch.tensor(
        rng.integers(0, head.weight.shape[0], 256))] \
        .detach().float().cpu().numpy()

    np.savez_compressed(
        f"{DATA}/logitlens_{tag}.npz", ids=np.array(ids), mass=mass,
        argmax_ctrl=argmax_ctrl, W_ctrl=W_ctrl.astype(np.float16),
        W_rand=W_rand.astype(np.float16), norm_g=g.astype(np.float16),
        ctrl_tokens=np.array(sorted(ctrl), dtype=object),
        ctrl_ids=cids.cpu().numpy())
    with open(f"{DATA}/logitlens_{tag}_top.json", "w") as f:
        json.dump(top_tokens, f)
    gate_data.commit()
    print(f">>> wrote logitlens_{tag}.npz + _top.json", flush=True)
    return int(n)


@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 40)
def logit_lens_mo(tag: str) -> int:
    return _lens_one(tag, MODEL_DIR, _load_model)


@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 40)
def logit_lens_hf(tag: str) -> int:
    def load(model_dir):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            f"/workspace/models/{tag}", torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").eval().cuda()
        tok = AutoTokenizer.from_pretrained(f"/workspace/models/{tag}")
        return model, tok
    return _lens_one(tag, None, lambda _: load(None))


def _cue_loop(tag, model, tok, layer_modules, build_run):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import layers as layers_mod

    df = pd.read_parquet(f"{DATA}/calib_features.parquet")
    df = df[(df["split"] == "calib") & df["escalate_label"].notna()]
    qs = (df.groupby("pool", group_keys=False)
          .apply(lambda g: g.sort_values("id").head(12))
          [["id", "pool", "query"]].to_dict("records"))
    print(f">>> {tag}: {len(qs)} queries x {len(CUES)} cue variants",
          flush=True)

    ids, hset = [], {v: [] for v in CUES}
    for k, q in enumerate(qs):
        ids.append(q["id"])
        for v, suffix in CUES.items():
            hl, _ = layers_mod.capture_prefill_layers(
                layer_modules, build_run(q["query"] + suffix))
            hset[v].append(hl.numpy().astype(np.float16))
        if k < 2 or k % 10 == 0:
            print(f"  [{k}] {q['pool']}", flush=True)

    np.savez_compressed(
        f"{DATA}/cues_{tag}.npz", ids=np.array(ids),
        **{f"h_{v}": np.stack(hset[v]) for v in CUES})
    gate_data.commit()
    print(f">>> wrote cues_{tag}.npz", flush=True)
    return len(ids)


@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def cue_capture_mo(tag: str) -> int:
    sys.path.insert(0, "/workspace/gate")
    import decode
    model, tok = _load_model(MODEL_DIR)
    kw = decode._chat_kwargs(model, tok)

    def build_run(text):
        return lambda: model.chat(msgs=[{"role": "user", "content": [text]}],
                                  max_new_tokens=1, **kw)
    return _cue_loop(tag, model, tok, list(model.llm.model.layers), build_run)


@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def cue_capture_hf(tag: str) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import hf_decode
    model = AutoModelForCausalLM.from_pretrained(
        f"/workspace/models/{tag}", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").eval().cuda()
    tok = AutoTokenizer.from_pretrained(f"/workspace/models/{tag}")

    def build_run(text):
        inputs = hf_decode._build_inputs(model, tok, text)
        return lambda: model(**inputs)
    return _cue_loop(tag, model, tok, list(model.model.layers), build_run)


@app.local_entrypoint()
def run_logitlens(tag: str):
    fn = logit_lens_mo if tag.startswith("minicpm") else logit_lens_hf
    print("done, n =", fn.remote(tag))


@app.local_entrypoint()
def run_cues(tag: str):
    fn = cue_capture_mo if tag.startswith("minicpm") else cue_capture_hf
    print("done, n =", fn.remote(tag))
