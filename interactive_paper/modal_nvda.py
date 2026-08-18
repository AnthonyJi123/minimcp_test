"""NVIDIA NemotronLabs-VoiceChat-11B transfer arm (2026-08-18, Jisen #3).

The §9 pre-registered second-duplex-family test: does the frozen-gate
METHODOLOGY (not the frozen weights — a new backbone needs its own
probe) transfer to a different open full-duplex model?

Pipeline (offline replay, one H100 pass gives features AND labels):
  1. download   : 44 GB HF checkpoint → nvda-weights Volume
  2. smoke      : load NemotronVoiceChat, run one wav, print the LLM
                  backbone module tree + hook a layer to verify shapes
  3. answer_shard : per pool — offline inference per wav; store the
                  agent text (→ local-floor label after judging) and
                  the eoth2-format hidden reads (last-K window at the
                  end of user audio + running mean over user-audio
                  frames) at NVDA_LAYERS
  4. judging + probe fit + layer sweep run LOCALLY / via modal_train2
                  conventions on the stored npz (CPU-only forever)

Backbone: Nemotron-Nano-9B-v2, hybrid Mamba2/attention — layer
indices are NOT comparable to MiniCPM's; the sweep restarts from
scratch (that is the point of the test).

Order:
  modal run modal_nvda.py::download
  modal run modal_nvda.py::smoke
  modal run modal_nvda.py::run_answers --pools striviaqa,... --limit 0
"""
import json
import os

import modal

app = modal.App("nvda-voicechat")

weights = modal.Volume.from_name("nvda-weights", create_if_missing=True)
gate_data = modal.Volume.from_name("gate-data")
DATA = "/data"
MODEL_DIR = "/workspace/nvda/NVIDIA-NemotronLabs-VoiceChat-11B"
VOLS = {"/workspace/nvda": weights, DATA: gate_data}
HF_REPO = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"

# filled from the model-card recipe (README 2026-08-18); NeMo Speech
# branch nemotron-labs-voicechat is the only supported loading path
nemo_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04",
                              add_python="3.11")
    .apt_install("git", "libsndfile1", "ffmpeg", "sox")
    .pip_install("torch==2.10.0", "torchvision==0.25.0",
                 "torchaudio==2.10.0")
    .run_commands(
        "git clone --depth 1 -b nemotron-labs-voicechat "
        "https://github.com/NVIDIA-NeMo/Speech.git /opt/nemo-speech",
        # [all] pulls ASR-eval C extensions (cdifflib/texterrors/pesq)
        # that fail to build and aren't needed for speechlm2 — install
        # core + the collection's actual deps instead
        "cd /opt/nemo-speech && pip install -e .",
        "pip install hydra-core omegaconf 'lightning==2.4.0' "
        "sentencepiece webdataset braceexpand editdistance "
        "sacremoses inflect")
    .pip_install("transformers==4.56.0", "tokenizers==0.22.0",
                 "lhotse==1.32.2", "huggingface-hub==0.34.4",
                 "hf-xet==1.1.9", "torchcodec==0.10.0",
                 "torch_audiomentations", "jinja2",
                 "ninja", "packaging", "wheel", "einops")
    .run_commands(
        "pip install --no-build-isolation --no-deps "
        "causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1")
    .pip_install("librosa", "soundfile", "pandas", "pyarrow",
                 "matplotlib", "pyloudnorm", "seaborn", "unidecode",
                 "attrdict3", "pypinyin", "nltk", "jieba",
                 "jiwer", "kaldiio", "pydub", "sox",
                 "pyannote.core", "pyannote.metrics", "sacrebleu",
                 "datasets", "whisper_normalizer", "ipython",
                 "editdistance", "resampy", "wandb",
                 "praat-parselmouth", "torchdiffeq", "pystoi",
                 "peft", "accelerate", "megatron-core")
    .env({"HF_HUB_DISABLE_XET": "1"})
)

dl_image = (modal.Image.debian_slim(python_version="3.11")
            .pip_install("huggingface_hub[hf_transfer]")
            .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))

# same recipe as modal_train2 eoth2: last-K window at the eot read +
# running mean over user-audio positions. 56-layer hybrid backbone
# (27x Mamba2 / 4x attn / 25x MLP), hidden 4480 — every 4th layer.
# Nemotron runs CACHELESS (full prefix per 80 ms frame), so the final
# frame's forward contains every position: capture once, slice.
K_EOT = 8
NVDA_LAYERS = list(range(2, 56, 4))          # 14 layers
FRAME_S = 0.08
TAIL_SIL_S = 12.0        # answer room appended to each query wav
SYS_PROMPT = ("You are a helpful voice assistant. Listen to the "
              "user's question and answer it directly and concisely. "
              "Do not greet the user; wait for the question.")


@app.function(image=dl_image, volumes={"/workspace/nvda": weights},
              timeout=60 * 60 * 2)
def download():
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, local_dir=MODEL_DIR)
    weights.commit()
    print("downloaded to", MODEL_DIR)


def _load_model(device="cuda"):
    import sys
    sys.path.insert(0, "/opt/nemo-speech")
    from nemo.collections.speechlm2.inference.utils.offline_voicechat         import build_model
    import torch
    m = build_model(MODEL_DIR, device=device)
    # bf16 on the STT stack only (perception + LLM): halves weight
    # bandwidth on the cacheless per-frame full-prefix forwards; the
    # TTS stack stays fp32 (its audio quality is irrelevant here but
    # its numerics are untested in bf16)
    m.stt_model.to(torch.bfloat16)
    return m


def _infer_batch(model, wav_paths, capture=True):
    """Batched offline inference (+ appended silence), capturing at
    NVDA_LAYERS from the FINAL cacheless forward. Batching is the big
    lever: the per-frame full-prefix forwards are bandwidth-bound at
    B=1. Returns a list of dicts (text, secs, eot, mean, ...)."""
    import re
    import time
    import numpy as np
    import librosa
    import torch
    from nemo.collections.speechlm2.inference.utils.offline_voicechat         import encode_system_prompt, run_offline_inference

    B = len(wav_paths)
    aus = []
    for w in wav_paths:
        au, _ = librosa.load(w, sr=16000, mono=True)
        aus.append(au)
    qlens = [len(a) for a in aus]
    tail = int(TAIL_SIL_S * 16000)
    full = max(qlens) + tail
    sig = torch.zeros(B, full)
    for b, a in enumerate(aus):
        sig[b, :len(a)] = torch.tensor(a)
    sig = sig.cuda()
    lens = torch.full((B,), full, dtype=torch.long, device="cuda")

    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    # LM-frame count of each QUERY (pre-silence) via the perception
    # front-end alone, true lengths
    with torch.no_grad(), amp:
        q_sig = torch.zeros(B, max(qlens), device="cuda")
        for b, a in enumerate(aus):
            q_sig[b, :len(a)] = torch.tensor(a, device="cuda")
        q_len = torch.tensor(qlens, dtype=torch.long, device="cuda")
        out = model.stt_model.perception(
            input_signal=q_sig, input_signal_length=q_len)
        n_frames_q = [int(x) for x in out[1]]

    prompt_tokens, prompt_token_lens = encode_system_prompt(
        model, SYS_PROMPT, device="cuda")
    if prompt_tokens.shape[0] == 1 and B > 1:
        prompt_tokens = prompt_tokens.expand(B, -1).contiguous()
        prompt_token_lens = prompt_token_lens.expand(B).contiguous()
    prompt_len = int(prompt_token_lens[0].item())

    store = {}
    handles = []
    if capture:
        def mk(L):
            def hook(_m, _i, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                store[L] = hs.detach()            # (B, T_cur, d) GPU
            return hook
        handles = [model.stt_model.llm.layers[L].register_forward_hook(
            mk(L)) for L in NVDA_LAYERS]

    t0 = time.time()
    try:
        with amp:
            result = run_offline_inference(
                model, input_signal=sig, input_signal_lens=lens,
                prompt_tokens=prompt_tokens,
                prompt_token_lens=prompt_token_lens)
    finally:
        for h in handles:
            h.remove()
    secs = time.time() - t0

    texts = result.get("text", [""] * B)
    outs = []
    for b in range(B):
        t_end = prompt_len + n_frames_q[b]
        eot = mean = None
        if capture and store:
            d = store[NVDA_LAYERS[0]].shape[-1]
            eot = np.zeros((len(NVDA_LAYERS), K_EOT, d),
                           dtype=np.float16)
            mean = np.zeros((len(NVDA_LAYERS), d), dtype=np.float16)
            for j, L in enumerate(NVDA_LAYERS):
                h = store[L][b].float()           # (T, d)
                hi = min(t_end, h.shape[0])
                lo = max(prompt_len, hi - K_EOT)
                w = h[lo:hi].cpu().numpy().astype(np.float16)
                eot[j, K_EOT - w.shape[0]:] = w
                mean[j] = (h[prompt_len:hi].mean(0).cpu().numpy()
                           .astype(np.float16))
        raw = texts[b] if b < len(texts) else ""
        clean = re.sub(r"<[^>]{0,24}>", " ", raw)
        clean = re.sub(r"  +", " ", clean).strip()
        outs.append(dict(text=clean, text_raw=raw,
                         secs=secs / B, batch_secs=secs,
                         n_frames_query=n_frames_q[b],
                         prompt_len=prompt_len, t_end=t_end,
                         eot=eot, mean=mean))
    store.clear()
    return outs


def _infer_one(model, wav_path, capture=True):
    return _infer_batch(model, [wav_path], capture=capture)[0]


@app.function(image=nemo_image, gpu="H100", volumes=VOLS,
              timeout=60 * 60)
def smoke(wav: str = ""):
    """Load, print backbone summary, run one wav, report shapes."""
    import glob
    import time
    import torch

    t0 = time.time()
    model = _load_model()
    print(f"loaded in {time.time() - t0:.0f}s")
    layers = model.stt_model.llm.layers
    print("backbone:", type(model.stt_model.llm).__name__,
          "| n_layers:", len(layers),
          "| block:", type(layers[0]).__name__,
          "| dtype:", next(model.stt_model.llm.parameters()).dtype)

    if not wav:
        cands = sorted(glob.glob(f"{DATA}/bench_audio/*.wav"))
        print("bench_audio wavs:", len(cands), cands[:3])
        wav = cands[0]
    r = _infer_one(model, wav)
    cands8 = sorted(glob.glob(f"{DATA}/bench_audio/sllama00*.wav"))[:8]
    rs = _infer_batch(model, cands8)
    print(f"batch8: {rs[0]['batch_secs']:.0f}s total, "
          f"{rs[0]['secs']:.1f}s/q")
    for x in rs[:4]:
        print("   ", repr(x["text"])[:90])
    print(f"wav={wav}")
    print(f"query frames={r['n_frames_query']} prompt={r['prompt_len']} "
          f"t_end={r['t_end']} infer={r['secs']:.1f}s")
    print("text:", repr(r["text"])[:400])
    print("eot shape:", None if r["eot"] is None else r["eot"].shape,
          "| mean shape:", None if r["mean"] is None else r["mean"].shape)
    import numpy as np
    if r["eot"] is not None:
        print("eot finite:", bool(np.isfinite(r["eot"]).all()),
              "| nonzero rows:",
              int((np.abs(r["eot"][0]).sum(-1) > 0).sum()))
    return r["text"]

# ---------------------------------------------------------------- bulk ----
# pool tag -> (wav dir, queries jsonl) on the gate-data volume; mirrors
# modal_train.AUDIO_DIRS / Q_FILES so features + labels stay same-audio
POOLS = {
    "frozen":     (f"{DATA}/audio_pool",      f"{DATA}/queries.jsonl"),
    "expansion":  (f"{DATA}/audio_expansion", f"{DATA}/queries_expansion.jsonl"),
    "expansion2": (f"{DATA}/audio_expansion2",
                   f"{DATA}/queries_expansion2.jsonl"),
    "striviaqa":  (f"{DATA}/bench_audio",     f"{DATA}/queries_striviaqa.jsonl"),
    "swebq":      (f"{DATA}/bench_audio",     f"{DATA}/queries_swebq.jsonl"),
    "sllama":     (f"{DATA}/bench_audio",     f"{DATA}/queries_sllama.jsonl"),
    "sreason":    (f"{DATA}/bench_audio",     f"{DATA}/queries_sreason.jsonl"),
    "sdqa":       (f"{DATA}/sdqa_audio",      f"{DATA}/queries_sdqa.jsonl"),
}
NVDA_H = f"{DATA}/nvda_h"                    # + _{tag}.shard{i}.npz
NVDA_ANS = f"{DATA}/nvda_answers"            # + _{tag}.shard{i}.jsonl


def _read_jsonl_local(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@app.function(image=nemo_image, gpu="H100", volumes=VOLS,
              timeout=60 * 60 * 4, max_containers=8)
def answer_shard(tag: str, shard: list, shard_id: int) -> int:
    """Offline NVDA inference per wav: agent text (-> local-floor label
    after judging) + eoth2-format hidden reads in the same pass."""
    import numpy as np

    model = _load_model()
    adir, _ = POOLS[tag]
    items = [(q, f"{adir}/{q['id']}.wav") for q in shard]
    items = [(q, w) for q, w in items if os.path.exists(w)]
    items.sort(key=lambda t: os.path.getsize(t[1]))   # length buckets
    BATCH = 8
    ids, E, M, rows = [], [], [], []
    for k in range(0, len(items), BATCH):
        chunk = items[k:k + BATCH]
        try:
            rs = _infer_batch(model, [w for _, w in chunk])
        except Exception as e:
            print(f"  !! batch@{k}: {type(e).__name__}: {str(e)[:150]}",
                  flush=True)
            continue
        for (q, _), r in zip(chunk, rs):
            if r["eot"] is None:
                continue
            ids.append(q["id"])
            E.append(r["eot"])
            M.append(r["mean"])
            rows.append({"id": q["id"], "answer": r["text"],
                         "answer_raw": r["text_raw"],
                         "secs": round(r["secs"], 2),
                         "n_frames_query": r["n_frames_query"]})
        if k == 0 or (k // BATCH) % 4 == 0:
            r0 = rs[0]
            print(f"  [{k}/{len(items)}] batch {r0['batch_secs']:.0f}s "
                  f"({r0['secs']:.1f}s/q) {repr(r0['text'])[:70]}",
                  flush=True)
    np.savez_compressed(f"{NVDA_H}_{tag}.shard{shard_id}.npz",
                        ids=np.array(ids), H_eot=np.stack(E),
                        H_mean=np.stack(M),
                        layers=np.array(NVDA_LAYERS))
    with open(f"{NVDA_ANS}_{tag}.shard{shard_id}.jsonl", "w",
              encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    gate_data.commit()
    print(f">>> wrote nvda_{tag} shard {shard_id} ({len(ids)})", flush=True)
    return len(ids)


@app.local_entrypoint()
def run_answers(tags: str = "striviaqa,swebq,sllama,sreason,sdqa,frozen",
                workers: int = 6, limit: int = 0):
    for tag in tags.split(","):
        tag = tag.strip()
        _, qfile = POOLS[tag]
        qs = _read_q.remote(qfile)
        if limit:
            qs = qs[:limit]
        w = min(workers, max(1, len(qs) // 25))
        shards = [qs[i::w] for i in range(w)]
        print(f">>> {tag}: {len(qs)} queries / {w} workers")
        total = sum(answer_shard.starmap(
            [(tag, shards[i], i) for i in range(w)]))
        print(f">>> {tag}: answered {total}")


@app.function(image=dl_image, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_q(path: str) -> list:
    return _read_jsonl_local(path)

# ---------------------------------------------------------------- judge ---
# reuse the project judges verbatim (no protocol drift): ours =
# src/escalate.judge_many (gpt-5.4-mini ref-anchored, OPENAI_API_KEY
# secret); OAB pools additionally get the official OpenAudioBench
# judge from modal_bench (run separately if needed)
judge_image = (modal.Image.debian_slim(python_version="3.11")
               .pip_install("openai", "pandas", "pyarrow")
               .add_local_dir("src", "/workspace/gate"))
OPENAI = modal.Secret.from_name("openai")


@app.function(image=judge_image, volumes={DATA: gate_data},
              secrets=[OPENAI], timeout=60 * 60)
def judge(tags: str = "striviaqa,swebq,sllama,sreason,sdqa,frozen",
          concurrency: int = 8):
    """Judge NVDA answers vs references -> nvda_{tag}.parquet with
    escalate_label (NVDA-specific never-arm fail)."""
    import asyncio
    import glob
    import sys
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    for tag in tags.split(","):
        tag = tag.strip()
        shards = sorted(glob.glob(f"{NVDA_ANS}_{tag}.shard*.jsonl"))
        if not shards:
            print(f"{tag}: no answer shards, skipped")
            continue
        rows = []
        for sh in shards:
            rows += _read_jsonl_local(sh)
        qs = {q["id"]: q for q in _read_jsonl_local(POOLS[tag][1])}
        jr = [{"id": r["id"], "query": qs[r["id"]].get("query")
               or qs[r["id"]].get("text"),
               "reference_answer": qs[r["id"]].get("reference_answer")
               or qs[r["id"]].get("reference"),
               "answer": r["answer"]} for r in rows if r["id"] in qs]
        labeled = asyncio.run(
            escalate.judge_many(jr, concurrency=concurrency))
        df = pd.DataFrame(rows).merge(
            pd.DataFrame([{"id": r["id"], "adequate": x["adequate"],
                           "escalate_label": x["escalate_label"]}
                          for r, x in zip(jr, labeled)]), on="id")
        df.to_parquet(f"{DATA}/nvda_{tag}.parquet")
        ok = df[df["adequate"].notna()]
        print(f"{tag}: n={len(df)} judged={len(ok)} "
              f"NVDA-fail-rate={1 - ok['adequate'].mean():.3f}")
    gate_data.commit()

# ------------------------------------------------------------ probe fit ---
fit_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("scikit-learn", "pandas", "pyarrow", "numpy"))


@app.function(image=fit_image, volumes={DATA: gate_data}, timeout=60 * 30)
def fit(calib_tags: str = "frozen", ext_tags: str =
        "striviaqa,swebq,sllama,sreason,sdqa"):
    """Layer sweep + probe fit on the NVDA hidden reads. Mirrors the
    v3 recipe: features per layer = eot_last / eot_mean8 / user_mean,
    logistic C=1e-4 on standardized features, 5-fold OOF on calib,
    straight transfer AUC on the externals. CPU-only."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def load_tag(tag):
        shards = sorted(glob.glob(f"{NVDA_H}_{tag}.shard*.npz"))
        ids, E, M = [], [], []
        for sh in shards:
            z = np.load(sh, allow_pickle=True)
            ids += list(z["ids"])
            E.append(z["H_eot"]); M.append(z["H_mean"])
        if not ids:
            return None
        lab = pd.read_parquet(f"{DATA}/nvda_{tag}.parquet")                 .set_index("id")["escalate_label"]
        E, M = np.concatenate(E), np.concatenate(M)
        keep = [i for i, q in enumerate(ids)
                if q in lab.index and pd.notna(lab.get(q))]
        y = np.array([int(lab[ids[i]]) for i in keep])
        return E[keep].astype(np.float32), M[keep].astype(np.float32), y

    def feats(E, M, j, modes):
        parts = []
        if "eot_last" in modes:
            parts.append(E[:, j, -1])
        if "eot_mean8" in modes:
            parts.append(E[:, j].mean(1))
        if "user_mean" in modes:
            parts.append(M[:, j])
        return np.concatenate(parts, axis=1) if len(parts) > 1             else parts[0]

    def clf():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1e-4, max_iter=4000))

    cal = [load_tag(t) for t in calib_tags.split(",")]
    cal = [c for c in cal if c is not None]
    Ec = np.concatenate([c[0] for c in cal])
    Mc = np.concatenate([c[1] for c in cal])
    yc = np.concatenate([c[2] for c in cal])
    print(f"calib n={len(yc)} fail-rate={yc.mean():.3f}")

    ext = {}
    for t in ext_tags.split(","):
        r = load_tag(t.strip())
        if r is not None:
            ext[t.strip()] = r
            print(f"{t}: n={len(r[2])} fail-rate={r[2].mean():.3f}")

    out = {"layers": {}, "combos": {}}
    kf = StratifiedKFold(5, shuffle=True, random_state=42)

    def oof_auc(X, y):
        p = np.zeros(len(y))
        for tr, te in kf.split(X, y):
            m = clf().fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        return roc_auc_score(y, p)

    layers = list(NVDA_LAYERS)
    print(chr(10)+"== layer sweep (eot_last, OOF AUC on calib) ==")
    for j, L in enumerate(layers):
        a = oof_auc(feats(Ec, Mc, j, ["eot_last"]), yc)
        out["layers"][L] = round(float(a), 4)
        print(f"  L{L:2d}: {a:.3f}")

    best_j = int(np.argmax([out["layers"][L] for L in layers]))
    bestL = layers[best_j]
    print(chr(10)+f"best layer L{bestL}")
    combos = [("eot_last", ["eot_last"]),
              ("eot_last+eot_mean8", ["eot_last", "eot_mean8"]),
              ("eot_last+eot_mean8+user_mean",
               ["eot_last", "eot_mean8", "user_mean"])]
    for name, modes in combos:
        a = oof_auc(feats(Ec, Mc, best_j, modes), yc)
        row = {"oof": round(float(a), 4), "ext": {}}
        m = clf().fit(feats(Ec, Mc, best_j, modes), yc)
        for t, (Ee, Me, ye) in ext.items():
            if len(set(ye)) < 2:
                continue
            pa = m.predict_proba(feats(Ee, Me, best_j, modes))[:, 1]
            row["ext"][t] = round(float(roc_auc_score(ye, pa)), 4)
        out["combos"][name] = row
        print(f"{name}: OOF {a:.3f} ext {row['ext']}")

    with open(f"{DATA}/nvda_probe_sweep.json", "w") as f:
        json.dump(out, f, indent=1)
    gate_data.commit()
    return out

