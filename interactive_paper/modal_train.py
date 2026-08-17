"""Probe v2 training (user go-ahead 2026-08-13): expand the calib pool
from public benchmarks and retrain the L22 gate ON THE DEPLOYED SIGNAL
(the streaming end-of-turn hidden), fixing two diagnosed gaps at once:
  1. calib coverage (360 queries / 5 pools -> ~800 more, 2 new families)
  2. train/deploy mismatch (v1 trained on chat-style full-prefill
     h_last, deployed on the eot read)
External eval pools (OpenAudioBench striviaqa/swebq rows we evaluate on,
SD-QA) stay strictly OUT of training. Frozen 600 pool untouched; the new
artifact is written as midlayer_gate_audio_v2.json (v1 intact).

Stages (run in order):
  modal run modal_train.py::build_expansion
  modal run modal_train.py::run_tts
  modal run modal_train.py::run_label
  modal run modal_train.py::run_eoth --tag expansion   (then frozen,
        striviaqa, swebq, sdqa)
  modal run modal_train.py::refit
  modal run modal_train.py::eval_transfer
"""
import json
import os
import re
import sys
import time

import modal

from modal_app import (app, gen_app, image, util_image, GPU_VOL, gate_data,
                       DATA, OPENAI, API_REGION, QUERIES, _read_jsonl)

HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(HERE, "modal_app.py")
image_st = image.add_local_file(_APP_PY, "/root/modal_app.py")
util_st = util_image.add_local_file(_APP_PY, "/root/modal_app.py")
dl_image = (modal.Image.debian_slim(python_version="3.11")
            .pip_install("datasets==2.21.0", "pandas", "pyarrow")
            .env({"HF_HUB_DISABLE_XET": "1"})
            .add_local_file(_APP_PY, "/root/modal_app.py"))

XQ = f"{DATA}/queries_expansion.jsonl"
XAUDIO = f"{DATA}/audio_expansion"
XLABELS = f"{DATA}/expansion_labels.parquet"
EOTH = f"{DATA}/eoth"                    # + _{tag}.shard{i}.npz
ART_V2 = f"{DATA}/midlayer_gate_audio_v2.json"
GATE_ART = f"{DATA}/midlayer_gate_audio.json"
LAYER = 22
TTS_VOICE = "alloy"

# 8q speakable filter (input-side)
_BSLASH = re.compile(r"\\[A-Za-z]{2,}")
_CARET = re.compile(r"\^\{|_\{|\^\(|\^-?\w")
_DOLLAR = re.compile(r"\$([^$]+)\$")


def speakable(q):
    if _BSLASH.search(q) or _CARET.search(q):
        return False
    m = _DOLLAR.search(q)
    if m and re.search(r"[\\^_=]", m.group(1)):
        return False
    return 30 <= len(q) <= 600


@app.function(image=dl_image, volumes={DATA: gate_data}, timeout=60 * 40)
def build_expansion():
    """~800 new calib queries from public benchmarks (no LLM-generated
    content — [[no-selfmade-datasets]]), speakable-filtered, deduped
    against the frozen pool. New families: NQ-open + ARC-Challenge."""
    import numpy as np
    from datasets import load_dataset

    rng = np.random.default_rng(43)          # NOT 42: decorrelate from eval
    frozen = {re.sub(r"\s+", " ", q["query"].lower())[:120]
              for q in _read_jsonl(QUERIES)}

    def fresh(text):
        key = re.sub(r"\s+", " ", str(text).lower())[:120]
        if key in frozen:
            return False
        frozen.add(key)
        return True

    def take(it, n, fmt):
        rows = []
        for ex in it:
            r = fmt(ex)
            if r is None:
                continue
            q, ref = r
            if not speakable(q) or not fresh(q):
                continue
            rows.append((q, ref))
            if len(rows) >= n:
                break
        return rows

    out = []

    def add(pool, rows):
        for q, ref in rows:
            out.append({"id": f"x{len(out):04d}", "pool": pool,
                        "query": q, "reference_answer": ref,
                        "split": "calib_x"})
        print(f">>> {pool}: +{len(rows)}", flush=True)

    tq = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext",
                      split="validation")
    idx = rng.permutation(len(tq))
    add("easy-fact", take((tq[int(i)] for i in idx), 150,
        lambda e: (e["question"], e["answer"]["value"])))

    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dl = [e for e in dolly if e["category"] in ("open_qa", "general_qa")
          and not e["context"]]
    idx = rng.permutation(len(dl))
    add("easy-chat", take((dl[int(i)] for i in idx), 150,
        lambda e: (e["instruction"], e["response"][:300])))

    sq = load_dataset("basicv8vc/SimpleQA", split="test")
    idx = rng.permutation(len(sq))
    add("trap", take((sq[int(i)] for i in idx), 100,
        lambda e: (e["problem"], e["answer"])))

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    idx = rng.permutation(len(gsm))
    add("hard-math", take((gsm[int(i)] for i in idx), 150,
        lambda e: (e["question"],
                   e["answer"].split("####")[-1].strip())))

    nq = load_dataset("google-research-datasets/nq_open",
                      split="validation")
    idx = rng.permutation(len(nq))
    add("know-open", take((nq[int(i)] for i in idx), 150,
        lambda e: (e["question"].rstrip("?") + "?",
                   "; ".join(e["answer"]))))

    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    idx = rng.permutation(len(arc))

    def arc_fmt(e):
        ch = e["choices"]
        opts = ", ".join(f"({l}) {t}" for l, t in zip(ch["label"],
                                                      ch["text"]))
        return (f"{e['question']} {opts}", e["answerKey"])
    add("know-arc", take((arc[int(i)] for i in idx), 100, arc_fmt))

    with open(XQ, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> wrote {len(out)} expansion queries -> {XQ}", flush=True)


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_x() -> list:
    return _read_jsonl(XQ)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 50, region=API_REGION)
def run_tts(limit: int = 0, concurrency: int = 8):
    """tts-1/alloy render (same engine as the frozen pool — exonerated
    in 8r, so consistency beats novelty)."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    qs = _read_jsonl(XQ)[:limit or None]
    os.makedirs(XAUDIO, exist_ok=True)
    client = OpenAI()

    def render(q):
        out = f"{XAUDIO}/{q['id']}.wav"
        if os.path.exists(out) and os.path.getsize(out) > 44:
            return "cached"
        text = q["query"][:4000]
        resp = client.audio.speech.create(model="tts-1", voice=TTS_VOICE,
                                          input=text, response_format="wav")
        with open(out, "wb") as fh:
            fh.write(resp.content)
        return "done"

    with ThreadPoolExecutor(concurrency) as ex:
        res = list(ex.map(render, qs))
    gate_data.commit()
    print(f">>> tts: {res.count('done')} rendered, {res.count('cached')} "
          f"cached / {len(qs)}", flush=True)


@app.function(image=image_st, gpu="H100", volumes=GPU_VOL, timeout=60 * 60 * 2)
def answer_shard(shard: list, shard_id: int) -> list:
    """MiniCPM answers each expansion query from AUDIO (6a collection
    style: content=[au], no text instruction)."""
    import glob as _glob
    import shutil
    import librosa
    import torch
    from transformers import AutoModel, AutoTokenizer

    from modal_app import MODEL_DIR
    cache = os.path.expanduser("~/.cache/huggingface/modules/"
                               "transformers_modules/"
                               + os.path.basename(MODEL_DIR))
    os.makedirs(cache, exist_ok=True)
    for f in _glob.glob(f"{MODEL_DIR}/*.py"):
        shutil.copy(f, cache)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=False,
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    import pandas as pd
    rows = []
    for k, q in enumerate(shard):
        au, _ = librosa.load(f"{XAUDIO}/{q['id']}.wav", sr=16000, mono=True)
        ans = model.chat(msgs=[{"role": "user", "content": [au]}],
                         tokenizer=tok, max_new_tokens=512, sampling=False,
                         use_tts_template=False, generate_audio=False)
        rows.append({"id": q["id"], "answer": str(ans).strip()})
        if k < 3 or k % 40 == 0:
            print(f"  [{k}] {q['id']} :: {rows[-1]['answer'][:60]!r}",
                  flush=True)
    # persist per shard: GPU work must survive a downstream failure
    pd.DataFrame(rows).to_parquet(
        f"{DATA}/expansion_answers.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote answers shard {shard_id} ({len(rows)})", flush=True)
    return len(rows)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 40, region=API_REGION)
def judge_expansion():
    """Reads the answer shards off the volume (cross-app .remote() does
    NOT hydrate, so answering and judging are separate `modal run`s)."""
    import glob as _glob
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    answers = pd.concat(
        [pd.read_parquet(s) for s in
         sorted(_glob.glob(f"{DATA}/expansion_answers.shard*.parquet"))],
        ignore_index=True).drop_duplicates(subset="id", keep="last")
    print(f">>> judging {len(answers)} answers", flush=True)
    byid = {q["id"]: q for q in _read_jsonl(XQ)}
    rows = [{"id": a["id"], "query": byid[a["id"]]["query"],
             "reference_answer": byid[a["id"]]["reference_answer"],
             "answer": a["answer"]} for _, a in answers.iterrows()]
    judged = asyncio.run(escalate.judge_many(rows, concurrency=8))
    df = pd.DataFrame(judged)
    df["pool"] = [byid[i]["pool"] for i in df["id"]]
    df.to_parquet(XLABELS)
    gate_data.commit()
    ok = df[df["adequate"].notna()]
    print(f"\n=== expansion labels (n={len(ok)}) ===", flush=True)
    for p, g in ok.groupby("pool"):
        print(f"  {p:12s} n={len(g):3d} fail-rate="
              f"{g['escalate_label'].mean():.2f}", flush=True)


@app.local_entrypoint()
def run_answer(workers: int = 3, limit: int = 0):
    """Then judge separately: modal run modal_train.py::judge_expansion"""
    qs = _read_x.remote()
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> answering {len(qs)} expansion queries / {workers} workers")
    total = sum(answer_shard.starmap(
        [(shards[i], i) for i in range(workers)]))
    print(f">>> answered {total}")


AUDIO_DIRS = {"expansion": XAUDIO, "frozen": f"{DATA}/audio_pool",
              "striviaqa": f"{DATA}/bench_audio",
              "swebq": f"{DATA}/bench_audio",
              "sdqa": f"{DATA}/sdqa_audio",
              "valpaca": f"{DATA}/bench_audio",
              "sllama": f"{DATA}/bench_audio",
              "sreason": f"{DATA}/bench_audio"}
Q_FILES = {"expansion": XQ, "frozen": QUERIES,
           "striviaqa": f"{DATA}/queries_striviaqa.jsonl",
           "swebq": f"{DATA}/queries_swebq.jsonl",
           "sdqa": f"{DATA}/queries_sdqa.jsonl",
           "valpaca": f"{DATA}/queries_valpaca.jsonl",
           "sllama": f"{DATA}/queries_sllama.jsonl",
           "sreason": f"{DATA}/queries_sreason.jsonl"}


@app.function(image=image_st, gpu="H100", volumes=GPU_VOL, timeout=60 * 60 * 2)
def eoth_shard(tag: str, shard: list, shard_id: int) -> int:
    """Streaming replay -> store the RAW eot-read hidden (L22 last token
    after the empty assistant turn), so probes can be refit and evaluated
    offline on every pool without further GPU."""
    import glob as _glob
    import inspect
    import shutil
    import numpy as np
    import librosa
    import torch
    from transformers import AutoModel, AutoTokenizer

    from modal_app import MODEL_DIR
    cache = os.path.expanduser("~/.cache/huggingface/modules/"
                               "transformers_modules/"
                               + os.path.basename(MODEL_DIR))
    os.makedirs(cache, exist_ok=True)
    for f in _glob.glob(f"{MODEL_DIR}/*.py"):
        shutil.copy(f, cache)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=False,
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    def call_def(fn, /, **kw):
        params = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in params})

    holder = {}

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        holder["h"] = hs[0, -1, :].detach().float().cpu().numpy()

    adir = AUDIO_DIRS[tag]
    ids, H = [], []
    for k, q in enumerate(shard):
        wav = f"{adir}/{q['id']}.wav"
        if not os.path.exists(wav):
            continue
        au, _ = librosa.load(wav, sr=16000, mono=True)
        chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        model.reset_session()
        sys_msg = call_def(model.get_sys_prompt, mode="omni", language="en")
        call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
                 tokenizer=tok)
        h = model.llm.model.layers[LAYER].register_forward_hook(hook)
        try:
            for i, ch in enumerate(chunks):
                if len(ch) < 16000:
                    ch = np.pad(ch, (0, 16000 - len(ch)))
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[{"role": "user",
                                "content": [ch.astype(np.float32)]}],
                         tokenizer=tok,
                         is_last_chunk=(i == len(chunks) - 1))
            got = None
            for content in ("", " "):
                try:
                    call_def(model.streaming_prefill, session_id="s1",
                             msgs=[{"role": "assistant",
                                    "content": [content]}],
                             tokenizer=tok, is_last_chunk=True)
                    got = holder["h"]
                    break
                except Exception:
                    continue
        finally:
            h.remove()
        if got is None:
            continue
        ids.append(q["id"])
        H.append(got.astype(np.float32))
        if k < 3 or k % 50 == 0:
            print(f"  [{k}] {q['id']}", flush=True)

    np.savez_compressed(f"{EOTH}_{tag}.shard{shard_id}.npz",
                        ids=np.array(ids), H=np.stack(H))
    gate_data.commit()
    print(f">>> wrote eoth_{tag} shard {shard_id} ({len(ids)})", flush=True)
    return len(ids)


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_q(tag: str) -> list:
    return _read_jsonl(Q_FILES[tag])


@app.local_entrypoint()
def run_eoth(tag: str = "expansion", workers: int = 3, limit: int = 0):
    qs = _read_q.remote(tag)
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> eoth capture {tag}: {len(qs)} wavs / {workers} workers")
    total = sum(eoth_shard.starmap(
        [(tag, shards[i], i if not limit else 99)
         for i in range(workers)]))
    print(f">>> captured {total}")


def _load_eoth(tag):
    import glob as _glob
    import numpy as np
    ids, H = [], []
    for s in sorted(_glob.glob(f"{EOTH}_{tag}.shard*.npz")):
        z = np.load(s, allow_pickle=True)
        ids += [str(x) for x in z["ids"]]
        H.append(z["H"])
    return ids, np.concatenate(H)


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 30)
def refit(c_sweep: str = "0.0003,0.001,0.003,0.01"):
    """Probe v2: logistic on eot hiddens, frozen-calib(360) + expansion;
    OOF C-sweep; thresholds = OOF quantiles at .15/.30/.50 budgets."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    ids_f, H_f = _load_eoth("frozen")
    ids_x, H_x = _load_eoth("expansion")

    feats = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "split", "escalate_label"]]
    lab_f = dict(zip(feats[feats["split"] == "calib"]["id"],
                     feats[feats["split"] == "calib"]["escalate_label"]))
    xl = pd.read_parquet(XLABELS)
    lab_x = dict(zip(xl["id"], xl["escalate_label"]))

    X, y, src = [], [], []
    for i, h in zip(ids_f, H_f):
        if lab_f.get(i) is not None and not pd.isna(lab_f.get(i)):
            X.append(h); y.append(int(lab_f[i])); src.append("frozen")
    for i, h in zip(ids_x, H_x):
        if lab_x.get(i) is not None and not pd.isna(lab_x.get(i)):
            X.append(h); y.append(int(lab_x[i])); src.append("x")
    X = np.stack(X); y = np.array(y)
    print(f">>> train: n={len(y)} (frozen {src.count('frozen')}, "
          f"expansion {src.count('x')}), fail-rate {y.mean():.2f}",
          flush=True)

    best = None
    for C in (float(c) for c in c_sweep.split(",")):
        oof = cross_val_predict(
            LogisticRegression(C=C, max_iter=5000), X, y,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            method="predict_proba")[:, 1]
        a = roc_auc_score(y, oof)
        print(f"  C={C}: OOF AUC={a:.3f}", flush=True)
        if best is None or a > best[1]:
            best = (C, a, oof)
    C, auc, oof = best
    clf = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    thr = {t: float(np.quantile(oof, 1 - b))
           for t, b in (("conservative", 0.15), ("balanced", 0.30),
                        ("aggressive", 0.50))}
    art = {"layer": LAYER, "modality": "audio", "signal": "eot_read",
           "version": 2, "n_calib": int(len(y)), "C": C,
           "oof_auc": float(auc), "eot_thresholds": thr,
           "thresholds": thr,
           "w": clf.coef_[0].astype(float).tolist(),
           "b": float(clf.intercept_[0]),
           "gate": {"k_consecutive": 1, "ema_alpha": 1.0,
                    "cooldown_steps": 10 ** 6}}
    with open(ART_V2, "w") as fh:
        json.dump(art, fh)
    gate_data.commit()
    print(f">>> v2 probe: n={len(y)} C={C} OOF AUC={auc:.3f} "
          f"thr={ {k: round(v, 3) for k, v in thr.items()} } -> {ART_V2}",
          flush=True)


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 30)
def ablate():
    """v2 changed TWO things — signal match (train on the eot read, not
    full-prefill h_last) and data volume (360 -> 1160). Disentangle by
    training the same recipe on each subset, and check whether the
    threshold-quantile transfer (diagnosed rate compression) improved."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    ids_f, H_f = _load_eoth("frozen")
    ids_x, H_x = _load_eoth("expansion")
    feats = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "split", "escalate_label"]]
    cal = feats[(feats["split"] == "calib") & feats["escalate_label"].notna()]
    lab_f = dict(zip(cal["id"], cal["escalate_label"].astype(int)))
    xl = pd.read_parquet(XLABELS)
    xl = xl[xl["escalate_label"].notna()]
    lab_x = dict(zip(xl["id"], xl["escalate_label"].astype(int)))

    def subset(ids, H, lab):
        keep = [j for j, i in enumerate(ids) if i in lab]
        return H[keep], np.array([lab[ids[j]] for j in keep])

    Xf, yf = subset(ids_f, H_f, lab_f)
    Xx, yx = subset(ids_x, H_x, lab_x)

    # external eval sets (labels = never-arm local fail)
    ext = {}
    for bench, tpath in (("striviaqa", f"{DATA}/striviaqa_traces.parquet"),
                         ("swebq", f"{DATA}/swebq_traces.parquet"),
                         ("sdqa", f"{DATA}/sdqa_traces.parquet")):
        ids, H = _load_eoth(bench)
        tr = pd.read_parquet(tpath)
        nev = tr[(tr["tier"] == "never") & tr["heard_ok"].notna()]
        lab = dict(zip(nev["id"], 1 - nev["heard_ok"].astype(int)))
        ext[bench] = subset(ids, H, lab)

    fits = {
        "frozen-360 only (signal match, no new data)": (Xf, yf),
        "expansion-800 only (new data, no frozen)": (Xx, yx),
        "both = v2 (1160)": (np.concatenate([Xf, Xx]),
                             np.concatenate([yf, yx])),
    }
    print(f"{'fit':45s} " + " ".join(f"{b:>10s}" for b in ext) + "   mean",
          flush=True)
    for name, (X, y) in fits.items():
        clf = LogisticRegression(C=0.0003, max_iter=5000).fit(X, y)
        aucs = [roc_auc_score(yy, clf.decision_function(XX))
                for XX, yy in ext.values()]
        print(f"{name:45s} " + " ".join(f"{a:10.3f}" for a in aucs)
              + f"   {np.mean(aucs):.3f}", flush=True)

    # threshold transfer: do v2 quantiles fire at the intended budgets?
    v2 = json.load(open(ART_V2))
    v1 = json.load(open(GATE_ART))
    sys.path.insert(0, "/workspace/gate")
    import gate as gate_mod
    P2 = gate_mod.Probe(v2["w"], v2["b"])
    P1 = gate_mod.Probe(v1["w"], v1["b"])
    print("\n=== fire rate at each tier (target 15/30/50%) ===", flush=True)
    print(f"{'pool':12s} " + " ".join(f"{t[:4]+' v1/v2':>14s}" for t in
                                      ("conservative", "balanced",
                                       "aggressive")), flush=True)
    for bench, (X, _) in ext.items():
        s1 = np.array([P1.score(h) for h in X])
        s2 = np.array([P2.score(h) for h in X])
        cells = []
        for t in ("conservative", "balanced", "aggressive"):
            r1 = (s1 >= v1["eot_thresholds"][t]).mean()
            r2 = (s2 >= v2["eot_thresholds"][t]).mean()
            cells.append(f"{r1:5.0%}/{r2:5.0%}   ")
        print(f"{bench:12s} " + " ".join(c.rjust(11) for c in cells),
              flush=True)


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 30)
def make_thresholds():
    """Per-domain quantile thresholds for the v2 probe (8t's deployment
    recommendation): a global quantile cannot follow the per-domain score
    shift, so each pool gets thresholds at ITS OWN score quantiles.
    Label-free — only the unlabeled score distribution is used.
    frozen: quantiles from the CALIB split (no test leakage).
    external: from that pool's own scores (no labels exist, none used).
    Writes gate_v2_{tag}.json = v2 weights + that pool's thresholds."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import gate as gate_mod

    v2 = json.load(open(ART_V2))
    P = gate_mod.Probe(v2["w"], v2["b"])
    budgets = (("conservative", .15), ("balanced", .30),
               ("aggressive", .50))

    feats = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "split"]]
    calib_ids = set(feats[feats["split"] == "calib"]["id"])

    for tag in ("frozen", "striviaqa", "swebq", "sdqa", "valpaca",
                "sllama", "sreason"):
        try:
            ids, H = _load_eoth(tag)
        except ValueError:
            print(f">>> {tag}: no eoth shards yet — skipped", flush=True)
            continue
        if tag == "frozen":
            keep = [j for j, i in enumerate(ids) if i in calib_ids]
            H = H[keep]
            src = f"calib split ({len(keep)})"
        else:
            src = f"own pool ({len(ids)})"
        s = np.array([float(P.score(h)) for h in H])
        thr = {t: float(np.quantile(s, 1 - b)) for t, b in budgets}
        art = dict(v2)
        art["eot_thresholds"] = thr
        art["thresholds"] = thr
        art["threshold_source"] = f"{tag}: per-domain quantiles, {src}"
        with open(f"{DATA}/gate_v2_{tag}.json", "w") as fh:
            json.dump(art, fh)
        print(f">>> {tag:10s} thresholds from {src}: "
              f"{ {k: round(v, 3) for k, v in thr.items()} }", flush=True)
    gate_data.commit()


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 30)
def matched_rate(budget: float = 0.30):
    """Deployment-form readout: set each pool's threshold at ITS OWN
    score quantile (label-free — needs only unlabeled score history), so
    both probes escalate the same budget, then compare precision (share
    of escalated rows the talker would have failed) and recall."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, "/workspace/gate")
    import gate as gate_mod

    v1 = json.load(open(GATE_ART))
    v2 = json.load(open(ART_V2))
    P = {"v1": gate_mod.Probe(v1["w"], v1["b"]),
         "v2": gate_mod.Probe(v2["w"], v2["b"])}

    print(f"=== per-domain quantile thresholds @ {budget:.0%} budget ===",
          flush=True)
    print(f"{'pool':12s} {'base':>6s} " +
          " ".join(f"{v+' prec':>9s} {v+' rec':>8s}" for v in P), flush=True)
    for bench, tpath in (("striviaqa", f"{DATA}/striviaqa_traces.parquet"),
                         ("swebq", f"{DATA}/swebq_traces.parquet"),
                         ("sdqa", f"{DATA}/sdqa_traces.parquet")):
        ids, H = _load_eoth(bench)
        tr = pd.read_parquet(tpath)
        nev = tr[(tr["tier"] == "never") & tr["heard_ok"].notna()]
        lab = dict(zip(nev["id"], 1 - nev["heard_ok"].astype(int)))
        keep = [j for j, i in enumerate(ids) if i in lab]
        y = np.array([lab[ids[j]] for j in keep])
        Hk = H[keep]
        cells = []
        for v, probe in P.items():
            s = np.array([probe.score(h) for h in Hk])
            thr = np.quantile(s, 1 - budget)
            fired = s >= thr
            prec = y[fired].mean() if fired.any() else float("nan")
            rec = (y & fired).sum() / max(1, y.sum())
            cells.append(f"{prec:9.2f} {rec:8.2f}")
        print(f"{bench:12s} {y.mean():6.2f} " + " ".join(cells), flush=True)


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 30)
def eval_transfer():
    """Old vs new probe on identical eot hiddens; labels = never-arm
    local fail. Also frozen-test in-mix check."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, "/workspace/gate")
    import gate as gate_mod

    old = json.load(open(GATE_ART))
    new = json.load(open(ART_V2))
    p_old = gate_mod.Probe(old["w"], old["b"])
    p_new = gate_mod.Probe(new["w"], new["b"])

    def score(P, H):
        return np.array([float(P.score(h)) for h in H])

    print(f"{'pool':10s} {'n':>4s} {'AUC v1':>7s} {'AUC v2':>7s}",
          flush=True)
    for bench, tpath in (("striviaqa", f"{DATA}/striviaqa_traces.parquet"),
                         ("swebq", f"{DATA}/swebq_traces.parquet"),
                         ("sdqa", f"{DATA}/sdqa_traces.parquet")):
        ids, H = _load_eoth(bench)
        tr = pd.read_parquet(tpath)
        nev = tr[(tr["tier"] == "never") & tr["heard_ok"].notna()]
        lab = dict(zip(nev["id"], 1 - nev["heard_ok"].astype(int)))
        keep = [j for j, i in enumerate(ids) if i in lab]
        y = np.array([lab[ids[j]] for j in keep])
        Hk = H[keep]
        a1 = roc_auc_score(y, score(p_old, Hk))
        a2 = roc_auc_score(y, score(p_new, Hk))
        print(f"{bench:10s} {len(y):4d} {a1:7.3f} {a2:7.3f}", flush=True)

    ids, H = _load_eoth("frozen")
    feats = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "split", "escalate_label"]]
    tst = feats[(feats["split"] == "test") &
                feats["escalate_label"].notna()]
    lab = dict(zip(tst["id"], tst["escalate_label"].astype(int)))
    keep = [j for j, i in enumerate(ids) if i in lab]
    y = np.array([lab[ids[j]] for j in keep])
    Hk = H[keep]
    a1 = roc_auc_score(y, score(p_old, Hk))
    a2 = roc_auc_score(y, score(p_new, Hk))
    print(f"{'frozen-test':10s} {len(y):4d} {a1:7.3f} {a2:7.3f}",
          flush=True)
