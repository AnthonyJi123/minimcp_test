"""Probe v3 (user decision 2026-08-16): RL/SFT REJECTED for the gate —
(1) single-step decision with BOTH counterfactuals observable offline
(never/always arms) = cost-sensitive supervised classification; RL
re-derives the same Bayes classifier at far worse sample efficiency;
(2) SFT on the backbone breaks the zero-training frozen-checkpoint
claim and invalidates every measured curve; (3) the binding constraint
is domain shift + judge label noise (OOF .878 vs external .76-.78),
which neither method addresses. Chosen levers instead:

  1. calib expansion2: ~1150 more public-benchmark queries, 7 NEW
     families (PopQA, TruthfulQA, CommonsenseQA, OpenBookQA, HotpotQA,
     SVAMP, MMLU) — 8t showed data is a real lever (+.031) and new
     families carry the transferable signal. [[no-selfmade-datasets]]
  2. multi-layer / multi-position features: L{14,18,22,26,30} x
     (eot last-8 tokens + user-turn mean) captured in ONE streaming
     replay per query (5d: transfer peaks mid-network — L22 .931 vs
     readout .366 — and mean-pooling survives the duplex readout
     cliff). All refits are then CPU-only.

Selection on train-OOF only; externals read once in eval_transfer3.
Pre-registered guard: frozen-test in-mix must not regress vs v2 (.860).
External eval pools stay strictly OUT of training; frozen 600 and all
v1/v2 artifacts untouched. New artifact: midlayer_gate_audio_v3.json.

Stages (run in order; eoth2 capture of EXISTING tags can start in
parallel with the expansion2 data pipeline):
  modal run modal_train2.py::build_expansion2
  modal run modal_train2.py::run_tts2
  modal run modal_train2.py::run_answer2
  modal run modal_train2.py::judge_expansion2
  modal run modal_train2.py::run_eoth2 --tags frozen,expansion,...
  modal run modal_train2.py::refit3
  modal run modal_train2.py::eval_transfer3
"""
import json
import os
import re
import sys

from modal_app import (app, gen_app, GPU_VOL, gate_data, DATA, OPENAI,
                       API_REGION, QUERIES, _read_jsonl)
from modal_train import (image_st, util_st, dl_image, AUDIO_DIRS, Q_FILES,
                         XQ, XLABELS, TTS_VOICE, speakable)

HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN_PY = os.path.join(HERE, "modal_train.py")
image_st2 = image_st.add_local_file(_TRAIN_PY, "/root/modal_train.py")
util_st2 = util_st.add_local_file(_TRAIN_PY, "/root/modal_train.py")
dl_image2 = dl_image.add_local_file(_TRAIN_PY, "/root/modal_train.py")

YQ = f"{DATA}/queries_expansion2.jsonl"
YAUDIO = f"{DATA}/audio_expansion2"
YLABELS = f"{DATA}/expansion2_labels.parquet"
EOTH2 = f"{DATA}/eoth2"                  # + _{tag}.shard{i}.npz
ART_V2 = f"{DATA}/midlayer_gate_audio_v2.json"
ART_V3 = f"{DATA}/midlayer_gate_audio_v3.json"
LAYERS2 = [14, 18, 22, 26, 30]           # 5d: mid-band strong, cliff at 32+
K_EOT = 8

AUDIO2 = dict(AUDIO_DIRS, expansion2=YAUDIO)
Q2_FILES = dict(Q_FILES, expansion2=YQ)

# External transfer eval: never-arm local fail, identical to 8t plus the
# two 8x pools (sllama official-judge English factoid, sreason Chinese).
EXT_TRACES = (("striviaqa", "striviaqa_traces.parquet"),
              ("swebq", "swebq_traces.parquet"),
              ("sdqa", "sdqa_traces.parquet"),
              ("sllama", "sllama_v2_traces.parquet"),
              ("sreason", "sreason_v2_traces.parquet"))


@app.function(image=dl_image2, volumes={DATA: gate_data}, memory=8192,
              timeout=60 * 60)
def build_expansion2():
    """~1150 new calib queries, 7 families NONE of which appear in the
    v1 calib mix, deduped against frozen + expansion + ALL external eval
    pools. Per-family try/except so one broken loader doesn't kill the
    build; shortfall backfilled from PopQA (14k rows of headroom)."""
    import numpy as np
    from datasets import load_dataset

    rng = np.random.default_rng(44)      # 42=eval, 43=expansion1

    seen = set()

    def key(text):
        return re.sub(r"\s+", " ", str(text).lower())[:120]

    for path in [QUERIES, XQ] + [Q2_FILES[t] for t in
                                 ("striviaqa", "swebq", "sdqa", "valpaca",
                                  "sllama", "sreason")]:
        try:
            for q in _read_jsonl(path):
                seen.add(key(q["query"]))
        except FileNotFoundError:
            print(f">>> dedup source missing (skipped): {path}", flush=True)

    def fresh(text):
        k = key(text)
        if k in seen:
            return False
        seen.add(k)
        return True

    def take(ds, fmt, n):
        rows = []
        for i in rng.permutation(len(ds)):
            r = fmt(ds[int(i)])
            if r is None:
                continue
            q, ref = r
            if not speakable(q) or not fresh(q):
                continue
            rows.append((q, ref))
            if len(rows) >= n:
                break
        return rows

    def mc(stem, choices):
        opts = ", ".join(f"({l}) {t}" for l, t in
                         zip(choices["label"], choices["text"]))
        return f"{stem} {opts}"

    out = []

    def add(pool, rows):
        for q, ref in rows:
            out.append({"id": f"y{len(out):04d}", "pool": pool,
                        "query": q, "reference_answer": ref,
                        "split": "calib_y"})
        print(f">>> {pool}: +{len(rows)}", flush=True)

    def family(pool, n, loader):
        try:
            add(pool, take(*loader(), n))
        except Exception as e:
            print(f">>> {pool} FAILED ({type(e).__name__}: {e}) — skipped",
                  flush=True)

    def _popqa():
        ds = load_dataset("akariasai/PopQA", split="test")

        def fmt(e):
            try:
                refs = json.loads(e["possible_answers"])
            except Exception:
                refs = [str(e["possible_answers"])]
            return (e["question"], "; ".join(str(r) for r in refs[:4]))
        return ds, fmt

    def _truthful():
        ds = load_dataset("truthfulqa/truthful_qa", "generation",
                          split="validation")
        return ds, lambda e: (e["question"], e["best_answer"])

    def _csqa():
        ds = load_dataset("tau/commonsense_qa", split="validation")

        def fmt(e):
            ch = e["choices"]
            ref = dict(zip(ch["label"], ch["text"])).get(e["answerKey"], "")
            return (mc(e["question"], ch), f"({e['answerKey']}) {ref}")
        return ds, fmt

    def _obqa():
        ds = load_dataset("allenai/openbookqa", "main", split="test")

        def fmt(e):
            ch = e["choices"]
            ref = dict(zip(ch["label"], ch["text"])).get(e["answerKey"], "")
            return (mc(e["question_stem"], ch), f"({e['answerKey']}) {ref}")
        return ds, fmt

    def _hotpot():
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                          split="validation", trust_remote_code=True)
        return ds, lambda e: (e["question"], e["answer"])

    def _svamp():
        ds = load_dataset("ChilleD/SVAMP", split="test")

        def fmt(e):
            a = e["Answer"]
            ref = str(int(a)) if float(a) == int(float(a)) else str(a)
            return (f"{e['Body'].strip()} {e['Question'].strip()}", ref)
        return ds, fmt

    def _mmlu():
        ds = load_dataset("cais/mmlu", "all", split="test")

        def fmt(e):
            ch = {"label": list("ABCD"), "text": e["choices"]}
            k = "ABCD"[int(e["answer"])]
            return (mc(e["question"], ch),
                    f"({k}) {e['choices'][int(e['answer'])]}")
        return ds, fmt

    family("know-longtail", 250, _popqa)
    family("trap-truthful", 150, _truthful)
    family("know-commonsense", 150, _csqa)
    family("know-openbook", 100, _obqa)
    family("hard-multihop", 200, _hotpot)
    family("easy-mathword", 100, _svamp)
    family("know-mmlu", 200, _mmlu)

    if len(out) < 1000:                  # backfill failed families
        family("know-longtail", 1150 - len(out), _popqa)

    with open(YQ, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> wrote {len(out)} expansion2 queries -> {YQ}", flush=True)


@app.function(image=util_st2, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_y() -> list:
    return _read_jsonl(YQ)


@app.function(image=util_st2, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_q2(tag: str) -> list:
    return _read_jsonl(Q2_FILES[tag])


@gen_app.function(image=util_st2, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 60, region=API_REGION)
def run_tts2(limit: int = 0, concurrency: int = 8):
    """tts-1/alloy, same engine as frozen + expansion (8r exonerated it
    for speakable content — consistency beats novelty)."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    qs = _read_jsonl(YQ)[:limit or None]
    os.makedirs(YAUDIO, exist_ok=True)
    client = OpenAI()

    def render(q):
        out = f"{YAUDIO}/{q['id']}.wav"
        if os.path.exists(out) and os.path.getsize(out) > 44:
            return "cached"
        resp = client.audio.speech.create(model="tts-1", voice=TTS_VOICE,
                                          input=q["query"][:4000],
                                          response_format="wav")
        with open(out, "wb") as fh:
            fh.write(resp.content)
        return "done"

    with ThreadPoolExecutor(concurrency) as ex:
        res = list(ex.map(render, qs))
    gate_data.commit()
    print(f">>> tts2: {res.count('done')} rendered, {res.count('cached')} "
          f"cached / {len(qs)}", flush=True)


@app.function(image=image_st2, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 2)
def answer2_shard(shard: list, shard_id: int) -> int:
    """MiniCPM answers each expansion2 query from AUDIO (6a collection
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
        au, _ = librosa.load(f"{YAUDIO}/{q['id']}.wav", sr=16000, mono=True)
        ans = model.chat(msgs=[{"role": "user", "content": [au]}],
                         tokenizer=tok, max_new_tokens=512, sampling=False,
                         use_tts_template=False, generate_audio=False)
        rows.append({"id": q["id"], "answer": str(ans).strip()})
        if k < 3 or k % 40 == 0:
            print(f"  [{k}] {q['id']} :: {rows[-1]['answer'][:60]!r}",
                  flush=True)
    pd.DataFrame(rows).to_parquet(
        f"{DATA}/expansion2_answers.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote answers shard {shard_id} ({len(rows)})", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_answer2(workers: int = 4, limit: int = 0):
    """Then judge separately: modal run modal_train2.py::judge_expansion2"""
    qs = _read_y.remote()
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> answering {len(qs)} expansion2 queries / {workers} workers")
    total = sum(answer2_shard.starmap(
        [(shards[i], i) for i in range(workers)]))
    print(f">>> answered {total}")


@gen_app.function(image=util_st2, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 60, region=API_REGION)
def judge_expansion2():
    import glob as _glob
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    answers = pd.concat(
        [pd.read_parquet(s) for s in
         sorted(_glob.glob(f"{DATA}/expansion2_answers.shard*.parquet"))],
        ignore_index=True).drop_duplicates(subset="id", keep="last")
    print(f">>> judging {len(answers)} answers", flush=True)
    byid = {q["id"]: q for q in _read_jsonl(YQ)}
    rows = [{"id": a["id"], "query": byid[a["id"]]["query"],
             "reference_answer": byid[a["id"]]["reference_answer"],
             "answer": a["answer"]} for _, a in answers.iterrows()]
    judged = asyncio.run(escalate.judge_many(rows, concurrency=8))
    df = pd.DataFrame(judged)
    df["pool"] = [byid[i]["pool"] for i in df["id"]]
    df.to_parquet(YLABELS)
    gate_data.commit()
    ok = df[df["adequate"].notna()]
    print(f"\n=== expansion2 labels (n={len(ok)}) ===", flush=True)
    for p, g in ok.groupby("pool"):
        print(f"  {p:16s} n={len(g):3d} fail-rate="
              f"{g['escalate_label'].mean():.2f}", flush=True)


@app.function(image=image_st2, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 2, max_containers=8)
def eoth2_shard(tag: str, shard: list, shard_id: int) -> int:
    """One streaming replay per query, capturing at L{14,18,22,26,30}:
      H_eot  (n, 5, 8, d) — last-8 tokens of the eot read (the final
              assistant-turn prefill forward; v2's signal = [:, L22, -1])
      H_mean (n, 5, d)    — mean over ALL user-audio-chunk positions
              (5d: mean-pooling survives the duplex readout cliff;
              deployable online as a running mean, zero eot latency)
    Stored raw float16 so every probe refit is CPU-only forever."""
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

    # "tail" is a rolling last-K_EOT-token window ACROSS forwards: the
    # streaming assistant prefill runs 1-token forwards, so a within-
    # forward slice degenerates to a single token (smoke-tested).
    state = {"accum": False, "tail": {}, "sum": {}, "cnt": 0}

    def mk_hook(L, count_here):
        def hook(_m, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            h = hs[0].detach().float()               # (T, d)
            t = h[-K_EOT:].cpu()
            prev = state["tail"].get(L)
            state["tail"][L] = (t if prev is None
                                else torch.cat([prev, t])[-K_EOT:])
            if state["accum"]:
                s = h.sum(0).cpu()
                prev = state["sum"].get(L)
                state["sum"][L] = s if prev is None else prev + s
                if count_here:
                    state["cnt"] += h.shape[0]
        return hook

    handles = [model.llm.model.layers[L].register_forward_hook(
        mk_hook(L, L == LAYERS2[0])) for L in LAYERS2]

    adir = AUDIO2[tag]
    ids, E, M, ELEN = [], [], [], []
    try:
        for k, q in enumerate(shard):
            wav = f"{adir}/{q['id']}.wav"
            if not os.path.exists(wav):
                continue
            au, _ = librosa.load(wav, sr=16000, mono=True)
            chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
            model.reset_session()
            state.update(accum=False, tail={}, sum={}, cnt=0)
            sys_msg = call_def(model.get_sys_prompt, mode="omni",
                               language="en")
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[sys_msg], tokenizer=tok)
            state["accum"] = True
            for i, ch in enumerate(chunks):
                if len(ch) < 16000:
                    ch = np.pad(ch, (0, 16000 - len(ch)))
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[{"role": "user",
                                "content": [ch.astype(np.float32)]}],
                         tokenizer=tok,
                         is_last_chunk=(i == len(chunks) - 1))
            state["accum"] = False
            got = None
            for content in ("", " "):
                try:
                    call_def(model.streaming_prefill, session_id="s1",
                             msgs=[{"role": "assistant",
                                    "content": [content]}],
                             tokenizer=tok, is_last_chunk=True)
                    got = {L: state["tail"][L] for L in LAYERS2}
                    break
                except Exception:
                    continue
            if got is None or state["cnt"] == 0:
                continue
            d = got[LAYERS2[0]].shape[1]
            eot = np.zeros((len(LAYERS2), K_EOT, d), dtype=np.float16)
            mean = np.zeros((len(LAYERS2), d), dtype=np.float16)
            elen = got[LAYERS2[0]].shape[0]
            for j, L in enumerate(LAYERS2):
                t = got[L].numpy()
                eot[j, K_EOT - t.shape[0]:] = t.astype(np.float16)
                mean[j] = (state["sum"][L].numpy()
                           / state["cnt"]).astype(np.float16)
            ids.append(q["id"])
            E.append(eot)
            M.append(mean)
            ELEN.append(elen)
            if k < 3 or k % 50 == 0:
                print(f"  [{k}] {q['id']}", flush=True)
    finally:
        for h in handles:
            h.remove()

    np.savez_compressed(f"{EOTH2}_{tag}.shard{shard_id}.npz",
                        ids=np.array(ids), H_eot=np.stack(E),
                        H_mean=np.stack(M),
                        eot_len=np.array(ELEN, dtype=np.int16),
                        layers=np.array(LAYERS2))
    gate_data.commit()
    print(f">>> wrote eoth2_{tag} shard {shard_id} ({len(ids)})", flush=True)
    return len(ids)


@app.local_entrypoint()
def run_eoth2(tags: str = "frozen,expansion,striviaqa,swebq,sdqa,valpaca,"
                          "sllama,sreason",
              workers: int = 3, limit: int = 0):
    """Flat starmap across all tags (max_containers bounds H100 use).
    expansion2 gets its own later run once its wavs exist."""
    items = []
    for tag in [t.strip() for t in tags.split(",") if t.strip()]:
        qs = _read_q2.remote(tag)
        if limit:
            qs = qs[:limit]
        w = 1 if limit else (workers if len(qs) > 400 else 2)
        shards = [qs[i::w] for i in range(w)]
        items += [(tag, shards[i], i if not limit else 99)
                  for i in range(w)]
        print(f">>> {tag}: {len(qs)} queries / {w} shards")
    total = sum(eoth2_shard.starmap(items))
    print(f">>> captured {total}")


def _load2(tag):
    import glob as _glob
    import numpy as np
    shards = sorted(_glob.glob(f"{EOTH2}_{tag}.shard*.npz"))
    if not shards:
        raise FileNotFoundError(f"no eoth2 shards for {tag}")
    ids, E, M, ELEN = [], [], [], []
    for s in shards:
        z = np.load(s, allow_pickle=True)
        ids += [str(x) for x in z["ids"]]
        E.append(z["H_eot"])
        M.append(z["H_mean"])
        ELEN.append(z["eot_len"])
    return (ids, np.concatenate(E), np.concatenate(M),
            np.concatenate(ELEN))


def _feat(E, M, ELEN, layers, modes):
    """Feature matrix for one (layers, modes) config. E (n,5,8,d) f16,
    M (n,5,d) f16, ELEN (n,). Modes: eot_last / eot_mean / user_mean."""
    import numpy as np
    parts = []
    for L in layers:
        j = LAYERS2.index(L)
        for m in modes:
            if m == "eot_last":
                parts.append(E[:, j, -1, :].astype(np.float32))
            elif m == "eot_mean":
                He = E[:, j].astype(np.float32)          # (n, 8, d)
                ln = np.clip(ELEN.astype(np.int32), 1, K_EOT)
                mask = (np.arange(K_EOT)[None, :]
                        >= (K_EOT - ln[:, None])).astype(np.float32)
                parts.append((He * mask[:, :, None]).sum(1)
                             / ln[:, None])
            elif m == "user_mean":
                parts.append(M[:, j].astype(np.float32))
    return np.concatenate(parts, axis=1)


CFGS = ([{"name": f"eot_last L{L}", "layers": [L], "modes": ["eot_last"]}
         for L in LAYERS2]
        + [{"name": f"eot_mean8 L{L}", "layers": [L], "modes": ["eot_mean"]}
           for L in LAYERS2]
        + [{"name": f"user_mean L{L}", "layers": [L], "modes": ["user_mean"]}
           for L in LAYERS2]
        + [{"name": "eot_last L18+22+26", "layers": [18, 22, 26],
            "modes": ["eot_last"]},
           {"name": "eot_last+user_mean L22", "layers": [22],
            "modes": ["eot_last", "user_mean"]},
           {"name": "eot_last+eot_mean+user_mean L22", "layers": [22],
            "modes": ["eot_last", "eot_mean", "user_mean"]},
           {"name": "eot_last+user_mean L18+22+26", "layers": [18, 22, 26],
            "modes": ["eot_last", "user_mean"]}])


def _train_xy(with_y=True):
    """Train matrices: frozen calib + expansion (+ expansion2). Returns
    (ids, E, M, ELEN, y) restricted to labeled rows. Never touches
    external pools."""
    import numpy as np
    import pandas as pd

    feats = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "split", "escalate_label"]]
    cal = feats[(feats["split"] == "calib") & feats["escalate_label"].notna()]
    lab = dict(zip(cal["id"], cal["escalate_label"].astype(int)))
    xl = pd.read_parquet(XLABELS)
    xl = xl[xl["escalate_label"].notna()]
    lab.update(dict(zip(xl["id"], xl["escalate_label"].astype(int))))
    tags = ["frozen", "expansion"]
    if with_y:
        yl = pd.read_parquet(YLABELS)
        yl = yl[yl["escalate_label"].notna()]
        lab.update(dict(zip(yl["id"], yl["escalate_label"].astype(int))))
        tags.append("expansion2")

    IDS, E_, M_, L_ = [], [], [], []
    for tag in tags:
        ids, E, M, ELEN = _load2(tag)
        keep = [j for j, i in enumerate(ids) if i in lab]
        IDS += [ids[j] for j in keep]
        E_.append(E[keep])
        M_.append(M[keep])
        L_.append(ELEN[keep])
    import numpy as np
    E = np.concatenate(E_)
    M = np.concatenate(M_)
    ELEN = np.concatenate(L_)
    y = np.array([lab[i] for i in IDS])
    return IDS, E, M, ELEN, y


@app.function(image=util_st2, volumes={DATA: gate_data}, timeout=60 * 120,
              cpu=16, memory=32768)
def refit3(base_c: float = 0.0003,
           c_sweep: str = "0.0001,0.0003,0.001,0.003"):
    """Config sweep on train-OOF ONLY (5-fold, fixed C), then C-sweep on
    the winner, fit, export ART_V3. Externals untouched here."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    IDS, E, M, ELEN, y = _train_xy(with_y=True)
    print(f">>> train n={len(y)}, fail-rate {y.mean():.2f}", flush=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    def oof_auc(X, C):
        oof = cross_val_predict(
            LogisticRegression(C=C, max_iter=5000), X, y, cv=cv,
            method="predict_proba")[:, 1]
        return roc_auc_score(y, oof), oof

    results = []
    for cfg in CFGS:
        X = _feat(E, M, ELEN, cfg["layers"], cfg["modes"])
        a, _ = oof_auc(X, base_c)
        results.append((a, cfg))
        print(f"  {cfg['name']:34s} d={X.shape[1]:6d} OOF AUC={a:.3f}",
              flush=True)
    results.sort(key=lambda r: -r[0])
    best_cfg = results[0][1]
    print(f">>> winner: {best_cfg['name']} "
          f"(OOF {results[0][0]:.3f} @ C={base_c})", flush=True)

    X = _feat(E, M, ELEN, best_cfg["layers"], best_cfg["modes"])
    best = None
    for C in (float(c) for c in c_sweep.split(",")):
        a, oof = oof_auc(X, C)
        print(f"  C={C}: OOF AUC={a:.3f}", flush=True)
        if best is None or a > best[1]:
            best = (C, a, oof)
    C, auc, oof = best
    clf = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    thr = {t: float(np.quantile(oof, 1 - b))
           for t, b in (("conservative", 0.15), ("balanced", 0.30),
                        ("aggressive", 0.50))}
    art = {"version": 3, "modality": "audio",
           "signal": "eot_read_multilayer",
           "layers_captured": LAYERS2, "k_eot": K_EOT,
           "layer_set": best_cfg["layers"], "modes": best_cfg["modes"],
           "config_name": best_cfg["name"],
           "n_calib": int(len(y)), "C": C, "oof_auc": float(auc),
           "eot_thresholds": thr, "thresholds": thr,
           "w": clf.coef_[0].astype(float).tolist(),
           "b": float(clf.intercept_[0]),
           "gate": {"k_consecutive": 1, "ema_alpha": 1.0,
                    "cooldown_steps": 10 ** 6}}
    with open(ART_V3, "w") as fh:
        json.dump(art, fh)
    gate_data.commit()
    print(f">>> v3 probe: {best_cfg['name']} n={len(y)} C={C} "
          f"OOF AUC={auc:.3f} -> {ART_V3}", flush=True)


@app.function(image=util_st2, volumes={DATA: gate_data}, timeout=60 * 60,
              cpu=16, memory=32768)
def eval_transfer3():
    """Externals read ONCE. Rows: stored-v2 sanity anchor + the 2x2
    ablation (recipe x data) with D = the shipped v3 artifact. Guard:
    frozen-test in-mix must not regress vs v2 (.860 in 8t)."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, "/workspace/gate")
    import gate as gate_mod

    v2 = json.load(open(ART_V2))
    v3 = json.load(open(ART_V3))
    P2 = gate_mod.Probe(v2["w"], v2["b"])
    best_layers, best_modes = v3["layer_set"], v3["modes"]

    # train matrices for the ablation fits
    _, E, M, ELEN, y = _train_xy(with_y=True)
    _, Eo, Mo, Lo, yo = _train_xy(with_y=False)      # frozen+x only

    def fit(E_, M_, L_, y_, layers, modes, C):
        X = _feat(E_, M_, L_, layers, modes)
        return LogisticRegression(C=C, max_iter=5000).fit(X, y_), \
            (layers, modes)

    fits = [
        ("A v2-recipe (L22 last, frozen+x)",
         *fit(Eo, Mo, Lo, yo, [22], ["eot_last"], 0.0003)),
        ("B data lever (L22 last, +expansion2)",
         *fit(E, M, ELEN, y, [22], ["eot_last"], 0.0003)),
        ("C feature lever (best cfg, frozen+x)",
         *fit(Eo, Mo, Lo, yo, best_layers, best_modes, v3["C"])),
        (f"D v3 = {v3['config_name']} + all data",
         *fit(E, M, ELEN, y, best_layers, best_modes, v3["C"])),
    ]

    # eval pools: externals (never-arm local fail) + frozen-test
    pools = []
    for bench, tpath in EXT_TRACES:
        try:
            ids, Ee, Me, Le = _load2(bench)
            tr = pd.read_parquet(f"{DATA}/{tpath}")
            nev = tr[(tr["tier"] == "never") & tr["heard_ok"].notna()]
            lab = dict(zip(nev["id"], 1 - nev["heard_ok"].astype(int)))
            keep = [j for j, i in enumerate(ids) if i in lab]
            if len(keep) < 30 or len({lab[ids[j]] for j in keep}) < 2:
                print(f">>> {bench}: unusable labels — skipped", flush=True)
                continue
            pools.append((bench, Ee[keep], Me[keep], Le[keep],
                          np.array([lab[ids[j]] for j in keep])))
        except FileNotFoundError as e:
            print(f">>> {bench}: {e} — skipped", flush=True)
    ids, Ef, Mf, Lf = _load2("frozen")
    feats = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "split", "escalate_label"]]
    tst = feats[(feats["split"] == "test") & feats["escalate_label"].notna()]
    lab = dict(zip(tst["id"], tst["escalate_label"].astype(int)))
    keep = [j for j, i in enumerate(ids) if i in lab]
    pools.append(("frozen-test", Ef[keep], Mf[keep], Lf[keep],
                  np.array([lab[ids[j]] for j in keep])))

    names = [p[0] for p in pools]
    print(f"\n{'fit':42s} " + " ".join(f"{n[:11]:>11s}" for n in names)
          + "   ext-mean", flush=True)

    # sanity anchor: stored v2 weights on the re-captured L22 eot_last
    row = []
    for _, Ee, Me, Le, yy in pools:
        s = _feat(Ee, Me, Le, [22], ["eot_last"]) @ np.array(v2["w"]) \
            + v2["b"]
        row.append(roc_auc_score(yy, s))
    ext = [a for n, a in zip(names, row) if n != "frozen-test"]
    print(f"{'v2 stored artifact (sanity)':42s} "
          + " ".join(f"{a:11.3f}" for a in row)
          + f"   {np.mean(ext):.3f}", flush=True)

    for label, clf, (layers, modes) in fits:
        row = []
        for _, Ee, Me, Le, yy in pools:
            X = _feat(Ee, Me, Le, layers, modes)
            row.append(roc_auc_score(yy, clf.decision_function(X)))
        ext = [a for n, a in zip(names, row) if n != "frozen-test"]
        print(f"{label:42s} " + " ".join(f"{a:11.3f}" for a in row)
              + f"   {np.mean(ext):.3f}", flush=True)

    ft = dict(zip(names, row))
    print(f"\n>>> GUARD frozen-test (v3 row D): {ft['frozen-test']:.3f} "
          f"vs v2 8t reference .860 — "
          f"{'OK' if ft['frozen-test'] >= 0.855 else 'REGRESSION'}",
          flush=True)


@app.function(image=util_st2, volumes={DATA: gate_data}, timeout=60 * 30,
              cpu=8, memory=32768)
def make_thresholds3():
    """Per-domain quantile thresholds for v3 (8t deployment finding: a
    global quantile cannot follow per-domain score shift). Label-free.
    Writes gate_v3_{tag}.json. Run only when a live re-run is decided."""
    import numpy as np
    import pandas as pd

    v3 = json.load(open(ART_V3))
    w = np.array(v3["w"])
    budgets = (("conservative", .15), ("balanced", .30),
               ("aggressive", .50))
    feats = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "split"]]
    calib_ids = set(feats[feats["split"] == "calib"]["id"])

    for tag in ("frozen", "striviaqa", "swebq", "sdqa", "valpaca",
                "sllama", "sreason"):
        try:
            ids, E, M, L = _load2(tag)
        except FileNotFoundError:
            print(f">>> {tag}: no eoth2 shards — skipped", flush=True)
            continue
        if tag == "frozen":
            keep = [j for j, i in enumerate(ids) if i in calib_ids]
            E, M, L = E[keep], M[keep], L[keep]
            src = f"calib split ({len(keep)})"
        else:
            src = f"own pool ({len(ids)})"
        z = _feat(E, M, L, v3["layer_set"], v3["modes"]) @ w + v3["b"]
        s = 1.0 / (1.0 + np.exp(-z))     # sigmoid: Probe.score scale
        thr = {t: float(np.quantile(s, 1 - b)) for t, b in budgets}
        art = dict(v3)
        art["eot_thresholds"] = thr
        art["thresholds"] = thr
        art["threshold_source"] = f"{tag}: per-domain quantiles, {src}"
        with open(f"{DATA}/gate_v3_{tag}.json", "w") as fh:
            json.dump(art, fh)
        print(f">>> {tag:10s} thresholds from {src}: "
              f"{ {k: round(v, 3) for k, v in thr.items()} }", flush=True)
    gate_data.commit()
