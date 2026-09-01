"""Calibration expansion 3 (post-8be): more rows for the native probe +
the zh axis. Motivated by two measured facts, not vibes:

  1. The NATIVE scaling curve is not saturated at 2310 (native_refit.
     json: external mean .643->.709, ~+.02 AUC per doubling, still
     rising). expansion3 doubles the en mix: ~2300 new public-benchmark
     queries, same 7 families as expansion2 (8t/8bb: new-family data is
     the transferable signal), deduped against EVERYTHING incl.
     expansion2.
  2. The zh axis: sreason fires 0% at every deployed tier and DECLINED
     when en calib grew (8bb -.039) — no amount of en data touches it.
     expansion3zh stages the ~355 OpenAudioBench reasoning_qa rows the
     202-row eval pool did NOT sample (excluded by source stem, so the
     two sets are disjoint), official audio, zh labels.

  METHOD NOTE: once exp3zh enters a probe's calib mix, sreason stops
  being an "external transfer" pool for that probe — same source
  distribution. Report it as zh in-domain, or keep exp3zh out of any
  probe scored for the external-transfer table.

Cost (est.): TTS ~$3, answers ~2-3 H100h ~$10, judge ~$4 API, native
dump ~15 H100h ~$60-75 -> ~$80-100 all-in for a predicted external
+.02-.03 AUC (en) + a live zh operating point.

Stages (run in order; then the native dump + refit):
  modal run modal_train3.py::build_expansion3
  modal run modal_train3.py::build_expansion3zh
  modal run modal_train3.py::run_tts3
  modal run modal_train3.py::run_answer3 --tag expansion3 --workers 4
  modal run modal_train3.py::run_answer3 --tag expansion3zh --workers 2
  modal run modal_train3.py::judge_expansion3 --tag expansion3
  modal run modal_train3.py::judge_expansion3 --tag expansion3zh
  modal run modal_native_dump.py::run_native --pool expansion3 \
      --tag exp3 --workers 8
  modal run modal_native_dump.py::run_native --pool expansion3zh \
      --tag exp3zh --workers 2
  python scripts/22_native_refit.py      # picks exp3/exp3zh up if present
  python scripts/26_pool_thresholds.py
  python scripts/27_probe_receipt_native.py
"""
import json
import os
import re
import sys
import time

from modal_app import (app, gen_app, GPU_VOL, gate_data, DATA, OPENAI,
                       API_REGION, QUERIES, _read_jsonl)
from modal_train import (image_st, util_st, dl_image, Q_FILES, XQ,
                         TTS_VOICE, speakable)
from modal_train2 import YQ  # expansion2 queries join the dedup set

HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN_PY = os.path.join(HERE, "modal_train.py")
_TRAIN2_PY = os.path.join(HERE, "modal_train2.py")


def _chain(img):
    return (img.add_local_file(_TRAIN_PY, "/root/modal_train.py")
            .add_local_file(_TRAIN2_PY, "/root/modal_train2.py"))


image_st3 = _chain(image_st)
util_st3 = _chain(util_st)
dl_image3 = _chain(dl_image)

ZQ = f"{DATA}/queries_expansion3.jsonl"
ZAUDIO = f"{DATA}/audio_expansion3"
ZHQ = f"{DATA}/queries_expansion3zh.jsonl"
ZHAUDIO = f"{DATA}/audio_expansion3zh"
SREASON_Q = f"{DATA}/queries_sreason.jsonl"

TAGS = {"expansion3": (ZQ, ZAUDIO),
        "expansion3zh": (ZHQ, ZHAUDIO)}


@app.function(image=dl_image3, volumes={DATA: gate_data}, memory=8192,
              timeout=60 * 60)
def build_expansion3():
    """~2300 new en calib queries: expansion2's 7 families at 2x the
    take, seed 45, deduped against frozen + expansion + expansion2 +
    ALL external eval pools. Same per-family try/except + PopQA
    backfill as build_expansion2."""
    import numpy as np
    from datasets import load_dataset

    rng = np.random.default_rng(45)      # 42=eval, 43=exp1, 44=exp2

    seen = set()

    def key(text):
        return re.sub(r"\s+", " ", str(text).lower())[:120]

    for path in [QUERIES, XQ, YQ] + [Q_FILES[t] for t in
                                     ("striviaqa", "swebq", "sdqa",
                                      "valpaca", "sllama", "sreason")]:
        try:
            for q in _read_jsonl(path):
                seen.add(key(q["query"]))
        except FileNotFoundError:
            print(f">>> dedup source missing (skipped): {path}",
                  flush=True)

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
            out.append({"id": f"z{len(out):04d}", "pool": pool,
                        "query": q, "reference_answer": ref,
                        "split": "calib_z"})
        print(f">>> {pool}: +{len(rows)}", flush=True)

    def family(pool, n, loader):
        try:
            add(pool, take(*loader(), n))
        except Exception as e:
            print(f">>> {pool} FAILED ({type(e).__name__}: {e}) — "
                  f"skipped", flush=True)

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
            ref = dict(zip(ch["label"], ch["text"])).get(e["answerKey"],
                                                         "")
            return (mc(e["question"], ch), f"({e['answerKey']}) {ref}")
        return ds, fmt

    def _obqa():
        ds = load_dataset("allenai/openbookqa", "main", split="test")

        def fmt(e):
            ch = e["choices"]
            ref = dict(zip(ch["label"], ch["text"])).get(e["answerKey"],
                                                         "")
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

    family("know-longtail", 500, _popqa)
    family("trap-truthful", 300, _truthful)
    family("know-commonsense", 300, _csqa)
    family("know-openbook", 200, _obqa)
    family("hard-multihop", 400, _hotpot)
    family("easy-mathword", 200, _svamp)
    family("know-mmlu", 400, _mmlu)

    if len(out) < 2000:                  # backfill failed families
        family("know-longtail", 2300 - len(out), _popqa)

    with open(ZQ, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> wrote {len(out)} expansion3 queries -> {ZQ}", flush=True)


@app.function(image=dl_image3, volumes={DATA: gate_data}, memory=8192,
              timeout=60 * 60)
def build_expansion3zh():
    """The zh calib slice: every OpenAudioBench reasoning_qa row the
    sreason eval pool did NOT sample (excluded by source stem — the
    two sets are disjoint by construction). Official audio resampled
    to 16 kHz, refs from the zh reference column. ~355 rows."""
    import numpy as np
    import librosa
    import pandas as pd
    import soundfile as sf
    from huggingface_hub import snapshot_download

    repo, sub = "baichuan-inc/OpenAudioBench", "reasoning_qa"
    root = None
    for attempt in range(4):
        try:
            root = snapshot_download(
                repo, repo_type="dataset", max_workers=2,
                allow_patterns=[f"eval_datas/{sub}/**"])
            break
        except Exception as e:
            wait = 90 * (attempt + 1)
            print(f">>> snapshot attempt {attempt}: {str(e)[:200]} — "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)
    if root is None:
        raise RuntimeError("snapshot_download kept failing")

    base = os.path.join(root, "eval_datas", sub)
    metas = [f for f in os.listdir(base)
             if f.endswith((".csv", ".tsv", ".jsonl"))]
    mpath = os.path.join(base, metas[0])
    meta = (pd.read_json(mpath, lines=True) if mpath.endswith(".jsonl")
            else pd.read_csv(mpath, sep="\t" if mpath.endswith(".tsv")
                             else ","))
    audir = os.path.join(base, "audios")
    files = {os.path.splitext(f)[0]: os.path.join(audir, f)
             for f in sorted(os.listdir(audir))}

    keycol = None                       # modal_bench keying, verbatim
    for c in meta.columns:
        vals = (meta[c].astype(str).str.split("/").str[-1]
                .str.replace(r"\.[A-Za-z0-9]{1,5}$", "", regex=True))
        if vals.isin(files).mean() > .9:
            keycol = c
            meta["_stem"] = vals
            break
    if keycol is None:
        raise RuntimeError(f"no column matches the audio stems "
                           f"(cols {list(meta.columns)})")

    norm_cols = {c.lower().strip(): c for c in meta.columns}
    q_field = next(norm_cols[c] for c in
                   ("prompt", "question", "query", "text") if c in
                   norm_cols)
    ref_field = norm_cols.get("参考答案") or norm_cols.get("answer")

    used_stems = set()
    used_text = set()
    for q in _read_jsonl(SREASON_Q):
        used_stems.add(str(q.get("source", "")).split("/")[-1])
        used_text.add(re.sub(r"\s+", " ", q["query"])[:120])
    meta = meta[meta["_stem"].isin(files)]
    meta = meta[~meta["_stem"].isin(used_stems)]
    meta = meta[~meta[q_field].astype(str).str.replace(
        r"\s+", " ", regex=True).str[:120].isin(used_text)]
    meta = meta.reset_index(drop=True)
    print(f">>> reasoning_qa leftovers after eval-pool exclusion: "
          f"{len(meta)}", flush=True)

    os.makedirs(ZHAUDIO, exist_ok=True)
    rng = np.random.default_rng(46)
    rows = []
    for j, di in enumerate(rng.permutation(len(meta))):
        r = meta.iloc[int(di)]
        qid = f"zh{j:04d}"
        arr, _sr = librosa.load(files[r["_stem"]], sr=16000, mono=True)
        sf.write(f"{ZHAUDIO}/{qid}.wav", arr.astype(np.float32), 16000)
        ref = r[ref_field] if ref_field else None
        rows.append({"id": qid, "pool": "zh-reasoning",
                     "source": f"{repo}/{sub}/{r['_stem']}",
                     "query": str(r[q_field]),
                     "reference_answer": (str(ref) if ref is not None
                                          else None),
                     "split": "calib_zh"})
    with open(ZHQ, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> wrote {len(rows)} expansion3zh queries -> {ZHQ}",
          flush=True)


@app.function(image=util_st3, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_tag(tag: str) -> list:
    return _read_jsonl(TAGS[tag][0])


@gen_app.function(image=util_st3, volumes={DATA: gate_data},
                  secrets=[OPENAI], timeout=60 * 60, region=API_REGION)
def run_tts3(limit: int = 0, concurrency: int = 8):
    """tts-1/alloy over expansion3 (en) only — expansion3zh ships the
    official OpenAudioBench audio, no TTS."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    qs = _read_jsonl(ZQ)[:limit or None]
    os.makedirs(ZAUDIO, exist_ok=True)
    client = OpenAI()

    def render(q):
        out = f"{ZAUDIO}/{q['id']}.wav"
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
    print(f">>> tts3: {res.count('done')} rendered, "
          f"{res.count('cached')} cached / {len(qs)}", flush=True)


@app.function(image=image_st3, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 2)
def answer3_shard(shard: list, shard_id: int, tag: str) -> int:
    """MiniCPM answers each query from AUDIO, turn-based (6a collection
    style, same as answer2_shard) — these are the LABELS, which stay
    turn-based by §8bb methodology."""
    import glob as _glob
    import shutil
    import librosa
    import torch
    from transformers import AutoModel, AutoTokenizer

    from modal_app import MODEL_DIR
    audio_dir = TAGS[tag][1]
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
        au, _ = librosa.load(f"{audio_dir}/{q['id']}.wav", sr=16000,
                             mono=True)
        ans = model.chat(msgs=[{"role": "user", "content": [au]}],
                         tokenizer=tok, max_new_tokens=512,
                         sampling=False, use_tts_template=False,
                         generate_audio=False)
        rows.append({"id": q["id"], "answer": str(ans).strip()})
        if k < 3 or k % 40 == 0:
            print(f"  [{k}] {q['id']} :: {rows[-1]['answer'][:60]!r}",
                  flush=True)
    pd.DataFrame(rows).to_parquet(
        f"{DATA}/{tag}_answers.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote answers shard {shard_id} ({len(rows)})", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_answer3(tag: str = "expansion3", workers: int = 4,
                limit: int = 0):
    """Then: modal run modal_train3.py::judge_expansion3 --tag <tag>"""
    qs = _read_tag.remote(tag)
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> answering {len(qs)} {tag} queries / {workers} workers")
    total = sum(answer3_shard.starmap(
        [(shards[i], i, tag) for i in range(workers)]))
    print(f">>> answered {total}")


@gen_app.function(image=util_st3, volumes={DATA: gate_data},
                  secrets=[OPENAI], timeout=60 * 60, region=API_REGION)
def judge_expansion3(tag: str = "expansion3"):
    """gpt judge (escalate.judge_many — handles zh; the sreason
    turn-based labels came through the same judge) ->
    {tag}_labels.parquet."""
    import glob as _glob
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    answers = pd.concat(
        [pd.read_parquet(s) for s in
         sorted(_glob.glob(f"{DATA}/{tag}_answers.shard*.parquet"))],
        ignore_index=True).drop_duplicates(subset="id", keep="last")
    print(f">>> judging {len(answers)} answers", flush=True)
    byid = {q["id"]: q for q in _read_jsonl(TAGS[tag][0])}
    rows = [{"id": a["id"], "query": byid[a["id"]]["query"],
             "reference_answer": byid[a["id"]]["reference_answer"],
             "answer": a["answer"]} for _, a in answers.iterrows()]
    judged = asyncio.run(escalate.judge_many(rows, concurrency=8))
    df = pd.DataFrame(judged)
    df["pool"] = [byid[i]["pool"] for i in df["id"]]
    df.to_parquet(f"{DATA}/{tag}_labels.parquet")
    gate_data.commit()
    ok = df[df["adequate"].notna()]
    print(f"\n=== {tag} labels (n={len(ok)}) ===", flush=True)
    for p, g in ok.groupby("pool"):
        print(f"  {p:16s} n={len(g):3d} fail-rate="
              f"{1 - g['adequate'].mean():.3f}", flush=True)
