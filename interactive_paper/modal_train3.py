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
     expansion3zh v2: the original OpenAudioBench-leftover plan is
     impossible (reasoning_qa has only 202 rows; the eval pool sampled
     ALL of them — measured 2026-09-01). Instead MGSM zh (250) + XCOPA
     zh (150), public benchmarks through the same TTS pipeline as the
     en families, source-disjoint from sreason — which therefore STAYS
     a fully external transfer pool.

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
    """The zh calib slice, v2. The original plan (OpenAudioBench
    reasoning_qa leftovers) is IMPOSSIBLE: the subset has only 202 rows
    and the sreason eval pool sampled ALL of them (measured 2026-09-01;
    the assumed ~355-row remainder does not exist). Instead: public zh
    benchmarks through the same TTS pipeline as the en families —
    MGSM zh (250, human-translated GSM8K math word problems) + XCOPA zh
    (150 of 500, causal commonsense MC). BOTH are source-disjoint from
    sreason, so it STAYS a fully external transfer pool — strictly
    cleaner than the original in-domain plan. ~400 rows."""
    import numpy as np
    from datasets import load_dataset

    rng = np.random.default_rng(46)
    rows = []

    def add(pool, items):
        for q, ref in items:
            rows.append({"id": f"zh{len(rows):04d}", "pool": pool,
                         "query": q, "reference_answer": ref,
                         "split": "calib_zh"})
        print(f">>> {pool}: +{len(items)}", flush=True)

    def take(ds, fmt, n):
        out, seen = [], set()
        for i in rng.permutation(len(ds)):
            r = fmt(ds[int(i)])
            if r is None:
                continue
            k = re.sub(r"\s+", " ", r[0])[:120]
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
            if len(out) >= n:
                break
        return out

    mgsm = load_dataset("juletxara/mgsm", "zh", split="test",
                        trust_remote_code=True)

    def fmt_mgsm(e):
        a = e["answer_number"]
        ref = (str(int(a)) if float(a) == int(float(a)) else str(a))
        return (e["question"].strip(), ref)
    add("zh-mathword", take(mgsm, fmt_mgsm, 250))

    xcopa = load_dataset("cambridgeltl/xcopa", "zh", split="test",
                         trust_remote_code=True)

    def fmt_xcopa(e):
        what = "原因" if e["question"] == "cause" else "结果"
        q = (f"{e['premise']} 更可能的{what}是哪一个？"
             f"(A) {e['choice1']} (B) {e['choice2']}")
        k = "AB"[int(e["label"])]
        ref = f"({k}) {e['choice1'] if k == 'A' else e['choice2']}"
        return (q, ref)
    add("zh-causal", take(xcopa, fmt_xcopa, 150))

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
    """tts-1/alloy over expansion3 (en) + expansion3zh (zh; alloy is
    the pinned en+zh voice — same as the frozen pool's zh items)."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    client = OpenAI()
    for qfile, audir in ((ZQ, ZAUDIO), (ZHQ, ZHAUDIO)):
        try:
            qs = _read_jsonl(qfile)[:limit or None]
        except FileNotFoundError:
            print(f">>> {qfile} missing — skipped", flush=True)
            continue
        os.makedirs(audir, exist_ok=True)

        def render(q):
            out = f"{audir}/{q['id']}.wav"
            if os.path.exists(out) and os.path.getsize(out) > 44:
                return "cached"
            resp = client.audio.speech.create(
                model="tts-1", voice=TTS_VOICE,
                input=q["query"][:4000], response_format="wav")
            with open(out, "wb") as fh:
                fh.write(resp.content)
            return "done"

        with ThreadPoolExecutor(concurrency) as ex:
            res = list(ex.map(render, qs))
        gate_data.commit()
        print(f">>> tts3 [{os.path.basename(audir)}]: "
              f"{res.count('done')} rendered, "
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
