"""External speech-QA benchmark arm (user request 2026-08-12): the live
gated sweep on OpenAudioBench renders — public audio, not our TTS.

Benches: striviaqa (Speech TriviaQA, official MiniCPM-o 4.5 = 75.5) and
swebq (Speech Web Questions, official = 70.2). 250 queries each (seed
42). Same frozen L22 gate + eot thresholds as the 600-pool sweep, zero
recalibration; same STALL/RELAY protocol; expert gpt-5.5 low via cache.
SD-QA (real speech) reuses modal_stream.py's own pipeline.

Order:
  modal run modal_bench.py::build
  modal run modal_bench.py::run_transcribe --bench striviaqa   (and swebq)
  modal run modal_bench.py::run_live --bench striviaqa --tier never  (x4 tiers)
  modal run modal_bench.py::report --bench striviaqa
"""
import json
import os
import sys
import time

import modal

from modal_app import (app, gen_app, image, util_image, GPU_VOL, gate_data,
                       DATA, OPENAI, API_REGION, _read_jsonl)

HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(HERE, "modal_app.py")
image_st = image.add_local_file(_APP_PY, "/root/modal_app.py")
util_st = util_image.add_local_file(_APP_PY, "/root/modal_app.py")
dl_image = (modal.Image.debian_slim(python_version="3.11")
            .apt_install("libsndfile1")
            .pip_install("datasets==2.21.0", "huggingface_hub[hf_transfer]",
                         "pandas", "pyarrow", "librosa", "soundfile")
            .env({"HF_HUB_DISABLE_XET": "1"})   # unauth Xet 429s/stalls
            .add_local_file(_APP_PY, "/root/modal_app.py"))

BENCHES = {
    "striviaqa": {"match": "trivia", "n": 250},
    "swebq": {"match": "web", "n": 250},
    # added 2026-08-13: short-factoid (llama) + execution-type failures
    # (reasoning) — the two failure species the external arms still miss
    "sllama": {"match": "llama", "n": 250},
    "sreason": {"match": "reasoning", "n": 202},
}
BENCH_AUDIO = f"{DATA}/bench_audio"

# every pool the live loop can sweep (v2 re-run covers all four)
POOLS = {
    "striviaqa": {"audio": BENCH_AUDIO,
                  "q": f"{DATA}/queries_striviaqa.jsonl",
                  "asr": f"{DATA}/striviaqa_transcripts", "split": None},
    "swebq": {"audio": BENCH_AUDIO,
              "q": f"{DATA}/queries_swebq.jsonl",
              "asr": f"{DATA}/swebq_transcripts", "split": None},
    "sdqa": {"audio": f"{DATA}/sdqa_audio",
             "q": f"{DATA}/queries_sdqa.jsonl",
             "asr": f"{DATA}/sdqa_transcripts", "split": None},
    "sllama": {"audio": BENCH_AUDIO,
               "q": f"{DATA}/queries_sllama.jsonl",
               "asr": f"{DATA}/sllama_transcripts", "split": None},
    "sreason": {"audio": BENCH_AUDIO,
                "q": f"{DATA}/queries_sreason.jsonl",
                "asr": f"{DATA}/sreason_transcripts", "split": None},
    "frozen": {"audio": f"{DATA}/audio_pool",
               "q": f"{DATA}/queries.jsonl",
               "asr": f"{DATA}/asr_minicpm-o45-audio", "split": "test"},
    # VoiceBench AlpacaEval: the one speech-QA row of the official
    # condensed matrix (MiniCPM-o 4.5 = 4.8). Open-ended, no reference
    # answers -> scored 1-5 by judge (valpaca_report), not adequate-bool.
    "valpaca": {"audio": BENCH_AUDIO,
                "q": f"{DATA}/queries_valpaca.jsonl",
                "asr": f"{DATA}/valpaca_transcripts", "split": None},
}


def _q_path(bench):
    return POOLS.get(bench, {}).get("q", f"{DATA}/queries_{bench}.jsonl")


def _asr_glob(bench):
    return POOLS.get(bench, {}).get("asr", f"{DATA}/{bench}_transcripts")


def _audio_dir(bench):
    return POOLS.get(bench, {}).get("audio", BENCH_AUDIO)


def _traces(bench, suffix=""):
    return f"{DATA}/{bench}{suffix}_traces.jsonl"


# ---- constants mirrored from modal_stream.py (keep in sync) ---------------
LAYER = 22
GATE_ART = f"{DATA}/midlayer_gate_audio.json"
EOT_SCORES = f"{DATA}/eot_scores"
STALL = "Hmm, that's a good question — give me a moment to check."
RELAY_TMPL = ("[SYSTEM NOTE] A trusted expert system has already verified "
              "the answer to the user's last spoken question.\n"
              "Expert answer: {ans}\n"
              "Relay this answer to the user in your own words, concisely "
              "and naturally, as a continuation of the conversation. Do not "
              "contradict the expert answer.")
ASR_INSTR = ("Transcribe the speech in the audio verbatim. Output ONLY the "
             "transcription. Do not answer any question it contains.")


@app.function(image=dl_image, volumes={DATA: gate_data}, timeout=60 * 40)
def build(n_override: int = 0):
    """Stage both benches from OpenAudioBench's repo layout
    (eval_datas/{name}/{name}.csv + eval_datas/{name}/audios/): 16 kHz
    wavs + queries jsonl in the frozen-pool schema. Introspective —
    prints csv columns and the audio-file keying before mapping."""
    import numpy as np
    import librosa
    import pandas as pd
    import soundfile as sf
    from huggingface_hub import snapshot_download

    repo = "baichuan-inc/OpenAudioBench"
    names = {"striviaqa": "trivia_qa", "swebq": "web_questions",
             "sllama": "llama_questions", "sreason": "reasoning_qa"}
    root = None
    for attempt in range(4):
        try:
            root = snapshot_download(
                repo, repo_type="dataset", max_workers=2,
                allow_patterns=[f"eval_datas/{s}/**"
                                for s in names.values()])
            break
        except Exception as e:
            wait = 90 * (attempt + 1)
            print(f">>> snapshot attempt {attempt}: {str(e)[:200]} — "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)
    if root is None:
        raise RuntimeError("snapshot_download kept failing (HF rate limit)")
    os.makedirs(BENCH_AUDIO, exist_ok=True)
    rng = np.random.default_rng(42)

    for bench, sub in names.items():
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
        print(f">>> {bench}: meta {metas[0]!r} cols={list(meta.columns)} "
              f"n={len(meta)} | {len(files)} audio files, first stems "
              f"{list(files)[:3]}", flush=True)
        print(f"    first row: {meta.iloc[0].to_dict()}", flush=True)

        # key column: values (sans .wav) that match the audio stems
        keycol = None
        for c in meta.columns:
            # strip ANY extension: pools mix .wav and .mp3 filenames
            vals = (meta[c].astype(str).str.split("/").str[-1]
                    .str.replace(r"\.[A-Za-z0-9]{1,5}$", "", regex=True))
            if vals.isin(files).mean() > .9:
                keycol = c
                meta["_stem"] = vals
                break
        if keycol is None:
            # NEVER fall back to row order — it silently pairs the wrong
            # audio with the wrong question (hit on reasoning_qa: .mp3
            # filenames vs a .wav-only strip)
            raise RuntimeError(
                f"{bench}: no column matches the audio stems "
                f"(cols {list(meta.columns)}; stems e.g. "
                f"{list(files)[:3]})")
        print(f">>> {bench}: keyed by {keycol!r}", flush=True)

        norm_cols = {c.lower().strip(): c for c in meta.columns}
        q_field = next((norm_cols[c] for c in
                        ("prompt", "question", "query", "text",
                         "instruction", "questions") if c in norm_cols),
                       None)
        ref_field = next((norm_cols[c] for c in
                          ("reference", "answer", "answers",
                           "answer_normalized_value", "gt",
                           "output", "response", "label",
                           "参考答案")          # reasoning_qa is Chinese
                          if c in norm_cols), None)
        alias_field = norm_cols.get("answer_normalized_aliases")
        if not q_field:
            raise RuntimeError(f"{bench}: no question column in "
                               f"{list(meta.columns)}")
        print(f">>> {bench}: q_field={q_field!r} ref_field={ref_field!r}",
              flush=True)

        n = n_override or BENCHES[bench]["n"]
        meta = meta[meta["_stem"].isin(files)].reset_index(drop=True)
        order = rng.permutation(len(meta))[:n]
        rows = []
        for j, di in enumerate(order):
            r = meta.iloc[int(di)]
            arr, sr = librosa.load(files[r["_stem"]], sr=16000, mono=True)
            qid = f"{bench}{j:04d}"
            sf.write(f"{BENCH_AUDIO}/{qid}.wav",
                     arr.astype(np.float32), 16000)
            ref = r[ref_field] if ref_field else None
            if isinstance(ref, (list, tuple)):
                ref = "; ".join(str(x) for x in ref)
            if ref is not None and alias_field and pd.notna(r[alias_field]):
                ref = f"{ref} (accepted aliases: {str(r[alias_field])[:200]})"
            rows.append({"id": qid, "pool": bench,
                         "source": f"{repo}/{sub}/{r['_stem']}",
                         "query": str(r[q_field]),
                         "reference_answer": (str(ref) if ref is not None
                                              else None),
                         "split": "test"})
        with open(_q_path(bench), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f">>> {bench}: wrote {len(rows)} queries + wavs "
              f"(refs: {'yes' if ref_field else 'NO'})", flush=True)
        for r in rows[:3]:
            print(f"    {r['id']}: {r['query'][:70]!r} -> "
                  f"{str(r['reference_answer'])[:40]!r}", flush=True)
    gate_data.commit()


@app.function(image=dl_image, volumes={DATA: gate_data}, timeout=60 * 30)
def build_valpaca():
    """Stage VoiceBench's alpacaeval subset (the official-matrix row):
    16 kHz wavs into bench_audio/ + queries_valpaca.jsonl. Introspective
    like build_sdqa."""
    import numpy as np
    import librosa
    import soundfile as sf
    from datasets import load_dataset, get_dataset_config_names

    configs = get_dataset_config_names("hlt-lab/voicebench")
    cfg = next((c for c in configs
                if "alpaca" in c.lower().replace("_", "")), None)
    if cfg is None:
        raise RuntimeError(f"no alpacaeval config in {configs}")
    ds = load_dataset("hlt-lab/voicebench", cfg)
    split = list(ds.keys())[0]
    d = ds[split]
    print(f">>> voicebench/{cfg}/{split}: n={len(d)} "
          f"features={d.features}", flush=True)
    q_field = next((f for f in ("prompt", "question", "instruction",
                                "text") if f in d.features), None)
    os.makedirs(BENCH_AUDIO, exist_ok=True)
    rows = []
    for i, ex in enumerate(d):
        au = ex["audio"]
        arr = np.asarray(au["array"], dtype=np.float32)
        if au["sampling_rate"] != 16000:
            arr = librosa.resample(arr, orig_sr=au["sampling_rate"],
                                   target_sr=16000)
        qid = f"valpaca{i:04d}"
        sf.write(f"{BENCH_AUDIO}/{qid}.wav", arr, 16000)
        rows.append({"id": qid, "pool": "valpaca",
                     "source": f"voicebench/{cfg}/{split}#{i}",
                     "query": ex[q_field], "reference_answer": None,
                     "split": "test"})
    with open(_q_path("valpaca"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> wrote {len(rows)} valpaca queries + wavs", flush=True)


# ---- 1-5 judge: VoiceBench's OWN meta_prompt_open, verbatim --------------
# Copied from MatthewCYM/VoiceBench api_judge.py so our absolute numbers
# are comparable to the officially reported AlpacaEval score (4.8).
# Their judge is gpt-4o-mini + this exact prompt, score-only output.
VB_JUDGE_MODEL = "gpt-4o-mini"
VB_META_PROMPT_OPEN = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user’s question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don’t contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user’s question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user’s query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user’s instruction and models’ response:
### [Instruction]: {prompt}
### [Response]: {response}

After evaluating, please output the score only without anything else.
You don’t need to provide any explanations.
"""


def _score_many(rows, concurrency=3):
    """rows: [{id, query, answer}] -> add 'score' (1-5, None on error).
    Uses VoiceBench's exact judge model + prompt (no structured output —
    the official prompt asks for a bare number)."""
    import asyncio
    import re as _re
    sys.path.insert(0, "/workspace/gate")
    import escalate

    client = escalate._async_client()
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        prompt = (VB_META_PROMPT_OPEN.replace("{prompt}", str(r["query"]))
                  .replace("{response}", str(r["answer"])))
        r["score"], r["score_err"] = None, ""
        for attempt in range(6):        # gpt-4o-mini 429s under batch load
            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=VB_JUDGE_MODEL, max_tokens=1024,
                        frequency_penalty=0, presence_penalty=0,
                        messages=[
                            {"role": "system", "content":
                             "You are a helpful assistant who tries to help "
                             "answer the user's question."},
                            {"role": "user", "content": prompt}],
                        user=escalate.USER_ID,
                    )
                    txt = (resp.choices[0].message.content or "").strip()
                    m = _re.search(r"\d+", txt)
                    v = int(m.group()) if m else None
                    if v is not None and 1 <= v <= 5:
                        r["score"] = v
                        return r
                    r["score_err"] = f"unparsable: {txt[:80]!r}"
                except Exception as e:
                    r["score_err"] = f"{type(e).__name__}: {str(e)[:150]}"
            await asyncio.sleep(min(60, 3 * 2 ** attempt))
        return r

    async def run():
        return await asyncio.gather(*(one(r) for r in rows))
    return asyncio.run(run())


# ---- OpenAudioBench's OWN judge, verbatim from tasks/trivia_qa_audio.py --
# (gpt-4o-2024-08-06, JSON analysis+judgment, "correct if it matches at
# least ONE reference alias"). Needed to know whether the officially
# reported numbers are reproducible with our answers.
OAB_JUDGE_MODEL = "gpt-4o-2024-08-06"
OAB_PATTERN = """
Your will be given a question, the reference answers to that question, and an answer to be judged. Your tasks is to judge whether the answer to be judged is correct, given the question and reference answers. An answer considered correct expresses or contains the same meaning as at least **one of** the reference answers. The format and the tone of the response does not matter.

You should respond in JSON format. First provide a one-sentence concise analysis for the judgement in field 'analysis', then your judgment in field 'judgment'. For example,
'''json
{{"analysis": "<a one-sentence concise analysis for the judgement>", "judgment": < your final judgment, "correct" or "incorrect">}}
'''

# Question
{instruction}

# Reference Answer
{targets}

# Answer To Be Judged
{answer}

"""


def _oab_judge(rows, concurrency=3):
    """rows: [{query, reference_answer, answer}] -> add 'oab_ok' (0/1)."""
    import asyncio
    import re as _re
    sys.path.insert(0, "/workspace/gate")
    import escalate

    client = escalate._async_client()
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        p = (OAB_PATTERN.replace("{instruction}", str(r["query"]))
             .replace("{targets}", str(r.get("reference_answer")))
             .replace("{answer}", str(r["answer"])))
        r["oab_ok"] = None
        for attempt in range(5):
            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=OAB_JUDGE_MODEL, max_tokens=512,
                        messages=[{"role": "user", "content": p}],
                        user=escalate.USER_ID)
                    txt = (resp.choices[0].message.content or "")
                    m = _re.search(r'"judgment"\s*:\s*"?(correct|incorrect)',
                                   txt, _re.I)
                    if m:
                        r["oab_ok"] = int(m.group(1).lower() == "correct")
                        return r
                except Exception as e:
                    r["oab_err"] = f"{type(e).__name__}: {str(e)[:120]}"
            await asyncio.sleep(min(60, 3 * 2 ** attempt))
        return r

    async def run():
        return await asyncio.gather(*(one(r) for r in rows))
    return asyncio.run(run())


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 60, region=API_REGION)
def oab_rejudge(bench: str = "striviaqa"):
    """Re-score the chat-mode answers with OpenAudioBench's own judge, to
    test whether the officially reported number is reproducible."""
    import pandas as pd

    df = pd.read_parquet(f"{DATA}/{bench}_chatmode.parquet")
    rows = df.to_dict("records")
    print(f">>> OAB-judging {len(rows)} {bench} chat-mode answers",
          flush=True)
    scored = _oab_judge(rows)
    out = pd.DataFrame(scored)
    out.to_parquet(f"{DATA}/{bench}_chatmode_oab.parquet")
    gate_data.commit()
    ok = out[out["oab_ok"].notna()]
    print(f">>> {bench}: OUR judge acc={df['score'].mean():.3f} | "
          f"OAB judge acc={ok['oab_ok'].mean():.3f} (n={len(ok)})",
          flush=True)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 90, region=API_REGION)
def oab_rejudge_live(bench: str = "striviaqa", suffix: str = "_v2"):
    """Re-score every live arm AND the gpt-5.5 ceiling with
    OpenAudioBench's own judge, so the officially reported numbers (and
    the other models' numbers in the same table) are on the same scale
    as our curves. Adds 'oab_ok' to {bench}{suffix}_traces.parquet and
    {bench}_ceiling.parquet."""
    import pandas as pd

    df = pd.read_parquet(f"{DATA}/{bench}{suffix}_traces.parquet")
    rows = [{"query": r["query"],
             "reference_answer": r["reference_answer"],
             "answer": (r.get("relay") if r["mode"] == "escalated"
                        else r.get("answer", "")),
             "_k": f"{r['id']}|{r['tier']}"}
            for _, r in df.iterrows()]
    print(f">>> OAB-judging {len(rows)} live {bench} answers", flush=True)
    scored = _oab_judge(rows)
    m = {r["_k"]: r["oab_ok"] for r in scored}
    df["oab_ok"] = [m.get(f"{r['id']}|{r['tier']}")
                    for _, r in df.iterrows()]
    df.to_parquet(f"{DATA}/{bench}{suffix}_traces.parquet")

    cp = f"{DATA}/{bench}_ceiling.parquet"
    ce = pd.read_parquet(cp)
    crows = [{"query": r["query"], "reference_answer": r["reference_answer"],
              "answer": r["answer"], "_k": r["id"]}
             for _, r in ce.iterrows() if r.get("answer")]
    print(f">>> OAB-judging {len(crows)} ceiling answers", flush=True)
    cs = _oab_judge(crows)
    cm = {r["_k"]: r["oab_ok"] for r in cs}
    ce["oab_ok"] = [cm.get(i) for i in ce["id"]]
    ce.to_parquet(cp)
    gate_data.commit()

    ok = df[df["oab_ok"].notna()]
    print(f"\n=== {bench} under the OFFICIAL (OpenAudioBench) judge ===",
          flush=True)
    for t, g in ok.groupby("tier"):
        old = g["heard_ok"].mean() if "heard_ok" in g else float("nan")
        print(f"  [{t}] n={len(g)} ours={old:.3f} -> official-judge "
              f"{g['oab_ok'].mean():.3f}", flush=True)
    print(f"  ceiling: {ce['oab_ok'].dropna().mean():.3f}", flush=True)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 60, region=API_REGION)
def valpaca_ceiling():
    """gpt-5.5 (low) answers each gold instruction; judged 1-5."""
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate
    from modal_app import EXPERT_CACHE

    qs = _read_jsonl(_q_path("valpaca"))
    answers = asyncio.run(escalate.ask_expert_many(
        [q["query"] for q in qs], concurrency=3, effort="low",
        cache_dir=EXPERT_CACHE))
    rows = [{"id": q["id"], "query": q["query"],
             "answer": a.get("answer") or "", "latency_s": a.get("latency_s")}
            for q, a in zip(qs, answers)]
    rows = _score_many([r for r in rows if r["answer"]])
    out = pd.DataFrame(rows)
    out.to_parquet(f"{DATA}/valpaca_ceiling.parquet")
    gate_data.commit()
    ok = out[out["score"].notna()]
    print(f">>> valpaca ceiling: mean score "
          f"{ok['score'].mean():.2f} (n={len(ok)})", flush=True)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 60, region=API_REGION)
def valpaca_report(suffix: str = "_v2", never_glob: str = ""):
    """Judge every heard answer 1-5, per-arm mean score -> parquet+json.
    never_glob: extra glob for never-arm traces (probe-independent)."""
    import glob as _glob
    import pandas as pd

    paths = sorted(_glob.glob(f"{_traces('valpaca', suffix)}.*.shard*"))
    if never_glob:
        paths += sorted(_glob.glob(never_glob))
    print(f">>> trace files: {paths}", flush=True)
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    df = pd.DataFrame(rows).drop_duplicates(subset=["id", "tier"],
                                            keep="last")
    heard = [{"id": f"{r['id']}|{r['tier']}", "query": r["query"],
              "answer": (r.get("relay") if r["mode"] == "escalated"
                         else r.get("answer", ""))}
             for _, r in df.iterrows()]
    scored = _score_many(heard)
    smap = {r["id"]: r["score"] for r in scored}
    emap = {r["id"]: r.get("score_err", "") for r in scored}
    key = [f"{r['id']}|{r['tier']}" for _, r in df.iterrows()]
    df["score"] = [smap.get(k) for k in key]
    df["score_err"] = [emap.get(k, "") for k in key]
    df.to_parquet(f"{DATA}/valpaca{suffix}_scored.parquet")
    gate_data.commit()
    ok = df[df["score"].notna()]
    stats = {}
    print(f"\n=== valpaca judge score (1-5) per arm ===", flush=True)
    for tname, g in ok.groupby("tier"):
        e = g[g["mode"] == "escalated"]
        stats[tname] = {"n": int(len(g)),
                        "esc": float(len(e) / max(1, len(g))),
                        "score": float(g["score"].mean())}
        print(f"  [{tname}] n={len(g)} esc={stats[tname]['esc']:.2f} "
              f"score {stats[tname]['score']:.2f}", flush=True)
    with open(f"{DATA}/valpaca{suffix}_live.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    gate_data.commit()


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 20, region=API_REGION)
def debug_judge(n: int = 3):
    """Print the RAW judge response for rows that scored None."""
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    df = pd.read_parquet(f"{DATA}/valpaca_v2_scored.parquet")
    bad = df[df["score"].isna()].head(n)
    client = escalate._client()
    for _, r in bad.iterrows():
        ans = r.get("relay") if r["mode"] == "escalated" else r.get("answer")
        prompt = (VB_META_PROMPT_OPEN.replace("{prompt}", str(r["query"]))
                  .replace("{response}", str(ans)))
        try:
            resp = client.chat.completions.create(
                model=VB_JUDGE_MODEL, max_tokens=1024,
                frequency_penalty=0, presence_penalty=0,
                messages=[{"role": "system", "content":
                           "You are a helpful assistant who tries to help "
                           "answer the user's question."},
                          {"role": "user", "content": prompt}],
                user=escalate.USER_ID)
            m = resp.choices[0].message
            print(f"[{r['id']}/{r['tier']}] finish={resp.choices[0].finish_reason} "
                  f"refusal={getattr(m, 'refusal', None)} "
                  f"content={m.content!r} ans_len={len(str(ans))}", flush=True)
        except Exception as e:
            print(f"[{r['id']}/{r['tier']}] EXC {type(e).__name__}: "
                  f"{str(e)[:300]}", flush=True)


@app.function(image=image_st, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 2)
def valpaca_chatmode(max_new_tokens: int = 1024,
                     bench: str = "valpaca") -> list:
    """Fairness control for every official number: answer the SAME wavs
    in offline chat mode (model.chat, one shot, no streaming loop, no
    chunked prefill, no EOT read, generous token budget) — the setting
    the official numbers are measured in. Judged with the same judge as
    the live arms, so the only difference vs our live floor is the loop.
    """
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
    qs = _read_jsonl(_q_path(bench))
    rows = []
    for k, q in enumerate(qs):
        au, _ = librosa.load(f"{_audio_dir(bench)}/{q['id']}.wav",
                             sr=16000, mono=True)
        ans = model.chat(msgs=[{"role": "user", "content": [au]}],
                         tokenizer=tok, max_new_tokens=max_new_tokens,
                         sampling=False, use_tts_template=False,
                         generate_audio=False)
        rows.append({"id": q["id"], "query": q["query"],
                     "reference_answer": q.get("reference_answer"),
                     "answer": str(ans).strip()})
        if k < 2 or k % 50 == 0:
            print(f"  [{k}] {rows[-1]['answer'][:70]!r}", flush=True)
    # persist on the volume before any downstream step touches it
    pd.DataFrame(rows).to_parquet(f"{DATA}/{bench}_chatmode_raw.parquet")
    gate_data.commit()
    print(f">>> wrote {len(rows)} {bench} chat-mode answers", flush=True)
    return len(rows)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 60, region=API_REGION)
def score_chatmode(bench: str = "valpaca"):
    """valpaca -> VoiceBench 1-5 judge; the QA pools -> the same
    adequate-bool judge the live arms use."""
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    rows = pd.read_parquet(
        f"{DATA}/{bench}_chatmode_raw.parquet").to_dict("records")
    print(f">>> scoring {len(rows)} {bench} chat-mode answers", flush=True)
    if bench == "valpaca":
        scored = _score_many(rows)
        df = pd.DataFrame(scored)
        metric = "score"
    else:
        judged = asyncio.run(escalate.judge_many(rows, concurrency=8))
        df = pd.DataFrame(judged)
        df["score"] = df["adequate"].map(
            lambda x: 1.0 if x is True or x == 1 else
            (0.0 if x is False or x == 0 else None))
        metric = "acc"
    df["len"] = df["answer"].fillna("").str.len()
    df.to_parquet(f"{DATA}/{bench}_chatmode.parquet")
    gate_data.commit()
    ok = df[df["score"].notna()]
    print(f">>> {bench} chat-mode (offline, official-style) {metric}="
          f"{ok['score'].mean():.3f} (n={len(ok)}), median len "
          f"{ok['len'].median():.0f}", flush=True)


@app.local_entrypoint()
def run_chatmode(max_new_tokens: int = 1024, bench: str = "valpaca"):
    n = valpaca_chatmode.remote(max_new_tokens, bench)
    print(f">>> answered {n}; now: modal run "
          f"modal_bench.py::score_chatmode --bench {bench}")


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_bench(bench: str) -> list:
    return _read_jsonl(_q_path(bench))


@app.function(image=image_st, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def transcribe(bench: str, shard: list, shard_id: int) -> int:
    """Talker self-transcription (same mechanism as sdqa_transcribe)."""
    import glob as _glob
    import shutil
    import librosa
    import pandas as pd
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

    rows = []
    for k, q in enumerate(shard):
        au, _ = librosa.load(f"{_audio_dir(bench)}/{q['id']}.wav", sr=16000,
                             mono=True)
        out = model.chat(msgs=[{"role": "user", "content": [au, ASR_INSTR]}],
                         tokenizer=tok, max_new_tokens=512, sampling=False,
                         use_tts_template=False, generate_audio=False)
        rows.append({"id": q["id"], "pool": q["pool"],
                     "transcript": str(out).strip()})
        if k < 3 or k % 25 == 0:
            print(f"  [{k}] {q['id']} :: {rows[-1]['transcript'][:70]!r}",
                  flush=True)
    pd.DataFrame(rows).to_parquet(
        f"{_asr_glob(bench)}.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote {bench} transcripts shard {shard_id} ({len(rows)})",
          flush=True)
    return len(rows)


@app.local_entrypoint()
def run_transcribe(bench: str = "striviaqa", workers: int = 2,
                   limit: int = 0):
    qs = _read_bench.remote(bench)
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {bench} self-transcription: {len(qs)} wavs / {workers} "
          f"workers")
    total = sum(transcribe.starmap(
        [(bench, shards[i], i if not limit else 99)
         for i in range(workers)]))
    print(f">>> transcribed {total}")


@gen_app.function(image=image_st, gpu="H100", volumes=GPU_VOL,
                  secrets=[OPENAI], timeout=60 * 60 * 3)
def bench_live(bench: str, tier: str = "balanced", limit: int = 0,
               shard: list = None, shard_id: int = -1,
               art_path: str = "", suffix: str = "",
               sys_suffix: str = "") -> list:
    """The gated live loop on any registered pool (POOLS). art_path
    selects the gate artifact (default = the v1 global-threshold one);
    suffix namespaces the trace files (e.g. '_v2'). sys_suffix appends
    an extra instruction to the stock omni system prompt (8ab Q4
    anti-hedge arm) — default empty = byte-identical behavior."""
    import glob as _glob
    import inspect
    import shutil
    import threading
    import numpy as np
    import torch
    import librosa
    import pandas as pd
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import escalate
    import gate as gate_mod

    from modal_app import MODEL_DIR, EXPERT_CACHE
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
    art = json.load(open(art_path or GATE_ART))
    probe = gate_mod.Probe(art["w"], art["b"])
    thr = {"never": 1e9, "always": -1e9}.get(tier) or \
        art["eot_thresholds"][tier]
    print(f">>> {bench} live: L{art.get('layer', art.get('layer_set'))} "
          f"eot read, probe v"
          f"{art.get('version', 1)}, tier={tier} thr={thr:.3f} "
          f"[{art.get('threshold_source', 'frozen calib quantiles')}], "
          f"expert effort=low", flush=True)

    asr = pd.concat([pd.read_parquet(s) for s in
                     sorted(_glob.glob(f"{_asr_glob(bench)}.shard*.parquet"))],
                    ignore_index=True).drop_duplicates(subset="id",
                                                       keep="last")
    transcript = dict(zip(asr["id"], asr["transcript"]))

    split = POOLS.get(bench, {}).get("split")
    if shard is not None:
        queries = shard
    else:
        queries = [q for q in _read_jsonl(_q_path(bench))
                   if q["id"] in transcript][:limit or None]
    queries = [q for q in queries if q["id"] in transcript
               and (split is None or q.get("split") == split)]
    print(f">>> {len(queries)} {bench} queries", flush=True)

    def call_def(fn, /, **kw):
        params = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in params})

    def gen_text(**kw):
        kw.setdefault("max_new_tokens", 512)
        res = call_def(model.streaming_generate, tokenizer=tok,
                       temperature=0.1, generate_audio=False, **kw)
        parts = []
        if inspect.isgenerator(res) or hasattr(res, "__next__"):
            for r in res:
                t = getattr(r, "text", None)
                if t is None and isinstance(r, dict):
                    t = r.get("text")
                if t is None and isinstance(r, (tuple, list)) and r:
                    t = r[0]
                if isinstance(t, str):
                    parts.append(t)
        else:
            parts.append(str(res))
        return "".join(parts).strip()

    holder = {}
    # v3 (8z) multi-position read: rolling last-K tail across forwards
    # (streaming assistant prefill is 1-token forwards) + running mean
    # over user-audio positions; feature = concat per art["modes"].
    v3 = art.get("version", 1) >= 3
    K3 = art.get("k_eot", 8)
    st3 = {"accum": False, "tail": None, "sum": None, "cnt": 0}

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if not v3:
            holder["h"] = hs[0, -1, :].detach().float().cpu().numpy()
            return
        h = hs[0].detach().float()
        t = h[-K3:].cpu()
        st3["tail"] = (t if st3["tail"] is None
                       else torch.cat([st3["tail"], t])[-K3:])
        if st3["accum"]:
            s = h.sum(0).cpu()
            st3["sum"] = s if st3["sum"] is None else st3["sum"] + s
            st3["cnt"] += h.shape[0]

    def score_now():
        if not v3:
            return float(probe.score(holder["h"]))
        parts = []
        for m in art["modes"]:
            if m == "eot_last":
                parts.append(st3["tail"][-1])
            elif m == "eot_mean":
                parts.append(st3["tail"].mean(0))
            elif m == "user_mean":
                parts.append(st3["sum"] / max(1, st3["cnt"]))
        return float(probe.score(torch.cat(parts).numpy()))

    traces = []
    for qi, q in enumerate(queries):
        au, _ = librosa.load(f"{_audio_dir(bench)}/{q['id']}.wav", sr=16000,
                             mono=True)
        chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        model.reset_session()
        sys_msg = call_def(model.get_sys_prompt, mode="omni", language="en")
        if sys_suffix:
            c = sys_msg.get("content")
            if isinstance(c, str):
                sys_msg["content"] = c + " " + sys_suffix
            elif isinstance(c, list):
                for i2, part in enumerate(c):
                    if isinstance(part, str):
                        c[i2] = part + " " + sys_suffix
                        break
                else:
                    c.append(sys_suffix)
        call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
                 tokenizer=tok)
        h = model.llm.model.layers[LAYER].register_forward_hook(hook)
        st3.update(tail=None, sum=None, cnt=0, accum=v3)
        scores = []
        try:
            for i, ch in enumerate(chunks):
                if len(ch) < 16000:
                    ch = np.pad(ch, (0, 16000 - len(ch)))
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[{"role": "user",
                                "content": [ch.astype(np.float32)]}],
                         tokenizer=tok,
                         is_last_chunk=(i == len(chunks) - 1))
                scores.append(round(score_now(), 4))
            st3["accum"] = False      # assistant tokens stay out of user_mean
            t_eot0 = time.time()
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "assistant", "content": [" "]}],
                     tokenizer=tok, is_last_chunk=True)
            eot_score = score_now()
            eot_ms = int((time.time() - t_eot0) * 1000)
        finally:
            h.remove()

        fired = eot_score >= thr
        t_eot = time.time()
        row = {"id": q["id"], "pool": q["pool"], "tier": tier,
               "audio_s": round(len(au) / 16000, 2),
               "n_chunks": len(chunks), "scores": scores,
               "eot_score": round(eot_score, 4), "eot_read_ms": eot_ms,
               "reference_answer": q.get("reference_answer"),
               "query": q["query"], "transcript": transcript[q["id"]]}
        if not fired:
            ans = gen_text(session_id="s1")
            row.update(mode="local", answer=ans,
                       answer_ms=int((time.time() - t_eot) * 1000))
        else:
            expert = {}

            def expert_call(query_text):
                t0 = time.time()
                r = escalate.ask_expert(query_text, effort="low",
                                        cache_dir=EXPERT_CACHE)
                expert["answer"] = (r.get("answer")
                                    or f"[error: {r.get('error')}]")
                expert["cached_latency_s"] = r.get("latency_s")
                expert["wall_s"] = time.time() - t0

            th = threading.Thread(target=expert_call,
                                  args=(transcript[q["id"]],), daemon=True)
            th.start()
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "assistant", "content": [STALL]}],
                     tokenizer=tok, is_last_chunk=True)
            t_stall = time.time()
            th.join(timeout=120)
            t_expert = time.time()
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "user", "content":
                            [RELAY_TMPL.format(
                                ans=expert.get("answer", ""))]}],
                     tokenizer=tok, is_last_chunk=True)
            relay = gen_text(session_id="s1")
            row.update(mode="escalated", relay=relay,
                       expert_answer=expert.get("answer", ""),
                       expert_latency_s=round(
                           expert.get("cached_latency_s") or
                           expert.get("wall_s", -1), 2),
                       stall_ms=int((t_stall - t_eot) * 1000),
                       relay_ms=int((time.time() - t_expert) * 1000))
        traces.append(row)
        print(f"  [{qi}] {q['id']} eot={eot_score:.3f} "
              f"{'ESC' if fired else 'local':>5s} "
              f"| {(row.get('relay') or row.get('answer', ''))[:60]!r}",
              flush=True)

    out = (f"{_traces(bench, suffix)}.{tier}.smoke" if shard_id < 0
           else f"{_traces(bench, suffix)}.{tier}.shard{shard_id}")
    with open(out, "a", encoding="utf-8") as fh:
        for r in traces:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> appended {len(traces)} traces to {out}", flush=True)
    return [r["id"] for r in traces]


@gen_app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_bench_gen(bench: str) -> list:
    return _read_jsonl(_q_path(bench))


@gen_app.local_entrypoint()
def run_live(bench: str = "striviaqa", tier: str = "balanced",
             workers: int = 3, limit: int = 0, art_path: str = "",
             suffix: str = ""):
    """workers=3 keeps worst-case expert concurrency at 3 (probation cap).
    v2 re-run: --art-path /data/gate_v2_{bench}.json --suffix _v2"""
    qs = _read_bench_gen.remote(bench)
    split = POOLS.get(bench, {}).get("split")
    if split:
        qs = [q for q in qs if q.get("split") == split]
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {bench} live tier={tier}: {len(qs)} queries, "
          f"{workers} workers, art={art_path or 'v1'}")
    done = list(bench_live.starmap(
        [(bench, tier, 0, shards[i], i if not limit else -1, art_path,
          suffix) for i in range(workers)]))
    print(f">>> tier {tier} complete: {sum(len(d) for d in done)} traces")


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 60, region=API_REGION)
def ceiling(bench: str = "striviaqa"):
    """Always-escalate ceiling anchor: gpt-5.5 (low) answers the GOLD
    question text for every pool query; judged with the standard judge.
    Cached via EXPERT_CACHE — never re-sent. Writes {bench}_ceiling.parquet."""
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate
    from modal_app import EXPERT_CACHE

    qpath = (f"{DATA}/queries_sdqa.jsonl" if bench == "sdqa"
             else _q_path(bench))
    qs = _read_jsonl(qpath)
    print(f">>> ceiling: {len(qs)} {bench} gold queries -> "
          f"{escalate.EXPERT_MODEL} low", flush=True)
    answers = asyncio.run(escalate.ask_expert_many(
        [q["query"] for q in qs], concurrency=3, effort="low",
        cache_dir=EXPERT_CACHE))
    rows = [{"id": q["id"], "query": q["query"],
             "reference_answer": q.get("reference_answer"),
             "answer": a.get("answer"),
             "latency_s": a.get("latency_s"), "error": a.get("error")}
            for q, a in zip(qs, answers)]
    judged = asyncio.run(escalate.judge_many(
        [dict(r) for r in rows if r["answer"]], concurrency=8))
    adq = {r["id"]: r["adequate"] for r in judged}
    out = pd.DataFrame(rows)
    out["adequate"] = [adq.get(i) for i in out["id"]]
    out.to_parquet(f"{DATA}/{bench}_ceiling.parquet")
    gate_data.commit()
    ok = out[out["adequate"].notna()]
    print(f">>> {bench} ceiling acc = {ok['adequate'].mean():.3f} "
          f"(n={len(ok)})", flush=True)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 40, region=API_REGION)
def report(bench: str = "striviaqa", suffix: str = "",
           never_glob: str = ""):
    """Judge heard answers, per-tier acc + latency, write
    {bench}{suffix}_live.json + {bench}{suffix}_traces.parquet.
    never_glob: extra glob for never-arm traces to fold in (the never
    arm has thr=1e9 so the probe never fires — those rows are
    probe-independent by construction, so the v2 re-run reuses v1's)."""
    import asyncio
    import glob as _glob
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    paths = sorted(_glob.glob(f"{_traces(bench, suffix)}.*.shard*"))
    if never_glob:
        paths += sorted(_glob.glob(never_glob))
    print(f">>> trace files: {paths}", flush=True)
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    df = pd.DataFrame(rows).drop_duplicates(subset=["id", "tier"],
                                            keep="last")
    print(f"=== {bench} traces: n={len(df)} "
          f"({df.groupby('tier').size().to_dict()}) ===", flush=True)

    heard = [{"query": r["query"], "reference_answer": r["reference_answer"],
              "answer": (r.get("relay") if r["mode"] == "escalated"
                         else r.get("answer", ""))} for _, r in df.iterrows()]
    labeled = asyncio.run(escalate.judge_many(heard, concurrency=8))
    df["heard_ok"] = [1 - x["escalate_label"]
                      if x["escalate_label"] is not None else None
                      for x in labeled]
    ok = df[df["heard_ok"].notna()]

    stats = {}
    print(f"\n=== {bench} heard accuracy (zero recalibration) ===",
          flush=True)
    for tname, g in ok.groupby("tier"):
        e, l = g[g["mode"] == "escalated"], g[g["mode"] == "local"]
        stats[tname] = {
            "n": int(len(g)), "esc": float(len(e) / max(1, len(g))),
            "heard": float(g["heard_ok"].mean()),
            "heard_escalated": (float(e["heard_ok"].mean()) if len(e)
                                else None),
            "heard_local": (float(l["heard_ok"].mean()) if len(l)
                            else None)}
        print(f"  [{tname}] n={len(g)} esc={stats[tname]['esc']:.2f} "
              f"overall {stats[tname]['heard']:.3f}", flush=True)

    art = json.load(open(GATE_ART))
    thr = art["eot_thresholds"]
    eot = (df.sort_values("tier").drop_duplicates(subset="id")
           ["eot_score"].astype(float).to_numpy())
    out = {"tiers": stats,
           "eot_p25_50_75": [float(np.percentile(eot, p))
                             for p in (25, 50, 75)],
           "fire_rates": {t: float((eot >= v).mean())
                          for t, v in thr.items()}}
    with open(f"{DATA}/{bench}{suffix}_live.json", "w") as fh:
        json.dump(out, fh, indent=1)
    df.to_parquet(f"{DATA}/{bench}{suffix}_traces.parquet")
    gate_data.commit()
    print(f">>> wrote {DATA}/{bench}{suffix}_live.json + "
          f"{bench}{suffix}_traces.parquet",
          flush=True)

ANTI_HEDGE = ("If you are not sure of the answer, say only 'I am not "
              "sure.' in five words or fewer. Never explain your "
              "uncertainty or give background; answer in one short "
              "sentence.")


@gen_app.local_entrypoint()
def run_nohedge(bench: str = "striviaqa", limit: int = 0):
    """8ab Q4: anti-hedge never arm. Traces -> {bench}_nohedge_traces."""
    print(bench_live.remote(bench, tier="never", limit=limit,
                            art_path="/data/midlayer_gate_audio_v3.json",
                            suffix="_nohedge", sys_suffix=ANTI_HEDGE))

# ---- 8ab token-level mechanism test (user go 2026-08-24) -----------------
# Per-step full-vocab entropy + P(stop) trajectories during local decode,
# for four groups of striviaqa queries. Predictions: hedged-wrong = high
# early entropy + suppressed stop-prob through the ramble; right = early
# stop spike; confident-wrong = trajectory ~ right (the token-level
# signature of the two error species, and why entropy cannot replace the
# probe).
@gen_app.function(image=image_st, gpu="H100", volumes=GPU_VOL,
                  timeout=60 * 60 * 2)
def entropy_replay(per_group: int = 27, seed: int = 42) -> int:
    import glob as _glob
    import inspect
    import re
    import shutil
    import numpy as np
    import pandas as pd
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
        init_vision=False, init_audio=True, init_tts=False).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # group labels from the measured never arm
    tr = pd.read_parquet(f"{DATA}/striviaqa_v3_traces.parquet")
    nev = tr[tr["tier"] == "never"].set_index("id")
    HED = re.compile(r"there is no|no widely|not a real|i'?m not sure|"
                     r"not sure|might be|likely|however|unfortunately|"
                     r"i don'?t know|as an ai|apolog|unclear|difficult to",
                     re.I)
    hedge = nev["answer"].str.contains(HED).fillna(False)
    wrong = nev["oab_ok"] == 0
    rng = np.random.default_rng(seed)

    def pick(mask, n):
        ids = sorted(nev.index[mask])
        return list(rng.choice(ids, size=min(n, len(ids)),
                               replace=False))
    groups = {"hedged_wrong": pick(wrong & hedge, per_group),
              "confident_wrong": pick(wrong & ~hedge, per_group),
              "right": pick(~wrong & ~hedge, per_group),
              "hedged_right": pick(~wrong & hedge, 12)}
    print({k: len(v) for k, v in groups.items()}, flush=True)

    # stop-token ids
    eids = set()
    gc = getattr(model.llm, "generation_config", None)
    if gc is not None:
        v = gc.eos_token_id
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(x, int):
                eids.add(x)
    if tok.eos_token_id is not None:
        eids.add(tok.eos_token_id)
    for t_ in ("<|im_end|>",):
        i_ = tok.convert_tokens_to_ids(t_)
        if isinstance(i_, int) and i_ >= 0:
            eids.add(i_)
    # the ACTUAL terminator streaming_generate stops on — found from the
    # argmax tail of the first run; absent from generation_config
    eids.add(151704)
    STOP = torch.tensor(sorted(eids), device="cuda")
    print("stop ids:", sorted(eids), flush=True)

    st = {"on": False, "ent": [], "stop": [], "tid": []}

    def hook(_m, _i, out):
        if not st["on"]:
            return
        lg = out[0, -1].float()
        p = torch.softmax(lg, -1)
        st["ent"].append(float(-(p * torch.log(p + 1e-12)).sum()))
        st["stop"].append(float(p[STOP].sum()))
        st["tid"].append(int(lg.argmax()))
    h = model.llm.lm_head.register_forward_hook(hook)

    def call_def(fn, /, **kw):
        prm = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in prm})

    rows = []
    try:
        for gname, ids in groups.items():
            for k, qid in enumerate(ids):
                wav = f"{DATA}/bench_audio/{qid}.wav"
                if not os.path.exists(wav):
                    continue
                au, _ = librosa.load(wav, sr=16000, mono=True)
                chunks = [au[i:i + 16000]
                          for i in range(0, len(au), 16000)]
                model.reset_session()
                st.update(on=False, ent=[], stop=[], tid=[])
                sys_msg = call_def(model.get_sys_prompt, mode="omni",
                                   language="en")
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[sys_msg], tokenizer=tok)
                for i, ch in enumerate(chunks):
                    if len(ch) < 16000:
                        ch = np.pad(ch, (0, 16000 - len(ch)))
                    call_def(model.streaming_prefill, session_id="s1",
                             msgs=[{"role": "user",
                                    "content": [ch.astype(np.float32)]}],
                             tokenizer=tok,
                             is_last_chunk=(i == len(chunks) - 1))
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[{"role": "assistant", "content": [" "]}],
                         tokenizer=tok, is_last_chunk=True)
                st["on"] = True
                res = call_def(model.streaming_generate,
                               session_id="s1", tokenizer=tok,
                               temperature=0.1, generate_audio=False,
                               max_new_tokens=512)
                parts = []
                if inspect.isgenerator(res) or hasattr(res, "__next__"):
                    for x in res:
                        t = getattr(x, "text", None)
                        if t is None and isinstance(x, dict):
                            t = x.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                else:
                    parts.append(str(res))
                st["on"] = False
                rows.append({"id": qid, "group": gname,
                             "n_steps": len(st["ent"]),
                             "ent": list(st["ent"]),
                             "p_stop": list(st["stop"]),
                             "tok_ids": list(st["tid"]),
                             "answer": "".join(parts).strip()})
                if k % 8 == 0:
                    print(f"  [{gname} {k}] {qid} steps="
                          f"{len(st['ent'])}", flush=True)
    finally:
        h.remove()
    pd.DataFrame(rows).to_parquet(f"{DATA}/entropy_traj.parquet")
    gate_data.commit()
    print(f">>> wrote entropy_traj.parquet ({len(rows)} rows)",
          flush=True)
    return len(rows)
