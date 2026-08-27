# -*- coding: utf-8 -*-
"""NVDA paper-table completion arm (2026-08-26).

Goal: give the frozen-recipe NVDA probe (8ac) the same table coverage
as tab:transfer -- per-pool escalation-tier re-mix under each pool's
OFFICIAL judge, plus the missing AlpacaEval pool.

API-only steps here; the valpaca GPU pass reuses
modal_nvda.py::run_answers (POOLS gained a valpaca entry).

  1. rejudge_local  : NVDA local answers on the OAB pools -> official
                      OAB judge (they only had our judge) -> oab_ok
                      column added to nvda_{tag}.parquet
  2. expert_fill    : gpt-5.5 (low, cached) on the heard transcripts of
                      every id with no measured expert answer in the
                      v3 traces -> nvda_expert_fill.parquet
  3. judge_experts  : ALL expert answers (measured + fill), judged
                      uniformly with the pool's official judge on the
                      gold query -> nvda_expert_outcomes.parquet
  4. vb_local_valpaca / dump_valpaca : VoiceBench 1-5 judge on the NVDA
                      valpaca answers + probe scores (winner combo,
                      calib=frozen 600) -> nvda_valpaca.parquet,
                      nvda_scores_valpaca.parquet

Judge prompts/models are copied VERBATIM from modal_bench.py (official
OAB gpt-4o-2024-08-06 JSON judge; official VoiceBench gpt-4o-mini
meta_prompt_open) so the numbers sit on the same scale as tab:transfer.
"""
import json
import sys

import modal

app = modal.App("nvda-ext")
gate_data = modal.Volume.from_name("gate-data")
DATA = "/data"
OPENAI = modal.Secret.from_name("openai")

util = (modal.Image.debian_slim(python_version="3.11")
        .pip_install("openai", "pandas", "pyarrow")
        .add_local_dir("src", "/workspace/gate"))
score_image = (modal.Image.debian_slim(python_version="3.11")
               .pip_install("scikit-learn", "pandas", "pyarrow", "numpy",
                            "transformers", "sentencepiece"))

QFILES = {t: f"{DATA}/queries_{t}.jsonl"
          for t in ("striviaqa", "swebq", "sllama", "sdqa", "valpaca")}
OAB_POOLS = ("striviaqa", "swebq", "sllama")


def _read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _gold(tag):
    qs = _read_jsonl(QFILES[tag])
    return {q["id"]: {"query": q.get("query") or q.get("text"),
                      "reference_answer": q.get("reference_answer")
                      or q.get("reference")} for q in qs}


# ---- judges: verbatim copies from modal_bench.py --------------------------
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


def _vb_judge(rows, concurrency=3):
    """rows: [{query, answer}] -> add 'score' (1-5, None on error)."""
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
        for attempt in range(6):
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


# ---- 1. official OAB judge over the NVDA local answers --------------------
@app.function(image=util, volumes={DATA: gate_data}, secrets=[OPENAI],
              timeout=60 * 60 * 2)
def rejudge_local(tags: str = "striviaqa,swebq,sllama"):
    import pandas as pd
    for tag in tags.split(","):
        tag = tag.strip()
        df = pd.read_parquet(f"{DATA}/nvda_{tag}.parquet")
        gold = _gold(tag)
        rows = [{"id": r["id"], "answer": r["answer"], **gold[r["id"]]}
                for _, r in df.iterrows() if r["id"] in gold]
        print(f">>> {tag}: OAB-judging {len(rows)} NVDA local answers",
              flush=True)
        scored = _oab_judge(rows)
        m = {r["id"]: r["oab_ok"] for r in scored}
        df["oab_ok"] = [m.get(i) for i in df["id"]]
        df.to_parquet(f"{DATA}/nvda_{tag}.parquet")
        ok = df[df["oab_ok"].notna()]
        print(f">>> {tag}: ours-fail={1 - df['adequate'].mean():.3f} "
              f"OAB-fail={1 - ok['oab_ok'].mean():.3f} (n={len(ok)})",
              flush=True)
    gate_data.commit()


# ---- 2. gpt-5.5 fills for ids with no measured expert answer --------------
@app.function(image=util, volumes={DATA: gate_data}, secrets=[OPENAI],
              timeout=60 * 60 * 3)
def expert_fill(infile: str = "nvda_expert_fill_in.jsonl"):
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    rows = _read_jsonl(f"{DATA}/{infile}")
    print(f">>> {len(rows)} expert fills (gpt-5.5 low, cached)", flush=True)
    res = asyncio.run(escalate.ask_expert_many(
        [r["transcript"] for r in rows], concurrency=3, effort="low",
        cache_dir=f"{DATA}/expert_cache"))
    out = pd.DataFrame([{**r, "expert_answer": x.get("answer"),
                         "expert_latency_s": x.get("latency_s"),
                         "expert_error": x.get("error")}
                        for r, x in zip(rows, res)])
    out.to_parquet(f"{DATA}/nvda_expert_fill.parquet")
    gate_data.commit()
    n_ok = out["expert_answer"].notna().sum()
    print(f">>> {n_ok}/{len(out)} answered; by pool:", flush=True)
    print(out.groupby("pool")["expert_answer"]
          .apply(lambda s: s.notna().sum()), flush=True)


# ---- 3. one uniform judge pass over ALL expert answers --------------------
@app.function(image=util, volumes={DATA: gate_data}, secrets=[OPENAI],
              timeout=60 * 60 * 3)
def judge_experts():
    """Measured (v3-trace escalated) + fill expert answers, judged with
    the pool's official judge on the GOLD query text. expert_ok for the
    QA pools, expert_score (1-5) for valpaca."""
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    fill = pd.read_parquet(f"{DATA}/nvda_expert_fill.parquet")
    frames = []
    for pool in ("striviaqa", "swebq", "sllama", "sdqa", "valpaca"):
        src = (f"{DATA}/valpaca_v3_scored.parquet" if pool == "valpaca"
               else f"{DATA}/{pool}_v3_traces.parquet")
        tr = pd.read_parquet(src)
        meas = (tr[(tr["mode"] == "escalated")
                   & tr["expert_answer"].notna()]
                .drop_duplicates("id")[["id", "expert_answer"]])
        meas["src"] = "measured"
        fl = fill[(fill["pool"] == pool)
                  & fill["expert_answer"].notna()][["id", "expert_answer"]]
        fl = fl[~fl["id"].isin(set(meas["id"]))].copy()
        fl["src"] = "fill"
        d = pd.concat([meas, fl], ignore_index=True)
        d["pool"] = pool
        gold = _gold(pool)
        d["query"] = d["id"].map(lambda i: gold[i]["query"])
        d["reference_answer"] = d["id"].map(
            lambda i: gold[i]["reference_answer"])
        rows = d.to_dict("records")
        print(f">>> {pool}: judging {len(rows)} expert answers "
              f"({(d['src'] == 'fill').sum()} fills)", flush=True)
        if pool in OAB_POOLS:
            scored = _oab_judge(
                [{"query": r["query"],
                  "reference_answer": r["reference_answer"],
                  "answer": r["expert_answer"]} for r in rows])
            d["expert_ok"] = [r["oab_ok"] for r in scored]
        elif pool == "sdqa":
            labeled = asyncio.run(escalate.judge_many(
                [{"query": r["query"],
                  "reference_answer": r["reference_answer"],
                  "answer": r["expert_answer"]} for r in rows],
                concurrency=8))
            d["expert_ok"] = [None if x["adequate"] is None
                              else int(x["adequate"]) for x in labeled]
        else:
            scored = _vb_judge(
                [{"query": r["query"], "answer": r["expert_answer"]}
                 for r in rows])
            d["expert_score"] = [r["score"] for r in scored]
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(f"{DATA}/nvda_expert_outcomes.parquet")
    gate_data.commit()
    for pool, g in out.groupby("pool"):
        col = "expert_score" if pool == "valpaca" else "expert_ok"
        v = g[col].dropna()
        print(f">>> {pool}: n={len(g)} {col}={v.mean():.3f}", flush=True)


# ---- 4. valpaca: VB judge on NVDA answers + probe scores ------------------
@app.function(image=util, volumes={DATA: gate_data}, secrets=[OPENAI],
              timeout=60 * 60)
def vb_local_valpaca():
    import glob
    import pandas as pd
    shards = sorted(glob.glob(f"{DATA}/nvda_answers_valpaca.shard*.jsonl"))
    rows = []
    for sh in shards:
        rows += _read_jsonl(sh)
    gold = _gold("valpaca")
    jr = [{"id": r["id"], "query": gold[r["id"]]["query"],
           "answer": r["answer"]} for r in rows if r["id"] in gold]
    print(f">>> valpaca: VB-judging {len(jr)} NVDA answers", flush=True)
    scored = _vb_judge(jr)
    m = {r["id"]: r["score"] for r in scored}
    df = pd.DataFrame(rows)
    df["score"] = [m.get(i) for i in df["id"]]
    df["adequate"] = None
    df.to_parquet(f"{DATA}/nvda_valpaca.parquet")
    gate_data.commit()
    ok = df[df["score"].notna()]
    print(f">>> valpaca: n={len(df)} judged={len(ok)} "
          f"NVDA local mean={ok['score'].mean():.3f}", flush=True)


@app.function(image=score_image, volumes={DATA: gate_data},
              timeout=60 * 30)
def dump_valpaca():
    """Probe scores for the valpaca hidden reads -- same winner combo and
    calib as modal_nvda.dump_scores -> nvda_scores_valpaca.parquet."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2", trust_remote_code=True)
    NVDA_LAYERS = list(range(2, 56, 4))
    jbest = NVDA_LAYERS.index(34)

    def load_tag(tag):
        shards = sorted(glob.glob(f"{DATA}/nvda_h_{tag}.shard*.npz"))
        ids, E, M = [], [], []
        for sh in shards:
            z = np.load(sh, allow_pickle=True)
            ids += [str(x) for x in z["ids"]]
            E.append(z["H_eot"]); M.append(z["H_mean"])
        return ids, np.concatenate(E).astype(np.float32), \
            np.concatenate(M).astype(np.float32)

    def feats(E, M):
        return np.concatenate([E[:, jbest, -1], E[:, jbest].mean(1),
                               M[:, jbest]], axis=1)

    ids_c, Ec, Mc = load_tag("frozen")
    lab = pd.read_parquet(f"{DATA}/nvda_frozen.parquet") \
        .set_index("id")["escalate_label"]
    keep = [i for i, q in enumerate(ids_c)
            if q in lab.index and pd.notna(lab.get(q))]
    yc = np.array([int(lab[ids_c[i]]) for i in keep])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1e-4, max_iter=4000))
    clf.fit(feats(Ec[keep], Mc[keep]), yc)
    print(f"calib n={len(yc)} fail={yc.mean():.3f}")

    ids, E, M = load_tag("valpaca")
    sc = clf.predict_proba(feats(E, M))[:, 1]
    ans = pd.read_parquet(f"{DATA}/nvda_valpaca.parquet").set_index("id")
    rows = []
    for i, qid in enumerate(ids):
        if qid not in ans.index:
            continue
        a = str(ans.loc[qid, "answer"] or "")
        rows.append({"pool": "valpaca", "id": qid, "score": float(sc[i]),
                     "n_tokens": len(tok.encode(
                         a, add_special_tokens=False)),
                     "vb_score": ans.loc[qid, "score"]})
    pd.DataFrame(rows).to_parquet(f"{DATA}/nvda_scores_valpaca.parquet")
    gate_data.commit()
    print(f">>> wrote nvda_scores_valpaca.parquet n={len(rows)}")
