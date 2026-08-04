"""Modal app: zero-training escalation gate for MiniCPM-o 4.5 (text modality).

System-1 / System-2 architecture. The small model (MiniCPM-o 4.5, 9B) answers
by default; a zero-training gate reads its per-step hidden states + logits and
decides when to escalate a distilled query to a large model (OpenAI GPT-5.x).

Reuses the existing `minicpm-o45-weights` Volume (weights already downloaded by
the sibling modal_app.py::download_weights) and the project's validated stack
(torch 2.8 + transformers 4.51, SDPA, no flash-attn). Standalone image so the
local-dir add is the final build step (Modal requires add_local_* last).

Phases (see PLAN.md):
    modal run interactive_paper/modal_app.py::smoke          # Phase 0: chat + introspect
    modal run interactive_paper/modal_app.py::signal_check   # Phase 1: signal sanity
    ...

The `src/` dir is mounted at /workspace/gate so container code imports it.
"""
import json
import os
import sys

import modal

MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"

HERE = os.path.dirname(os.path.abspath(__file__))

app = modal.App("think-gate")
# The OpenAI-dependent judge (`label`) lives on a separate app so that running
# a non-OpenAI function (e.g. build_public_queries) doesn't eagerly resolve the
# `openai` secret — Modal hydrates secrets per invoked app.
gen_app = modal.App("think-gate-gen")

weights = modal.Volume.from_name("minicpm-o45-weights")
gate_data = modal.Volume.from_name("gate-data", create_if_missing=True)

DATA = "/data"

# Mirrors modal_bench.py's validated stack (torch 2.8 + transformers 4.51, SDPA,
# no flash-attn), built standalone so the local-dir add is last. Adds openai
# (GPT-5.x judge + escalation), scikit-learn (linear probe), pyarrow/pandas
# (feature store). No MiniCPM-o-Demo clone / duplex harness — the gate uses the
# raw AutoModel + a custom text decode loop.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsndfile1")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install(
        "minicpmo-utils[all]",       # librosa 0.9.0 + audio/video stack
        "transformers==4.51.0",      # 4.52+ breaks MiniCPM's Resampler init
        "accelerate==1.12.0",
        "setuptools<81",             # librosa 0.9 imports pkg_resources
        "pydantic>=2.11",
        "PyYAML",
        "soundfile",
        "opencv-python-headless",
        "huggingface_hub[hf_transfer]",
        "scikit-learn",
        "pandas",
        "pyarrow",
        "openai",
        "sentencepiece",   # mistral tokenizer (cross-backbone replication)
    )
    .add_local_dir(os.path.join(HERE, "src"), "/workspace/gate")
)

GPU_VOL = {"/workspace/models": weights, DATA: gate_data}

# CPU image for dataset pulls + OpenAI calls (no torch). datasets 2.21 keeps
# script-based datasets (trivia_qa) working with trust_remote_code.
util_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets==2.21.0", "huggingface_hub[hf_transfer]",
                 "pandas", "pyarrow", "openai", "scikit-learn", "matplotlib")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(os.path.join(HERE, "src"), "/workspace/gate")
)

Q_PUBLIC = f"{DATA}/queries_public.jsonl"
QUERIES = f"{DATA}/queries.jsonl"
SIGNALS = f"{DATA}/signals.jsonl"
FEATURES = f"{DATA}/calib_features.parquet"
GATE_CFG = f"{DATA}/gate_config.json"
EVAL_EXPERT = f"{DATA}/eval_expert.parquet"
EVAL_PARA = f"{DATA}/eval_paraphrase.parquet"
EXPERT_CACHE = f"{DATA}/expert_cache"   # gpt-5.5 answer cache (anti-distillation-
                                        # flag: no query is ever sent twice)
OPENAI = modal.Secret.from_name("openai")

# Single egress region for the API-heavy CPU functions — the 2026-07 suspension
# review flags rotating datacenter IPs; GPU functions stay unpinned (H100
# capacity) since their API volume is small.
API_REGION = "us-east"


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 30)
def smoke():
    """Phase 0: text-only chat + speed baseline + model-internals introspection.

    Prints the LLM-backbone attribute path, layer count, hidden size, and vocab
    so Phase 1's hidden-state hook targets the right module.
    """
    import time
    import torch
    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=False, init_tts=False,  # text-only: save VRAM
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    load_s = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f">>> loaded in {load_s:.1f}s | VRAM {vram:.1f} GB", flush=True)

    # ---- introspection: locate the LLM backbone for the Phase-1 hook ----------
    print("\n=== MODEL STRUCTURE ===", flush=True)
    print("top type:", type(model).__name__, flush=True)
    print("named_children:", [n for n, _ in model.named_children()], flush=True)
    llm = getattr(model, "llm", None)
    if llm is not None:
        cfg = llm.config
        print("model.llm type:", type(llm).__name__, flush=True)
        print("  num_hidden_layers:", getattr(cfg, "num_hidden_layers", "?"),
              "| hidden_size:", getattr(cfg, "hidden_size", "?"),
              "| vocab_size:", getattr(cfg, "vocab_size", "?"), flush=True)
        print("  llm.named_children:", [n for n, _ in llm.named_children()], flush=True)
    else:
        print("!! no model.llm attribute — dumping full module tree (depth 2)",
              flush=True)
        for n, _ in model.named_modules():
            if n.count(".") <= 1:
                print("   ", n, flush=True)

    # ---- 3 probe queries: chit-chat / GSM8K-ish / GPQA-ish --------------------
    probes = [
        ("chat", "What's a fun fact about octopuses? Answer in two sentences."),
        ("math", "Natalia sold clips to 48 friends in April, then half as many in "
                 "May. How many clips did she sell altogether? Show your work."),
        ("gpqa", "In quantum mechanics, what is the physical significance of the "
                 "commutator [x, p] = i*hbar? Explain briefly."),
    ]
    print("\n=== CHAT PROBES ===", flush=True)
    for tag, q in probes:
        msgs = [{"role": "user", "content": [q]}]
        t = time.time()
        # introspect chat signature once (kwargs vary across MiniCPM-o builds)
        import inspect
        params = set(inspect.signature(model.chat).parameters)
        kw = {k: v for k, v in dict(
            do_sample=False, max_new_tokens=256, generate_audio=False,
            use_tts_template=False, enable_thinking=False).items() if k in params}
        if "tokenizer" in params:
            kw["tokenizer"] = tok
        out = model.chat(msgs=msgs, **kw)
        dt = time.time() - t
        n_out = len(tok(out).input_ids)
        print(f"\n[{tag}] {n_out} tok in {dt:.1f}s = {n_out/dt:.1f} tok/s",
              flush=True)
        print(f"  Q: {q}", flush=True)
        print(f"  A: {out[:400]}", flush=True)

    print("\n>>> smoke OK", flush=True)


def _load_model(model_dir: str = MODEL_DIR):
    import glob as _glob
    import shutil
    import torch
    from transformers import AutoModel, AutoTokenizer
    # Pre-seed the dynamic-module cache: 4.51's relative-import copier misses
    # image_processing_minicpmv.py on the o2.6 snapshot (chain written for 4.44).
    cache = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/"
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


@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 30)
def signal_check():
    """Phase 1: verify hook-based signal capture — faithfulness, overhead, sanity."""
    import time
    sys.path.insert(0, "/workspace/gate")
    import decode

    model, tok = _load_model()
    print(">>> model loaded", flush=True)

    easy = "What's the capital of France?"
    hard = ("A particle is in a 1-D infinite square well of width L. Give the "
            "expectation value <x^2> for the n-th stationary state and briefly "
            "derive it.")

    # ---- overhead: plain chat vs hooked chat on the SAME query ---------------
    kw = decode._chat_kwargs(model, tok)
    q = hard
    t = time.time(); plain = model.chat(msgs=[{"role": "user", "content": [q]}],
                                        max_new_tokens=256, **kw)
    dt_plain = time.time() - t
    n_plain = len(tok(plain).input_ids)

    t = time.time(); r = decode.chat_with_signals(model, tok, q, k=16,
                                                  max_new_tokens=256)
    dt_hook = time.time() - t
    n_hook = len(tok(r["text"]).input_ids)

    print(f"\n=== OVERHEAD (hard query) ===", flush=True)
    print(f"plain : {n_plain} tok in {dt_plain:.2f}s = {n_plain/dt_plain:.1f} tok/s",
          flush=True)
    print(f"hooked: {n_hook} tok in {dt_hook:.2f}s = {n_hook/dt_hook:.1f} tok/s",
          flush=True)
    slow = (dt_hook/max(n_hook,1)) / (dt_plain/max(n_plain,1)) - 1
    print(f"per-token slowdown: {slow*100:+.1f}%  (gate: <30%)", flush=True)
    # faithfulness: hooked text should match plain (same greedy decode)
    print(f"text identical to plain chat: {r['text'] == plain.strip()}", flush=True)

    # ---- sanity: entropy curves, hard vs easy -------------------------------
    print(f"\n=== SIGNAL SANITY ===", flush=True)
    for tag, query in [("easy", easy), ("hard", hard)]:
        s = decode.chat_with_signals(model, tok, query, k=16, max_new_tokens=256)
        ent = s["entropy"]
        mean_ent = sum(ent) / len(ent) if ent else 0.0
        print(f"\n[{tag}] n_forward={s['n_forward']} "
              f"h_prompt_dim={len(s['h_prompt']) if s['h_prompt'] else 0} "
              f"mean_entropy@16={mean_ent:.3f}", flush=True)
        print(f"  entropy: {[round(e,2) for e in ent]}", flush=True)
        print(f"  margin : {[round(m,2) for m in s['margin']]}", flush=True)
        print(f"  A: {s['text'][:200]}", flush=True)

    print("\n>>> signal_check done", flush=True)


# ============================ Phase 2.1: query pool ========================
# All five pools now come from PUBLIC datasets (user decision 2026-07-08: no
# model-generated queries — the GPT-generated trap pool failed its design with
# a 0.102 escalate rate, and generator==judge is a circularity reviewers would
# flag). easy-chat = dolly-15k (en) + alpaca-zh (zh); trap = SimpleQA
# (confident-wrong short facts; PopQA long-tail as fallback).
N_GSM8K, N_MATH500 = 100, 50           # hard-math = 150
N_MMLU_PRO, N_GPQA = 130, 20           # hard-knowledge = 150 (GPQA if reachable)
N_TRIVIA = 100                         # easy-fact = 100
N_EASY_CHAT, N_TRAP = 150, 50          # easy-chat (75 en + 75 zh) + trap


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 60)
def build_public_queries():
    """Phase 2.1: sample ALL pools from public datasets → queries_public.jsonl.

    hard-math   : GSM8K test tail + MATH-500 sample
    hard-knowledge: MMLU-Pro sample (+ GPQA-main if the gated repo is reachable)
    easy-fact   : TriviaQA sample
    easy-chat   : dolly-15k short instructions (en) + alpaca-zh (zh; BELLE fallback)
    trap        : SimpleQA short confident-wrong facts (PopQA fallback)
    """
    import random
    from datasets import load_dataset
    sys.path.insert(0, "/workspace/gate")
    import queries as Q

    out = []

    # --- hard-math: GSM8K tail ------------------------------------------------
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for i in range(len(gsm) - N_GSM8K, len(gsm)):
        r = gsm[i]
        out.append({"pool": "hard-math", "source": "gsm8k",
                    "query": r["question"] + "\n\nShow your work and end with the "
                             "final numeric answer.",
                    "reference_answer": Q.gsm8k_reference(r["answer"])})
    # --- hard-math: MATH-500 --------------------------------------------------
    try:
        m5 = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rng = random.Random(42)
        for i in rng.sample(range(len(m5)), N_MATH500):
            r = m5[i]
            out.append({"pool": "hard-math", "source": "math500",
                        "query": r["problem"] + "\n\nSolve and give the final answer.",
                        "reference_answer": str(r["answer"])})
    except Exception as e:
        print(f"!! MATH-500 skipped: {e}", flush=True)

    # --- hard-knowledge: MMLU-Pro --------------------------------------------
    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rng = random.Random(43)
    n_mmlu = N_MMLU_PRO
    for i in rng.sample(range(len(mmlu)), n_mmlu):
        r = mmlu[i]
        out.append({"pool": "hard-knowledge", "source": f"mmlu-pro/{r['category']}",
                    "query": Q.mcq_prompt(r["question"], r["options"]),
                    "reference_answer": Q.mcq_reference(r["options"], r["answer_index"])})
    # --- hard-knowledge: GPQA (gated — try, else top up from MMLU-Pro) --------
    got_gpqa = 0
    try:
        gpqa = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        rng = random.Random(44)
        for i in rng.sample(range(len(gpqa)), N_GPQA):
            r = gpqa[i]
            opts = [r["Correct Answer"], r["Incorrect Answer 1"],
                    r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            order = list(range(4)); random.Random(44 + i).shuffle(order)
            shuffled = [opts[j] for j in order]
            correct_pos = order.index(0)
            out.append({"pool": "hard-knowledge", "source": "gpqa-main",
                        "query": Q.mcq_prompt(r["Question"], shuffled),
                        "reference_answer": Q.mcq_reference(shuffled, correct_pos)})
            got_gpqa += 1
    except Exception as e:
        print(f"!! GPQA skipped ({e}); topping up hard-knowledge from MMLU-Pro",
              flush=True)
        used = set(rng.sample(range(len(mmlu)), n_mmlu))  # avoid reuse best-effort
        rng2 = random.Random(45)
        extra = [i for i in rng2.sample(range(len(mmlu)), N_GPQA * 3)
                 if i not in used][:N_GPQA]
        for i in extra:
            r = mmlu[i]
            out.append({"pool": "hard-knowledge", "source": f"mmlu-pro/{r['category']}",
                        "query": Q.mcq_prompt(r["question"], r["options"]),
                        "reference_answer": Q.mcq_reference(r["options"], r["answer_index"])})

    # --- easy-fact: TriviaQA --------------------------------------------------
    try:
        tv = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext",
                          split="validation", trust_remote_code=True)
        rng = random.Random(46)
        for i in rng.sample(range(len(tv)), N_TRIVIA):
            r = tv[i]
            out.append({"pool": "easy-fact", "source": "triviaqa",
                        "query": r["question"],
                        "reference_answer": r["answer"]["value"]})
    except Exception as e:
        print(f"!! TriviaQA skipped: {e}", flush=True)

    # --- easy-chat (en): dolly-15k short no-context instructions ---------------
    n_en = N_EASY_CHAT // 2
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    easy_cats = {"general_qa", "brainstorming", "creative_writing", "open_qa"}
    cand = [i for i in range(len(dolly))
            if dolly[i]["category"] in easy_cats
            and not dolly[i]["context"].strip()
            and 15 <= len(dolly[i]["instruction"]) <= 200]
    rng = random.Random(47)
    for i in rng.sample(cand, n_en):
        out.append({"pool": "easy-chat", "source": "dolly-15k",
                    "query": dolly[i]["instruction"], "reference_answer": None})

    # --- easy-chat (zh): alpaca-zh, BELLE fallback -----------------------------
    n_zh = N_EASY_CHAT - n_en
    try:
        zh = load_dataset("shibing624/alpaca-zh", split="train")
        zh_src = "alpaca-zh"
    except Exception as e:
        print(f"!! alpaca-zh unavailable ({e}); falling back to BELLE", flush=True)
        zh = load_dataset("BelleGroup/train_0.5M_CN", split="train")
        zh_src = "belle-0.5m"
    def _zh_fields(r):
        ins = r.get("instruction", "")
        inp = r.get("input", "") or ""
        return ins, inp
    cand = [i for i in range(min(len(zh), 100_000))
            if (lambda t: not t[1].strip() and 8 <= len(t[0]) <= 120)(_zh_fields(zh[i]))]
    rng = random.Random(48)
    for i in rng.sample(cand, n_zh):
        out.append({"pool": "easy-chat", "source": zh_src,
                    "query": _zh_fields(zh[i])[0], "reference_answer": None})

    # --- trap: SimpleQA (PopQA fallback) ---------------------------------------
    try:
        sq = load_dataset("basicv8vc/SimpleQA", split="test")
        rng = random.Random(49)
        for i in rng.sample(range(len(sq)), N_TRAP):
            r = sq[i]
            out.append({"pool": "trap", "source": "simpleqa",
                        "query": r["problem"],
                        "reference_answer": r["answer"]})
    except Exception as e:
        print(f"!! SimpleQA unavailable ({e}); falling back to PopQA", flush=True)
        pq = load_dataset("akariasai/PopQA", split="test")
        rng = random.Random(49)
        for i in rng.sample(range(len(pq)), N_TRAP):
            r = pq[i]
            ans = r["possible_answers"]
            if isinstance(ans, str):
                try:
                    ans = json.loads(ans)[0]
                except Exception:
                    pass
            elif isinstance(ans, list):
                ans = ans[0]
            out.append({"pool": "trap", "source": "popqa",
                        "query": r["question"], "reference_answer": str(ans)})

    _write_jsonl(Q_PUBLIC, out)
    gate_data.commit()
    from collections import Counter
    print(f">>> {len(out)} public queries | pools: "
          f"{dict(Counter(r['pool'] for r in out))} | gpqa={got_gpqa}", flush=True)
    print("sample:", json.dumps(out[0], ensure_ascii=False)[:200], flush=True)


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 10)
def finalize_queries():
    """Assign ids + calib/test split (seed 42) over the public pools."""
    sys.path.insert(0, "/workspace/gate")
    import queries as Q

    rows = _read_jsonl(Q_PUBLIC)
    for j, r in enumerate(rows):
        r["id"] = f"q{j:04d}"
    Q.make_split(rows)
    _write_jsonl(QUERIES, rows)
    gate_data.commit()
    from collections import Counter
    print(f">>> {len(rows)} total queries → {QUERIES}", flush=True)
    print("pools:", dict(Counter(r["pool"] for r in rows)), flush=True)
    print("split:", dict(Counter(r["split"] for r in rows)), flush=True)


@app.function(image=util_image, volumes={DATA: gate_data})
def _load_queries() -> list:
    return _read_jsonl(QUERIES)


# ============================ Phase 2.2: signals ===========================
@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60 * 2)
def collect_signals(shard: list, shard_id: int) -> int:
    """Greedy-answer each query in the shard + capture first-16 signals.

    Writes signals.shard{id}.parquet with per-query answer + entropy/margin
    lists + h_prompt/h_mean8 vectors (labels added later by `label`).
    """
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import decode

    model, tok = _load_model()
    print(f">>> shard {shard_id}: {len(shard)} queries", flush=True)
    rows = []
    for k, q in enumerate(shard):
        s = decode.chat_with_signals(model, tok, q["query"], k=16, max_new_tokens=512)
        rows.append({**{f: q[f] for f in
                        ("id", "pool", "source", "query", "reference_answer", "split")},
                     "answer": s["text"], "n_forward": s["n_forward"],
                     "entropy": s["entropy"], "margin": s["margin"],
                     "h_prompt": s["h_prompt"], "h_mean8": s["h_mean8"]})
        if k < 2 or k % 50 == 0:
            print(f"  [{k}] {q['pool']} :: {s['text'][:60]!r}", flush=True)
    out = f"{DATA}/signals.shard{shard_id}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    gate_data.commit()
    print(f">>> wrote {out} ({len(rows)} rows)", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_signals(workers: int = 4):
    """Shard the query pool across H100 workers and collect signals."""
    qs = _load_queries.remote()
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {len(qs)} queries / {workers} H100 workers")
    total = sum(collect_signals.starmap(
        [(shards[i], i) for i in range(workers)]))
    print(f">>> collected signals for {total} queries")


# ====================== Phase 2.2: judge labels + features =================
@gen_app.function(image=util_image, volumes={DATA: gate_data},
                  secrets=[OPENAI], timeout=60 * 60, region=API_REGION)
def label(concurrency: int = 8):
    """gpt-5.4-mini judges each answer → adequate; merge into calib_features.parquet."""
    import asyncio
    import glob
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    shards = sorted(glob.glob(f"{DATA}/signals.shard*.parquet"))
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    print(f">>> {len(df)} rows from {len(shards)} shards; judging...", flush=True)

    rows = [{"query": r["query"],
             "reference_answer": (None if pd.isna(r["reference_answer"])
                                  else r["reference_answer"]),
             "answer": r["answer"]} for _, r in df.iterrows()]
    labeled = asyncio.run(escalate.judge_many(rows, concurrency=concurrency))
    df["adequate"] = [x["adequate"] for x in labeled]
    df["judge_reason"] = [x["judge_reason"] for x in labeled]
    df["escalate_label"] = [x["escalate_label"] for x in labeled]

    n_err = sum(1 for x in labeled if x["adequate"] is None)
    df.to_parquet(FEATURES)
    gate_data.commit()
    from collections import Counter
    ok = df[df["adequate"].notna()]
    print(f">>> labeled {len(df)} ({n_err} judge errors)", flush=True)
    print("escalate_label by pool (mean = small-model failure rate):", flush=True)
    for pool, g in ok.groupby("pool"):
        print(f"   {pool:16s} n={len(g):4d}  escalate_rate={g['escalate_label'].mean():.3f}",
              flush=True)


# ============================ Phase 2.3b: overfit audit ====================
@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 20)
def audit():
    """Overfit audit for the probe, CALIB SPLIT ONLY (test stays frozen).

    1. in-sample vs CV AUC gap (raw parameter overfit)
    2. calib-only 5-fold CV (the number the plan actually wanted)
    3. pool-base-rate oracle (how much AUC does pool identity alone buy?)
    4. leave-one-pool-out (does the probe generalize across query types?)
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score

    df = pd.read_parquet(FEATURES)
    df = df[df["escalate_label"].notna()].reset_index(drop=True)
    df = df[df["split"] == "calib"].reset_index(drop=True)   # test never touched
    y = df["escalate_label"].astype(int).values
    X = np.array([list(v) for v in df["h_prompt"]], float)
    pools = df["pool"].values
    print(f">>> calib-only audit: n={len(df)} escalate_rate={y.mean():.3f}", flush=True)

    lr = LogisticRegression(max_iter=2000)

    # 1. in-sample ceiling vs honest CV
    ins = roc_auc_score(y, lr.fit(X, y).predict_proba(X)[:, 1])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cv,
                          method="predict_proba")[:, 1]
    cvauc = roc_auc_score(y, p)
    print(f"[1] in-sample AUC = {ins:.3f} | 5-fold CV AUC = {cvauc:.3f} "
          f"(gap = {ins - cvauc:.3f})", flush=True)

    # 2 is cvauc above (calib-only, the plan-compliant headline)

    # 3. pool-base-rate oracle: score = pool's mean failure rate
    rate = {pl: y[pools == pl].mean() for pl in set(pools)}
    oracle = np.array([rate[pl] for pl in pools])
    print(f"[3] pool-identity-only AUC = {roc_auc_score(y, oracle):.3f} "
          f"(composition shortcut ceiling)", flush=True)

    # 4. leave-one-pool-out: train on 4 pools, eval on the held-out one
    print("[4] leave-one-pool-out (trained WITHOUT that pool):", flush=True)
    for pl in sorted(set(pools)):
        tr, te = pools != pl, pools == pl
        if len(set(y[te])) < 2:
            print(f"    {pl:16s} n={te.sum():3d} single-class "
                  f"(esc={y[te].mean():.2f}) — AUC n/a; "
                  f"mean score={LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1].mean():.3f}",
                  flush=True)
            continue
        pte = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        print(f"    {pl:16s} n={te.sum():3d} esc={y[te].mean():.2f} "
              f"LOPO-AUC={roc_auc_score(y[te], pte):.3f}", flush=True)


# ===================== Phase 2.3c: audit deep-dive =========================
@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 20)
def audit2():
    """Deep-dive on the audit's two open questions + bootstrap CIs. CPU, ~free.

    A. math LOPO inversion root cause — what drives AUC 0.372: source
       (gsm8k vs math500), prompt length, and score-by-outcome breakdowns.
    B. pool-classifier evidence chain — can h_prompt identify the pool (5-way)?
       Plus pool-controlled residual: dummies-only vs dummies+h_prompt OOF AUC
       and within-pool AUC of the plain probe's OOF scores.
    C. bootstrap CIs (2000 resamples, seed 42) for the calib OOF AUC and the
       Phase-5 test headline numbers (small/big/paraphrase/tier accuracies).
    """
    import json as _json
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, "/workspace/gate")
    from gate import Probe

    df = pd.read_parquet(FEATURES)
    df = df[df["escalate_label"].notna()].reset_index(drop=True)
    calib = df[df["split"] == "calib"].reset_index(drop=True)
    y = calib["escalate_label"].astype(int).to_numpy()
    X = np.array([list(v) for v in calib["h_prompt"]], float)
    pools = np.array(calib["pool"].tolist())
    rng = np.random.RandomState(42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    # ---------- A. math LOPO inversion root cause ------------------------------
    print("=== [A] math LOPO inversion root cause (calib only) ===", flush=True)
    tr, te = pools != "hard-math", pools == "hard-math"
    lopo = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    oof = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cv,
                            method="predict_proba")[:, 1]        # in-distribution ref
    m = calib[te].reset_index(drop=True)
    ym, oofm = y[te], oof[te]
    qlen = m["query"].str.len().to_numpy(float)
    print(f"  math calib n={len(m)} fail_rate={ym.mean():.3f} | "
          f"LOPO AUC={roc_auc_score(ym, lopo):.3f} | in-dist OOF AUC={roc_auc_score(ym, oofm):.3f}",
          flush=True)
    for src in sorted(set(m["source"])):
        s = (m["source"] == src).to_numpy()
        line = (f"  {src:8s} n={s.sum():3d} fail={ym[s].mean():.2f} "
                f"len(mean)={qlen[s].mean():5.0f} | LOPO score "
                f"fail={lopo[s & (ym == 1)].mean() if (s & (ym == 1)).any() else float('nan'):.3f} "
                f"pass={lopo[s & (ym == 0)].mean() if (s & (ym == 0)).any() else float('nan'):.3f}")
        if len(set(ym[s])) == 2:
            line += (f" | LOPO AUC={roc_auc_score(ym[s], lopo[s]):.3f}"
                     f" OOF AUC={roc_auc_score(ym[s], oofm[s]):.3f}")
        print(line, flush=True)
    # is the inversion just "source" (surface style) or within-source too?
    src01 = (m["source"] == "math500").astype(int).to_numpy()
    print(f"  corr(LOPO score, is_math500)={np.corrcoef(lopo, src01)[0,1]:+.3f} | "
          f"corr(LOPO score, len)={np.corrcoef(lopo, qlen)[0,1]:+.3f} | "
          f"corr(fail, is_math500)={np.corrcoef(ym, src01)[0,1]:+.3f} | "
          f"corr(fail, len)={np.corrcoef(ym, qlen)[0,1]:+.3f}", flush=True)
    print(f"  len(chars): fail={qlen[ym==1].mean():.0f} pass={qlen[ym==0].mean():.0f}",
          flush=True)

    # ---------- B. pool-classifier evidence chain ------------------------------
    print("\n=== [B] does h_prompt encode the pool? ===", flush=True)
    acc = cross_val_score(LogisticRegression(max_iter=2000),
                          X, pools, cv=5, scoring="accuracy")
    print(f"  5-way pool classifier 5-fold acc = {acc.mean():.3f} "
          f"(folds: {np.round(acc, 3).tolist()})", flush=True)

    dum = pd.get_dummies(pools).values.astype(float)
    oof_dum = cross_val_predict(LogisticRegression(max_iter=2000), dum, y, cv=cv,
                                method="predict_proba")[:, 1]
    both = np.hstack([dum * 10.0, X])   # scale dummies so L2 doesn't drown them
    oof_both = cross_val_predict(LogisticRegression(max_iter=2000), both, y, cv=cv,
                                 method="predict_proba")[:, 1]
    print(f"  OOF AUC: pool-dummies-only={roc_auc_score(y, oof_dum):.3f} | "
          f"h_prompt={roc_auc_score(y, oof):.3f} | "
          f"dummies+h_prompt={roc_auc_score(y, oof_both):.3f}", flush=True)
    print("  within-pool AUC of plain-probe OOF scores (type shortcut removed):",
          flush=True)
    wp = []
    for pl in sorted(set(pools)):
        s = pools == pl
        if len(set(y[s])) < 2:
            print(f"    {pl:16s} single-class (fail={y[s].mean():.2f}) — n/a", flush=True)
            continue
        a = roc_auc_score(y[s], oof[s]); wp.append(a)
        print(f"    {pl:16s} n={s.sum():3d} within-AUC={a:.3f}", flush=True)
    print(f"    macro-mean within-pool AUC = {np.mean(wp):.3f} "
          f"(vs aggregate {roc_auc_score(y, oof):.3f})", flush=True)

    # ---------- C. bootstrap CIs ------------------------------------------------
    print("\n=== [C] bootstrap 95% CIs (2000 resamples) ===", flush=True)

    def bs_ci(fn, n, B=2000):
        vals = []
        for _ in range(B):
            idx = rng.randint(0, n, n)
            v = fn(idx)
            if v is not None:
                vals.append(v)
        return np.percentile(vals, [2.5, 97.5])

    lo, hi = bs_ci(lambda i: roc_auc_score(y[i], oof[i])
                   if len(set(y[i])) == 2 else None, len(y))
    print(f"  calib OOF probe AUC = {roc_auc_score(y, oof):.3f} [{lo:.3f}, {hi:.3f}]",
          flush=True)

    # Phase-5 test headline CIs (reuses eval_assemble's merge)
    cfg = _json.load(open(GATE_CFG))
    probe = Probe.from_config(cfg)
    test = df[df["split"] == "test"].reset_index(drop=True)
    exp = pd.read_parquet(EVAL_EXPERT).set_index("id")
    para = pd.read_parquet(EVAL_PARA).set_index("id")
    ids = test["id"].values
    b = lambda x: 1 if x is True or x == 1 else 0
    s = np.array([b(x) for x in test["adequate"].values])
    e = np.array([b(exp.loc[i, "expert_adequate"]) for i in ids])
    p = np.array([b(para.loc[i, "paraphrase_adequate"]) for i in ids])
    sc = np.array([probe.score(list(h)) for h in test["h_prompt"]])
    n = len(test)
    for name, acc_fn in [("small-only", lambda i: s[i].mean()),
                         ("big-only(expert)", lambda i: e[i].mean()),
                         ("full-paraphrase", lambda i: p[i].mean())]:
        lo, hi = bs_ci(acc_fn, n)
        print(f"  {name:18s} = {acc_fn(np.arange(n)):.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    for tier in ("conservative", "balanced", "aggressive"):
        t = cfg["thresholds"][tier]
        hyb = lambda i, t=t: np.where(sc[i] >= t, e[i], s[i]).mean()
        lo, hi = bs_ci(hyb, n)
        print(f"  hybrid {tier:12s} = {hyb(np.arange(n)):.3f} [{lo:.3f}, {hi:.3f}] "
              f"(esc={float((sc >= t).mean()):.3f})", flush=True)

    # area-vs-random CI: recompute the whole sweep per resample
    def area(i):
        si, ei, sci = s[i], e[i], sc[i]
        ts = np.concatenate([[np.inf], np.unique(sci)[::-1], [-np.inf]])
        pts = np.array([[(sci >= t).mean(),
                         np.where(sci >= t, ei, si).mean()] for t in ts])
        o = np.argsort(pts[:, 0])
        rate, accs = pts[o, 0], pts[o, 1]
        randl = (1 - rate) * si.mean() + rate * ei.mean()
        d = accs - randl
        return float(np.sum((d[1:] + d[:-1]) / 2 * np.diff(rate)))
    lo, hi = bs_ci(area, n, B=500)   # sweep per resample — keep B moderate
    print(f"  gate-vs-random area = {area(np.arange(n)):+.4f} [{lo:+.4f}, {hi:+.4f}]",
          flush=True)


# ==================== p(True) verbalized self-eval baseline ================
PTRUE_PRE = ("I am going to show you a question. Do NOT answer it. "
             "Judge honestly whether you yourself would answer it correctly.\n\n"
             "Question:\n{q}\n\n"
             "Would you answer this question correctly? "
             "Reply with exactly one word: Yes or No.")
PTRUE_POST = ("Here is a question and a proposed answer.\n\n"
              "Question:\n{q}\n\n"
              "Proposed answer:\n{a}\n\n"
              "Is the proposed answer correct? "
              "Reply with exactly one word: Yes or No.")


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 10)
def _load_ptrue_rows() -> list:
    """id/query/answer rows from the feature store (answers already generated)."""
    import pandas as pd
    df = pd.read_parquet(FEATURES)
    return [{"id": r["id"], "query": r["query"],
             "answer": r["answer"] if isinstance(r["answer"], str) else ""}
            for _, r in df.iterrows()]


@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_ptrue(shard: list, shard_id: int) -> int:
    """p(True) self-eval per query: pre-answer ("would you get this right?") and
    post-answer ("is this answer correct?"), each = P(Yes) read off the first
    generated token's logits. Zero training, zero calibration."""
    import pandas as pd
    import torch
    sys.path.insert(0, "/workspace/gate")
    import decode

    model, tok = _load_model()

    def tok_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    YES = tok_ids(["Yes", "yes", "YES", " Yes", " yes", "是", "能", "对", "会"])
    NO = tok_ids(["No", "no", "NO", " No", " no", "否", "不", "错"])
    print(f">>> shard {shard_id}: {len(shard)} queries | "
          f"|YES|={len(YES)} |NO|={len(NO)}", flush=True)

    def p_yes(prompt):
        logits = decode.first_token_logits(model, tok, prompt)
        p = torch.softmax(logits, dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        arg = tok.decode([int(logits.argmax())])
        return py / (py + pn) if (py + pn) > 0 else 0.5, py + pn, arg

    rows = []
    for k, q in enumerate(shard):
        pre, pre_mass, pre_arg = p_yes(PTRUE_PRE.format(q=q["query"]))
        post, post_mass, post_arg = p_yes(
            PTRUE_POST.format(q=q["query"], a=q["answer"][:4000]))
        rows.append({"id": q["id"], "p_yes_pre": pre, "p_yes_post": post,
                     "mass_pre": pre_mass, "mass_post": post_mass})
        if k < 3 or k % 50 == 0:
            print(f"  [{k}] pre={pre:.3f} (mass {pre_mass:.2f}, argmax {pre_arg!r}) "
                  f"post={post:.3f} (mass {post_mass:.2f}, argmax {post_arg!r})",
                  flush=True)
    out = f"{DATA}/ptrue.shard{shard_id}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    gate_data.commit()
    print(f">>> wrote {out} ({len(rows)} rows)", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_ptrue(workers: int = 4):
    """Collect p(True) self-eval signals for all queries across H100 workers."""
    qs = _load_ptrue_rows.remote()
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {len(qs)} queries / {workers} H100 workers")
    total = sum(collect_ptrue.starmap([(shards[i], i) for i in range(workers)]))
    print(f">>> collected p(True) for {total} queries")


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 20)
def ptrue_analyze():
    """Compare p(True) self-eval vs the linear probe — overall, per-pool, and on
    the probe's failure cases (math, trap). p(True) needs NO calibration pool,
    so its per-pool AUC is already the 'transfer' condition the probe fails."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    df = pd.read_parquet(FEATURES)
    df = df[df["escalate_label"].notna()].reset_index(drop=True)
    pt = pd.concat([pd.read_parquet(s) for s in
                    sorted(glob.glob(f"{DATA}/ptrue.shard*.parquet"))],
                   ignore_index=True)
    df = df.merge(pt, on="id", validate="one_to_one")
    y = df["escalate_label"].astype(int).to_numpy()
    pools = np.array(df["pool"].tolist())
    # escalation score = P(model says it CAN'T) = 1 - P(Yes)
    s_pre = 1.0 - df["p_yes_pre"].to_numpy()
    s_post = 1.0 - df["p_yes_post"].to_numpy()
    print(f">>> n={len(df)} | Yes/No mass: pre median={df['mass_pre'].median():.3f} "
          f"post median={df['mass_post'].median():.3f}", flush=True)

    calib = (df["split"] == "calib").to_numpy()
    X = np.array([list(v) for v in df["h_prompt"]], float)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.full(len(df), np.nan)
    oof[calib] = cross_val_predict(LogisticRegression(max_iter=2000),
                                   X[calib], y[calib], cv=cv,
                                   method="predict_proba")[:, 1]

    print("\n=== overall AUC (calib rows, so probe OOF is comparable) ===", flush=True)
    yc = y[calib]
    for name, sc in [("probe OOF", oof[calib]), ("ptrue_pre", s_pre[calib]),
                     ("ptrue_post", s_post[calib]),
                     ("pre+post mean", (s_pre + s_post)[calib] / 2)]:
        print(f"  {name:14s} AUC={roc_auc_score(yc, sc):.3f}", flush=True)
    print(f"  (all 600, no-training signals) ptrue_pre={roc_auc_score(y, s_pre):.3f} "
          f"ptrue_post={roc_auc_score(y, s_post):.3f}", flush=True)

    print("\n=== per-pool AUC — the transfer test ===", flush=True)
    print(f"  {'pool':16s} {'n':>4s} {'fail':>5s} {'probeOOF':>9s} {'LOPO':>6s} "
          f"{'ptrue_pre':>10s} {'ptrue_post':>11s}", flush=True)
    for pl in sorted(set(pools)):
        s = pools == pl
        scb = calib & s
        # probe LOPO score for this pool (train on other calib pools)
        trm = calib & (pools != pl)
        lopo = LogisticRegression(max_iter=2000).fit(X[trm], y[trm]) \
            .predict_proba(X[s])[:, 1]
        def a(scores, mask):
            return (f"{roc_auc_score(y[mask], scores[mask]):.3f}"
                    if len(set(y[mask])) == 2 else "  n/a")
        def a_arr(scores):
            return (f"{roc_auc_score(y[s], scores):.3f}"
                    if len(set(y[s])) == 2 else "  n/a")
        probecell = a(oof, scb) if scb.any() else "  n/a"
        print(f"  {pl:16s} {s.sum():4d} {y[s].mean():5.2f} {probecell:>9s} "
              f"{a_arr(lopo):>6s} {a(s_pre, s):>10s} {a(s_post, s):>11s}", flush=True)

    print("\n=== trap pool (100% fail): would each signal escalate? ===", flush=True)
    s = pools == "trap"
    trm = calib & (pools != "trap")
    lopo_trap = LogisticRegression(max_iter=2000).fit(X[trm], y[trm]) \
        .predict_proba(X[s])[:, 1]
    nontrap_pre = s_pre[~s]
    print(f"  probe-LOPO mean score on trap = {lopo_trap.mean():.3f} (known: ~0.23)",
          flush=True)
    print(f"  ptrue_pre  mean score on trap = {s_pre[s].mean():.3f} "
          f"(vs non-trap mean {nontrap_pre.mean():.3f})", flush=True)
    print(f"  ptrue_post mean score on trap = {s_post[s].mean():.3f} "
          f"(vs non-trap mean {s_post[~s].mean():.3f})", flush=True)
    # rank check: share of trap queries above the non-trap 70th percentile
    thr = np.quantile(nontrap_pre, 0.7)
    print(f"  ptrue_pre: {float((s_pre[s] >= thr).mean()):.2f} of trap above "
          f"non-trap P70 (probe-LOPO puts ~0 there)", flush=True)


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 20)
def ptrue_gate_eval():
    """RQ2 rerun with p(True) gates: does the better AUC translate into a better
    accuracy-vs-escalation curve on the frozen test split? Same protocol as
    eval_assemble (stored outcomes, threshold sweep); tier thresholds are score
    quantiles on CALIB rows (no fitting — p(True) needs none). Writes
    figures/tradeoff_ptrue.png."""
    import glob
    import json as _json
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, "/workspace/gate")
    from gate import Probe

    df = pd.read_parquet(FEATURES)
    df = df[df["escalate_label"].notna()].reset_index(drop=True)
    pt = pd.concat([pd.read_parquet(s) for s in
                    sorted(glob.glob(f"{DATA}/ptrue.shard*.parquet"))],
                   ignore_index=True)
    df = df.merge(pt, on="id", validate="one_to_one")
    cfg = _json.load(open(GATE_CFG))
    probe = Probe.from_config(cfg)
    df["probe_score"] = [probe.score(list(h)) for h in df["h_prompt"]]
    df["ptrue_pre_score"] = 1.0 - df["p_yes_pre"]
    df["ptrue_post_score"] = 1.0 - df["p_yes_post"]

    test = df[df["split"] == "test"].reset_index(drop=True)
    calib = df[df["split"] == "calib"]
    exp = pd.read_parquet(EVAL_EXPERT).set_index("id")
    ids = test["id"].values
    b = lambda x: 1 if x is True or x == 1 else 0
    s = np.array([b(x) for x in test["adequate"].values])
    e = np.array([b(exp.loc[i, "expert_adequate"]) for i in ids])
    small_acc, big_acc = s.mean(), e.mean()
    print(f">>> TEST n={len(test)} small={small_acc:.3f} big={big_acc:.3f}",
          flush=True)

    def curve(scores):
        ts = np.concatenate([[np.inf], np.unique(scores)[::-1], [-np.inf]])
        pts = np.array([[(scores >= t).mean(),
                         np.where(scores >= t, e, s).mean()] for t in ts])
        o = np.argsort(pts[:, 0])
        return pts[o]

    def area(c):
        randl = (1 - c[:, 0]) * small_acc + c[:, 0] * big_acc
        d = c[:, 1] - randl
        return float(np.sum((d[1:] + d[:-1]) / 2 * np.diff(c[:, 0])))

    plt.figure(figsize=(7, 6))
    print(f"\n{'signal':16s} {'area':>8s} | hybrid acc @ esc rate "
          f"{{.15/.30/.50}} (thr from calib quantiles)", flush=True)
    for name, col in [("probe", "probe_score"), ("ptrue_pre", "ptrue_pre_score"),
                      ("ptrue_post", "ptrue_post_score")]:
        sc = test[col].to_numpy()
        c = curve(sc)
        accs = []
        for rate in (0.15, 0.30, 0.50):
            t = float(np.quantile(calib[col].to_numpy(), 1 - rate))
            esc = sc >= t
            accs.append(f"{np.where(esc, e, s).mean():.3f}(esc {esc.mean():.2f})")
        print(f"{name:16s} {area(c):+.4f} | " + "  ".join(accs), flush=True)
        plt.plot(c[:, 0], c[:, 1], "-", lw=2, label=f"{name} (area {area(c):+.3f})")

    r = np.linspace(0, 1, 101)
    plt.plot(r, (1 - r) * small_acc + r * big_acc, "k--", lw=1,
             label="random escalation")
    plt.xlabel("escalation rate"); plt.ylabel("accuracy (expert-inject)")
    plt.title("Gate signals on frozen test (n=240)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{DATA}/tradeoff_ptrue.png", dpi=140)
    gate_data.commit()
    print(f">>> wrote {DATA}/tradeoff_ptrue.png", flush=True)


# ================= cross-backbone replication (generality) ================
# Does the Phase-2/audit finding (probe AUC high in-distribution, but driven by
# type shortcut + no LOPO transfer) replicate on other backbones?
HF_MODELS = {
    "qwen3-8b": "Qwen/Qwen3-8B",                       # MiniCPM's backbone family, raw
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",  # different family, ungated
    # --- duplex/omni generalization (2026-07-17): does the omni fine-tune
    # destroying probe transfer generalize? Matched pairs: one raw backbone vs
    # its audio/omni/duplex fine-tunes (all Qwen2.5-7B-based), + a 2nd family.
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",          # raw control for the pairs below
    "qwen2.5-omni-7b": "Qwen/Qwen2.5-Omni-7B",         # omni full-finetune (talker-thinker)
    "minicpm-o26": "openbmb/MiniCPM-o-2_6",            # true duplex, Qwen2.5-7B base
    "kimi-audio-7b": "moonshotai/Kimi-Audio-7B-Instruct",  # audio-only finetune (no duplex)
    # kimi's dual-stream forward (audio input_ids + text_input_ids) breaks
    # vanilla generate() — deferred. Qwen2-Audio fills the same cell (audio FT,
    # no duplex) with a native transformers class + its own raw control:
    "qwen2-audio-7b": "Qwen/Qwen2-Audio-7B-Instruct",  # audio FT of Qwen2-7B
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",              # raw control for it
    "glm4-voice-9b": "zai-org/glm-4-voice-9b",         # 2nd family: voice finetune
    "glm4-9b-chat-hf": "zai-org/glm-4-9b-chat-hf",     # 2nd family: raw control
    # (glm-4-9b-chat non-hf was downloaded first but is ChatGLM remote code —
    #  incompatible with hf_decode's vanilla layout; the -hf port is native.)
}


@app.function(image=util_image, volumes={"/workspace/models": weights},
              timeout=60 * 60)
def download_hf_model(tag: str) -> str:
    """Snapshot-download an HF backbone onto the weights volume (idempotent)."""
    import os as _os
    _os.environ["HF_HUB_DISABLE_XET"] = "1"   # unauthenticated Xet stalls (2026-07-14)
    from huggingface_hub import snapshot_download
    repo = HF_MODELS[tag]
    dest = f"/workspace/models/{tag}"
    # no config.json short-circuit: a killed run can leave a partial snapshot;
    # snapshot_download itself verifies existing files and fills the gaps.
    snapshot_download(repo, local_dir=dest,
                      ignore_patterns=["*.pth", "original/*", "*.gguf",
                                       "consolidated*"])  # 403s via public xet
    weights.commit()
    print(f">>> downloaded {repo} -> {dest}", flush=True)
    return dest


@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60 * 2)
def collect_signals_hf(shard: list, shard_id: int, tag: str) -> int:
    """collect_signals for a vanilla HF backbone (hf_decode hooks)."""
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import hf_decode

    model_dir = f"/workspace/models/{tag}"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").eval().cuda()
    tok = AutoTokenizer.from_pretrained(model_dir)
    print(f">>> {tag} shard {shard_id}: {len(shard)} queries", flush=True)

    rows = []
    for k, q in enumerate(shard):
        s = hf_decode.chat_with_signals(model, tok, q["query"], k=16,
                                        max_new_tokens=512)
        rows.append({**{f: q[f] for f in
                        ("id", "pool", "source", "query", "reference_answer", "split")},
                     "answer": s["text"], "n_forward": s["n_forward"],
                     "entropy": s["entropy"], "margin": s["margin"],
                     "h_prompt": s["h_prompt"], "h_mean8": s["h_mean8"]})
        if k < 2 or k % 50 == 0:
            print(f"  [{k}] {q['pool']} :: {s['text'][:60]!r}", flush=True)
    out = f"{DATA}/signals_{tag}.shard{shard_id}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    gate_data.commit()
    print(f">>> wrote {out} ({len(rows)} rows)", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_signals_hf(tag: str, workers: int = 4, limit: int = 0):
    """Download the backbone (if needed) + collect signals for all queries."""
    download_hf_model.remote(tag)
    qs = _load_queries.remote()
    if limit:
        qs = qs[:limit]
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {tag}: {len(qs)} queries / {workers} H100 workers")
    total = sum(collect_signals_hf.starmap(
        [(shards[i], i, tag) for i in range(workers)]))
    print(f">>> collected signals for {total} queries")


@gen_app.function(image=util_image, volumes={DATA: gate_data},
                  secrets=[OPENAI], timeout=60 * 60, region=API_REGION)
def label_hf(tag: str, concurrency: int = 8):
    """Judge the backbone's answers (same rubric/judge as Phase 2) →
    features_{tag}.parquet."""
    import asyncio
    import glob
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    shards = sorted(glob.glob(f"{DATA}/signals_{tag}.shard*.parquet"))
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    print(f">>> {tag}: {len(df)} rows from {len(shards)} shards; judging...",
          flush=True)
    rows = [{"query": r["query"],
             "reference_answer": (None if pd.isna(r["reference_answer"])
                                  else r["reference_answer"]),
             "answer": r["answer"]} for _, r in df.iterrows()]
    labeled = asyncio.run(escalate.judge_many(rows, concurrency=concurrency))
    df["adequate"] = [x["adequate"] for x in labeled]
    df["judge_reason"] = [x["judge_reason"] for x in labeled]
    df["escalate_label"] = [x["escalate_label"] for x in labeled]
    n_err = sum(1 for x in labeled if x["adequate"] is None)
    df.to_parquet(f"{DATA}/features_{tag}.parquet")
    gate_data.commit()
    ok = df[df["adequate"].notna()]
    print(f">>> labeled {len(df)} ({n_err} judge errors)", flush=True)
    for pool, g in ok.groupby("pool"):
        print(f"   {pool:16s} n={len(g):4d} escalate_rate="
              f"{g['escalate_label'].mean():.3f}", flush=True)


@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_ptrue_hf(shard: list, shard_id: int, tag: str) -> int:
    """p(True) self-eval for an HF backbone (same prompts as collect_ptrue)."""
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import hf_decode

    model_dir = f"/workspace/models/{tag}"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").eval().cuda()
    tok = AutoTokenizer.from_pretrained(model_dir)

    def tok_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    YES = tok_ids(["Yes", "yes", "YES", " Yes", " yes", "是", "能", "对", "会"])
    NO = tok_ids(["No", "no", "NO", " No", " no", "否", "不", "错"])
    print(f">>> {tag} shard {shard_id}: {len(shard)} | |YES|={len(YES)} "
          f"|NO|={len(NO)}", flush=True)

    def p_yes(prompt):
        logits = hf_decode.first_token_logits(model, tok, prompt)
        p = torch.softmax(logits, dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        arg = tok.decode([int(logits.argmax())])
        return py / (py + pn) if (py + pn) > 0 else 0.5, py + pn, arg

    rows = []
    for k, q in enumerate(shard):
        pre, pre_mass, pre_arg = p_yes(PTRUE_PRE.format(q=q["query"]))
        post, post_mass, post_arg = p_yes(
            PTRUE_POST.format(q=q["query"], a=q["answer"][:4000]))
        rows.append({"id": q["id"], "p_yes_pre": pre, "p_yes_post": post,
                     "mass_pre": pre_mass, "mass_post": post_mass})
        if k < 3 or k % 50 == 0:
            print(f"  [{k}] pre={pre:.3f} (mass {pre_mass:.2f}, {pre_arg!r}) "
                  f"post={post:.3f} (mass {post_mass:.2f}, {post_arg!r})", flush=True)
    out = f"{DATA}/ptrue_{tag}.shard{shard_id}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    gate_data.commit()
    print(f">>> wrote {out}", flush=True)
    return len(rows)


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 10)
def _load_ptrue_rows_hf(tag: str) -> list:
    import pandas as pd
    df = pd.read_parquet(f"{DATA}/features_{tag}.parquet")
    return [{"id": r["id"], "query": r["query"],
             "answer": r["answer"] if isinstance(r["answer"], str) else ""}
            for _, r in df.iterrows()]


@app.local_entrypoint()
def run_ptrue_hf(tag: str, workers: int = 4):
    qs = _load_ptrue_rows_hf.remote(tag)
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {tag}: {len(qs)} queries / {workers} workers")
    total = sum(collect_ptrue_hf.starmap(
        [(shards[i], i, tag) for i in range(workers)]))
    print(f">>> collected p(True) for {total} queries")


# ====== omni/duplex text-mode collection (duplex-generalization pairs) ======
# Qwen2.5-Omni's classes need transformers>=4.52; the main GPU image pins 4.51
# for MiniCPM-o 4.5, so omni collection gets its own image. Text-only: load the
# Thinker (talker/audio never enter the generate path); hf_decode's hooks fit
# its layout (model.model.layers[-1] / model.lm_head).
omni_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0")
    .pip_install("transformers==4.57.6", "accelerate==1.12.0", "pandas",
                 "pyarrow", "huggingface_hub[hf_transfer]")
    .add_local_dir(os.path.join(HERE, "src"), "/workspace/gate")
)


def _load_omni_text(tag: str):
    """Load an omni checkpoint's text path (Thinker) + tokenizer."""
    import torch
    from transformers import AutoTokenizer
    model_dir = f"/workspace/models/{tag}"
    if tag == "qwen2.5-omni-7b":
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").eval().cuda()
    elif tag == "qwen2-audio-7b":
        # text-only path = the audio-FT'd language_model, a plain
        # Qwen2ForCausalLM — hf_decode's hooks/generate fit it directly.
        from transformers import Qwen2AudioForConditionalGeneration
        full = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").eval().cuda()
        model = full.language_model
    else:
        raise ValueError(f"no omni text loader for {tag}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    return model, tok


@app.function(image=omni_image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60 * 2)
def collect_signals_omni(shard: list, shard_id: int, tag: str) -> int:
    """collect_signals_hf for an omni backbone's Thinker."""
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import hf_decode

    model, tok = _load_omni_text(tag)
    print(f">>> {tag} shard {shard_id}: {len(shard)} queries", flush=True)
    rows = []
    for k, q in enumerate(shard):
        s = hf_decode.chat_with_signals(model, tok, q["query"], k=16,
                                        max_new_tokens=512)
        rows.append({**{f: q[f] for f in
                        ("id", "pool", "source", "query", "reference_answer", "split")},
                     "answer": s["text"], "n_forward": s["n_forward"],
                     "entropy": s["entropy"], "margin": s["margin"],
                     "h_prompt": s["h_prompt"], "h_mean8": s["h_mean8"]})
        if k < 2 or k % 50 == 0:
            print(f"  [{k}] {q['pool']} :: {s['text'][:60]!r}", flush=True)
    out = f"{DATA}/signals_{tag}.shard{shard_id}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    gate_data.commit()
    print(f">>> wrote {out} ({len(rows)} rows)", flush=True)
    return len(rows)


@app.function(image=omni_image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_ptrue_omni(shard: list, shard_id: int, tag: str) -> int:
    """p(True) for an omni backbone's Thinker (same prompts as collect_ptrue)."""
    import pandas as pd
    import torch
    sys.path.insert(0, "/workspace/gate")
    import hf_decode

    model, tok = _load_omni_text(tag)

    def tok_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    YES = tok_ids(["Yes", "yes", "YES", " Yes", " yes", "是", "能", "对", "会"])
    NO = tok_ids(["No", "no", "NO", " No", " no", "否", "不", "错"])
    print(f">>> {tag} shard {shard_id}: {len(shard)} | |YES|={len(YES)} "
          f"|NO|={len(NO)}", flush=True)

    def p_yes(prompt):
        logits = hf_decode.first_token_logits(model, tok, prompt)
        p = torch.softmax(logits, dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        arg = tok.decode([int(logits.argmax())])
        return py / (py + pn) if (py + pn) > 0 else 0.5, py + pn, arg

    rows = []
    for k, q in enumerate(shard):
        pre, pre_mass, pre_arg = p_yes(PTRUE_PRE.format(q=q["query"]))
        post, post_mass, post_arg = p_yes(
            PTRUE_POST.format(q=q["query"], a=q["answer"][:4000]))
        rows.append({"id": q["id"], "p_yes_pre": pre, "p_yes_post": post,
                     "mass_pre": pre_mass, "mass_post": post_mass})
        if k < 3 or k % 50 == 0:
            print(f"  [{k}] pre={pre:.3f} (mass {pre_mass:.2f}, {pre_arg!r}) "
                  f"post={post:.3f} (mass {post_mass:.2f}, {post_arg!r})", flush=True)
    out = f"{DATA}/ptrue_{tag}.shard{shard_id}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    gate_data.commit()
    print(f">>> wrote {out}", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_signals_omni(tag: str, workers: int = 4, limit: int = 0):
    qs = _load_queries.remote()
    if limit:
        qs = qs[:limit]
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {tag}: {len(qs)} queries / {workers} H100 workers")
    total = sum(collect_signals_omni.starmap(
        [(shards[i], i, tag) for i in range(workers)]))
    print(f">>> collected signals for {total} queries")


@app.local_entrypoint()
def run_ptrue_omni(tag: str, workers: int = 4):
    qs = _load_ptrue_rows_hf.remote(tag)
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {tag}: {len(qs)} queries / {workers} workers")
    total = sum(collect_ptrue_omni.starmap(
        [(shards[i], i, tag) for i in range(workers)]))
    print(f">>> collected p(True) for {total} queries")


# ---- MiniCPM-o 2.6 (duplex FT of Qwen2.5-7B): same remote-code path as 4.5 ----
@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60 * 2)
def collect_signals_mo(shard: list, shard_id: int, tag: str) -> int:
    """collect_signals for another MiniCPM-o checkpoint (decode.py hooks)."""
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import decode

    model, tok = _load_model(f"/workspace/models/{tag}")
    print(f">>> {tag} shard {shard_id}: {len(shard)} queries", flush=True)
    rows = []
    for k, q in enumerate(shard):
        s = decode.chat_with_signals(model, tok, q["query"], k=16,
                                     max_new_tokens=512)
        rows.append({**{f: q[f] for f in
                        ("id", "pool", "source", "query", "reference_answer", "split")},
                     "answer": s["text"], "n_forward": s["n_forward"],
                     "entropy": s["entropy"], "margin": s["margin"],
                     "h_prompt": s["h_prompt"], "h_mean8": s["h_mean8"]})
        if k < 2 or k % 50 == 0:
            print(f"  [{k}] {q['pool']} :: {s['text'][:60]!r}", flush=True)
    out = f"{DATA}/signals_{tag}.shard{shard_id}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    gate_data.commit()
    print(f">>> wrote {out} ({len(rows)} rows)", flush=True)
    return len(rows)


@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_ptrue_mo(shard: list, shard_id: int, tag: str) -> int:
    """p(True) for another MiniCPM-o checkpoint (decode.first_token_logits)."""
    import pandas as pd
    import torch
    sys.path.insert(0, "/workspace/gate")
    import decode

    model, tok = _load_model(f"/workspace/models/{tag}")

    def tok_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    YES = tok_ids(["Yes", "yes", "YES", " Yes", " yes", "是", "能", "对", "会"])
    NO = tok_ids(["No", "no", "NO", " No", " no", "否", "不", "错"])
    print(f">>> {tag} shard {shard_id}: {len(shard)} | |YES|={len(YES)} "
          f"|NO|={len(NO)}", flush=True)

    def p_yes(prompt):
        logits = decode.first_token_logits(model, tok, prompt)
        p = torch.softmax(logits, dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        arg = tok.decode([int(logits.argmax())])
        return py / (py + pn) if (py + pn) > 0 else 0.5, py + pn, arg

    rows = []
    for k, q in enumerate(shard):
        pre, pre_mass, pre_arg = p_yes(PTRUE_PRE.format(q=q["query"]))
        post, post_mass, post_arg = p_yes(
            PTRUE_POST.format(q=q["query"], a=q["answer"][:4000]))
        rows.append({"id": q["id"], "p_yes_pre": pre, "p_yes_post": post,
                     "mass_pre": pre_mass, "mass_post": post_mass})
        if k < 3 or k % 50 == 0:
            print(f"  [{k}] pre={pre:.3f} (mass {pre_mass:.2f}, {pre_arg!r}) "
                  f"post={post:.3f} (mass {post_mass:.2f}, {post_arg!r})", flush=True)
    out = f"{DATA}/ptrue_{tag}.shard{shard_id}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    gate_data.commit()
    print(f">>> wrote {out}", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_signals_mo(tag: str, workers: int = 4, limit: int = 0):
    qs = _load_queries.remote()
    if limit:
        qs = qs[:limit]
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {tag}: {len(qs)} queries / {workers} H100 workers")
    total = sum(collect_signals_mo.starmap(
        [(shards[i], i, tag) for i in range(workers)]))
    print(f">>> collected signals for {total} queries")


@app.local_entrypoint()
def run_ptrue_mo(tag: str, workers: int = 4):
    qs = _load_ptrue_rows_hf.remote(tag)
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {tag}: {len(qs)} queries / {workers} workers")
    total = sum(collect_ptrue_mo.starmap(
        [(shards[i], i, tag) for i in range(workers)]))
    print(f">>> collected p(True) for {total} queries")


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 20)
def xmodel_report(tag: str):
    """Replication report for a backbone: headline AUCs + oracle + LOPO
    (mirrors audit/audit2's key checks, calib rows only)."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
    from sklearn.metrics import roc_auc_score

    df = pd.read_parquet(f"{DATA}/features_{tag}.parquet")
    df = df[df["escalate_label"].notna()].reset_index(drop=True)
    calib = df[df["split"] == "calib"].reset_index(drop=True)
    y = calib["escalate_label"].astype(int).to_numpy()
    X = np.array([list(v) for v in calib["h_prompt"]], float)
    pools = np.array(calib["pool"].tolist())
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    print(f"=== {tag} replication (calib n={len(calib)}, "
          f"fail_rate={y.mean():.3f}) ===", flush=True)
    for pl in sorted(set(pools)):
        s = pools == pl
        print(f"   {pl:16s} n={s.sum():3d} fail_rate={y[s].mean():.3f}", flush=True)

    oof = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cv,
                            method="predict_proba")[:, 1]
    ent = np.array([max(list(v)[:4]) if len(list(v)) else np.nan
                    for v in calib["entropy"]], float)
    ent_auc = roc_auc_score(y[~np.isnan(ent)], ent[~np.isnan(ent)])
    rate = {pl: y[pools == pl].mean() for pl in set(pools)}
    oracle = np.array([rate[pl] for pl in pools])
    acc = cross_val_score(LogisticRegression(max_iter=2000), X, pools, cv=5)
    print(f"\n  probe OOF AUC      = {roc_auc_score(y, oof):.3f}", flush=True)
    print(f"  max_entropy@4 AUC  = {ent_auc:.3f}", flush=True)
    print(f"  pool-oracle AUC    = {roc_auc_score(y, oracle):.3f}", flush=True)
    print(f"  pool-classifier acc= {acc.mean():.3f}", flush=True)

    print("\n  LOPO (trained WITHOUT that pool):", flush=True)
    for pl in sorted(set(pools)):
        tr, te = pools != pl, pools == pl
        sc = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]) \
            .predict_proba(X[te])[:, 1]
        if len(set(y[te])) < 2:
            print(f"    {pl:16s} single-class (fail={y[te].mean():.2f}) "
                  f"mean score={sc.mean():.3f}", flush=True)
        else:
            print(f"    {pl:16s} LOPO-AUC={roc_auc_score(y[te], sc):.3f} "
                  f"(fail={y[te].mean():.2f})", flush=True)

    # --- p(True) comparison (if collected for this backbone) ------------------
    import glob
    shards = sorted(glob.glob(f"{DATA}/ptrue_{tag}.shard*.parquet"))
    if not shards:
        print("\n  (no p(True) shards for this tag yet)", flush=True)
        return
    pt = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    mall = df.merge(pt, on="id", validate="one_to_one")
    ya = mall["escalate_label"].astype(int).to_numpy()
    pa = np.array(mall["pool"].tolist())
    s_pre = 1.0 - mall["p_yes_pre"].to_numpy()
    s_post = 1.0 - mall["p_yes_post"].to_numpy()
    print(f"\n  p(True) (all {len(mall)} rows): "
          f"pre AUC={roc_auc_score(ya, s_pre):.3f} "
          f"post AUC={roc_auc_score(ya, s_post):.3f}", flush=True)
    for pl in sorted(set(pa)):
        sm = pa == pl
        if len(set(ya[sm])) < 2:
            print(f"    {pl:16s} single-class fail={ya[sm].mean():.2f} | "
                  f"pre mean={s_pre[sm].mean():.3f} (non-pool {s_pre[~sm].mean():.3f})",
                  flush=True)
            continue
        print(f"    {pl:16s} pre AUC={roc_auc_score(ya[sm], s_pre[sm]):.3f} "
              f"post AUC={roc_auc_score(ya[sm], s_post[sm]):.3f}", flush=True)


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 10)
def trap_examples(n: int = 8):
    """Show trap queries side by side: pre-answer self-eval vs the confident-wrong
    answer the model actually generated (and its post-answer self-endorsement)."""
    import glob
    import pandas as pd
    df = pd.read_parquet(FEATURES)
    pt = pd.concat([pd.read_parquet(s) for s in
                    sorted(glob.glob(f"{DATA}/ptrue.shard*.parquet"))],
                   ignore_index=True)
    df = df.merge(pt, on="id")
    t = df[df["pool"] == "trap"].sort_values("p_yes_pre").reset_index(drop=True)
    for _, r in t.head(n).iterrows():
        print("=" * 78, flush=True)
        print(f"[{r['id']}] Q: {r['query'][:180]}", flush=True)
        print(f"  REF: {str(r['reference_answer'])[:120]}", flush=True)
        print(f"  模型答案 (judge: 错): {str(r['answer'])[:220]}", flush=True)
        print(f"  pre  『你能答对吗?』 P(Yes)={r['p_yes_pre']:.3f}  -> 升级分 {1-r['p_yes_pre']:.3f}", flush=True)
        print(f"  post 『这答案对吗?』 P(Yes)={r['p_yes_post']:.3f} -> 升级分 {1-r['p_yes_post']:.3f}", flush=True)
    # and the self-persuasion tail: traps where post most endorses a wrong answer
    print("\n" + "#" * 22 + " 自我说服最严重的 3 条 (post P(Yes) 最高) " + "#" * 22, flush=True)
    for _, r in t.sort_values("p_yes_post", ascending=False).head(3).iterrows():
        print("=" * 78, flush=True)
        print(f"[{r['id']}] Q: {r['query'][:180]}", flush=True)
        print(f"  REF: {str(r['reference_answer'])[:120]}", flush=True)
        print(f"  模型答案 (judge: 错): {str(r['answer'])[:220]}", flush=True)
        print(f"  pre P(Yes)={r['p_yes_pre']:.3f} | post P(Yes)={r['p_yes_post']:.3f}", flush=True)


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 20)
def lopo_matched():
    """Disambiguate WHY qwen3-8b's LOPO transfers but MiniCPM's inverts.

    Confound: qwen fails 37–51% on every non-math pool (broad positive
    coverage) while MiniCPM fails 19–48%. Subsample qwen's LOPO *training*
    pools to MiniCPM's per-pool fail rates and re-test LOPO-math:
      - stays high  -> representation difference (omni fine-tune) is the story
      - collapses   -> label coverage was the story, not the representation
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(42)
    mini_rate = {"easy-chat": 0.207, "easy-fact": 0.340,
                 "hard-knowledge": 0.480, "trap": 1.00}

    df = pd.read_parquet(f"{DATA}/features_qwen3-8b.parquet")
    df = df[df["escalate_label"].notna()].reset_index(drop=True)
    calib = df[df["split"] == "calib"].reset_index(drop=True)
    y = calib["escalate_label"].astype(int).to_numpy()
    X = np.array([list(v) for v in calib["h_prompt"]], float)
    pools = np.array(calib["pool"].tolist())
    te = pools == "hard-math"

    # baseline (unmatched) for reference
    lr = LogisticRegression(max_iter=2000)
    base = lr.fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    print(f">>> qwen LOPO-math unmatched: AUC={roc_auc_score(y[te], base):.3f} "
          f"(train n={int((~te).sum())}, fail={y[~te].mean():.3f})", flush=True)

    # matched: subsample each training pool to MiniCPM's fail rate
    keep = np.zeros(len(calib), bool)
    for pl, r in mini_rate.items():
        s = np.where(pools == pl)[0]
        fails, passes = s[y[s] == 1], s[y[s] == 0]
        if r >= 1.0:                       # trap: keep fails only (MiniCPM: all fail)
            keep[fails] = True
            continue
        n_fail = min(len(fails), int(round(r / (1 - r) * len(passes))))
        keep[passes] = True
        keep[rng.choice(fails, n_fail, replace=False)] = True
    tr = keep & ~te
    print(f">>> matched train n={int(tr.sum())} fail={y[tr].mean():.3f} "
          f"(per-pool:", flush=True)
    for pl in mini_rate:
        s = tr & (pools == pl)
        print(f"      {pl:16s} n={int(s.sum()):3d} fail={y[s].mean():.3f} "
              f"(target {mini_rate[pl]:.2f})", flush=True)
    sc = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    print(f">>> qwen LOPO-math MATCHED: AUC={roc_auc_score(y[te], sc):.3f}",
          flush=True)
    # repeat 5× with different subsample seeds for stability
    aucs = []
    for seed in range(5):
        r2 = np.random.RandomState(100 + seed)
        keep2 = np.zeros(len(calib), bool)
        for pl, r in mini_rate.items():
            s = np.where(pools == pl)[0]
            fails, passes = s[y[s] == 1], s[y[s] == 0]
            if r >= 1.0:
                keep2[fails] = True
                continue
            n_fail = min(len(fails), int(round(r / (1 - r) * len(passes))))
            keep2[passes] = True
            keep2[r2.choice(fails, n_fail, replace=False)] = True
        t2 = keep2 & ~te
        sc2 = LogisticRegression(max_iter=2000).fit(X[t2], y[t2]) \
            .predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], sc2))
    print(f">>> matched LOPO-math over 5 seeds: mean={np.mean(aucs):.3f} "
          f"min={min(aucs):.3f} max={max(aucs):.3f}", flush=True)


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 20)
def lopo_matched2(tag_ref: str, tag_ft: str):
    """Generalized lopo_matched: subsample tag_ref's LOPO training pools to
    tag_ft's per-pool fail rates (up- or down-matching), re-test LOPO on
    hard-math and hard-knowledge. If tag_ref's transfer survives the handicap,
    the pair's LOPO gap is representational, not label-coverage."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def calib_of(tag):
        d = pd.read_parquet(f"{DATA}/features_{tag}.parquet")
        d = d[d["escalate_label"].notna()]
        return d[d["split"] == "calib"].reset_index(drop=True)

    ref, ft = calib_of(tag_ref), calib_of(tag_ft)
    yf = ft["escalate_label"].astype(int).to_numpy()
    fp = np.array(ft["pool"].tolist())
    ft_rate = {pl: float(yf[fp == pl].mean()) for pl in sorted(set(fp))}
    print(f"=== match {tag_ref} -> {tag_ft} rates: "
          + " ".join(f"{p}={r:.2f}" for p, r in ft_rate.items()), flush=True)

    y = ref["escalate_label"].astype(int).to_numpy()
    X = np.array([list(v) for v in ref["h_prompt"]], float)
    pools = np.array(ref["pool"].tolist())

    def matched_mask(rng, exclude_pool):
        keep = np.zeros(len(ref), bool)
        for pl, r in ft_rate.items():
            if pl == exclude_pool:
                continue
            s = np.where(pools == pl)[0]
            fails, passes = s[y[s] == 1], s[y[s] == 0]
            if r >= 1.0 or len(passes) == 0:
                keep[fails] = True
                continue
            if len(fails) == 0:
                keep[passes] = True
                continue
            natural = len(fails) / len(s)
            if r <= natural:               # down-match: subsample fails
                n_f = min(len(fails), int(round(r / (1 - r) * len(passes))))
                keep[passes] = True
                keep[rng.choice(fails, max(n_f, 1), replace=False)] = True
            else:                          # up-match: subsample passes
                n_p = min(len(passes), int(round((1 - r) / r * len(fails))))
                keep[fails] = True
                keep[rng.choice(passes, max(n_p, 1), replace=False)] = True
        return keep

    for test_pool in ("hard-math", "hard-knowledge"):
        te = pools == test_pool
        base = LogisticRegression(max_iter=2000).fit(X[~te], y[~te]) \
            .predict_proba(X[te])[:, 1]
        aucs = []
        for seed in range(5):
            tr = matched_mask(np.random.RandomState(100 + seed), test_pool) & ~te
            sc = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]) \
                .predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], sc))
        print(f"  {test_pool:16s} unmatched={roc_auc_score(y[te], base):.3f}  "
              f"matched mean={np.mean(aucs):.3f} "
              f"[{min(aucs):.3f},{max(aucs):.3f}] (train n≈{int(tr.sum())})",
              flush=True)


# ================= Phase 5d: layer x position sweep (2026-07-20) ============
# 5c read ONE point of the network (last layer, last prompt token) and found
# its transfer degrades with duplex-ness. Is the difficulty info DESTROYED, or
# RELOCATED away from that readout (late layers / last token plausibly get
# repurposed for streaming turn control)? Prefill-only forward, all layers,
# last-token + mean-pooled — labels reused from the 5b/5c judge runs, so no
# generation and no new API spend.

def _collect_layers_loop(shard, shard_id, tag, layer_modules, make_run):
    """Shared shard loop: capture per-layer prefill hiddens -> npz on volume."""
    import numpy as np
    sys.path.insert(0, "/workspace/gate")
    import layers as layers_mod

    ids, lasts, means = [], [], []
    for k, q in enumerate(shard):
        hl, hm = layers_mod.capture_prefill_layers(layer_modules, make_run(q))
        ids.append(q["id"])
        lasts.append(hl.numpy().astype(np.float16))
        means.append(hm.numpy().astype(np.float16))
        if k < 2 or k % 50 == 0:
            print(f"  [{k}] {q['pool']} layers={hl.shape[0]} d={hl.shape[1]}",
                  flush=True)
    out = f"{DATA}/layers_{tag}.shard{shard_id}.npz"
    np.savez_compressed(out, ids=np.array(ids), h_last=np.stack(lasts),
                        h_mean=np.stack(means))
    gate_data.commit()
    print(f">>> wrote {out} ({len(ids)} rows)", flush=True)
    return len(ids)


@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_layers_hf(shard: list, shard_id: int, tag: str) -> int:
    """Layer sweep for a vanilla HF backbone (prefill-only, no generation)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import hf_decode

    model_dir = f"/workspace/models/{tag}"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").eval().cuda()
    tok = AutoTokenizer.from_pretrained(model_dir)
    print(f">>> {tag} shard {shard_id}: {len(shard)} queries", flush=True)

    def make_run(q):
        inputs = hf_decode._build_inputs(model, tok, q["query"])
        return lambda: model(**inputs)

    return _collect_layers_loop(shard, shard_id, tag,
                                list(model.model.layers), make_run)


@app.function(image=omni_image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_layers_omni(shard: list, shard_id: int, tag: str) -> int:
    """Layer sweep for an omni backbone's text path (Thinker)."""
    sys.path.insert(0, "/workspace/gate")
    import hf_decode

    model, tok = _load_omni_text(tag)
    print(f">>> {tag} shard {shard_id}: {len(shard)} queries", flush=True)

    def make_run(q):
        inputs = hf_decode._build_inputs(model, tok, q["query"])
        return lambda: model(**inputs)

    return _collect_layers_loop(shard, shard_id, tag,
                                list(model.model.layers), make_run)


@app.function(image=image, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_layers_mo(shard: list, shard_id: int, tag: str) -> int:
    """Layer sweep for a MiniCPM-o checkpoint (model.chat prefill, remote code
    builds the prompt — same faithfulness argument as decode.py)."""
    sys.path.insert(0, "/workspace/gate")
    import decode

    model_dir = MODEL_DIR if tag == "minicpm-o45" else f"/workspace/models/{tag}"
    model, tok = _load_model(model_dir)
    kw = decode._chat_kwargs(model, tok)
    print(f">>> {tag} shard {shard_id}: {len(shard)} queries", flush=True)

    def make_run(q):
        return lambda: model.chat(msgs=[{"role": "user", "content": [q["query"]]}],
                                  max_new_tokens=1, **kw)

    return _collect_layers_loop(shard, shard_id, tag,
                                list(model.llm.model.layers), make_run)


def _run_layers(collect_fn, tag: str, workers: int, limit: int):
    qs = _load_queries.remote()
    if limit:
        qs = qs[:limit]
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> {tag}: {len(qs)} queries / {workers} H100 workers")
    total = sum(collect_fn.starmap([(shards[i], i, tag) for i in range(workers)]))
    print(f">>> layer-swept {total} queries")


@app.local_entrypoint()
def run_layers_hf(tag: str, workers: int = 2, limit: int = 0):
    _run_layers(collect_layers_hf, tag, workers, limit)


@app.local_entrypoint()
def run_layers_omni(tag: str, workers: int = 2, limit: int = 0):
    _run_layers(collect_layers_omni, tag, workers, limit)


@app.local_entrypoint()
def run_layers_mo(tag: str, workers: int = 2, limit: int = 0):
    _run_layers(collect_layers_mo, tag, workers, limit)


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 60 * 2,
              cpu=8)
def layer_sweep_report(tag: str):
    """Per-layer, per-pooling probe metrics on calib rows (mirrors
    xmodel_report's estimators): OOF AUC + LOPO per pool + trap mean score.
    Saves curves to layer_sweep_{tag}.json for the cross-model figure."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    feats = FEATURES if tag == "minicpm-o45" else f"{DATA}/features_{tag}.parquet"
    df = pd.read_parquet(feats)[["id", "pool", "split", "escalate_label"]]

    shards = sorted(glob.glob(f"{DATA}/layers_{tag}.shard*.npz"))
    ids, hl, hm = [], [], []
    for s in shards:
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
        hm.append(z["h_mean"])
    hl, hm = np.concatenate(hl), np.concatenate(hm)      # (n, L, d) float16
    m = pd.DataFrame({"id": ids, "row": range(len(ids))}).merge(df, on="id")
    m = m[(m["split"] == "calib") & m["escalate_label"].notna()]
    y = m["escalate_label"].astype(int).to_numpy()
    pools = m["pool"].to_numpy()
    rows_idx = m["row"].to_numpy()
    n_layers = hl.shape[1]
    print(f"=== {tag} layer sweep (calib n={len(m)}, layers={n_layers}, "
          f"d={hl.shape[2]}) ===", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    curves = []
    for li in range(n_layers):
        rec = {"layer": li}
        for name, H in (("last", hl), ("mean", hm)):
            X = H[rows_idx, li, :].astype(np.float32)
            oof = cross_val_predict(LogisticRegression(max_iter=2000), X, y,
                                    cv=cv, method="predict_proba")[:, 1]
            rec[f"{name}_oof"] = float(roc_auc_score(y, oof))
            for pl in sorted(set(pools)):
                tr, te = pools != pl, pools == pl
                sc = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]) \
                    .predict_proba(X[te])[:, 1]
                if len(set(y[te])) < 2:
                    rec[f"{name}_lopo_{pl}_meanscore"] = float(sc.mean())
                else:
                    rec[f"{name}_lopo_{pl}"] = float(roc_auc_score(y[te], sc))
        curves.append(rec)
        print(f"  L{li:02d} last: oof={rec['last_oof']:.3f} "
              f"math={rec.get('last_lopo_hard-math', float('nan')):.3f} "
              f"know={rec.get('last_lopo_hard-knowledge', float('nan')):.3f} | "
              f"mean: oof={rec['mean_oof']:.3f} "
              f"math={rec.get('mean_lopo_hard-math', float('nan')):.3f} "
              f"know={rec.get('mean_lopo_hard-knowledge', float('nan')):.3f}",
              flush=True)

    out = f"{DATA}/layer_sweep_{tag}.json"
    with open(out, "w") as f:
        json.dump({"tag": tag, "n_layers": n_layers, "curves": curves}, f)
    gate_data.commit()
    print(f">>> wrote {out}", flush=True)


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 10)
def layer_sweep_fig(tags: str):
    """Cross-model figure: LOPO transfer vs relative depth. `tags` comma-sep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("lopo_hard-math", "LOPO hard-math AUC"),
               ("lopo_hard-knowledge", "LOPO hard-knowledge AUC"),
               ("oof", "in-mix OOF AUC")]
    poolings = [("last", "last prompt token"), ("mean", "mean over prompt")]
    fig, axes = plt.subplots(len(poolings), len(metrics),
                             figsize=(14, 7), sharex=True, sharey=True)
    for ti, tag in enumerate(tags.split(",")):
        with open(f"{DATA}/layer_sweep_{tag}.json") as f:
            data = json.load(f)
        L = data["n_layers"]
        x = [(li + 1) / L for li in range(L)]
        for pi, (pool, plabel) in enumerate(poolings):
            for mi, (met, mlabel) in enumerate(metrics):
                ax = axes[pi][mi]
                ys = [c.get(f"{pool}_{met}") for c in data["curves"]]
                ax.plot(x, ys, label=tag, color=f"C{ti}")
                if pi == 0:
                    ax.set_title(mlabel)
                if mi == 0:
                    ax.set_ylabel(f"{plabel}\nAUC")
    for ax in axes.flat:
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("relative depth")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Probe transfer by layer: is difficulty info destroyed or relocated?")
    fig.tight_layout()
    out = f"{DATA}/layer_sweep.png"
    fig.savefig(out, dpi=150)
    gate_data.commit()
    print(f">>> wrote {out}", flush=True)


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 30,
              cpu=8)
def midlayer_gate_eval(layer: int = 22, pooling: str = "last"):
    """RQ2 rerun with the MID-LAYER probe (Phase 5e): does 5d's finding — the
    self-knowledge signal survives mid-network on the duplex model — convert
    into a better accuracy-vs-escalation curve on the frozen test split?

    Layer choice must come from CALIB-only evidence (the 5d sweep used calib
    rows exclusively); default L22 = calib LOPO-math peak for minicpm-o45.
    Nearby layers are swept for sensitivity. Same protocol/curve/area code as
    ptrue_gate_eval; deployed final-layer probe + p(True) reproduced as
    references on the same rows."""
    import glob
    import json as _json
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    sys.path.insert(0, "/workspace/gate")
    from gate import Probe

    df = pd.read_parquet(FEATURES)
    df = df[df["escalate_label"].notna()].reset_index(drop=True)
    pt = pd.concat([pd.read_parquet(s) for s in
                    sorted(glob.glob(f"{DATA}/ptrue.shard*.parquet"))],
                   ignore_index=True)
    df = df.merge(pt, on="id", validate="one_to_one")
    cfg = _json.load(open(GATE_CFG))
    probe = Probe.from_config(cfg)
    df["probe_score"] = [probe.score(list(h)) for h in df["h_prompt"]]
    df["ptrue_pre_score"] = 1.0 - df["p_yes_pre"]
    df["ptrue_post_score"] = 1.0 - df["p_yes_post"]

    # 5d layer capture (all 600 rows, all layers, float16)
    ids, H = [], []
    key = "h_last" if pooling == "last" else "h_mean"
    for s in sorted(glob.glob(f"{DATA}/layers_minicpm-o45.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        H.append(z[key])
    H = np.concatenate(H)
    row_of = {i: k for k, i in enumerate(ids)}
    df["lrow"] = df["id"].map(row_of)

    test = df[df["split"] == "test"].reset_index(drop=True)
    calib = df[df["split"] == "calib"].reset_index(drop=True)
    exp = pd.read_parquet(EVAL_EXPERT).set_index("id")
    b = lambda x: 1 if x is True or x == 1 else 0
    s = np.array([b(x) for x in test["adequate"].values])
    e = np.array([b(exp.loc[i, "expert_adequate"]) for i in test["id"].values])
    small_acc, big_acc = s.mean(), e.mean()
    print(f">>> TEST n={len(test)} small={small_acc:.3f} big={big_acc:.3f} "
          f"(pooling={pooling})", flush=True)

    def curve(scores):
        ts = np.concatenate([[np.inf], np.unique(scores)[::-1], [-np.inf]])
        pts = np.array([[(scores >= t).mean(),
                         np.where(scores >= t, e, s).mean()] for t in ts])
        return pts[np.argsort(pts[:, 0])]

    def area(c):
        randl = (1 - c[:, 0]) * small_acc + c[:, 0] * big_acc
        d = c[:, 1] - randl
        return float(np.sum((d[1:] + d[:-1]) / 2 * np.diff(c[:, 0])))

    def mid_scores(li):
        y = calib["escalate_label"].astype(int).to_numpy()
        Xc = H[calib["lrow"].to_numpy(), li, :].astype(np.float32)
        Xt = H[test["lrow"].to_numpy(), li, :].astype(np.float32)
        lr = LogisticRegression(max_iter=2000).fit(Xc, y)
        return lr.predict_proba(Xc)[:, 1], lr.predict_proba(Xt)[:, 1]

    plt.figure(figsize=(7, 6))
    print(f"\n{'signal':18s} {'area':>8s} | hybrid acc @ esc rate "
          f"{{.15/.30/.50}} (thr from calib quantiles)", flush=True)
    rows = [("probe_final(cfg)", calib["probe_score"].to_numpy(),
             test["probe_score"].to_numpy(), False),
            ("ptrue_pre", calib["ptrue_pre_score"].to_numpy(),
             test["ptrue_pre_score"].to_numpy(), True),
            ("ptrue_post", calib["ptrue_post_score"].to_numpy(),
             test["ptrue_post_score"].to_numpy(), True)]
    for li in sorted({layer, 18, 20, 24, 26, 30}):
        cs, tsc = mid_scores(li)
        rows.insert(-1, (f"midlayer_L{li}", cs, tsc, li == layer))
    for name, csc, tsc, plot in rows:
        c = curve(tsc)
        accs = []
        for rate in (0.15, 0.30, 0.50):
            t = float(np.quantile(csc, 1 - rate))
            esc = tsc >= t
            accs.append(f"{np.where(esc, e, s).mean():.3f}(esc {esc.mean():.2f})")
        print(f"{name:18s} {area(c):+.4f} | " + "  ".join(accs), flush=True)
        if plot or name.startswith("probe_final"):
            plt.plot(c[:, 0], c[:, 1], "-", lw=2,
                     label=f"{name} (area {area(c):+.3f})")

    r = np.linspace(0, 1, 101)
    plt.plot(r, (1 - r) * small_acc + r * big_acc, "k--", lw=1,
             label="random escalation")
    plt.xlabel("escalation rate"); plt.ylabel("accuracy (expert-inject)")
    plt.title(f"Mid-layer probe (L{layer}, {pooling}) vs deployed gate signals")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{DATA}/tradeoff_midlayer.png", dpi=140)
    gate_data.commit()
    print(f">>> wrote {DATA}/tradeoff_midlayer.png", flush=True)


# ============================ Phase 2.3: calibration =======================
@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 20)
def calibrate():
    """Phase 2.3: ROC-AUC of each zero-training signal for predicting escalation."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_parquet(FEATURES)
    df = df[df["escalate_label"].notna()].reset_index(drop=True)
    y = df["escalate_label"].astype(int).values
    calib = df["split"] == "calib"
    print(f">>> {len(df)} labeled | calib={calib.sum()} test={(~calib).sum()} | "
          f"overall escalate_rate={y.mean():.3f}", flush=True)

    def arr(col, K):
        return np.array([ (list(v)[:K] + [np.nan] * K)[:K] for v in df[col] ], float)

    feats = {}
    for K in (4, 8, 16):
        ent = arr("entropy", K); mar = arr("margin", K)
        feats[f"mean_entropy@{K}"] = np.nanmean(ent, 1)
        feats[f"max_entropy@{K}"] = np.nanmax(ent, 1)
        feats[f"min_margin@{K}"] = np.nanmin(mar, 1)
        feats[f"mean_margin@{K}"] = np.nanmean(mar, 1)
    # entropy/margin: higher entropy & lower margin -> more likely to escalate.
    # roc_auc_score expects higher score = positive; flip margins by negating.
    scalar_auc = {}
    for name, v in feats.items():
        s = -v if "margin" in name else v
        scalar_auc[name] = roc_auc_score(y, s)

    # --- probes on hidden states (5-fold CV within the full set) --------------
    H_prompt = np.array([list(v) for v in df["h_prompt"]], float)
    H_mean8 = np.array([list(v) for v in df["h_mean8"]], float)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    probe_auc = {}
    probe_scores = {}
    for name, X in [("probe_h_prompt", H_prompt), ("probe_h_mean8", H_mean8)]:
        p = cross_val_predict(
            LogisticRegression(max_iter=2000, C=1.0), X, y, cv=cv,
            method="predict_proba")[:, 1]
        probe_auc[name] = roc_auc_score(y, p)
        probe_scores[name] = p

    # --- combined: best scalars + best probe score through one LR -------------
    best_scalar = max(scalar_auc, key=lambda k: max(scalar_auc[k], 1 - scalar_auc[k]))
    combo_X = np.column_stack([
        feats["max_entropy@16"], -feats["min_margin@16"],
        probe_scores["probe_h_prompt"]])
    combo = cross_val_predict(LogisticRegression(max_iter=2000), combo_X, y, cv=cv,
                              method="predict_proba")[:, 1]
    combo_auc = roc_auc_score(y, combo)

    all_auc = {**scalar_auc, **probe_auc, "combined": combo_auc}
    print("\n=== ROC-AUC (predicting escalate_label = small-model fails) ===",
          flush=True)
    for k in sorted(all_auc, key=all_auc.get, reverse=True):
        print(f"   {k:20s} AUC={all_auc[k]:.3f}", flush=True)
    best = max(all_auc, key=all_auc.get)
    print(f"\n>>> BEST signal: {best} = {all_auc[best]:.3f}", flush=True)

    # --- per-pool AUC of entropy vs probe (the trap-pool story) ---------------
    print("\n=== per-pool: does the probe beat entropy where entropy fails? ===",
          flush=True)
    for pool, g in df.groupby("pool"):
        yy = g["escalate_label"].astype(int).values
        if len(set(yy)) < 2:
            print(f"   {pool:16s} n={len(g):4d} (single class — AUC n/a, "
                  f"escalate_rate={yy.mean():.2f})", flush=True)
            continue
        idx = g.index.values
        ent_auc = roc_auc_score(yy, feats["max_entropy@16"][idx])
        prb_auc = roc_auc_score(yy, probe_scores["probe_h_prompt"][idx])
        print(f"   {pool:16s} n={len(g):4d} esc={yy.mean():.2f} | "
              f"max_entropy@16 AUC={ent_auc:.3f} | probe AUC={prb_auc:.3f}", flush=True)

    # --- ROC figure ----------------------------------------------------------
    plt.figure(figsize=(6, 6))
    for name in [best_scalar, "probe_h_prompt", "combined"]:
        sc = (combo if name == "combined"
              else probe_scores[name] if name.startswith("probe")
              else (-feats[best_scalar] if "margin" in best_scalar else feats[best_scalar]))
        fpr, tpr, _ = roc_curve(y, sc)
        plt.plot(fpr, tpr, label=f"{name} (AUC={all_auc.get(name, combo_auc):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.3)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Zero-training escalation signals — MiniCPM-o 4.5")
    plt.legend(); plt.tight_layout()
    fig = f"{DATA}/roc.png"
    plt.savefig(fig, dpi=120)
    gate_data.commit()
    print(f"\n>>> ROC figure → {fig}", flush=True)

    # go/no-go verdict
    top = all_auc[best]
    verdict = ("GO (>=0.75)" if top >= 0.75 else
               "GO-WEAK (0.65-0.75; flat tradeoff expected)" if top >= 0.65 else
               "NO-GO (<0.65) — STOP, report failure analysis")
    print(f"\n>>> PHASE-2 VERDICT: best AUC {top:.3f} → {verdict}", flush=True)


# ============================ Phase 3: online gate =========================
@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 10)
def fit_gate():
    """Phase 3: fit the deployable probe and pick 3 thresholds. CALIB ONLY.

    calibrate() measures AUC via CV but fits nothing shippable. This fits the ONE
    LogisticRegression on calib h_prompt that the online gate ships with, then
    picks three operating points and persists gate_config.json.

    Two deviations from the plan's naive recipe, both forced by the data:
      1. C-regularization sweep. With 4096 dims / n=360 the C=1.0 probe MEMORIZES
         (in-sample AUC 1.000), so its shipped score scale wouldn't match the OOF
         scale the thresholds live on. We pick C by 5-fold OOF AUC (tie -> smaller
         C = more regularization), which de-memorizes so thresholds transfer.
      2. Tiers by ESCALATION BUDGET, not precision target. At base rate 0.32 and
         AUC 0.82 the plan's "precision >= 0.80" is only reachable at ~0 recall
         (degenerate). Escalation rate is the real cost knob and the exact axis
         Phase 5 sweeps, so tiers = target escalate rate {.15/.30/.50}.

    Test split stays frozen (Phase 5 only). Thresholds are chosen on OOF scores;
    gate_eval() reports the realized precision/recall per tier.
    """
    import json as _json
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    df = pd.read_parquet(FEATURES)
    df = df[df["escalate_label"].notna()]
    calib = df[df["split"] == "calib"].reset_index(drop=True)
    y = calib["escalate_label"].astype(int).values
    X = np.array([list(v) for v in calib["h_prompt"]], float)
    print(f">>> fit_gate: calib n={len(calib)} escalate_rate={y.mean():.3f}",
          flush=True)

    # 1. pick C by OOF AUC (iterate ascending; strict-improve keeps smaller C on ties)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    best = None  # (C, oof_auc, oof_scores)
    for C in (0.001, 0.01, 0.1, 1.0):
        oof = cross_val_predict(LogisticRegression(max_iter=2000, C=C), X, y,
                                cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, oof)
        ins = roc_auc_score(y, LogisticRegression(max_iter=2000, C=C)
                            .fit(X, y).predict_proba(X)[:, 1])
        print(f"    C={C:<6g} OOF AUC={auc:.3f}  in-sample={ins:.3f}  "
              f"gap={ins - auc:.3f}", flush=True)
        if best is None or auc > best[1]:
            best = (C, auc, oof)
    C, oof_auc, scores = best
    lr = LogisticRegression(max_iter=2000, C=C).fit(X, y)
    insample_auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
    print(f">>> selected C={C} OOF AUC={oof_auc:.3f} in-sample={insample_auc:.3f}",
          flush=True)

    # 2. tiers by target escalation rate (threshold = OOF-score quantile)
    def by_rate(rate):
        return float(np.quantile(scores, 1.0 - rate))
    tiers = {
        "conservative": by_rate(0.15),
        "balanced": by_rate(0.30),
        "aggressive": by_rate(0.50),
    }

    cfg = {
        "signal": "probe_h_prompt",
        "probe": {
            "weight": lr.coef_[0].tolist(),
            "bias": float(lr.intercept_[0]),
            "dim": int(X.shape[1]),
            "C": float(C),
            "fit": f"calib-only LogisticRegression(C={C}, max_iter=2000) on h_prompt",
        },
        "thresholds": tiers,
        "tier_target_rate": {"conservative": 0.15, "balanced": 0.30, "aggressive": 0.50},
        "gate": {
            "mode": "pre_decode_single_shot",
            "k_consecutive": 1,
            "ema_alpha": 1.0,
            "cooldown_steps": 64,
        },
        "calib": {
            "n": int(len(calib)),
            "escalate_rate": float(y.mean()),
            "oof_auc": float(oof_auc),
            "insample_auc": float(insample_auc),
            "threshold_basis": "5-fold OOF scores (seed 42), tiers by escalation budget",
        },
    }
    with open(GATE_CFG, "w") as fh:
        _json.dump(cfg, fh, indent=2)
    gate_data.commit()
    print(">>> thresholds:", {k: round(v, 3) for k, v in tiers.items()}, flush=True)
    print(f">>> gate config → {GATE_CFG} (probe dim {cfg['probe']['dim']}, "
          f"OOF AUC {cfg['calib']['oof_auc']:.3f})", flush=True)


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 10)
def gate_eval():
    """Phase 3: report the gate's realized operating points + per-pool trigger
    rates on the calib split. CPU only — h_prompt is already stored, so scoring
    is a deterministic replay; no GPU decode is needed to validate trigger logic.

    Two score paths, both honest:
      - the pure-Python Probe scores calib IN-SAMPLE — reported only to show it
        memorizes (AUC ~1.0), which is WHY thresholds live on OOF scores. This
        also exercises the deployment scorer (Probe.score) on real 4096-d states.
      - 5-fold OOF scores (same CV as fit_gate) give the deployment-like operating
        points the thresholds were chosen for.
    Also asserts the shipped EscalationGate (single-shot) reproduces
    `score >= threshold` on every calib row.
    """
    import json as _json
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    sys.path.insert(0, "/workspace/gate")
    from gate import EscalationGate, Probe

    cfg = _json.load(open(GATE_CFG))
    probe = Probe.from_config(cfg)
    df = pd.read_parquet(FEATURES)
    df = df[(df["escalate_label"].notna()) & (df["split"] == "calib")].reset_index(drop=True)
    y = df["escalate_label"].astype(int).values
    X = np.array([list(v) for v in df["h_prompt"]], float)

    # deployment scorer (pure Python) on real hiddens — the shipped probe's own
    # in-sample scores (regularized, so no longer fully memorized):
    insample = np.array([probe.score(list(h)) for h in df["h_prompt"]])
    # deployment-like generalization: OOF scores at the shipped C (thresholds
    # were picked on these).
    C = cfg["probe"].get("C", 1.0)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_predict(
        LogisticRegression(max_iter=2000, C=C), X, y, cv=cv,
        method="predict_proba")[:, 1]
    print(f">>> gate_eval: calib n={len(df)} escalate_rate={y.mean():.3f} | C={C} | "
          f"Probe in-sample AUC={roc_auc_score(y, insample):.3f} | "
          f"OOF AUC={roc_auc_score(y, scores):.3f} (deployment-like)", flush=True)

    def one_shot(gate, s):
        gate.reset()
        return gate.update(float(s))

    for tier in ("conservative", "balanced", "aggressive"):
        t = cfg["thresholds"][tier]
        pred = scores >= t
        # the shipped class (single-shot) must match the vectorized threshold
        gate = EscalationGate.from_config(cfg, tier)
        gpred = np.array([one_shot(gate, s) for s in scores])
        assert (gpred == pred).all(), f"EscalationGate != threshold at {tier}"

        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"\n[{tier}] thr={t:.3f} escalate_rate={pred.mean():.3f} "
              f"precision={prec:.3f} recall={rec:.3f}", flush=True)
        for pool, g in df.groupby("pool"):
            idx = g.index.values
            print(f"    {pool:16s} n={len(g):3d} esc={y[idx].mean():.2f} "
                  f"trigger_rate={pred[idx].mean():.3f}", flush=True)

    print("\n>>> gate_eval OK — EscalationGate reproduces the threshold decision "
          "on all calib rows; single-shot online gate validated.", flush=True)


# ============================ Phase 4: escalation chain E2E ================
@gen_app.function(image=image, gpu="H100", volumes=GPU_VOL, secrets=[OPENAI],
                  timeout=60 * 30)
def e2e_demo(n: int = 9):
    """Phase 4: full escalation chain on a few hard TEST queries, printed as a
    readable trace: small answer + gate score/decision → distilled query →
    gpt-5.5 expert answer → small-model paraphrase. Human-checks distill quality
    and paraphrase faithfulness (PLAN Phase-4 go/no-go)."""
    import json as _json
    sys.path.insert(0, "/workspace/gate")
    import decode
    import distill
    import escalate
    import inject
    from gate import Probe

    model, tok = _load_model()
    cfg = _json.load(open(GATE_CFG))
    probe = Probe.from_config(cfg)
    thr = cfg["thresholds"]["balanced"]

    qs = [q for q in _read_jsonl(QUERIES) if q["split"] == "test"
          and q["pool"] in ("hard-knowledge", "hard-math", "trap")]
    # a few from each hard pool
    pick, seen = [], {}
    for q in qs:
        c = seen.get(q["pool"], 0)
        if c < n // 3:
            pick.append(q); seen[q["pool"]] = c + 1
    print(f">>> e2e_demo on {len(pick)} hard test queries (balanced thr={thr:.3f})\n",
          flush=True)

    for q in pick:
        s = decode.chat_with_signals(model, tok, q["query"], k=16, max_new_tokens=512)
        score = probe.score(s["h_prompt"])
        fired = score >= thr
        dq = distill.distill_query(model, tok, q["query"])
        exp = escalate.ask_expert(dq, cache_dir=EXPERT_CACHE)  # escalate the DISTILLED query
        ea = exp["answer"]
        final = (inject.paraphrase(model, tok, q["query"], ea) if ea
                 else f"[no expert answer: {exp['error']}]")
        print("=" * 78, flush=True)
        print(f"[{q['pool']}] {q['id']}  gate_score={score:.3f} → "
              f"{'ESCALATE' if fired else 'keep small'}", flush=True)
        print(f"  Q         : {q['query'][:200]}", flush=True)
        print(f"  ref       : {q.get('reference_answer')}", flush=True)
        print(f"  small     : {s['text'][:200]}", flush=True)
        print(f"  distilled : {dq[:200]}", flush=True)
        print(f"  expert(5.5): {(ea or '['+str(exp['error'])+']')[:220]}  "
              f"({exp['latency_s']:.1f}s)", flush=True)
        print(f"  paraphrase: {final[:220]}", flush=True)
    print("\n>>> e2e_demo done — inspect distilled-query fidelity + paraphrase "
          "faithfulness above.", flush=True)


# ============================ Phase 5: system evaluation ===================
@gen_app.function(image=util_image, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 60, region=API_REGION)
def eval_expert(concurrency: int = 3, force: bool = False):
    """Phase 5 big-only: gpt-5.5 answers every TEST query (original, single-turn
    → already standalone), judged for adequacy. → eval_expert.parquet.
    Answers are cached on the volume — a re-run only hits the API for misses."""
    import asyncio
    import os
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    if os.path.exists(EVAL_EXPERT) and not force:
        print(">>> eval_expert.parquet exists — expert results are FROZEN "
              "(anti-distillation-flag: never bulk re-collect). "
              "--force true to override.", flush=True)
        return

    df = pd.read_parquet(FEATURES)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f">>> eval_expert: {len(test)} test queries → gpt-5.5", flush=True)

    res = asyncio.run(escalate.ask_expert_many(test["query"].tolist(),
                                               concurrency=concurrency,
                                               cache_dir=EXPERT_CACHE))
    n_err = sum(1 for r in res if r["error"])
    print(f">>> expert done ({n_err} errors); judging...", flush=True)

    jrows = [{"query": test["query"][i],
              "reference_answer": (None if pd.isna(test["reference_answer"][i])
                                   else test["reference_answer"][i]),
              "answer": res[i]["answer"] or ""} for i in range(len(test))]
    labeled = asyncio.run(escalate.judge_many(jrows, concurrency=8))

    out = pd.DataFrame({
        "id": test["id"], "pool": test["pool"],
        "expert_answer": [r["answer"] for r in res],
        "expert_latency": [r["latency_s"] for r in res],
        "expert_prompt_tokens": [r["prompt_tokens"] for r in res],
        "expert_completion_tokens": [r["completion_tokens"] for r in res],
        "expert_error": [r["error"] for r in res],
        "expert_adequate": [x["adequate"] for x in labeled],
    })
    out.to_parquet(EVAL_EXPERT)
    gate_data.commit()
    acc = out["expert_adequate"].dropna().astype(bool).mean()
    print(f">>> big-only accuracy = {acc:.3f}", flush=True)
    for pool, g in out.groupby("pool"):
        print(f"    {pool:16s} n={len(g):3d} "
              f"acc={g['expert_adequate'].dropna().astype(bool).mean():.3f}", flush=True)


@gen_app.function(image=image, gpu="H100", volumes=GPU_VOL, secrets=[OPENAI],
                  timeout=60 * 60)
def eval_paraphrase():
    """Phase 5 hybrid outcome: the small model paraphrases each expert answer
    (inject), judged for adequacy. → eval_paraphrase.parquet. Run after
    eval_expert. The gap vs big-only accuracy = the faithfulness cost of relaying
    the expert answer through the small model."""
    import time
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate
    import inject

    model, tok = _load_model()
    exp = pd.read_parquet(EVAL_EXPERT).set_index("id")
    df = pd.read_parquet(FEATURES)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f">>> eval_paraphrase: {len(test)} test queries", flush=True)

    finals, lats = [], []
    for k, r in test.iterrows():
        ea = exp.loc[r["id"], "expert_answer"]
        if not isinstance(ea, str) or not ea:
            finals.append(None); lats.append(0.0); continue
        t0 = time.time()
        finals.append(inject.paraphrase(model, tok, r["query"], ea))
        lats.append(time.time() - t0)
        if k < 2 or k % 60 == 0:
            print(f"  [{k}] {r['pool']} :: {str(finals[-1])[:60]!r}", flush=True)

    jrows = [{"query": test["query"][i],
              "reference_answer": (None if pd.isna(test["reference_answer"][i])
                                   else test["reference_answer"][i]),
              "answer": finals[i] or ""} for i in range(len(test))]
    labeled = asyncio.run(escalate.judge_many(jrows, concurrency=8))

    out = pd.DataFrame({
        "id": test["id"], "pool": test["pool"],
        "paraphrase_answer": finals, "paraphrase_latency": lats,
        "paraphrase_adequate": [x["adequate"] for x in labeled],
    })
    out.to_parquet(EVAL_PARA)
    gate_data.commit()
    acc = out["paraphrase_adequate"].dropna().astype(bool).mean()
    print(f">>> hybrid(paraphrase) accuracy @ full-escalation = {acc:.3f}", flush=True)


@app.function(image=util_image, volumes={DATA: gate_data}, timeout=60 * 20)
def eval_assemble():
    """Phase 5 RQ2: assemble the accuracy-vs-escalation-rate tradeoff from the
    stored small answers (calib_features), expert answers (eval_expert), and
    paraphrases (eval_paraphrase). Sweeps the gate threshold to draw the hybrid
    curve, compares to the random-escalation and pool-oracle (type-only)
    baselines, prints latency/cost, and writes figures/tradeoff.png."""
    import json as _json
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, "/workspace/gate")
    from gate import Probe

    cfg = _json.load(open(GATE_CFG))
    probe = Probe.from_config(cfg)
    df = pd.read_parquet(FEATURES)
    test = df[df["split"] == "test"].reset_index(drop=True)
    exp = pd.read_parquet(EVAL_EXPERT).set_index("id")
    para = pd.read_parquet(EVAL_PARA).set_index("id")

    ids = test["id"].values
    def b(x):  # None/NaN → 0 (unjudged or failed = not adequate)
        return 1 if x is True or x == 1 else 0
    s = np.array([b(x) for x in test["adequate"].values])          # small-only
    e = np.array([b(exp.loc[i, "expert_adequate"]) for i in ids])  # expert (raw)
    p = np.array([b(para.loc[i, "paraphrase_adequate"]) for i in ids])  # paraphrased
    scores = np.array([probe.score(list(h)) for h in test["h_prompt"]])
    pools = test["pool"].values

    small_acc, big_acc, para_full = s.mean(), e.mean(), p.mean()
    print(f">>> TEST n={len(test)} | small-only={small_acc:.3f} | "
          f"big-only(expert)={big_acc:.3f} | full-escalate(paraphrase)={para_full:.3f}",
          flush=True)

    # --- hybrid-gate curve: sweep threshold over observed scores --------------
    def hybrid_acc(escalated, outcome):
        return np.where(escalated, outcome, s).mean()
    ts = np.concatenate([[np.inf], np.unique(scores)[::-1], [-np.inf]])
    curve = []  # (rate, acc_expert_inject, acc_paraphrase)
    for t in ts:
        esc = scores >= t
        curve.append((esc.mean(), hybrid_acc(esc, e), hybrid_acc(esc, p)))
    curve = np.array(curve)

    # --- named tiers ----------------------------------------------------------
    print("\n=== hybrid-gate operating points (test) ===", flush=True)
    print(f"{'tier':13s} {'thr':>6s} {'esc':>6s} {'acc(expert)':>12s} "
          f"{'acc(parashr)':>13s}", flush=True)
    tier_pts = {}
    for tier in ("conservative", "balanced", "aggressive"):
        t = cfg["thresholds"][tier]
        esc = scores >= t
        ae, ap = hybrid_acc(esc, e), hybrid_acc(esc, p)
        tier_pts[tier] = (esc.mean(), ae, ap)
        print(f"{tier:13s} {t:6.3f} {esc.mean():6.3f} {ae:12.3f} {ap:13.3f}",
              flush=True)

    # --- random baseline (expectation lines) + empirical area gap -------------
    # random escalation at rate r: E[acc] = (1-r)*small + r*overall_expert(or para)
    def rand_line(overall):
        r = np.linspace(0, 1, 101)
        return r, (1 - r) * small_acc + r * overall
    # area between gate curve and random line (expert-inject), trapezoid over rate
    order = np.argsort(curve[:, 0])
    rate_s, acc_e_s = curve[order, 0], curve[order, 1]
    rand_e = (1 - rate_s) * small_acc + rate_s * big_acc
    diff = acc_e_s - rand_e
    lift_area = float(np.sum((diff[1:] + diff[:-1]) / 2 * np.diff(rate_s)))
    print(f"\n>>> gate-vs-random area (expert-inject, ∫(acc_gate−acc_rand)d(rate)) "
          f"= {lift_area:+.4f}", flush=True)

    # --- pool-oracle (type-only) baseline ------------------------------------
    # score = the query's pool CALIB fail rate (true pool label, no test
    # leakage): the upper bound of any router that only recognizes query type.
    # Ties within a pool → the swept points sit at pool boundaries; straight
    # segments between them = expectation over random within-pool order.
    calib = df[df["split"] == "calib"]
    pool_fail = {pl: float(np.mean([1 - b(x) for x in
                                    calib.loc[calib["pool"] == pl, "adequate"]]))
                 for pl in np.unique(pools)}
    o_scores = np.array([pool_fail[pl] for pl in pools])
    ocurve = []
    for t in np.concatenate([[np.inf], np.unique(o_scores)[::-1], [-np.inf]]):
        esc = o_scores >= t
        ocurve.append((esc.mean(), hybrid_acc(esc, e), hybrid_acc(esc, p)))
    ocurve = np.array(ocurve)
    rand_o = (1 - ocurve[:, 0]) * small_acc + ocurve[:, 0] * big_acc
    d = ocurve[:, 1] - rand_o
    oracle_area = float(np.sum((d[1:] + d[:-1]) / 2 * np.diff(ocurve[:, 0])))
    print(f">>> pool-oracle-vs-random area (type-only upper bound) = "
          f"{oracle_area:+.4f} → internal-signal residual = "
          f"{lift_area - oracle_area:+.4f}", flush=True)
    print(f"    calib pool fail rates: "
          f"{ {k: round(v, 2) for k, v in sorted(pool_fail.items())} }", flush=True)

    # --- latency + cost -------------------------------------------------------
    el = exp["expert_latency"].values
    pl = para["paraphrase_latency"].replace(0.0, np.nan).dropna().values
    print(f"\n=== latency (s) ===\n  expert(gpt-5.5) P50={np.percentile(el,50):.1f} "
          f"P95={np.percentile(el,95):.1f}\n  paraphrase(small) P50="
          f"{np.percentile(pl,50):.1f} P95={np.percentile(pl,95):.1f}", flush=True)
    pt = exp["expert_prompt_tokens"].dropna().sum()
    ct = exp["expert_completion_tokens"].dropna().sum()
    # gpt-5.5 price: $5 / $30 per 1M in/out (July-2026)
    expert_cost = (pt * 5 + ct * 30) / 1e6
    print(f"\n=== cost (expert only; judge extra) ===\n  gpt-5.5 tokens "
          f"in={int(pt)} out={int(ct)} → ${expert_cost:.2f} for {len(exp)} queries "
          f"= ${expert_cost/len(exp)*100:.2f}/100q (big-only). Hybrid scales with "
          f"escalation rate.", flush=True)

    # --- figure ---------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.plot(curve[:, 0], curve[:, 1], "-o", ms=3, label="hybrid-gate (expert-inject)")
    plt.plot(curve[:, 0], curve[:, 2], "-o", ms=3, label="hybrid-gate (paraphrase)")
    rr, rl = rand_line(big_acc)
    plt.plot(rr, rl, "k--", alpha=.5, label="random escalate (expert)")
    plt.plot(ocurve[:, 0], ocurve[:, 1], "-s", ms=4, color="tab:purple",
             alpha=.85, label="pool-oracle escalate (type-only, expert)")
    for tier, (r, ae, ap) in tier_pts.items():
        plt.annotate(tier, (r, ae), fontsize=8,
                     textcoords="offset points", xytext=(4, 4))
        plt.scatter([r], [ae], c="red", zorder=5, s=20)
    plt.axhline(small_acc, color="gray", ls=":", alpha=.5)
    plt.axhline(big_acc, color="green", ls=":", alpha=.5)
    plt.xlabel("escalation rate"); plt.ylabel("accuracy (judge-adequate)")
    plt.title("Accuracy vs escalation rate — hybrid gate vs random\nMiniCPM-o 4.5 + gpt-5.5")
    plt.legend(loc="lower right"); plt.grid(alpha=.3); plt.tight_layout()
    fig = f"{DATA}/tradeoff.png"
    plt.savefig(fig, dpi=120)
    gate_data.commit()
    print(f"\n>>> tradeoff figure → {fig}", flush=True)
