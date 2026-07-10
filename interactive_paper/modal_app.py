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
OPENAI = modal.Secret.from_name("openai")


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


def _load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=False, init_tts=False,
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
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
                  secrets=[OPENAI], timeout=60 * 60)
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
        exp = escalate.ask_expert(dq)                       # escalate the DISTILLED query
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
                  timeout=60 * 60)
def eval_expert(concurrency: int = 6):
    """Phase 5 big-only: gpt-5.5 answers every TEST query (original, single-turn
    → already standalone), judged for adequacy. → eval_expert.parquet."""
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    df = pd.read_parquet(FEATURES)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f">>> eval_expert: {len(test)} test queries → gpt-5.5", flush=True)

    res = asyncio.run(escalate.ask_expert_many(test["query"].tolist(),
                                               concurrency=concurrency))
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
    curve, compares to the random-escalation baseline, prints latency/cost, and
    writes figures/tradeoff.png."""
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
