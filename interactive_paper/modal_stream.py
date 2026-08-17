"""Phase 8 (Part 2, milestone 1) — chat_gated: the gate goes live.

The full duplex loop, headless, per query: stream the user's AUDIO in 1s
chunks -> per-chunk mid-layer (L22) probe score -> EMA/hysteresis gate
(src/gate.py, its intended Phase-6 use) -> on fire, launch the gpt-5.5
expert in a background thread WHILE the model keeps listening -> at end of
turn, teacher-force the stall phrase (streaming_smoke's verified control
point) -> inject the expert result as an authority-framed text turn in the
SAME session -> the model relays it. Timers on every segment.

Scope guards (milestone 1): no barge-in, no video, no spoken output
(generate_audio=False), expert query = the pool's text form (live ASR is
step-2 follow-up work).

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run modal_stream.py::fit_midlayer_gate        # CPU $0 -> gate artifact
  modal run modal_stream.py::run_chat_gated --limit 3 # smoke
  modal run modal_stream.py::run_chat_gated           # full (24 q)
  modal run modal_stream.py::gated_report             # timings + judged acc
"""
import json
import os
import sys
import time

import modal

from modal_app import (app, gen_app, image, util_image, GPU_VOL, gate_data,
                       DATA, OPENAI, QUERIES, FEATURES, API_REGION,
                       GATE_CFG, EVAL_EXPERT, _read_jsonl)

HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(HERE, "modal_app.py")
image_st = image.add_local_file(_APP_PY, "/root/modal_app.py")
util_st = util_image.add_local_file(_APP_PY, "/root/modal_app.py")

AUDIO_DIR = f"{DATA}/audio_pool"
GATE_ART = f"{DATA}/midlayer_gate_audio.json"
TRACES = f"{DATA}/gated_traces.jsonl"
LAYER = 22
STALL = "Hmm, that's a good question — give me a moment to check."
RELAY_TMPL = ("[SYSTEM NOTE] A trusted expert system has already verified "
              "the answer to the user's last spoken question.\n"
              "Expert answer: {ans}\n"
              "Relay this answer to the user in your own words, concisely "
              "and naturally, as a continuation of the conversation. Do not "
              "second-guess or re-derive it.")


# ============================ gate artifact ================================
@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 20,
              cpu=8)
def fit_midlayer_gate():
    """L22 last-token probe on AUDIO calib rows (the live loop's modality)
    -> midlayer_gate_audio.json. Phase-3 recipe: C=0.001 score compression,
    thresholds = OOF-score quantiles at escalation budgets .15/.30/.50."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    ids, hl = [], []
    for s in sorted(glob.glob(f"{DATA}/layers_minicpm-o45-audio.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    hl = np.concatenate(hl)
    f = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "pool", "split", "escalate_label"]]
    m = pd.DataFrame({"id": ids, "row": range(len(ids))}).merge(f, on="id")
    m = m[(m["split"] == "calib") & m["escalate_label"].notna()]
    X = hl[m["row"].to_numpy(), LAYER, :].astype(np.float32)
    y = m["escalate_label"].astype(int).to_numpy()

    clf = LogisticRegression(C=0.001, max_iter=5000).fit(X, y)
    oof = cross_val_predict(LogisticRegression(C=0.001, max_iter=5000), X, y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)
    thr = {t: float(np.quantile(oof, 1 - b))
           for t, b in (("conservative", 0.15), ("balanced", 0.30),
                        ("aggressive", 0.50))}
    art = {"layer": LAYER, "modality": "audio", "n_calib": int(len(y)),
           "oof_auc": float(auc), "thresholds": thr,
           "w": clf.coef_[0].astype(float).tolist(),
           "b": float(clf.intercept_[0]),
           "gate": {"k_consecutive": 1, "ema_alpha": 1.0,
                    "cooldown_steps": 10 ** 6}}
    with open(GATE_ART, "w") as fh:
        json.dump(art, fh)
    gate_data.commit()
    print(f">>> L{LAYER} audio probe: calib n={len(y)} OOF AUC={auc:.3f} "
          f"thresholds={ {k: round(v, 3) for k, v in thr.items()} }", flush=True)


# ============================ chunk-score calibration ======================
CHUNK_SCORES = f"{DATA}/chunk_scores"


@app.function(image=image_st, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_chunk_scores(shard: list, shard_id: int) -> int:
    """Milestone-1 find #3 fix, step 1: run the EXACT live streaming procedure
    (sys prompt, 1s chunks, padded tail, L22 hook) on CALIB queries and store
    the per-chunk score sequence — no generation. The live gate's firing
    statistic is then calibrated on these, not on offline full-prefill scores."""
    import glob as _glob
    import inspect
    import shutil
    import numpy as np
    import pandas as pd
    import torch
    import librosa
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import gate as gate_mod

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

    art = json.load(open(GATE_ART))
    probe = gate_mod.Probe(art["w"], art["b"])

    def call_def(fn, /, **kw):
        params = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in params})

    holder = {}

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        holder["h"] = hs[0, -1, :].detach().float().cpu().numpy()

    rows = []
    for k, q in enumerate(shard):
        au, _ = librosa.load(f"{AUDIO_DIR}/{q['id']}.wav", sr=16000, mono=True)
        chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        model.reset_session()
        sys_msg = call_def(model.get_sys_prompt, mode="omni", language="en")
        call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
                 tokenizer=tok)
        h = model.llm.model.layers[LAYER].register_forward_hook(hook)
        scores, padded = [], []
        try:
            for i, ch in enumerate(chunks):
                pad = len(ch) < 16000
                if pad:
                    ch = np.pad(ch, (0, 16000 - len(ch)))
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[{"role": "user",
                                "content": [ch.astype(np.float32)]}],
                         tokenizer=tok,
                         is_last_chunk=(i == len(chunks) - 1))
                scores.append(round(float(probe.score(holder["h"])), 4))
                padded.append(pad)
        finally:
            h.remove()
        rows.append({"id": q["id"], "pool": q["pool"], "scores": scores,
                     "padded": padded})
        if k < 2 or k % 30 == 0:
            print(f"  [{k}] {q['pool']} n_chunks={len(scores)}", flush=True)
    pd.DataFrame(rows).to_parquet(f"{CHUNK_SCORES}.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote chunk scores shard {shard_id} ({len(rows)})", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_chunk_scores(workers: int = 4):
    qs = [q for q in _read_jsonl_vol.remote() if q["split"] == "calib"]
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> chunk-score calibration pass: {len(qs)} calib queries")
    total = sum(collect_chunk_scores.starmap(
        [(shards[i], i) for i in range(workers)]))
    print(f">>> collected {total}")


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_jsonl_vol() -> list:
    return _read_jsonl(QUERIES)


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 10)
def calibrate_chunk_gate():
    """Step 2: pick the firing statistic by AUC, re-quantile thresholds on it,
    write chunk_thresholds into the gate artifact."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.concat([pd.read_parquet(s) for s in
                    sorted(glob.glob(f"{CHUNK_SCORES}.shard*.parquet"))],
                   ignore_index=True)
    f = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "escalate_label"]]
    df = df.merge(f, on="id")
    df = df[df["escalate_label"].notna()]
    y = df["escalate_label"].astype(int).to_numpy()

    def unpadded(r):
        s = [x for x, p in zip(r["scores"], r["padded"]) if not p]
        return s if s else list(r["scores"])

    stats = {
        "max": df.apply(lambda r: max(r["scores"]), axis=1),
        "max_unpadded": df.apply(lambda r: max(unpadded(r)), axis=1),
        "last_unpadded": df.apply(lambda r: unpadded(r)[-1], axis=1),
        "mean_top2": df.apply(
            lambda r: float(np.mean(sorted(r["scores"])[-2:])), axis=1),
    }
    print(f"=== firing-statistic AUCs (calib n={len(df)}) ===", flush=True)
    aucs = {}
    for name, s in stats.items():
        aucs[name] = roc_auc_score(y, s)
        print(f"  {name:14s} AUC={aucs[name]:.3f}", flush=True)
    best = max(aucs, key=aucs.get)
    s = stats[best].to_numpy()
    thr = {t: float(np.quantile(s, 1 - b))
           for t, b in (("conservative", 0.15), ("balanced", 0.30),
                        ("aggressive", 0.50))}
    per_pool = {p: float((stats[best][df["pool"] == p] >= thr["balanced"]).mean())
                for p in sorted(df["pool"].unique())}
    print(f">>> best={best} thresholds={ {k: round(v, 3) for k, v in thr.items()} }",
          flush=True)
    print(f">>> predicted fire rate @balanced per pool: "
          f"{ {k: round(v, 2) for k, v in per_pool.items()} }", flush=True)

    art = json.load(open(GATE_ART))
    art["chunk_thresholds"] = thr
    art["chunk_statistic"] = best
    art["chunk_stat_auc"] = float(aucs[best])
    with open(GATE_ART, "w") as fh:
        json.dump(art, fh)
    gate_data.commit()
    print(f">>> updated {GATE_ART}", flush=True)


# ============================ the live loop ================================
@gen_app.function(image=image_st, gpu="H100", volumes=GPU_VOL,
                  secrets=[OPENAI], timeout=60 * 60 * 2)
def chat_gated(limit: int = 24, tier: str = "balanced") -> list:
    """Direct-run (gen_app + secret, e2e_demo precedent):
    modal run modal_stream.py::chat_gated --limit 3"""
    import glob as _glob
    import inspect
    import shutil
    import threading
    import numpy as np
    import torch
    import librosa
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import escalate
    import gate as gate_mod

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
    art = json.load(open(GATE_ART))
    probe = gate_mod.Probe(art["w"], art["b"])
    thr_src = ("chunk_thresholds" if "chunk_thresholds" in art
               else "thresholds")
    print(f">>> gate: L{art['layer']} audio probe (OOF {art['oof_auc']:.3f}), "
          f"tier={tier} thr={art[thr_src][tier]:.3f} [{thr_src}]", flush=True)

    qs = [q for q in _read_jsonl(QUERIES) if q["split"] == "test"
          and os.path.exists(f"{AUDIO_DIR}/{q['id']}.wav")]
    per_pool = max(1, limit // 5)
    queries, seen = [], {}
    for q in qs:
        if seen.get(q["pool"], 0) < per_pool:
            queries.append(q)
            seen[q["pool"]] = seen.get(q["pool"], 0) + 1
    print(f">>> {len(queries)} test queries ({seen})", flush=True)

    def call_def(fn, /, **kw):
        params = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in params})

    def gen_text(**kw):
        res = call_def(model.streaming_generate, tokenizer=tok,
                       temperature=0.1, generate_audio=False, **kw)
        parts = []
        if inspect.isgenerator(res) or hasattr(res, "__next__"):
            for r in res:
                t = getattr(r, "text", None)
                if t is None and isinstance(r, dict):
                    t = r.get("text")
                if t is None and isinstance(r, (tuple, list)) and r:
                    t = r[0]              # streaming yields (text, is_final)
                if isinstance(t, str):
                    parts.append(t)
        else:
            parts.append(str(res))
        return "".join(parts).strip()

    holder = {}

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        holder["h"] = hs[0, -1, :].detach().float().cpu().numpy()

    traces = []
    for qi, q in enumerate(queries):
        au, _ = librosa.load(f"{AUDIO_DIR}/{q['id']}.wav", sr=16000, mono=True)
        chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        g = gate_mod.EscalationGate(
            threshold=art[thr_src][tier],
            k_consecutive=art["gate"]["k_consecutive"],
            ema_alpha=art["gate"]["ema_alpha"],
            cooldown_steps=art["gate"]["cooldown_steps"])

        model.reset_session()
        sys_msg = call_def(model.get_sys_prompt, mode="omni", language="en")
        call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
                 tokenizer=tok)

        expert = {}

        def expert_call(query_text):
            t0 = time.time()
            try:
                r = escalate.ask_expert(query_text)
                expert["answer"] = (r.get("answer") if isinstance(r, dict)
                                    else str(r)) or ""
            except Exception as e:
                expert["answer"] = f"[expert error: {e}]"
            expert["s"] = time.time() - t0

        h = model.llm.model.layers[LAYER].register_forward_hook(hook)
        fired_at, scores, th = None, [], None
        try:
            for i, ch in enumerate(chunks):
                if len(ch) < 16000:
                    ch = np.pad(ch, (0, 16000 - len(ch)))
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[{"role": "user",
                                "content": [ch.astype(np.float32)]}],
                         tokenizer=tok,
                         is_last_chunk=(i == len(chunks) - 1))
                s = probe.score(holder["h"])
                scores.append(round(float(s), 4))
                if fired_at is None and g.update(s):
                    fired_at = i
                    th = threading.Thread(target=expert_call,
                                          args=(q["query"],), daemon=True)
                    th.start()          # expert races the rest of the turn
        finally:
            h.remove()

        t_eot = time.time()
        row = {"id": q["id"], "pool": q["pool"], "tier": tier,
               "audio_s": round(len(au) / 16000, 2), "n_chunks": len(chunks),
               "scores": scores, "fired_at_chunk": fired_at,
               "reference_answer": q.get("reference_answer"),
               "query": q["query"]}
        if fired_at is None:
            ans = gen_text(session_id="s1")
            row.update(mode="local", answer=ans,
                       answer_ms=int((time.time() - t_eot) * 1000))
        else:
            try:
                # canned-filler semantics: the stall phrase enters history as
                # the model's own utterance (tts_filler plays it in deployment)
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[{"role": "assistant", "content": [STALL]}],
                         tokenizer=tok, is_last_chunk=True)
                stall = STALL
            except Exception as e:
                print(f"  assistant-prefill stall failed ({type(e).__name__}); "
                      f"falling back to teacher forcing", flush=True)
                stall = gen_text(session_id="s1", teacher_forcing_text=STALL,
                                 max_new_tokens=8)
            t_stall = time.time()
            th.join(timeout=60)
            t_expert = time.time()
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "user", "content":
                            [RELAY_TMPL.format(ans=expert.get("answer", ""))]}],
                     tokenizer=tok, is_last_chunk=True)
            relay = gen_text(session_id="s1")
            row.update(mode="escalated", stall_text=stall, relay=relay,
                       expert_answer=expert.get("answer", ""),
                       stall_ms=int((t_stall - t_eot) * 1000),
                       expert_wait_ms=int((t_expert - t_stall) * 1000),
                       expert_total_s=round(expert.get("s", -1), 2),
                       relay_ms=int((time.time() - t_expert) * 1000))
        traces.append(row)
        print(f"  [{qi}] {q['id']} {q['pool']:14s} "
              f"{'ESC@' + str(fired_at) if fired_at is not None else 'local':>8s} "
              f"| {(row.get('relay') or row.get('answer', ''))[:60]!r}",
              flush=True)

    with open(TRACES, "a", encoding="utf-8") as fh:
        for r in traces:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> appended {len(traces)} traces to {TRACES}", flush=True)
    return [r["id"] for r in traces]


# (chat_gated is run directly: modal run modal_stream.py::chat_gated)


# ============ worklist ②a: end-of-turn second read (trap fix) ==============
# Grilling decision 2: the end-of-turn read becomes the DECIDING gate; the
# per-chunk gate is demoted to speculative prefetch. This pass validates the
# streaming end-of-turn position on calib before touching the live loop:
# after the last audio chunk, prefill an (empty) assistant turn — the hook
# then reads the probe at the assistant-start position, the streaming
# approximation of the offline h_prompt read whose trap catch was strong.
EOT_SCORES = f"{DATA}/eot_scores"


@app.function(image=image_st, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_eot_scores(shard: list, shard_id: int) -> int:
    """Replay the exact live streaming procedure on calib queries, recording
    per-chunk scores AND the end-of-turn read, so ③ can simulate the whole
    prefetch+decide policy from one file."""
    import glob as _glob
    import inspect
    import shutil
    import numpy as np
    import torch
    import librosa
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import gate as gate_mod

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

    art = json.load(open(GATE_ART))
    probe = gate_mod.Probe(art["w"], art["b"])

    def call_def(fn, /, **kw):
        params = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in params})

    holder = {}

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        holder["h"] = hs[0, -1, :].detach().float().cpu().numpy()

    rows = []
    for k, q in enumerate(shard):
        au, _ = librosa.load(f"{AUDIO_DIR}/{q['id']}.wav", sr=16000, mono=True)
        chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        model.reset_session()
        sys_msg = call_def(model.get_sys_prompt, mode="omni", language="en")
        call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
                 tokenizer=tok)
        h = model.llm.model.layers[LAYER].register_forward_hook(hook)
        scores, padded, eot, eot_err = [], [], None, None
        try:
            for i, ch in enumerate(chunks):
                pad = len(ch) < 16000
                if pad:
                    ch = np.pad(ch, (0, 16000 - len(ch)))
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[{"role": "user",
                                "content": [ch.astype(np.float32)]}],
                         tokenizer=tok,
                         is_last_chunk=(i == len(chunks) - 1))
                scores.append(round(float(probe.score(holder["h"])), 4))
                padded.append(pad)
            for content in ("", " "):     # empty assistant turn = header-only
                try:
                    call_def(model.streaming_prefill, session_id="s1",
                             msgs=[{"role": "assistant", "content": [content]}],
                             tokenizer=tok, is_last_chunk=True)
                    eot = round(float(probe.score(holder["h"])), 4)
                    break
                except Exception as e:
                    eot_err = f"{type(e).__name__}: {e}"
        finally:
            h.remove()
        rows.append({"id": q["id"], "pool": q["pool"], "scores": scores,
                     "padded": padded, "eot_score": eot, "eot_err": eot_err})
        if k < 3 or k % 30 == 0:
            print(f"  [{k}] {q['pool']:14s} last={scores[-1]:.3f} "
                  f"eot={eot if eot is not None else eot_err}", flush=True)

    import pandas as pd
    pd.DataFrame(rows).to_parquet(f"{EOT_SCORES}.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote eot shard {shard_id} ({len(rows)})", flush=True)
    return len(rows)


@app.local_entrypoint()
def run_eot_scores(workers: int = 4, limit: int = 0):
    qs = [q for q in _read_jsonl_vol.remote() if q["split"] == "calib"]
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> end-of-turn read pass: {len(qs)} calib queries")
    total = sum(collect_eot_scores.starmap(
        [(shards[i], i if not limit else 99) for i in range(workers)]))
    print(f">>> collected {total}")


# ========== option-B validation: false-premise trap pool (FalseQA) =========
# A trap pool whose failure mode is premise-checking, not obscure-entity
# recall — every word common, transcription-fair BY DESIGN (criterion fixed
# before data selection; the anti-cherry-pick answer). Public dataset only
# (no-selfmade-datasets rule). Validation protocol = the standard pool
# audition: small-model fail rate (text+audio) + eot gate signal + expert
# answerability.
FALSEQA_POOL = f"{DATA}/falseqa_pool.jsonl"
FALSEQA_AUDIO = f"{DATA}/falseqa_audio"
FALSEQA_EVAL = f"{DATA}/falseqa_eval.jsonl"


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 30)
def build_falseqa_pool(n: int = 60):
    import io
    import urllib.request
    import numpy as np
    import pandas as pd

    df = None
    try:
        from datasets import load_dataset
        df = load_dataset("thunlp/FalseQA", split="test").to_pandas()
        print(">>> loaded HF thunlp/FalseQA", flush=True)
    except Exception as e:
        print(f"  HF load failed ({type(e).__name__}); trying GitHub raw",
              flush=True)
        for url in (
            "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/test.csv",
            "https://raw.githubusercontent.com/thunlp/FalseQA/master/dataset/test.csv",
            "https://raw.githubusercontent.com/thunlp/FalseQA/main/data/test.csv",
            "https://raw.githubusercontent.com/thunlp/FalseQA/master/data/test.csv",
            "https://raw.githubusercontent.com/thunlp/FalseQA/main/FalseQA/test.csv",
        ):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    df = pd.read_csv(io.BytesIO(r.read()))
                print(f">>> loaded {url}", flush=True)
                break
            except Exception as e2:
                print(f"  {url}: {type(e2).__name__}", flush=True)
    if df is None:
        raise RuntimeError("could not fetch FalseQA from any source")

    df.columns = [c.lower().strip() for c in df.columns]
    print(f">>> columns: {list(df.columns)}, n={len(df)}", flush=True)
    qcol = next(c for c in df.columns if "question" in c)
    acol = next(c for c in df.columns if "answer" in c)
    lcol = next((c for c in df.columns if "label" in c), None)
    if lcol is not None:
        df = df[df[lcol] == 1]                    # 1 = false premise
        print(f">>> false-premise rows: {len(df)}", flush=True)
    df = df[df[qcol].str.len().between(15, 220)]
    rng = np.random.default_rng(42)
    pick = df.iloc[rng.choice(len(df), size=min(n, len(df)), replace=False)]
    rows = [{"id": f"fp{i:04d}", "pool": "false-premise",
             "source": "falseqa", "query": str(r[qcol]).strip(),
             "reference_answer": str(r[acol]).strip()}
            for i, (_, r) in enumerate(pick.iterrows())]
    with open(FALSEQA_POOL, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    for r in rows[:5]:
        print(f"  {r['id']}: {r['query'][:80]!r}", flush=True)
    print(f">>> wrote {len(rows)} false-premise queries", flush=True)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 30, region=API_REGION)
def tts_falseqa(limit: int = 0, concurrency: int = 8):
    """Render the candidate pool to wav (tts-1, alloy — same as the frozen
    pool; idempotent)."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    qs = _read_jsonl(FALSEQA_POOL)
    if limit:
        qs = qs[:limit]
    os.makedirs(FALSEQA_AUDIO, exist_ok=True)
    client = OpenAI()

    def render(q):
        out = f"{FALSEQA_AUDIO}/{q['id']}.wav"
        if os.path.exists(out):
            return "skip"
        resp = client.audio.speech.create(model="tts-1", voice="alloy",
                                          input=q["query"][:4000],
                                          response_format="wav")
        with open(out, "wb") as fh:
            fh.write(resp.content)
        return "done"

    with ThreadPoolExecutor(concurrency) as ex:
        results = list(ex.map(render, qs))
    gate_data.commit()
    print(f">>> tts_falseqa: {results.count('done')} rendered, "
          f"{results.count('skip')} skipped", flush=True)


@app.function(image=image_st, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 2)
def falseqa_eval():
    """The audition: text answer + live-audio answer + end-of-turn gate
    score per candidate query."""
    import glob as _glob
    import inspect
    import shutil
    import numpy as np
    import librosa
    import torch
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import gate as gate_mod

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
    art = json.load(open(GATE_ART))
    probe = gate_mod.Probe(art["w"], art["b"])

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

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        holder["h"] = hs[0, -1, :].detach().float().cpu().numpy()

    rows = []
    qs = [q for q in _read_jsonl(FALSEQA_POOL)
          if os.path.exists(f"{FALSEQA_AUDIO}/{q['id']}.wav")]
    print(f">>> auditioning {len(qs)} false-premise queries", flush=True)
    for i, q in enumerate(qs):
        txt = model.chat(msgs=[{"role": "user", "content": [q["query"]]}],
                         tokenizer=tok, max_new_tokens=512, sampling=False,
                         use_tts_template=False, generate_audio=False)
        au, _ = librosa.load(f"{FALSEQA_AUDIO}/{q['id']}.wav", sr=16000,
                             mono=True)
        chunks = [au[j:j + 16000] for j in range(0, len(au), 16000)]
        model.reset_session()
        sys_msg = call_def(model.get_sys_prompt, mode="omni", language="en")
        call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
                 tokenizer=tok)
        h = model.llm.model.layers[LAYER].register_forward_hook(hook)
        try:
            for j, ch in enumerate(chunks):
                if len(ch) < 16000:
                    ch = np.pad(ch, (0, 16000 - len(ch)))
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[{"role": "user",
                                "content": [ch.astype(np.float32)]}],
                         tokenizer=tok,
                         is_last_chunk=(j == len(chunks) - 1))
            for content in ("", " "):
                try:
                    call_def(model.streaming_prefill, session_id="s1",
                             msgs=[{"role": "assistant",
                                    "content": [content]}],
                             tokenizer=tok, is_last_chunk=True)
                    eot = float(probe.score(holder["h"]))
                    break
                except Exception:
                    eot = None
        finally:
            h.remove()
        audio_ans = gen_text(session_id="s1")
        rows.append({**q, "text_answer": str(txt).strip(),
                     "audio_answer": audio_ans,
                     "eot_score": round(eot, 4) if eot is not None else None})
        if i < 3 or i % 15 == 0:
            print(f"  [{i}] eot={eot:.3f} | {audio_ans[:60]!r}", flush=True)
    with open(FALSEQA_EVAL, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> wrote {len(rows)} audition rows", flush=True)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 30, region=API_REGION)
def falseqa_report(n_expert: int = 20):
    """Audition verdict: fail rates, gate-signal fire rate, expert
    answerability."""
    import asyncio
    import numpy as np
    sys.path.insert(0, "/workspace/gate")
    import escalate

    from modal_app import EXPERT_CACHE
    rows = _read_jsonl(FALSEQA_EVAL)
    art = json.load(open(GATE_ART))
    thr = art["eot_thresholds"]

    for arm in ("text_answer", "audio_answer"):
        judged = asyncio.run(escalate.judge_many(
            [{"query": r["query"], "reference_answer": r["reference_answer"],
              "answer": r[arm]} for r in rows], concurrency=8))
        for r, j in zip(rows, judged):
            r[f"{arm}_ok"] = j["adequate"]
    t_fail = np.mean([not r["text_answer_ok"] for r in rows
                      if r["text_answer_ok"] is not None])
    a_fail = np.mean([not r["audio_answer_ok"] for r in rows
                      if r["audio_answer_ok"] is not None])
    eot = np.array([r["eot_score"] for r in rows
                    if r["eot_score"] is not None])
    print(f"=== false-premise audition (n={len(rows)}) ===", flush=True)
    print(f"  small-model fail rate: text {t_fail:.2f} | audio {a_fail:.2f} "
          f"(SimpleQA trap reference: ~1.00)", flush=True)
    print(f"  eot score: mean {eot.mean():.3f} P25/50/75 "
          f"{np.percentile(eot, 25):.3f}/{np.median(eot):.3f}/"
          f"{np.percentile(eot, 75):.3f}", flush=True)
    for t in ("conservative", "balanced", "aggressive"):
        print(f"  fire@{t}: {(eot >= thr[t]).mean():.2f}", flush=True)
    fails = [r for r in rows if r["audio_answer_ok"] is False
             and r["eot_score"] is not None]
    if fails:
        print(f"  eot mean on audio-FAILED subset: "
              f"{np.mean([r['eot_score'] for r in fails]):.3f}", flush=True)

    exp = asyncio.run(escalate.ask_expert_many(
        [r["query"] for r in rows[:n_expert]], concurrency=3, effort="low",
        cache_dir=EXPERT_CACHE))
    ej = asyncio.run(escalate.judge_many(
        [{"query": r["query"], "reference_answer": r["reference_answer"],
          "answer": e.get("answer") or ""} for r, e in zip(rows, exp)],
        concurrency=8))
    e_ok = np.mean([j["adequate"] for j in ej
                    if j["adequate"] is not None])
    print(f"  expert (gpt-5.5 low) adequacy on {n_expert}: {e_ok:.2f} "
          f"(SimpleQA trap reference: .65-.71)", flush=True)
    with open(FALSEQA_EVAL, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(">>> audition rows updated with judgments", flush=True)


# ============ worklist ⑤: query-feature router baseline ====================
@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 20,
              cpu=8)
def router_baseline():
    """The RouteLLM-style external baseline the grilling decided to add
    (closes the 'type recognition is all you need' objection with data):
    a text-only classifier over the query surface (TF-IDF word+char
    n-grams -> LR), trained on calib TEXT labels, evaluated three ways —
    in-mix AUC, LOPO transfer (predicted: collapses, it has ONLY the type
    shortcut), and the Phase-5-style tradeoff area on test (expert-inject,
    vs the random-escalation line). $0 — frozen data only."""
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import FeatureUnion, Pipeline

    from modal_app import EVAL_EXPERT
    f = pd.read_parquet(FEATURES)[["id", "pool", "split", "escalate_label"]]
    qtext = {q["id"]: q["query"] for q in _read_jsonl(QUERIES)}
    f = f[f["escalate_label"].notna() & f["id"].isin(qtext)]
    f["text"] = f["id"].map(qtext)
    cal = f[f["split"] == "calib"].reset_index(drop=True)
    tst = f[f["split"] == "test"].reset_index(drop=True)
    y_cal = cal["escalate_label"].astype(int).to_numpy()

    def make_pipe():
        return Pipeline([
            ("tfidf", FeatureUnion([
                ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                ("c", TfidfVectorizer(analyzer="char_wb",
                                      ngram_range=(3, 5), min_df=2)),
            ])),
            ("lr", LogisticRegression(C=1.0, max_iter=3000)),
        ])

    oof = cross_val_predict(make_pipe(), cal["text"], y_cal,
                            cv=StratifiedKFold(5, shuffle=True,
                                               random_state=42),
                            method="predict_proba")[:, 1]
    print(f"=== router (text-only) ===", flush=True)
    print(f"  calib OOF AUC = {roc_auc_score(y_cal, oof):.3f} "
          f"(probe h_prompt .828 / pool-oracle .715)", flush=True)

    # ---- training receipt (method / params / losses / accuracy) ----
    fit_full = make_pipe().fit(cal["text"], y_cal)
    p_train = fit_full.predict_proba(cal["text"])[:, 1]
    n_feat = int(len(fit_full.named_steps["tfidf"].get_feature_names_out()))
    receipt = {
        "method": "TF-IDF (word 1-2gram + char_wb 3-5gram, min_df=2) -> "
                  "LogisticRegression(C=1.0, penalty=l2, solver=lbfgs, "
                  "max_iter=3000); 5-fold stratified OOF (seed 42)",
        "n_train": int(len(cal)), "n_test": 0,  # n_test filled below
        "escalate_rate_train": float(y_cal.mean()),
        "n_features": n_feat,
        "train_logloss": float(log_loss(y_cal, p_train)),
        "train_acc": float(accuracy_score(y_cal, p_train >= .5)),
        "oof_logloss": float(log_loss(y_cal, oof)),
        "oof_acc": float(accuracy_score(y_cal, oof >= .5)),
        "majority_acc_calib": float(max(y_cal.mean(), 1 - y_cal.mean())),
    }
    print("--- training receipt ---", flush=True)
    print(f"  {receipt['method']}", flush=True)
    print(f"  n_train={receipt['n_train']} (escalate rate "
          f"{receipt['escalate_rate_train']:.3f}), features={n_feat}",
          flush=True)
    print(f"  train logloss={receipt['train_logloss']:.3f} "
          f"acc={receipt['train_acc']:.3f} | eval (OOF) "
          f"logloss={receipt['oof_logloss']:.3f} "
          f"acc={receipt['oof_acc']:.3f} "
          f"(majority {receipt['majority_acc_calib']:.3f})", flush=True)

    print("--- LOPO (train without a pool, test on it) ---", flush=True)
    for p in sorted(cal["pool"].unique()):
        tr = cal[cal["pool"] != p]
        te = cal[cal["pool"] == p]
        if te["escalate_label"].nunique() < 2:
            print(f"  {p:15s} single-class (fail rate "
                  f"{te['escalate_label'].mean():.2f}) — AUC undefined; "
                  f"mean score {make_pipe().fit(tr['text'], tr['escalate_label'].astype(int)).predict_proba(te['text'])[:, 1].mean():.3f}",
                  flush=True)
            continue
        pipe = make_pipe().fit(tr["text"], tr["escalate_label"].astype(int))
        s = pipe.predict_proba(te["text"])[:, 1]
        print(f"  {p:15s} AUC={roc_auc_score(te['escalate_label'].astype(int), s):.3f}",
              flush=True)

    pipe = fit_full
    tst = tst.merge(pd.read_parquet(EVAL_EXPERT)[["id", "expert_adequate"]],
                    on="id")
    s_test = pipe.predict_proba(tst["text"])[:, 1]
    small_ok = 1 - tst["escalate_label"].astype(int).to_numpy()
    exp_ok = tst["expert_adequate"].astype(float).to_numpy()
    order = np.argsort(-s_test)
    n = len(tst)
    accs, rand = [], []
    a0, a1 = small_ok.mean(), exp_ok.mean()
    for k in range(n + 1):
        esc = np.zeros(n, dtype=bool)
        esc[order[:k]] = True
        accs.append(np.where(esc, exp_ok, small_ok).mean())
        rand.append(a0 + (a1 - a0) * k / n)
    area = float(np.trapezoid(np.array(accs) - np.array(rand),
                              dx=1 / n) if hasattr(np, "trapezoid")
                 else np.trapz(np.array(accs) - np.array(rand), dx=1 / n))
    test_auc = roc_auc_score(tst["escalate_label"].astype(int), s_test)
    y_tst = tst["escalate_label"].astype(int).to_numpy()
    receipt["n_test"] = int(n)
    receipt["test_logloss"] = float(log_loss(y_tst, s_test))
    receipt["test_acc"] = float(accuracy_score(y_tst, s_test >= .5))
    receipt["majority_acc_test"] = float(max(y_tst.mean(), 1 - y_tst.mean()))
    print(f"--- test tradeoff (expert-inject) ---", flush=True)
    print(f"  test AUC = {test_auc:.3f}; area over random = {area:+.3f} "
          f"(probe_final +.054 / midlayer_L22 +.064 / ptrue_post +.068)",
          flush=True)
    print(f"  test n={n} logloss={receipt['test_logloss']:.3f} "
          f"acc={receipt['test_acc']:.3f} "
          f"(majority {receipt['majority_acc_test']:.3f})", flush=True)
    for k_frac in (0.15, 0.30, 0.50):
        k = int(round(k_frac * n))
        print(f"  acc @ {k_frac:.0%} escalation = {accs[k]:.3f}", flush=True)

    out = {"oof_auc": float(roc_auc_score(y_cal, oof)),
           "test_auc": float(test_auc), "area": area,
           "curve": [float(a) for a in accs],
           "receipt": receipt}
    with open(f"{DATA}/router_baseline.json", "w") as fh:
        json.dump(out, fh)
    gate_data.commit()
    print(f">>> wrote {DATA}/router_baseline.json", flush=True)


# ============ worklist ⑤b: RouterBench score for the same router ===========
@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 90,
              cpu=16, memory=32768)
def router_bench():
    """Public-benchmark grounding for the 8f router (grilling ask: 'what
    would this router score on a routing bench?'). RouterBench 0-shot
    (withmartian, ~30k prompts, 11 models): pair routing weak=mixtral-8x7b
    -> strong=gpt-4, label = weak model incorrect (same semantics as our
    escalate_label). Three readouts: (1) in-domain — the same TF-IDF+LR
    recipe trained on RouterBench itself, 5-fold OOF AUC/acc/logloss;
    (2) leave-one-benchmark-out — the LOPO mirror on public data;
    (3) zero-shot transfer — our calib-trained router scored on
    RouterBench. $0 GPU — CPU only."""
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import FeatureUnion, Pipeline

    def make_pipe():  # identical recipe to router_baseline (8f)
        return Pipeline([
            ("tfidf", FeatureUnion([
                ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                ("c", TfidfVectorizer(analyzer="char_wb",
                                      ngram_range=(3, 5), min_df=2)),
            ])),
            ("lr", LogisticRegression(C=1.0, max_iter=3000)),
        ])

    pkl = hf_hub_download("withmartian/routerbench",
                          "routerbench_0shot.pkl", repo_type="dataset")
    df = pd.read_pickle(pkl)
    print(f"columns: {list(df.columns)}", flush=True)

    def score_col(tag):
        hits = [c for c in df.columns
                if tag in c.lower() and "cost" not in c.lower()
                and "response" not in c.lower()]
        assert len(hits) == 1, f"{tag} -> {hits}"
        return hits[0]

    weak_c, strong_c = score_col("mixtral"), score_col("gpt-4")
    df = df[["prompt", "eval_name", weak_c, strong_c]].dropna()
    text = df["prompt"].astype(str)
    weak_ok = (df[weak_c].astype(float) >= .5).to_numpy()
    strong_ok = (df[strong_c].astype(float) >= .5).to_numpy()
    y = (~weak_ok).astype(int)
    print(f"=== RouterBench 0-shot, n={len(df)}: weak={weak_c} "
          f"(correct {weak_ok.mean():.3f}) strong={strong_c} "
          f"(correct {strong_ok.mean():.3f}); escalate rate {y.mean():.3f}",
          flush=True)

    oof = cross_val_predict(make_pipe(), text, y,
                            cv=StratifiedKFold(5, shuffle=True,
                                               random_state=42),
                            method="predict_proba", n_jobs=5)[:, 1]
    in_auc = roc_auc_score(y, oof)
    in_acc = accuracy_score(y, oof >= .5)
    print(f"--- in-domain (trained on RouterBench) ---", flush=True)
    print(f"  OOF AUC={in_auc:.3f} acc={in_acc:.3f} "
          f"logloss={log_loss(y, oof):.3f} "
          f"(majority {max(y.mean(), 1 - y.mean()):.3f})", flush=True)

    # deferral curve in our paper's currency: escalate top-k by router score
    def pair_area(scores):
        order = np.argsort(-scores)
        n = len(scores)
        a0, a1 = weak_ok.mean(), strong_ok.mean()
        accs = []
        for k in range(0, n + 1, max(1, n // 200)):
            esc = np.zeros(n, dtype=bool)
            esc[order[:k]] = True
            accs.append(np.where(esc, strong_ok, weak_ok).mean())
        xs = np.linspace(0, 1, len(accs))
        rand = a0 + (a1 - a0) * xs
        return (float(np.trapezoid(np.array(accs) - rand, xs)
                      if hasattr(np, "trapezoid")
                      else np.trapz(np.array(accs) - rand, xs)), accs, xs)

    in_area, in_curve, xs = pair_area(oof)
    for f_target in (.15, .30, .50):
        i = int(round(f_target * (len(in_curve) - 1)))
        print(f"  pair acc @ {f_target:.0%} escalation = "
              f"{in_curve[i]:.3f}", flush=True)
    print(f"  area over random = {in_area:+.3f}", flush=True)

    print("--- leave-one-benchmark-out (LOPO mirror) ---", flush=True)
    lobo = {}
    for b in sorted(df["eval_name"].unique()):
        m = (df["eval_name"] == b).to_numpy()
        if len(set(y[m])) < 2:
            print(f"  {b:15s} single-class (fail rate {y[m].mean():.2f})",
                  flush=True)
            continue
        pipe = make_pipe().fit(text[~m], y[~m])
        s = pipe.predict_proba(text[m])[:, 1]
        lobo[b] = float(roc_auc_score(y[m], s))
        print(f"  {b:15s} AUC={lobo[b]:.3f} (n={m.sum()}, "
              f"fail rate {y[m].mean():.2f})", flush=True)

    # zero-shot transfer: our calib-trained router scored on RouterBench
    f = pd.read_parquet(FEATURES)[["id", "split", "escalate_label"]]
    qtext = {q["id"]: q["query"] for q in _read_jsonl(QUERIES)}
    f = f[(f["split"] == "calib") & f["escalate_label"].notna()
          & f["id"].isin(qtext)]
    ours = make_pipe().fit(f["id"].map(qtext),
                           f["escalate_label"].astype(int))
    s_ours = ours.predict_proba(text)[:, 1]
    tr_auc = roc_auc_score(y, s_ours)
    tr_area, _, _ = pair_area(s_ours)
    print(f"--- zero-shot transfer (our calib router -> RouterBench) ---",
          flush=True)
    print(f"  AUC={tr_auc:.3f} area over random={tr_area:+.3f}", flush=True)

    out = {"n": int(len(df)), "weak": weak_c, "strong": strong_c,
           "escalate_rate": float(y.mean()),
           "in_domain": {"oof_auc": float(in_auc), "oof_acc": float(in_acc),
                         "oof_logloss": float(log_loss(y, oof)),
                         "area": in_area},
           "lobo_auc": lobo,
           "transfer": {"auc": float(tr_auc), "area": tr_area}}
    with open(f"{DATA}/router_bench.json", "w") as fh:
        json.dump(out, fh)
    gate_data.commit()
    print(f">>> wrote {DATA}/router_bench.json", flush=True)


# ============ worklist ⑤c: same receipt for the internal probes ============
@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 20,
              cpu=8)
def probe_receipt():
    """The router receipt (8j), applied to our own gates — symmetric
    accounting. Two probes, exact shipped recipes: (a) text h_prompt probe
    (gate_config.json: LR C=0.001 on 4096-d, the Phase-4 gate), (b) audio
    mid-layer L22 probe (fit_midlayer_gate: LR C=0.001, the live gate).
    Per probe: train / 5-fold-OOF / test logloss + acc@0.5 vs majority +
    AUC, plus test classification acc at the budget thresholds
    (OOF-score quantiles .15/.30/.50 — the gate's actual operating mode;
    0.5 is not how either signal is used). $0, frozen features."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    def receipt(name, Xc, yc, Xt, yt, C, max_iter):
        lr = LogisticRegression(C=C, max_iter=max_iter)
        oof = cross_val_predict(LogisticRegression(C=C, max_iter=max_iter),
                                Xc, yc,
                                cv=StratifiedKFold(5, shuffle=True,
                                                   random_state=42),
                                method="predict_proba")[:, 1]
        lr.fit(Xc, yc)
        p_tr = lr.predict_proba(Xc)[:, 1]
        r = {"name": name, "C": C, "dim": int(Xc.shape[1]),
             "n_train": int(len(yc)), "escalate_rate": float(yc.mean()),
             "train_logloss": float(log_loss(yc, p_tr)),
             "train_acc": float(accuracy_score(yc, p_tr >= .5)),
             "oof_auc": float(roc_auc_score(yc, oof)),
             "oof_logloss": float(log_loss(yc, oof)),
             "oof_acc": float(accuracy_score(yc, oof >= .5)),
             "majority_calib": float(max(yc.mean(), 1 - yc.mean()))}
        print(f"=== {name}: LR(C={C}) dim={r['dim']} n={r['n_train']} "
              f"(esc {r['escalate_rate']:.3f}) ===", flush=True)
        print(f"  train logloss={r['train_logloss']:.3f} "
              f"acc={r['train_acc']:.3f} | OOF AUC={r['oof_auc']:.3f} "
              f"logloss={r['oof_logloss']:.3f} acc={r['oof_acc']:.3f} "
              f"(majority {r['majority_calib']:.3f})", flush=True)
        if Xt is not None and len(set(yt)) == 2:
            s = lr.predict_proba(Xt)[:, 1]
            r.update({"n_test": int(len(yt)),
                      "test_auc": float(roc_auc_score(yt, s)),
                      "test_logloss": float(log_loss(yt, s)),
                      "test_acc": float(accuracy_score(yt, s >= .5)),
                      "majority_test": float(max(yt.mean(),
                                                 1 - yt.mean()))})
            print(f"  test n={r['n_test']} AUC={r['test_auc']:.3f} "
                  f"logloss={r['test_logloss']:.3f} "
                  f"acc={r['test_acc']:.3f} "
                  f"(majority {r['majority_test']:.3f})", flush=True)
            r["budget_ops"] = {}
            for b in (.15, .30, .50):
                thr = float(np.quantile(oof, 1 - b))
                fire = s >= thr
                r["budget_ops"][str(b)] = {
                    "acc": float(accuracy_score(yt, fire)),
                    "realized_rate": float(fire.mean())}
                print(f"  @budget {b:.0%}: classification acc="
                      f"{accuracy_score(yt, fire):.3f} "
                      f"(realized escalation {fire.mean():.3f})", flush=True)
        return r

    out = []
    f = pd.read_parquet(FEATURES)
    f = f[f["escalate_label"].notna()]
    cal = f[f["split"] == "calib"]
    tst = f[f["split"] == "test"]
    out.append(receipt(
        "text h_prompt probe (Phase-4 gate)",
        np.array([list(v) for v in cal["h_prompt"]], float),
        cal["escalate_label"].astype(int).to_numpy(),
        np.array([list(v) for v in tst["h_prompt"]], float),
        tst["escalate_label"].astype(int).to_numpy(),
        C=0.001, max_iter=2000))

    ids, hl = [], []
    for s in sorted(glob.glob(f"{DATA}/layers_minicpm-o45-audio.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    hl = np.concatenate(hl)
    fa = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "split", "escalate_label"]]
    m = pd.DataFrame({"id": ids, "row": range(len(ids))}).merge(fa, on="id")
    m = m[m["escalate_label"].notna()]
    mc = m[m["split"] == "calib"]
    mt = m[m["split"] == "test"]
    out.append(receipt(
        "audio mid-layer L22 probe (live gate)",
        hl[mc["row"].to_numpy(), LAYER, :].astype(np.float32),
        mc["escalate_label"].astype(int).to_numpy(),
        hl[mt["row"].to_numpy(), LAYER, :].astype(np.float32)
        if len(mt) else None,
        mt["escalate_label"].astype(int).to_numpy(),
        C=0.001, max_iter=5000))

    with open(f"{DATA}/probe_receipt.json", "w") as fh:
        json.dump(out, fh)
    gate_data.commit()
    print(f">>> wrote {DATA}/probe_receipt.json", flush=True)


# ============ worklist ⑤f: RouteLLM released-checkpoint baseline ===========
# torch + the routellm package (pulled from git; the pip release lags).
rllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.8.0")
    .pip_install("git+https://github.com/lm-sys/RouteLLM.git",
                 "openai", "datasets", "pandas", "pyarrow", "scikit-learn")
    .add_local_file(_APP_PY, "/root/modal_app.py")
)


@app.function(image=rllm_image, volumes={DATA: gate_data}, timeout=60 * 30,
              cpu=8, secrets=[OPENAI])
def routellm_baseline():
    """User challenge: compare against a REAL preference-trained router,
    not just our same-data recipe. Zero-shot eval of RouteLLM's released
    checkpoints — bert_gpt4_augmented and mf_gpt4_augmented, trained on
    ~100k GPT-4-vs-mixtral preference pairs — on our labeled pool.
    Score = calculate_strong_win_rate(prompt) (P(strong model needed)),
    used directly as the escalation score against escalate_label."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from routellm.routers.routers import ROUTER_CLS

    from modal_app import EVAL_EXPERT
    f = pd.read_parquet(FEATURES)[["id", "pool", "split", "escalate_label"]]
    qtext = {q["id"]: q["query"] for q in _read_jsonl(QUERIES)}
    f = f[f["escalate_label"].notna() & f["id"].isin(qtext)]
    f["text"] = f["id"].map(qtext).str.slice(0, 4000)
    exp = pd.read_parquet(EVAL_EXPERT)[["id", "expert_adequate"]]

    out = {}
    for name, ckpt in (("bert", "routellm/bert_gpt4_augmented"),
                       ("mf", "routellm/mf_gpt4_augmented")):
        r = ROUTER_CLS[name](ckpt)
        f[name] = [float(r.calculate_strong_win_rate(t)) for t in f["text"]]
        cal = f[f["split"] == "calib"]
        tst = f[f["split"] == "test"].merge(exp, on="id")
        y_cal = cal["escalate_label"].astype(int).to_numpy()
        y_tst = tst["escalate_label"].astype(int).to_numpy()
        s = tst[name].to_numpy()
        small_ok = 1 - y_tst
        exp_ok = tst["expert_adequate"].astype(float).to_numpy()
        order = np.argsort(-s)
        n = len(tst)
        accs, rand = [], []
        a0, a1 = small_ok.mean(), exp_ok.mean()
        for k in range(n + 1):
            esc = np.zeros(n, dtype=bool)
            esc[order[:k]] = True
            accs.append(np.where(esc, exp_ok, small_ok).mean())
            rand.append(a0 + (a1 - a0) * k / n)
        area = float(np.trapezoid(np.array(accs) - np.array(rand), dx=1 / n)
                     if hasattr(np, "trapezoid")
                     else np.trapz(np.array(accs) - np.array(rand), dx=1 / n))
        out[name] = {
            "calib_auc": float(roc_auc_score(y_cal, cal[name])),
            "test_auc": float(roc_auc_score(y_tst, s)),
            "area": area,
            "acc_at_30": float(accs[int(round(.3 * n))]),
            "pool_mean_score": {p: float(g[name].mean())
                                for p, g in f.groupby("pool")},
        }
        print(f"=== RouteLLM {name} ({ckpt}) zero-shot on our pool ===",
              flush=True)
        print(f"  calib AUC={out[name]['calib_auc']:.3f} "
              f"test AUC={out[name]['test_auc']:.3f} "
              f"area={area:+.3f} acc@30%={out[name]['acc_at_30']:.3f} "
              f"(our router .669/.721/+.040/.717; probe .828//+.054)",
              flush=True)
        for p, v in sorted(out[name]["pool_mean_score"].items()):
            print(f"    {p:15s} mean score {v:.3f}", flush=True)

    # same scores vs the AUDIO labels (TTS-read versions of the same
    # queries — apples-to-apples with the audio L22 probe .843/.879)
    fa = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "split", "escalate_label"]]
    fa = fa[fa["escalate_label"].notna()].rename(
        columns={"split": "split_a", "escalate_label": "y_audio"})
    fm = f.merge(fa, on="id")
    for name in ("bert", "mf"):
        for sp in ("calib", "test"):
            d = fm[fm["split_a"] == sp]
            auc = roc_auc_score(d["y_audio"].astype(int), d[name])
            out[name][f"audio_{sp}_auc"] = float(auc)
            print(f"  {name} vs AUDIO labels [{sp}] AUC={auc:.3f} "
                  f"(audio probe {'OOF .843' if sp == 'calib' else '.879'})",
                  flush=True)

    with open(f"{DATA}/routellm_baseline.json", "w") as fh:
        json.dump(out, fh)
    gate_data.commit()
    print(f">>> wrote {DATA}/routellm_baseline.json", flush=True)


# ============ worklist ⑤g: audio-modality router baseline ==================
@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 20,
              cpu=8)
def router_audio():
    """User challenge: 'is there an audio router?' There wasn't — every
    router number so far is text-modality. This closes it: the same
    TF-IDF+LR recipe against the AUDIO labels (the live gate's own
    label set, esc rate .41), two input variants: (a) gold query text —
    the router's best case; (b) the talker's own ASR transcript — what
    an external router would actually see in a live audio session.
    Opponent: audio L22 probe (OOF .843 / test .879). $0, frozen data."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import FeatureUnion, Pipeline

    def make_pipe():  # identical recipe to 8f
        return Pipeline([
            ("tfidf", FeatureUnion([
                ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                ("c", TfidfVectorizer(analyzer="char_wb",
                                      ngram_range=(3, 5), min_df=2)),
            ])),
            ("lr", LogisticRegression(C=1.0, max_iter=3000)),
        ])

    fa = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "pool", "split", "escalate_label"]]
    fa = fa[fa["escalate_label"].notna()]
    qtext = {q["id"]: q["query"] for q in _read_jsonl(QUERIES)}
    asr = pd.concat([pd.read_parquet(s) for s in
                     sorted(glob.glob(f"{DATA}/asr_minicpm-o45-audio"
                                      f".shard*.parquet"))],
                    ignore_index=True)
    tr = dict(zip(asr["id"], asr["transcript"]))
    print(f"audio rows {len(fa)}; with self-ASR transcript "
          f"{fa['id'].isin(tr).sum()}", flush=True)

    out = {}
    for name, text_of in (("gold_text", qtext), ("asr_transcript", tr)):
        d = fa[fa["id"].isin(text_of)].copy()
        d["text"] = d["id"].map(text_of).fillna("")
        cal = d[d["split"] == "calib"]
        tst = d[d["split"] == "test"]
        y_cal = cal["escalate_label"].astype(int).to_numpy()
        y_tst = tst["escalate_label"].astype(int).to_numpy()
        oof = cross_val_predict(make_pipe(), cal["text"], y_cal,
                                cv=StratifiedKFold(5, shuffle=True,
                                                   random_state=42),
                                method="predict_proba")[:, 1]
        s = make_pipe().fit(cal["text"], y_cal).predict_proba(
            tst["text"])[:, 1]
        out[name] = {
            "n_calib": int(len(cal)), "n_test": int(len(tst)),
            "oof_auc": float(roc_auc_score(y_cal, oof)),
            "oof_acc": float(accuracy_score(y_cal, oof >= .5)),
            "majority_calib": float(max(y_cal.mean(), 1 - y_cal.mean())),
            "test_auc": float(roc_auc_score(y_tst, s)),
            "test_acc": float(accuracy_score(y_tst, s >= .5)),
            "majority_test": float(max(y_tst.mean(), 1 - y_tst.mean()))}
        r = out[name]
        print(f"=== audio-label router [{name}] n={r['n_calib']}/"
              f"{r['n_test']} ===", flush=True)
        print(f"  OOF AUC={r['oof_auc']:.3f} acc={r['oof_acc']:.3f} "
              f"(maj {r['majority_calib']:.3f}) | test "
              f"AUC={r['test_auc']:.3f} acc={r['test_acc']:.3f} "
              f"(maj {r['majority_test']:.3f})  "
              f"[audio probe: OOF .843 / test .879]", flush=True)

    with open(f"{DATA}/router_audio.json", "w") as fh:
        json.dump(out, fh)
    gate_data.commit()
    print(f">>> wrote {DATA}/router_audio.json", flush=True)


# ============ worklist ⑤e: router fairness sweep ===========================
@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 20,
              cpu=8)
def router_sweep():
    """User challenge on 8f: 'prove you didn't train a deliberately weak
    router to flatter the probe.' Grid over feature space x min_df x
    regularization (24 configs), each scored by the same 5-fold OOF AUC
    protocol as 8f/probes. If no config beats .669 by more than noise,
    .669 is the query-surface ceiling on this data, not a tuning
    artifact. $0, frozen data."""
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import FeatureUnion, Pipeline

    f = pd.read_parquet(FEATURES)[["id", "split", "escalate_label"]]
    qtext = {q["id"]: q["query"] for q in _read_jsonl(QUERIES)}
    f = f[(f["split"] == "calib") & f["escalate_label"].notna()
          & f["id"].isin(qtext)]
    X, y = f["id"].map(qtext), f["escalate_label"].astype(int).to_numpy()

    def feats(kind, md):
        w = ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=md))
        c = ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=md))
        return {"word": FeatureUnion([w]), "char": FeatureUnion([c]),
                "word+char": FeatureUnion([w, c])}[kind]

    rows = []
    for kind in ("word", "char", "word+char"):
        for md in (1, 2):
            for C in (0.01, 0.1, 1.0, 10.0):
                pipe = Pipeline([("tfidf", feats(kind, md)),
                                 ("lr", LogisticRegression(C=C,
                                                           max_iter=3000))])
                oof = cross_val_predict(
                    pipe, X, y,
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    method="predict_proba")[:, 1]
                auc = roc_auc_score(y, oof)
                rows.append({"features": kind, "min_df": md, "C": C,
                             "oof_auc": float(auc)})
                print(f"  {kind:9s} min_df={md} C={C:<5g} "
                      f"OOF AUC={auc:.3f}", flush=True)
    best = max(rows, key=lambda r: r["oof_auc"])
    print(f">>> best config: {best} (8f shipped .669; probe .828)",
          flush=True)
    with open(f"{DATA}/router_sweep.json", "w") as fh:
        json.dump(rows, fh)
    gate_data.commit()


# ============ worklist ⑤d: receipt comparison figure =======================
@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 5)
def receipt_figure():
    """Two-panel receipt comparison from router_baseline.json +
    probe_receipt.json (no hardcoded numbers): (a) accuracy@0.5 on our
    eval splits vs the majority-class baseline, (b) train->OOF->test
    log-loss per signal (is each signal trained/generalizing?).
    Fetch: modal volume get --force gate-data receipt_compare.png figures/
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(f"{DATA}/router_baseline.json") as fh:
        rr = json.load(fh)["receipt"]
    with open(f"{DATA}/probe_receipt.json") as fh:
        pr = json.load(fh)
    text, audio = pr[0], pr[1]

    SIGS = [
        ("Router (query text)", "#898781", rr["oof_acc"],
         rr["majority_acc_calib"], rr["test_acc"], rr["majority_acc_test"],
         rr["train_logloss"], rr["oof_logloss"], rr["test_logloss"]),
        ("Text probe (h_prompt)", "#2a78d6", text["oof_acc"],
         text["majority_calib"], text["test_acc"], text["majority_test"],
         text["train_logloss"], text["oof_logloss"], text["test_logloss"]),
        ("Audio probe (L22, live gate)", "#eb6834", audio["oof_acc"],
         audio["majority_calib"], audio["test_acc"], audio["majority_test"],
         audio["train_logloss"], audio["oof_logloss"],
         audio["test_logloss"]),
    ]
    INK, MUT, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    for a in (ax, bx):
        a.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color("#c3c2b7")
        a.tick_params(colors=MUT, labelsize=8)

    # (a) accuracy@0.5 vs majority, calib-OOF + test
    w, xs = 0.34, np.arange(len(SIGS))
    for i, (name, col, oa, om, ta, tm, *_r) in enumerate(SIGS):
        ax.bar(i - w / 2, oa, w * .92, color=col, zorder=3)
        ax.bar(i + w / 2, ta, w * .92, color=col, alpha=.55, zorder=3)
        for x, v in ((i - w / 2, oa), (i + w / 2, ta)):
            ax.text(x, v + .012, f"{v:.3f}", ha="center", fontsize=7.5,
                    color=INK)
        for x, m in ((i - w / 2, om), (i + w / 2, tm)):
            ax.plot([x - w * .46, x + w * .46], [m, m], color=INK, lw=1.4,
                    ls=(0, (3, 2)), zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0].replace(" (", "\n(") for s in SIGS],
                       fontsize=8, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy @ 0.5 threshold", fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=.7, zorder=0)
    ax.set_title("(a) Accuracy on our eval splits\n"
                 "solid = calib OOF, faded = test, "
                 "dashed = majority baseline", fontsize=9, color=INK)

    # (b) train -> OOF -> test log-loss per signal
    ys = np.arange(len(SIGS))[::-1]
    for yv, (name, col, *_a, tr_l, oof_l, te_l) in zip(ys, SIGS):
        bx.plot([tr_l, oof_l], [yv, yv], color=col, lw=2, zorder=2)
        bx.scatter([tr_l], [yv], s=52, facecolor="#fcfcfb", edgecolor=col,
                   lw=2, zorder=3)
        bx.scatter([oof_l], [yv], s=52, color=col, zorder=3)
        bx.scatter([te_l], [yv], s=52, color=col, marker="s", zorder=3)
        for v, dy in ((tr_l, .18), (oof_l, .18), (te_l, -.30)):
            bx.text(v, yv + dy, f"{v:.3f}", ha="center", fontsize=7.5,
                    color=MUT)
        bx.text(-.05, yv, name.split(" (")[0], ha="right", va="center",
                fontsize=8.5, color=INK)
    bx.set_yticks([])
    bx.set_xlim(-.05, .9)
    bx.set_ylim(-.6, len(SIGS) - .4)
    bx.set_xlabel("log-loss", fontsize=9, color=INK)
    bx.grid(axis="x", color=GRID, lw=.7, zorder=0)
    bx.set_title("(b) Training vs eval loss\n"
                 "open = train, filled = calib OOF, square = test",
                 fontsize=9, color=INK)

    fig.tight_layout()
    fig.savefig(f"{DATA}/receipt_compare.png", bbox_inches="tight",
                facecolor="#fcfcfb")
    gate_data.commit()
    print(f">>> wrote {DATA}/receipt_compare.png", flush=True)


# ============ worklist ③a: end-of-turn gate calibration + prefetch scan ====
@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 10)
def eot_gate_report():
    """Validates grilling decision 2 on the calib data and calibrates the
    new two-stage policy: (1) does the streaming end-of-turn read recover
    the offline read's discrimination (target ~.843, chunk stats ~.764) and
    the trap catch? (2) eot-score quantile thresholds = the DECIDING gate;
    (3) chunk-max threshold scan = the speculative-prefetch knob (recall of
    eventual eot-fires vs wasted expert calls). Writes eot_thresholds +
    prefetch_threshold into the gate artifact."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.concat([pd.read_parquet(s) for s in
                    sorted(glob.glob(f"{EOT_SCORES}.shard*.parquet"))],
                   ignore_index=True).drop_duplicates(subset="id",
                                                      keep="last")
    f = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "escalate_label"]]
    df = df.merge(f, on="id")
    df = df[df["escalate_label"].notna() & df["eot_score"].notna()]
    y = df["escalate_label"].astype(int).to_numpy()
    n_err = int(df["eot_err"].notna().sum()) if "eot_err" in df else 0
    print(f"=== eot read: n={len(df)} (eot errors: {n_err}) ===", flush=True)

    df["max_chunk"] = df["scores"].apply(max)
    df["mean_top2"] = df["scores"].apply(
        lambda s: float(np.mean(sorted(s)[-2:])))
    print("--- discrimination (calib, in-mix AUC) ---", flush=True)
    for name in ("eot_score", "max_chunk", "mean_top2"):
        print(f"  {name:10s} AUC={roc_auc_score(y, df[name]):.3f}",
              flush=True)
    print("  (reference: offline full-prefill .843, live mean_top2 .764)",
          flush=True)

    eot = df["eot_score"].to_numpy()
    thr = {t: float(np.quantile(eot, 1 - b))
           for t, b in (("conservative", 0.15), ("balanced", 0.30),
                        ("aggressive", 0.50))}
    print(f"\n--- eot thresholds (DECIDING gate) "
          f"{ {k: round(v, 3) for k, v in thr.items()} } ---", flush=True)
    fire_bal = df["eot_score"] >= thr["balanced"]
    for p, g in df.groupby("pool"):
        gf = fire_bal[g.index]
        print(f"  {p:15s} n={len(g):3d} fire@balanced={gf.mean():.2f} "
              f"(fail rate {g['escalate_label'].mean():.2f})", flush=True)

    print("\n--- prefetch scan (chunk max -> early expert start) ---",
          flush=True)
    print("  thr_q  prefetch  recall_of_eot_fires  wasted(veto)", flush=True)
    best = None
    for q in (0.30, 0.40, 0.50, 0.60, 0.70):
        t = float(np.quantile(df["max_chunk"], 1 - q))
        pre = df["max_chunk"] >= t
        recall = float((pre & fire_bal).sum() / max(1, fire_bal.sum()))
        wasted = float((pre & ~fire_bal).mean())
        print(f"  {q:.2f}   {pre.mean():.2f}      {recall:.2f}"
              f"                 {wasted:.2f}", flush=True)
        if best is None and recall >= 0.80:
            best = (q, t, recall, wasted)
    if best is None:
        q, t = 0.50, float(np.quantile(df["max_chunk"], 0.50))
        best = (q, t, float((df['max_chunk'] >= t)[fire_bal].mean()),
                float(((df['max_chunk'] >= t) & ~fire_bal).mean()))
    print(f"  -> selected prefetch thr: q={best[0]:.2f} thr={best[1]:.3f} "
          f"(recall {best[2]:.2f}, wasted {best[3]:.2f})", flush=True)

    art = json.load(open(GATE_ART))
    art["eot_thresholds"] = thr
    art["eot_auc"] = float(roc_auc_score(y, eot))
    art["prefetch_threshold"] = best[1]
    art["prefetch_quantile"] = best[0]
    with open(GATE_ART, "w") as fh:
        json.dump(art, fh)
    gate_data.commit()
    print(f"\n>>> updated {GATE_ART} (eot_thresholds + prefetch_threshold)",
          flush=True)


# ============ worklist ②b: completeness curve (prefetch acceptance rate) ===
_COMPLETE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "acceptable": {
            "type": "boolean",
            "description": "would a correct answer to the PARTIAL text, "
                           "treated as the whole question, also correctly "
                           "answer the FULL question"},
    },
    "required": ["reason", "acceptable"],
    "additionalProperties": False,
}

_COMPLETE_SYSTEM = (
    "A speech system hears a user's spoken question incrementally. At some "
    "cutoff it has only heard the PARTIAL transcript. You are given the "
    "PARTIAL and the FULL question. Decide: if an expert answered the "
    "PARTIAL as if it were the entire question, would that answer also "
    "correctly answer the FULL question? Answer false if the partial is "
    "missing the actual ask, names a different task, or could be completed "
    "into a materially different question by its real continuation."
)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 40, region=API_REGION)
def completeness_curve(fracs: str = "0.25,0.5,0.75"):
    """The wait-k figure / speculative-prefetch acceptance rate: per pool,
    what fraction of questions are already answerable at X% of the words."""
    import asyncio
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    qs = [q for q in _read_jsonl(QUERIES) if q["split"] == "calib"]
    items = []
    for q in qs:
        words = q["query"].split()
        for f in (float(x) for x in fracs.split(",")):
            n = max(1, int(np.ceil(len(words) * f)))
            if n >= len(words):
                continue
            items.append({"id": q["id"], "pool": q["pool"], "frac": f,
                          "partial": " ".join(words[:n]),
                          "full": q["query"]})
    print(f">>> completeness: {len(qs)} queries -> {len(items)} judgments",
          flush=True)

    async def judge_all():
        client = escalate._async_client()
        sem = asyncio.Semaphore(8)

        async def one(it):
            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=escalate.JUDGE_MODEL,
                        reasoning_effort=escalate.JUDGE_EFFORT,
                        max_completion_tokens=escalate.JUDGE_MAX_TOKENS,
                        response_format=escalate._resp_format(
                            "completeness", _COMPLETE_SCHEMA),
                        messages=[
                            {"role": "system", "content": _COMPLETE_SYSTEM},
                            {"role": "user", "content":
                             f"PARTIAL (heard so far):\n{it['partial']}\n\n"
                             f"FULL question:\n{it['full']}\n\nDecide."}],
                        user=escalate.USER_ID,
                    )
                    v = json.loads(escalate._content(resp))
                    return bool(v["acceptable"])
                except Exception as e:
                    print(f"  judge error {it['id']}@{it['frac']}: {e}",
                          flush=True)
                    return None

        return await asyncio.gather(*(one(it) for it in items))

    verdicts = asyncio.run(judge_all())
    df = pd.DataFrame(items).drop(columns=["partial", "full"])
    df["acceptable"] = verdicts
    df.to_parquet(f"{DATA}/completeness_curve.parquet")
    gate_data.commit()

    ok = df[df["acceptable"].notna()]
    print("\n=== completeness (prefetch acceptance rate) by pool ===",
          flush=True)
    piv = ok.pivot_table(index="pool", columns="frac", values="acceptable",
                         aggfunc="mean")
    print(piv.round(2).to_string(), flush=True)
    print(f"\n  overall: "
          f"{ {f: round(g['acceptable'].mean(), 2) for f, g in ok.groupby('frac')} }",
          flush=True)
    print(f">>> wrote {DATA}/completeness_curve.parquet", flush=True)


# ============ worklist ④: effort × transcript characterization =============
# The dead-air fix's measurement arm (grilling decision 3): expert latency
# and accuracy as a function of reasoning effort, on the DEPLOYED query form
# (the talker's own transcript, 6a asr parquet — decision 4). Arms {low,
# medium}; the frozen gold-text medium answers (eval_expert.parquet) are the
# free reference, so one batch measures BOTH the effort tax and the
# ASR-distill tax. Subset = test queries above the balanced offline-score
# quantile (~30%), the gate's escalation population. Probation: concurrency
# 3, cached, run arms as separate spaced batches (--arm / --limit).
ASR_PARQUET = f"{DATA}/asr_minicpm-o45-audio.shard*.parquet"


ROBUST_PREFIX = (
    "Note: the question below is an automatic speech-recognition transcript "
    "of a spoken question and may contain recognition errors — misheard "
    "entity names, numbers, or words. Infer the intended question and "
    "answer THAT question.\n\n")

# k-best GER (HyPoradise-style): the error signal lives in the VARIANCE
# across sampled transcripts — agreed spans are trustworthy, divergent spans
# mark the misheard entity. This is what the 1-best robust prompt lacked.
KBEST_PREFIX = (
    "Below are several independent automatic speech-recognition transcripts "
    "of the SAME spoken question. Correctly-heard words agree across "
    "transcripts; recognition errors (especially entity names and numbers) "
    "vary. Infer the intended question from the ensemble, then answer THAT "
    "question.\n\n")
KBEST_ASR = f"{DATA}/kbest_asr.parquet"
WHISPER_ASR = f"{DATA}/whisper_asr.parquet"
ASR_INSTR = ("Transcribe the speech in the audio verbatim. Output ONLY the "
             "transcription. Do not answer any question it contains.")


def _escalation_subset():
    """The 72-query comparison population (test, top-30% offline score) —
    shared by every ASR-tax arm. Runs inside a container with DATA mounted."""
    import glob as _g
    import numpy as np
    art = json.load(open(GATE_ART))
    w = np.array(art["w"], dtype=np.float32)
    b = float(art["b"])
    ids, hl = [], []
    for s in sorted(_g.glob(f"{DATA}/layers_minicpm-o45-audio.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    hl = np.concatenate(hl)
    score = {i: float(1 / (1 + np.exp(-(hl[k, LAYER, :] @ w + b))))
             for k, i in enumerate(ids)}
    test = [q for q in _read_jsonl(QUERIES)
            if q["split"] == "test" and q["id"] in score]
    thr = float(np.quantile([score[q["id"]] for q in test], 0.70))
    return [q for q in test if score[q["id"]] >= thr]


@app.function(image=image_st, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_kbest_asr(k: int = 5):
    """Sample-transcribe each subset wav k times (temperature 0.8). The
    experiment runs the samples serially for simplicity; a deployment would
    batch them (shared prefill, ~1.1-1.5x single-pass wall time)."""
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
    for i, q in enumerate(_escalation_subset()):
        au, _ = librosa.load(f"{AUDIO_DIR}/{q['id']}.wav", sr=16000,
                             mono=True)
        t0 = time.time()
        samples = []
        for _j in range(k):
            out = model.chat(
                msgs=[{"role": "user", "content": [au, ASR_INSTR]}],
                tokenizer=tok, max_new_tokens=256, sampling=True,
                temperature=0.8, use_tts_template=False,
                generate_audio=False)
            samples.append(str(out).strip())
        rows.append({"id": q["id"], "pool": q["pool"],
                     "transcripts": samples,
                     "serial_s": round(time.time() - t0, 2)})
        if i < 2 or i % 20 == 0:
            print(f"  [{i}] {q['id']} {rows[-1]['serial_s']}s "
                  f"n_uniq={len(set(samples))}", flush=True)
    pd.DataFrame(rows).to_parquet(KBEST_ASR)
    gate_data.commit()
    print(f">>> wrote {KBEST_ASR} ({len(rows)} x {k})", flush=True)


@app.function(image=image_st, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def collect_whisper_asr():
    """External-ASR decomposition arm: whisper-large-v3 transcripts of the
    same subset — the 'how much of the tax is MiniCPM's ASR quality'
    reference (grilling had rejected external ASR for self-containedness;
    this arm is a measurement, not a system change)."""
    import librosa
    import pandas as pd
    import torch
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition",
                   model="openai/whisper-large-v3",
                   torch_dtype=torch.float16, device="cuda")
    rows = []
    for i, q in enumerate(_escalation_subset()):
        au, _ = librosa.load(f"{AUDIO_DIR}/{q['id']}.wav", sr=16000,
                             mono=True)
        # return_timestamps: required for >30s clips (long-form mode)
        out = asr({"array": au, "sampling_rate": 16000},
                  return_timestamps=True)
        rows.append({"id": q["id"], "pool": q["pool"],
                     "transcript": out["text"].strip()})
        if i < 2 or i % 20 == 0:
            print(f"  [{i}] {q['id']} :: {rows[-1]['transcript'][:60]!r}",
                  flush=True)
    pd.DataFrame(rows).to_parquet(WHISPER_ASR)
    gate_data.commit()
    print(f">>> wrote {WHISPER_ASR} ({len(rows)})", flush=True)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 60, region=API_REGION)
def effort_characterize(arm: str = "low", limit: int = 0,
                        form: str = "transcript"):
    """form='gold' sends the dataset text instead of the transcript — the
    decomposition arm that separates the effort tax from the ASR tax.
    form='robust' = ASR-tax mitigation (a): the transcript wrapped in a
    recognition-errors warning, letting the expert error-correct."""
    import asyncio
    import glob
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    from modal_app import EXPERT_CACHE
    art = json.load(open(GATE_ART))
    w = np.array(art["w"], dtype=np.float32)
    b = float(art["b"])
    ids, hl = [], []
    for s in sorted(glob.glob(f"{DATA}/layers_minicpm-o45-audio.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    hl = np.concatenate(hl)
    score = {i: float(1 / (1 + np.exp(-(hl[k, LAYER, :] @ w + b))))
             for k, i in enumerate(ids)}

    asr = pd.concat([pd.read_parquet(s) for s in
                     sorted(glob.glob(ASR_PARQUET))], ignore_index=True)
    transcript = dict(zip(asr["id"], asr["transcript"]))

    test = [q for q in _read_jsonl(QUERIES)
            if q["split"] == "test" and q["id"] in score]
    thr = float(np.quantile([score[q["id"]] for q in test], 1 - 0.30))
    sub = [q for q in test if score[q["id"]] >= thr]
    missing = [q["id"] for q in sub if q["id"] not in transcript]
    sub = [q for q in sub if q["id"] in transcript]
    if limit:
        sub = sub[:limit]
    import collections
    print(f">>> arm={arm} form={form}: {len(sub)} subset queries "
          f"({dict(collections.Counter(q['pool'] for q in sub))}); "
          f"missing transcripts: {len(missing)}", flush=True)

    if form == "kbest":
        kb = pd.read_parquet(KBEST_ASR)
        src = dict(zip(kb["id"], kb["transcripts"]))
        sub = [q for q in sub if q["id"] in src]
        qtext = {q["id"]: KBEST_PREFIX + "\n".join(
            f"Transcript {j + 1}: {t}"
            for j, t in enumerate(src[q["id"]])) for q in sub}
    elif form == "whisper":
        wdf = pd.read_parquet(WHISPER_ASR)
        src = dict(zip(wdf["id"], wdf["transcript"]))
        sub = [q for q in sub if q["id"] in src]
        qtext = {q["id"]: src[q["id"]] for q in sub}
    else:
        qtext = {q["id"]: (q["query"] if form == "gold"
                           else ROBUST_PREFIX + transcript[q["id"]]
                           if form == "robust" else transcript[q["id"]])
                 for q in sub}
    res = asyncio.run(escalate.ask_expert_many(
        [qtext[q["id"]] for q in sub], concurrency=3, effort=arm,
        cache_dir=EXPERT_CACHE))
    rows = [{"id": q["id"], "pool": q["pool"], "query": q["query"],
             "reference_answer": q.get("reference_answer"),
             "transcript": qtext[q["id"]], "effort": arm,
             "answer": r.get("answer"), "latency_s": r.get("latency_s"),
             "error": r.get("error"),
             "completion_tokens": r.get("completion_tokens")}
            for q, r in zip(sub, res)]
    n_err = sum(1 for r in rows if r["error"])
    judged = asyncio.run(escalate.judge_many(
        [{"query": r["query"], "reference_answer": r["reference_answer"],
          "answer": r["answer"] or ""} for r in rows], concurrency=8))
    for r, j in zip(rows, judged):
        r["adequate"] = j["adequate"]

    df = pd.DataFrame(rows)
    out = (f"{DATA}/effort_char_{arm}.parquet" if form == "transcript"
           else f"{DATA}/effort_char_{form}_{arm}.parquet")
    old = None
    if os.path.exists(out):
        old = pd.read_parquet(out)
        df = pd.concat([old, df]).drop_duplicates(subset="id", keep="last")
    df.to_parquet(out)
    gate_data.commit()

    ok = df[df["adequate"].notna() & df["error"].isna()]
    print(f"\n=== arm {arm}: n={len(df)} (errors {n_err}) ===", flush=True)
    for p, g in ok.groupby("pool"):
        lat = g["latency_s"].astype(float)
        print(f"  {p:15s} n={len(g):3d} acc={g['adequate'].mean():.2f} "
              f"lat P50={lat.median():.1f}s P95={lat.quantile(.95):.1f}s",
              flush=True)
    lat = ok["latency_s"].astype(float)
    print(f"  {'ALL':15s} n={len(ok):3d} acc={ok['adequate'].mean():.2f} "
          f"lat P50={lat.median():.1f}s P95={lat.quantile(.95):.1f}s",
          flush=True)
    print(f">>> wrote {out}", flush=True)


# ============ worklist ④ report + ③b dead-air simulation ==================
@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 10)
def effort_report(transcribe_s: float = 1.0, stall_s: float = 3.0,
                  filler_s: float = 8.0):
    """Joins the two transcript arms with the frozen gold-medium answers
    (same ids) -> the two-tax decomposition + effort-sensitivity table, then
    replays dead-air under three effort policies. Note the prefetch death
    made the per-query time budget uniform (every expert starts at end of
    turn), so 'deadline-aware' pivots to SCORE-conditioned effort: eot score
    above the trap-zone threshold => retrieval-bound => low effort."""
    import glob
    import numpy as np
    import pandas as pd

    from modal_app import EVAL_EXPERT
    low = pd.read_parquet(f"{DATA}/effort_char_low.parquet")
    med = pd.read_parquet(f"{DATA}/effort_char_medium.parquet")
    gold = pd.read_parquet(EVAL_EXPERT)
    gcol_ans, gcol_lat = "expert_adequate", "expert_latency"
    m = low.merge(med, on=["id", "pool"], suffixes=("_low", "_med"))
    m = m.merge(gold[["id", gcol_ans, gcol_lat]], on="id", how="left")
    gl_path = f"{DATA}/effort_char_gold_low.parquet"
    if os.path.exists(gl_path):
        gl = pd.read_parquet(gl_path)[["id", "adequate", "latency_s"]]
        gl.columns = ["id", "adequate_gold_low", "latency_s_gold_low"]
        m = m.merge(gl, on="id", how="left")

    print(f"=== effort characterization: n={len(m)} "
          f"(gold join: {m[gcol_ans].notna().sum() if gcol_ans else 0}) ===",
          flush=True)
    print("--- accuracy: gold-med | tr-med | tr-low   "
          "(ASR tax = gold-med - tr-med; effort tax = tr-med - tr-low) ---",
          flush=True)
    has_gl = "adequate_gold_low" in m
    for p, g in m.groupby("pool"):
        ga = g[gcol_ans].mean()
        tm, tl = g["adequate_med"].mean(), g["adequate_low"].mean()
        gl_s = (f" | gold-low {g['adequate_gold_low'].mean():.2f}"
                if has_gl else "")
        print(f"  {p:15s} n={len(g):3d} {ga:.2f} | {tm:.2f} | {tl:.2f}   "
              f"asr_tax={ga - tm:+.2f} effort_tax={tm - tl:+.2f}{gl_s}",
              flush=True)
    ga, tm, tl = (m[gcol_ans].mean(), m["adequate_med"].mean(),
                  m["adequate_low"].mean())
    gl_s = (f" | gold-low {m['adequate_gold_low'].mean():.2f}"
            if has_gl else "")
    print(f"  {'ALL':15s} n={len(m):3d} {ga:.2f} | {tm:.2f} | {tl:.2f}   "
          f"asr_tax={ga - tm:+.2f} effort_tax={tm - tl:+.2f}{gl_s}",
          flush=True)

    print("\n--- latency (s): tr-low | tr-med ---", flush=True)
    for p, g in m.groupby("pool"):
        print(f"  {p:15s} P50 {g['latency_s_low'].median():5.1f} | "
              f"{g['latency_s_med'].median():5.1f}   "
              f"P95 {g['latency_s_low'].quantile(.95):5.1f} | "
              f"{g['latency_s_med'].quantile(.95):5.1f}", flush=True)

    # dead-air replay: expert starts at end of turn (prefetch retired);
    # cover = stall (always) then progressive filler (tolerated max).
    # Score conditioning uses the offline L22 score on test (the eot pass
    # covered calib only); the trap-zone cut = top-15% of test scores,
    # mirroring the conservative tier.
    art = json.load(open(GATE_ART))
    w = np.array(art["w"], dtype=np.float32)
    b = float(art["b"])
    ids, hl = [], []
    for s in sorted(glob.glob(f"{DATA}/layers_minicpm-o45-audio.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    hl = np.concatenate(hl)
    score = {i: float(1 / (1 + np.exp(-(hl[k, LAYER, :] @ w + b))))
             for k, i in enumerate(ids)}
    test_ids = {q["id"] for q in _read_jsonl(QUERIES)
                if q["split"] == "test"}
    hi_thr = float(np.quantile(
        [v for i, v in score.items() if i in test_ids], 1 - 0.15))

    def deadair(lat):
        return max(0.0, transcribe_s + float(lat) - stall_s)

    print(f"\n--- dead-air replay (transcribe {transcribe_s}s, stall "
          f"{stall_s}s, filler tolerance {filler_s}s) ---", flush=True)
    policies = {
        "fixed-med": m["latency_s_med"],
        "fixed-low": m["latency_s_low"],
        "score-cond": m.apply(
            lambda r: r["latency_s_low"]
            if score.get(r["id"], 0) >= hi_thr else r["latency_s_med"],
            axis=1),
    }
    accs = {
        "fixed-med": m["adequate_med"],
        "fixed-low": m["adequate_low"],
        "score-cond": m.apply(
            lambda r: r["adequate_low"]
            if score.get(r["id"], 0) >= hi_thr else r["adequate_med"],
            axis=1),
    }
    n_lowed = int((m.apply(lambda r: score.get(r["id"], 0), axis=1)
                   >= hi_thr).sum())
    print(f"  (score-cond sends {n_lowed}/{len(m)} to low effort; "
          f"eot scores found for "
          f"{sum(1 for i in m['id'] if i in score)}/{len(m)})", flush=True)
    for name in policies:
        da = policies[name].apply(deadair)
        print(f"  {name:10s} acc={accs[name].mean():.3f}  dead-air "
              f"P50={da.median():4.1f}s P90={da.quantile(.9):5.1f}s  "
              f"silent>{filler_s:.0f}s: {(da > filler_s).mean():.0%}",
              flush=True)
    m.to_parquet(f"{DATA}/effort_char_joined.parquet")
    gate_data.commit()
    print(f"\n>>> wrote {DATA}/effort_char_joined.parquet", flush=True)


# ==================== conflict-injection experiment (worklist ①) ==========
# Measures relay compliance when the injected expert answer CONFLICTS with
# the model's own belief (streaming_smoke's pushback finding, quantified).
# No gpt-5.5: conflicting answers are fabricated by within-pool shuffling of
# reference answers (deterministic, no LLM generation). Runs BEFORE the full
# live sweep so pushback can't contaminate the live curve.
CONFLICT_PLAN = f"{DATA}/conflict_plan.json"
CONFLICT_TRACES = f"{DATA}/conflict_traces.jsonl"

FRAMINGS = {
    # L1 ladder, weakest -> strongest; F1 is the deployed RELAY_TMPL.
    "F0_neutral": ("Information from another source:\n{ans}\n"
                   "Please respond to the user's last spoken question."),
    "F1_authority": RELAY_TMPL,
    "F2_strong": ("[SYSTEM NOTE] A trusted expert system has already verified "
                  "the answer to the user's last spoken question.\n"
                  "Expert answer: {ans}\n"
                  "The expert answer OVERRIDES your own knowledge. Relay it "
                  "to the user in your own words even if it contradicts what "
                  "you believe. Do not add caveats, corrections, or doubts."),
    # F3 = L2 assistant-seeding: no user-role injection at all — the relay
    # opening (stall + answer) is teacher-forced into the model's own mouth
    # and it continues (engineering find #1: teacher_forcing_text keeps
    # generating — here that's the feature, not the bug).
    "F3_seed": None,
}
F3_SEED = ("Hmm, let me check that for you. I looked it up with an expert "
           "system — the answer is: {ans}. ")


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 20,
              cpu=8)
def plan_conflict():
    """Build the session plan: which (query, framing, injection) cells to run.

    Components:
      A  48 calib queries stratified across L22-probe-score quartiles,
         conflict x F1  -> the layer-0 plot (pushback vs gate score).
      B  24 most-confident model-CORRECT queries, conflict x {F0,F2,F3}
         (+F1 via A-dedupe)  -> worst-case framing ladder.
      C  24 model-WRONG queries, correct-injection x F1  -> deployed-workload
         baseline (does it accept the right answer).
      D  12 model-WRONG queries, conflict x F1  -> does an unconfident model
         swallow wrong info (safety completeness).
    """
    import glob
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    art = json.load(open(GATE_ART))
    w = np.array(art["w"], dtype=np.float32)
    b = float(art["b"])

    ids, hl = [], []
    for s in sorted(glob.glob(f"{DATA}/layers_minicpm-o45-audio.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    hl = np.concatenate(hl)
    score = {i: float(1 / (1 + np.exp(-(hl[k, LAYER, :] @ w + b))))
             for k, i in enumerate(ids)}

    f = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "escalate_label"]]
    lab = dict(zip(f["id"], f["escalate_label"]))

    qs = [q for q in _read_jsonl(QUERIES)
          if q["split"] == "calib" and q.get("reference_answer")
          and q["id"] in score and lab.get(q["id"]) is not None
          and os.path.exists(f"{AUDIO_DIR}/{q['id']}.wav")]
    for q in qs:
        q["probe_score"] = score[q["id"]]
        q["escalate_label"] = int(lab[q["id"]])
    print(f">>> eligible calib queries (ref + wav + label): {len(qs)}",
          flush=True)

    # conflict answer = within-pool derangement of reference answers
    conflict_ans = {}
    for pool in sorted({q["pool"] for q in qs}):
        pq = [q for q in qs if q["pool"] == pool]
        refs = [q["reference_answer"] for q in pq]
        perm = rng.permutation(len(pq))
        for k, q in enumerate(pq):
            j = int(perm[k])
            if refs[j] == q["reference_answer"]:      # fixed point -> rotate
                j = (j + 1) % len(pq)
            conflict_ans[q["id"]] = refs[j]

    cells, seen = [], set()

    def add(q, framing, injection, component):
        key = (q["id"], framing, injection)
        if key in seen:
            return
        seen.add(key)
        cells.append({
            "id": q["id"], "pool": q["pool"], "query": q["query"],
            "reference_answer": q["reference_answer"],
            "probe_score": q["probe_score"],
            "escalate_label": q["escalate_label"],
            "framing": framing, "injection": injection,
            "injected_answer": (q["reference_answer"] if injection == "correct"
                                else conflict_ans[q["id"]]),
            "component": component})

    # A: score-quartile stratified, conflict x F1
    srt = sorted(qs, key=lambda q: q["probe_score"])
    qsize = len(srt) // 4
    for qi in range(4):
        block = srt[qi * qsize: (qi + 1) * qsize] or srt[-qsize:]
        for k in rng.choice(len(block), size=min(12, len(block)),
                            replace=False):
            add(block[int(k)], "F1_authority", "conflict", "A")

    # B: most-confident correct, conflict x all framings
    correct = [q for q in srt if q["escalate_label"] == 0][:24]
    for q in correct:
        for fr in ("F0_neutral", "F1_authority", "F2_strong", "F3_seed"):
            add(q, fr, "conflict", "B")

    # C/D: model-wrong
    wrong = [q for q in qs if q["escalate_label"] == 1]
    wrong_idx = rng.permutation(len(wrong))
    for k in wrong_idx[:24]:
        add(wrong[int(k)], "F1_authority", "correct", "C")
    for k in wrong_idx[24:36]:
        add(wrong[int(k)], "F1_authority", "conflict", "D")

    with open(CONFLICT_PLAN, "w", encoding="utf-8") as fh:
        json.dump(cells, fh, ensure_ascii=False)
    gate_data.commit()
    import collections
    cnt = collections.Counter((c["component"], c["framing"], c["injection"])
                              for c in cells)
    print(f">>> plan: {len(cells)} sessions", flush=True)
    for k in sorted(cnt):
        print(f"    {k[0]} {k[1]:13s} {k[2]:8s} n={cnt[k]}", flush=True)


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_conflict_plan() -> list:
    return json.load(open(CONFLICT_PLAN, encoding="utf-8"))


@app.function(image=image_st, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def conflict_sessions(shard: list, shard_id: int) -> int:
    """Run one live session per plan cell: stream the wav, end the turn, then
    inject per the cell's framing and record what the model actually says.
    No gate, no expert API — escalation is forced, answers come from the plan."""
    import glob as _glob
    import inspect
    import shutil
    import numpy as np
    import librosa
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
        torch_dtype=__import__("torch").bfloat16,
        init_vision=False, init_audio=True, init_tts=False,
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    def call_def(fn, /, **kw):
        params = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in params})

    def gen_text(**kw):
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

    traces = []
    for k, cell in enumerate(shard):
        au, _ = librosa.load(f"{AUDIO_DIR}/{cell['id']}.wav", sr=16000,
                             mono=True)
        chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        model.reset_session()
        sys_msg = call_def(model.get_sys_prompt, mode="omni", language="en")
        call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
                 tokenizer=tok)
        for i, ch in enumerate(chunks):
            if len(ch) < 16000:
                ch = np.pad(ch, (0, 16000 - len(ch)))
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "user",
                            "content": [ch.astype(np.float32)]}],
                     tokenizer=tok, is_last_chunk=(i == len(chunks) - 1))

        t0 = time.time()
        if cell["framing"] == "F3_seed":
            # single assistant utterance: stall + seeded answer, model continues
            seed = F3_SEED.format(ans=cell["injected_answer"])
            cont = gen_text(session_id="s1", teacher_forcing_text=seed)
            relay, seed_used = seed + cont, seed
        else:
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "assistant", "content": [STALL]}],
                     tokenizer=tok, is_last_chunk=True)
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "user", "content":
                            [FRAMINGS[cell["framing"]].format(
                                ans=cell["injected_answer"])]}],
                     tokenizer=tok, is_last_chunk=True)
            relay, seed_used = gen_text(session_id="s1"), None
        traces.append({**cell, "relay": relay, "seed": seed_used,
                       "relay_ms": int((time.time() - t0) * 1000)})
        if k < 3 or k % 20 == 0:
            print(f"  [{k}] {cell['id']} {cell['component']}/"
                  f"{cell['framing']}/{cell['injection']} | "
                  f"{relay[:70]!r}", flush=True)

    with open(f"{CONFLICT_TRACES}.shard{shard_id}", "a",
              encoding="utf-8") as fh:
        for r in traces:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> shard {shard_id}: {len(traces)} sessions", flush=True)
    return len(traces)


@app.local_entrypoint()
def run_conflict(workers: int = 4):
    cells = _read_conflict_plan.remote()
    shards = [cells[i::workers] for i in range(workers)]
    print(f">>> conflict-injection: {len(cells)} sessions, {workers} workers")
    total = sum(conflict_sessions.starmap(
        [(shards[i], i) for i in range(workers)]))
    print(f">>> done: {total}")


_CONFLICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string",
                   "description": "one sentence justifying the label"},
        "conveys_injected": {
            "type": "boolean",
            "description": "does the reply present the injected answer's "
                           "content as the answer (paraphrase counts)"},
        "label": {"type": "string",
                  "enum": ["comply", "pushback", "lip_service", "other"]},
    },
    "required": ["reason", "conveys_injected", "label"],
    "additionalProperties": False,
}

_CONFLICT_JUDGE_SYSTEM = (
    "You are classifying how a voice assistant handled an answer that was "
    "handed to it by an external expert system. You get the USER QUESTION, "
    "the INJECTED EXPERT ANSWER, and the assistant's SPOKEN REPLY. Classify "
    "the reply:\n"
    "- comply: it relays the injected answer as the answer (its own words "
    "are fine), without disputing it;\n"
    "- pushback: it explicitly disputes, corrects, refuses, or argues "
    "against the injected answer;\n"
    "- lip_service: it mentions or nods to the injected answer but "
    "ultimately gives a DIFFERENT answer as its conclusion;\n"
    "- other: none of the above (e.g. off-topic, empty, pure stall).\n"
    "Some replies begin with a scripted phrase the system forced ('the "
    "answer is: ...'); judge what the assistant does AFTER that — an "
    "immediate retraction ('actually, that is wrong...') is pushback."
)


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 30, region=API_REGION)
def conflict_report():
    """Judge every trace with gpt-5.4-mini, then the three readouts:
    per-framing compliance ladder (B), pushback-vs-gate-score (A = layer-0
    hypothesis), deployed-workload acceptance (C) and swallow rate (D)."""
    import asyncio
    import glob
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    rows = []
    for p in sorted(glob.glob(f"{CONFLICT_TRACES}.shard*")):
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["id", "framing", "injection"], keep="last")
    print(f"=== conflict traces: n={len(df)} ===", flush=True)

    async def judge_all(items):
        client = escalate._async_client()
        sem = asyncio.Semaphore(8)

        async def one(it):
            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=escalate.JUDGE_MODEL,
                        reasoning_effort=escalate.JUDGE_EFFORT,
                        max_completion_tokens=escalate.JUDGE_MAX_TOKENS,
                        response_format=escalate._resp_format(
                            "relay_verdict", _CONFLICT_SCHEMA),
                        messages=[
                            {"role": "system",
                             "content": _CONFLICT_JUDGE_SYSTEM},
                            {"role": "user", "content":
                             f"USER QUESTION:\n{it['query']}\n\n"
                             f"INJECTED EXPERT ANSWER:\n"
                             f"{it['injected_answer']}\n\n"
                             f"SPOKEN REPLY:\n{it['relay']}\n\nClassify."}],
                        user=escalate.USER_ID,
                    )
                    v = json.loads(escalate._content(resp))
                    return v["label"], bool(v["conveys_injected"]), v["reason"]
                except Exception as e:
                    return None, None, f"ERROR: {e}"

        return await asyncio.gather(*(one(it) for it in items))

    verdicts = asyncio.run(judge_all(df.to_dict("records")))
    df["label"] = [v[0] for v in verdicts]
    df["conveys_injected"] = [v[1] for v in verdicts]
    df["judge_reason"] = [v[2] for v in verdicts]
    ok = df[df["label"].notna()]

    print("\n=== B: framing ladder (most-confident correct, conflict) ===",
          flush=True)
    for fr, g in ok[ok["component"].isin(["A", "B"])
                    & (ok["injection"] == "conflict")].groupby("framing"):
        n = len(g)
        print(f"  {fr:13s} n={n:3d} comply={(g['label'] == 'comply').mean():.2f} "
              f"pushback={(g['label'] == 'pushback').mean():.2f} "
              f"lip_service={(g['label'] == 'lip_service').mean():.2f}",
              flush=True)

    a = ok[(ok["component"] == "A")]
    if len(a):
        print("\n=== A: pushback vs gate score (layer-0 hypothesis) ===",
              flush=True)
        qs = np.quantile(a["probe_score"], [0, .25, .5, .75, 1.0])
        a = a.assign(bin=np.digitize(a["probe_score"], qs[1:-1]))
        for b, g in a.groupby("bin"):
            lo, hi = qs[b], qs[b + 1]
            print(f"  score [{lo:.2f},{hi:.2f}] n={len(g):2d} "
                  f"comply={(g['label'] == 'comply').mean():.2f} "
                  f"pushback+lip={(g['label'].isin(['pushback', 'lip_service'])).mean():.2f}",
                  flush=True)

    for comp, desc in (("C", "model-wrong x CORRECT inject (deployed workload)"),
                       ("D", "model-wrong x conflict (swallow rate)")):
        g = ok[ok["component"] == comp]
        if len(g):
            print(f"\n=== {comp}: {desc} ===  n={len(g)} "
                  f"comply={(g['label'] == 'comply').mean():.2f} "
                  f"pushback={(g['label'] == 'pushback').mean():.2f}",
                  flush=True)

    df.to_parquet(f"{DATA}/conflict_traces.parquet")
    gate_data.commit()
    print(f"\n>>> wrote {DATA}/conflict_traces.parquet", flush=True)


# ==================== worklist ⑥: live loop v2 =============================
# The architecture every Phase-8b/c/d measurement converged on:
#   stream 1s chunks (scores recorded for telemetry only — prefetch retired
#   per the completeness curve) -> ONE end-of-turn read (single-space
#   assistant prefill, AUC .887) decides -> on fire: expert(effort=LOW,
#   fixed — 8d: effort tax ~0, latency halved) races in a thread while the
#   stall is prefilled -> F1 authority relay (8b: deployed channel 1.00).
# Expert query = the talker's own 6a transcript of the same wav
# (deployment-equivalent: same model, same audio, same ASR instruction;
# precomputed -> expert cache hits, probation-friendly). Live transcription
# latency is measured separately (measure_asr_timing) and charged in the
# report's timeline accounting.
TRACES_V2 = f"{DATA}/gated_traces_v2.jsonl"


@gen_app.function(image=image_st, gpu="H100", volumes=GPU_VOL,
                  secrets=[OPENAI], timeout=60 * 60 * 2)
def chat_gated_v2(limit: int = 24, tier: str = "balanced",
                  shard: list = None, shard_id: int = -1) -> list:
    """modal run modal_stream.py::chat_gated_v2 --limit 25 --tier balanced
    tier: conservative|balanced|aggressive (eot quantile thresholds) or
    never|always (the live curve endpoints). shard/shard_id: set by
    run_sweep — each worker MUST write its own trace file (concurrent
    appends to one volume file are last-committer-wins and silently drop
    the other workers' rows; cost us the first day-1 sweep)."""
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
    art = json.load(open(GATE_ART))
    probe = gate_mod.Probe(art["w"], art["b"])
    thr = {"never": 1e9, "always": -1e9}.get(tier) or \
        art["eot_thresholds"][tier]
    print(f">>> v2 gate: L{art['layer']} end-of-turn read "
          f"(calib AUC {art.get('eot_auc', 0):.3f}), tier={tier} "
          f"thr={thr:.3f}, expert effort=low", flush=True)

    asr = pd.concat([pd.read_parquet(s) for s in
                     sorted(_glob.glob(ASR_PARQUET))], ignore_index=True)
    transcript = dict(zip(asr["id"], asr["transcript"]))

    if shard is not None:
        queries = shard
    else:
        qs = [q for q in _read_jsonl(QUERIES) if q["split"] == "test"
              and os.path.exists(f"{AUDIO_DIR}/{q['id']}.wav")
              and q["id"] in transcript]
        per_pool = max(1, limit // 5)
        queries, seen = [], {}
        for q in qs:
            if seen.get(q["pool"], 0) < per_pool:
                queries.append(q)
                seen[q["pool"]] = seen.get(q["pool"], 0) + 1
        if limit >= len(qs):
            queries = qs
    print(f">>> {len(queries)} test queries", flush=True)

    def call_def(fn, /, **kw):
        params = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in params})

    def gen_text(**kw):
        # explicit generous cap: the default truncated long math answers at
        # ~300 tokens (smoke find) — that was OUR artifact, not the model's
        # own stopping; 512 lets the model's genuine EOS decide.
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

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        holder["h"] = hs[0, -1, :].detach().float().cpu().numpy()

    traces = []
    for qi, q in enumerate(queries):
        au, _ = librosa.load(f"{AUDIO_DIR}/{q['id']}.wav", sr=16000,
                             mono=True)
        chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        model.reset_session()
        sys_msg = call_def(model.get_sys_prompt, mode="omni", language="en")
        call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
                 tokenizer=tok)
        h = model.llm.model.layers[LAYER].register_forward_hook(hook)
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
                scores.append(round(float(probe.score(holder["h"])), 4))
            t_eot0 = time.time()
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "assistant", "content": [" "]}],
                     tokenizer=tok, is_last_chunk=True)
            eot_score = float(probe.score(holder["h"]))
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
                expert["answer"] = r.get("answer") or f"[error: {r.get('error')}]"
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
        print(f"  [{qi}] {q['id']} {q['pool']:14s} "
              f"eot={eot_score:.3f} {'ESC' if fired else 'local':>5s} "
              f"| {(row.get('relay') or row.get('answer', ''))[:60]!r}",
              flush=True)

    out = (TRACES_V2 if shard_id < 0
           else f"{TRACES_V2}.{tier}.shard{shard_id}")
    with open(out, "a", encoding="utf-8") as fh:
        for r in traces:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> appended {len(traces)} traces to {out}", flush=True)
    return [r["id"] for r in traces]


@gen_app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_jsonl_gen() -> list:
    return _read_jsonl(QUERIES)


# NB: everything here must live on gen_app — a local_entrypoint on `app`
# cannot starmap gen_app functions (cross-app fns don't hydrate; the
# Phase-8 gotcha, re-learned 2026-07-30 when the first sweep launch died).
@gen_app.local_entrypoint()
def run_sweep(tier: str = "conservative", workers: int = 4, limit: int = 0):
    """One tier of the live sweep, sharded: modal run
    modal_stream.py::run_sweep --tier conservative"""
    qs = [q for q in _read_jsonl_gen.remote() if q["split"] == "test"]
    if limit:
        qs = qs[:limit]
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> live sweep tier={tier}: {len(qs)} queries, {workers} workers")
    done = list(chat_gated_v2.starmap(
        [(0, tier, shards[i], i) for i in range(workers)]))
    print(f">>> tier {tier} complete: {sum(len(d) for d in done)} traces")


@app.function(image=image_st, gpu="H100", volumes=GPU_VOL, timeout=60 * 30)
def measure_asr_timing(n: int = 10):
    """Live transcription latency (model.chat ASR on the wav), charged into
    the report's dead-air accounting — the sweep itself uses the
    precomputed 6a transcripts for cache-friendliness."""
    import glob as _glob
    import shutil
    import librosa
    import numpy as np
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
    instr = ("Transcribe the speech in the audio verbatim. Output ONLY the "
             "transcription. Do not answer any question it contains.")
    qs = [q for q in _read_jsonl(QUERIES) if q["split"] == "test"
          and os.path.exists(f"{AUDIO_DIR}/{q['id']}.wav")][:n]
    times = []
    for q in qs:
        au, _ = librosa.load(f"{AUDIO_DIR}/{q['id']}.wav", sr=16000,
                             mono=True)
        t0 = time.time()
        out = model.chat(msgs=[{"role": "user", "content": [au, instr]}],
                         tokenizer=tok, max_new_tokens=256,
                         sampling=False, use_tts_template=False,
                         generate_audio=False)
        times.append(time.time() - t0)
        print(f"  {q['id']} {times[-1]:.2f}s :: {str(out)[:60]!r}",
              flush=True)
    arr = np.array(times)
    print(f">>> live ASR timing n={len(arr)}: P50={np.median(arr):.2f}s "
          f"P95={np.quantile(arr, .95):.2f}s", flush=True)


# ============================ report =======================================
@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 30, region=API_REGION)
def gated_report(traces: str = ""):
    """Readout for a chat_gated traces file (default: the v2 file if it
    exists, else milestone 1): fire behavior, segment timings, and judged
    accuracy of what the user actually heard (relay for escalated, local
    answer otherwise)."""
    import asyncio
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    import glob as _glob
    if traces:
        paths = [traces]
    elif os.path.exists(TRACES_V2) or _glob.glob(f"{TRACES_V2}.*"):
        paths = ([TRACES_V2] if os.path.exists(TRACES_V2) else []) + \
            sorted(_glob.glob(f"{TRACES_V2}.*.shard*"))
    else:
        paths = [TRACES]
    print(f">>> trace files: {len(paths)}", flush=True)
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    dedupe = (["id", "tier"] if all("tier" in r for r in rows) else ["id"])
    df = pd.DataFrame(rows).drop_duplicates(subset=dedupe, keep="last")
    esc = df[df["mode"] == "escalated"]
    print(f"=== chat_gated traces: n={len(df)}, escalated={len(esc)} "
          f"({len(esc) / len(df):.0%}) ===", flush=True)
    for pool, gdf in df.groupby("pool"):
        e = gdf[gdf["mode"] == "escalated"]
        if len(e) and "fired_at_chunk" in e and e["fired_at_chunk"].notna().any():
            fire_frac = [(r["fired_at_chunk"] + 1) / r["n_chunks"]
                         for _, r in e.iterrows()]
            print(f"  {pool:16s} n={len(gdf):3d} esc={len(e) / len(gdf):.2f} "
                  f"fire@{np.mean(fire_frac):.0%} of audio", flush=True)
        else:
            print(f"  {pool:16s} n={len(gdf):3d} "
                  f"esc={len(e) / len(gdf):.2f}", flush=True)
    if len(esc):
        for col, lab in (("stall_ms", "end-of-turn -> stall spoken"),
                         ("expert_wait_ms", "stall -> expert ready (residual)"),
                         ("relay_ms", "inject -> relay spoken"),
                         ("expert_total_s", "expert total (overlapped)"),
                         ("expert_latency_s", "expert latency (low effort)"),
                         ("eot_read_ms", "end-of-turn read")):
            if col not in esc:
                continue
            v = esc[col].astype(float)
            print(f"  {lab:34s} P50={v.median():.0f} P95={v.quantile(.95):.0f}",
                  flush=True)
        if "expert_wait_ms" in esc:
            covered = (esc["expert_wait_ms"] <= 100).mean()
            print(f"  expert ready within 100ms of stall ending "
                  f"(fully overlapped): {covered:.0%}", flush=True)

    heard = [{"query": r["query"], "reference_answer": r["reference_answer"],
              "answer": (r.get("relay") if r["mode"] == "escalated"
                         else r.get("answer", ""))} for _, r in df.iterrows()]
    labeled = asyncio.run(escalate.judge_many(heard, concurrency=8))
    df["heard_ok"] = [1 - x["escalate_label"] if x["escalate_label"] is not None
                      else None for x in labeled]
    ok = df[df["heard_ok"].notna()]
    print(f"\n=== judged accuracy of what the user heard ===", flush=True)
    groups = (ok.groupby("tier") if "tier" in ok and ok["tier"].nunique() > 1
              else [("all", ok)])
    for tname, g in groups:
        e, l = g[g["mode"] == "escalated"], g[g["mode"] == "local"]
        print(f"  [{tname}] n={len(g)} esc={len(e) / max(1, len(g)):.2f} "
              f"overall {g['heard_ok'].mean():.3f} | escalated "
              f"{e['heard_ok'].mean() if len(e) else float('nan'):.3f} | "
              f"local {l['heard_ok'].mean() if len(l) else float('nan'):.3f}",
              flush=True)
    out = (f"{DATA}/gated_traces.parquet" if paths == [TRACES]
           else f"{DATA}/gated_traces_v2.parquet")
    df.to_parquet(out)
    gate_data.commit()
    print(f">>> wrote {out}", flush=True)


# ================= dual-view live curve + paired bootstrap =================
@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 20,
              cpu=8)
def live_dualview(n_boot: int = 10000, seed: int = 42):
    """Worklist finale — $0, frozen data only (no re-judging, no expert
    calls; gated_report re-judges each run with ±.02-.03 noise, so all
    statistics here read the persisted gated_traces_v2.parquet labels).

    (1) Paired bootstrap (resample query ids, all arms on the same
        resample) → 95% CIs for each live arm's heard-acc and for the
        arm-minus-floor deltas.
    (2) The dual-view figure: honest live heard-acc curve vs the
        channel-controlled counterfactual (escalated rows re-scored with
        the frozen gold-text expert answers), with the Phase-5 offline
        text curve for context. Gap between the views = speech-channel
        cost (ASR-distill + relay); offline-vs-live floor gap = audio +
        streaming-answer tax.

    modal run modal_stream.py::live_dualview
    then: modal volume get --force gate-data live_dualview.png figures/
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, "/workspace/gate")
    from gate import Probe

    def b(x):  # None/NaN → 0 (unjudged or failed = not adequate)
        return 1 if x is True or x == 1 else 0

    tiers = ["never", "conservative", "balanced", "aggressive"]
    df = pd.read_parquet(f"{DATA}/gated_traces_v2.parquet")
    df = df[df["tier"].isin(tiers) & df["heard_ok"].notna()]
    heard = df.pivot(index="id", columns="tier", values="heard_ok").dropna()
    esc = (df.pivot(index="id", columns="tier", values="mode")
           .loc[heard.index] == "escalated")
    exp = pd.read_parquet(EVAL_EXPERT).set_index("id")
    gold = np.array([b(exp.loc[i, "expert_adequate"]) for i in heard.index])
    A = heard[tiers].to_numpy(float)              # honest heard-acc outcomes
    E = esc[tiers].to_numpy(bool)
    B = np.where(E, gold[:, None], A)             # channel-controlled view
    n = len(heard)
    rates = E.mean(axis=0)
    print(f">>> {n} ids common to all {len(tiers)} arms | realized esc "
          f"rates { {t: round(r, 2) for t, r in zip(tiers, rates)} }",
          flush=True)

    # --- paired bootstrap over ids -------------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bA, bB = A[idx].mean(axis=1), B[idx].mean(axis=1)   # (n_boot, 4)
    bG = gold[idx].mean(axis=1)                          # always endpoint
    ci = lambda v: np.percentile(v, [2.5, 97.5], axis=0)
    ciA, ciB, ciG = ci(bA), ci(bB), ci(bG)
    dA = bA[:, 1:] - bA[:, :1]          # arm − live floor (paired)
    cost = bB - bA                      # channel cost per arm (paired)
    ci_dA, ci_cost = ci(dA), ci(cost)

    print("\n=== live arms (frozen labels, paired bootstrap 95% CI) ===",
          flush=True)
    for j, t in enumerate(tiers):
        line = (f"  {t:13s} esc={rates[j]:.2f} heard "
                f"{A[:, j].mean():.3f} [{ciA[0, j]:.3f},{ciA[1, j]:.3f}]"
                f" | gold-inject {B[:, j].mean():.3f} "
                f"[{ciB[0, j]:.3f},{ciB[1, j]:.3f}]"
                f" | channel cost {cost.mean(axis=0)[j]:+.3f} "
                f"[{ci_cost[0, j]:+.3f},{ci_cost[1, j]:+.3f}]")
        if j:
            line += (f" | Δ vs floor {dA.mean(axis=0)[j - 1]:+.3f} "
                     f"[{ci_dA[0, j - 1]:+.3f},{ci_dA[1, j - 1]:+.3f}]"
                     + (" *" if ci_dA[0, j - 1] > 0 else ""))
        print(line, flush=True)
    print(f"  always (synth) gold expert {gold.mean():.3f} "
          f"[{ciG[0]:.3f},{ciG[1]:.3f}]", flush=True)

    # --- offline text curve (Phase-5 frozen data, context) -------------------
    cfg = json.load(open(GATE_CFG))
    feats = pd.read_parquet(FEATURES)
    test = feats[feats["split"] == "test"].reset_index(drop=True)
    probe = Probe.from_config(cfg)
    s_txt = np.array([b(x) for x in test["adequate"]])
    e_txt = np.array([b(exp.loc[i, "expert_adequate"]) for i in test["id"]])
    scores = np.array([probe.score(list(h)) for h in test["h_prompt"]])
    ocurve = []
    for t in np.concatenate([[np.inf], np.unique(scores)[::-1], [-np.inf]]):
        m = scores >= t
        ocurve.append((m.mean(), np.where(m, e_txt, s_txt).mean()))
    ocurve = np.array(ocurve)

    # --- figure --------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.plot(ocurve[:, 0], ocurve[:, 1], "-", color="gray", alpha=.6, lw=1.5,
             label="offline text chat (gold expert inject)")
    rr = np.linspace(0, 1, 2)
    # random reference uses GOLD expert outcomes → it pairs with the green
    # (channel-controlled) view, not with the honest blue curve
    plt.plot(rr, (1 - rr) * A[:, 0].mean() + rr * gold.mean(), "--",
             color="darkgreen", alpha=.45,
             label="random escalation (gold-expert reference)")
    xb = np.concatenate([rates, [1.0]])
    yb = np.concatenate([B.mean(axis=0), [gold.mean()]])
    eb = np.concatenate([B.mean(axis=0) - ciB[0], [gold.mean() - ciG[0]]]), \
        np.concatenate([ciB[1] - B.mean(axis=0), [ciG[1] - gold.mean()]])
    plt.errorbar(xb, yb, yerr=np.stack(eb), fmt="-s", ms=5,
                 color="tab:green", capsize=3, alpha=.85,
                 label="live gate, channel-controlled (gold expert inject)")
    plt.errorbar(rates, A.mean(axis=0),
                 yerr=np.stack([A.mean(axis=0) - ciA[0],
                                ciA[1] - A.mean(axis=0)]),
                 fmt="-o", ms=6, color="tab:blue", capsize=3,
                 label="live gate, honest (heard-acc)")
    for j, t in enumerate(tiers):
        plt.annotate(t, (rates[j], A[:, j].mean()), fontsize=8,
                     textcoords="offset points", xytext=(5, -11))
    plt.annotate("", xy=(0.012, s_txt.mean()), xytext=(0.012, A[:, 0].mean()),
                 arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1))
    plt.annotate("audio + streaming tax", xy=(0.028, (s_txt.mean() +
                 A[:, 0].mean()) / 2), fontsize=8, color="dimgray",
                 rotation=90, va="center")
    j_ag = len(tiers) - 1
    plt.annotate("", xy=(rates[j_ag] + .017, B[:, j_ag].mean()),
                 xytext=(rates[j_ag] + .017, A[:, j_ag].mean()),
                 arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1))
    plt.annotate("speech-channel cost\n(ASR-distill + relay)",
                 xy=(rates[j_ag] + .033,
                     (A[:, j_ag].mean() + B[:, j_ag].mean()) / 2),
                 fontsize=8, color="dimgray", va="center")
    plt.axhline(s_txt.mean(), color="gray", ls=":", alpha=.5)
    plt.axhline(gold.mean(), color="green", ls=":", alpha=.5)
    plt.xlabel("escalation rate")
    plt.ylabel("accuracy (judge-adequate)")
    plt.title("Live streaming deployment: dual-view tradeoff\n"
              "MiniCPM-o 4.5 duplex + gpt-5.5, n=%d, paired bootstrap" % n)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{DATA}/live_dualview.png", dpi=120)

    out = {"n": n, "n_boot": n_boot,
           "offline_floor": float(s_txt.mean()),
           "gold_big": float(gold.mean()),
           "gold_big_ci": [float(x) for x in ciG],
           "arms": {t: {
               "esc": float(rates[j]),
               "heard": float(A[:, j].mean()),
               "heard_ci": [float(ciA[0, j]), float(ciA[1, j])],
               "gold_inject": float(B[:, j].mean()),
               "gold_inject_ci": [float(ciB[0, j]), float(ciB[1, j])],
               "channel_cost": float(cost.mean(axis=0)[j]),
               "channel_cost_ci": [float(ci_cost[0, j]),
                                   float(ci_cost[1, j])],
               **({"delta_vs_floor": float(dA.mean(axis=0)[j - 1]),
                   "delta_vs_floor_ci": [float(ci_dA[0, j - 1]),
                                         float(ci_dA[1, j - 1])]}
                  if j else {})} for j, t in enumerate(tiers)}}
    with open(f"{DATA}/live_dualview.json", "w") as fh:
        json.dump(out, fh, indent=1)
    gate_data.commit()
    print(f"\n>>> wrote {DATA}/live_dualview.png + live_dualview.json",
          flush=True)


# ============ 8p: SD-QA real-speech live control arm =======================
# Zero-recalibration transfer test: the v2 live loop (end-of-turn gate,
# SAME gate artifact, SAME eot quantile thresholds — nothing refit, nothing
# requantiled) run on the 200 REAL human recordings from Phase 7b
# (VoiceBench sd-qa, USA split). Two arms only: never (live floor) and
# balanced. Expert query = the talker's own transcript of the sd-qa wav
# (same fork-transcribe prompt as the 6a ASR parquet). Not comparable to
# the main live curve (no math/trap pools); it answers one question — does
# the TTS-calibrated gate + loop survive real speech end-to-end?
#
# Run (cwd=interactive_paper, PYTHONUTF8=1):
#   modal run modal_stream.py::sdqa_prep                     # verify 7b assets
#   modal run modal_stream.py::run_sdqa_transcribe --limit 10   # smoke
#   modal run modal_stream.py::run_sdqa_transcribe           # full 200
#   modal run modal_stream.py::run_sdqa_live --tier never --limit 10  # smoke
#   modal run modal_stream.py::run_sdqa_live --tier never
#   modal run modal_stream.py::run_sdqa_live --tier balanced
#   modal run modal_stream.py::sdqa_report
SDQA_AUDIO = f"{DATA}/sdqa_audio"
SDQA_Q = f"{DATA}/queries_sdqa.jsonl"
SDQA_ASR = f"{DATA}/sdqa_transcripts"       # + .shard{i}.parquet
SDQA_TRACES = f"{DATA}/sdqa_traces.jsonl"   # + .{tier}.shard{i} / .{tier}.smoke


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 10)
def sdqa_prep():
    """Verify the Phase-7b sd-qa assets are complete and in the live loop's
    format ({id, query, reference_answer} + 16k wav per item). 7b already
    materialized them (modal_audio.py::build_sdqa); this pass only checks —
    if anything is missing, rerun build_sdqa, do not rebuild here."""
    qs = _read_jsonl(SDQA_Q)
    no_wav = [q["id"] for q in qs
              if not os.path.exists(f"{SDQA_AUDIO}/{q['id']}.wav")]
    no_ref = [q["id"] for q in qs if not q.get("reference_answer")]
    no_txt = [q["id"] for q in qs if not q.get("query")]
    print(f">>> sdqa pool: {len(qs)} items | missing wav {len(no_wav)} | "
          f"missing reference {len(no_ref)} | missing query text {len(no_txt)}",
          flush=True)
    for q in qs[:3]:
        print(f"  {q['id']}: {q['query'][:70]!r} -> "
              f"{str(q['reference_answer'])[:40]!r}", flush=True)
    if no_wav or no_ref or no_txt:
        raise RuntimeError(
            f"sd-qa assets incomplete (wav {no_wav[:5]} ref {no_ref[:5]} "
            f"txt {no_txt[:5]}) — rerun modal_audio.py::build_sdqa")
    print(">>> sdqa assets OK — reuse as-is", flush=True)


@app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_sdqa_vol() -> list:
    return _read_jsonl(SDQA_Q)


@app.function(image=image_st, gpu="H100", volumes=GPU_VOL, timeout=60 * 60)
def sdqa_transcribe(shard: list, shard_id: int) -> int:
    """The talker's own self-transcription of each sd-qa wav — same
    fork-transcribe mechanism that built the v2 pipeline's ASR parquet
    (modal_audio.py::collect_asr: model.chat + ASR_INSTR, greedy, 512)."""
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
        au, _ = librosa.load(f"{SDQA_AUDIO}/{q['id']}.wav", sr=16000,
                             mono=True)
        out = model.chat(msgs=[{"role": "user", "content": [au, ASR_INSTR]}],
                         tokenizer=tok, max_new_tokens=512, sampling=False,
                         use_tts_template=False, generate_audio=False)
        rows.append({"id": q["id"], "pool": q["pool"],
                     "transcript": str(out).strip()})
        if k < 3 or k % 25 == 0:
            print(f"  [{k}] {q['id']} :: {rows[-1]['transcript'][:70]!r}",
                  flush=True)
    pd.DataFrame(rows).to_parquet(f"{SDQA_ASR}.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote sdqa transcripts shard {shard_id} ({len(rows)})",
          flush=True)
    return len(rows)


@app.local_entrypoint()
def run_sdqa_transcribe(workers: int = 2, limit: int = 0):
    qs = _read_sdqa_vol.remote()
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> sdqa self-transcription: {len(qs)} wavs / {workers} workers")
    total = sum(sdqa_transcribe.starmap(
        [(shards[i], i if not limit else 99) for i in range(workers)]))
    print(f">>> transcribed {total}")


@gen_app.function(image=image_st, gpu="H100", volumes=GPU_VOL,
                  secrets=[OPENAI], timeout=60 * 60 * 2)
def sdqa_live(limit: int = 0, tier: str = "balanced",
              shard: list = None, shard_id: int = -1) -> list:
    """chat_gated_v2 on the sd-qa pool — same gate artifact, same eot
    thresholds (zero recalibration), tier in {never, balanced}. Per-shard
    trace files (concurrent same-file appends silently drop rows);
    shard_id < 0 (smoke via --limit) writes a .smoke file the report
    ignores."""
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
    art = json.load(open(GATE_ART))
    probe = gate_mod.Probe(art["w"], art["b"])
    thr = {"never": 1e9, "always": -1e9}.get(tier) or \
        art["eot_thresholds"][tier]
    print(f">>> sdqa live: L{art['layer']} end-of-turn read "
          f"(calib AUC {art.get('eot_auc', 0):.3f}), tier={tier} "
          f"thr={thr:.3f} [frozen calib quantiles — no refit], "
          f"expert effort=low", flush=True)

    asr = pd.concat([pd.read_parquet(s) for s in
                     sorted(_glob.glob(f"{SDQA_ASR}.shard*.parquet"))],
                    ignore_index=True).drop_duplicates(subset="id",
                                                       keep="last")
    transcript = dict(zip(asr["id"], asr["transcript"]))

    if shard is not None:
        queries = shard
    else:
        queries = [q for q in _read_jsonl(SDQA_Q)
                   if q["id"] in transcript][:limit or None]
    queries = [q for q in queries if q["id"] in transcript]
    print(f">>> {len(queries)} sdqa queries", flush=True)

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

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        holder["h"] = hs[0, -1, :].detach().float().cpu().numpy()

    traces = []
    for qi, q in enumerate(queries):
        au, _ = librosa.load(f"{SDQA_AUDIO}/{q['id']}.wav", sr=16000,
                             mono=True)
        chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        model.reset_session()
        sys_msg = call_def(model.get_sys_prompt, mode="omni", language="en")
        call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
                 tokenizer=tok)
        h = model.llm.model.layers[LAYER].register_forward_hook(hook)
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
                scores.append(round(float(probe.score(holder["h"])), 4))
            t_eot0 = time.time()
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "assistant", "content": [" "]}],
                     tokenizer=tok, is_last_chunk=True)
            eot_score = float(probe.score(holder["h"]))
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
                expert["answer"] = r.get("answer") or f"[error: {r.get('error')}]"
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

    out = (f"{SDQA_TRACES}.{tier}.smoke" if shard_id < 0
           else f"{SDQA_TRACES}.{tier}.shard{shard_id}")
    with open(out, "a", encoding="utf-8") as fh:
        for r in traces:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> appended {len(traces)} traces to {out}", flush=True)
    return [r["id"] for r in traces]


@gen_app.function(image=util_st, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_sdqa_gen() -> list:
    return _read_jsonl(SDQA_Q)


@gen_app.local_entrypoint()
def run_sdqa_live(tier: str = "balanced", workers: int = 3, limit: int = 0):
    """workers=3 keeps worst-case expert concurrency at 3 (probation cap);
    --limit N runs a single-worker smoke into the .smoke trace file."""
    qs = _read_sdqa_gen.remote()
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> sdqa live tier={tier}: {len(qs)} queries, {workers} workers")
    done = list(sdqa_live.starmap(
        [(0, tier, shards[i], i if not limit else -1)
         for i in range(workers)]))
    print(f">>> tier {tier} complete: {sum(len(d) for d in done)} traces")


@gen_app.function(image=util_st, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 30, region=API_REGION)
def sdqa_report(n_boot: int = 10000, seed: int = 42):
    """Judge heard answers (gpt-5.4-mini, same prompt as gated_report),
    per-tier heard-acc + realized escalation, paired bootstrap for the
    balanced-minus-never delta, and the threshold-transfer readout (sdqa
    eot-score distribution vs the calib distribution the thresholds were
    quantiled on). Writes sdqa_live.json + sdqa_traces.parquet."""
    import asyncio
    import glob as _glob
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    paths = sorted(_glob.glob(f"{SDQA_TRACES}.*.shard*"))
    print(f">>> trace files: {paths}", flush=True)
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    df = pd.DataFrame(rows).drop_duplicates(subset=["id", "tier"],
                                            keep="last")
    print(f"=== sdqa traces: n={len(df)} "
          f"({df.groupby('tier').size().to_dict()}) ===", flush=True)

    heard = [{"query": r["query"], "reference_answer": r["reference_answer"],
              "answer": (r.get("relay") if r["mode"] == "escalated"
                         else r.get("answer", ""))} for _, r in df.iterrows()]
    labeled = asyncio.run(escalate.judge_many(heard, concurrency=8))
    df["heard_ok"] = [1 - x["escalate_label"] if x["escalate_label"] is not None
                      else None for x in labeled]
    ok = df[df["heard_ok"].notna()]

    print("\n=== heard accuracy (real human speech, zero recalibration) ===",
          flush=True)
    stats = {}
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
              f"overall {stats[tname]['heard']:.3f} | escalated "
              f"{e['heard_ok'].mean() if len(e) else float('nan'):.3f} | "
              f"local {l['heard_ok'].mean() if len(l) else float('nan'):.3f}",
              flush=True)
        if len(e):
            lat = e["expert_latency_s"].astype(float)
            print(f"    expert latency P50={lat.median():.1f}s "
                  f"P95={lat.quantile(.95):.1f}s", flush=True)

    # paired bootstrap: ids common to both tiers, balanced − never
    piv = ok.pivot(index="id", columns="tier", values="heard_ok").dropna()
    A = piv[["never", "balanced"]].to_numpy(float)
    n = len(piv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = A[idx].mean(axis=1)                     # (n_boot, 2)
    ci = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5], axis=0)]
    d = bs[:, 1] - bs[:, 0]
    delta = float(A[:, 1].mean() - A[:, 0].mean())
    d_ci = ci(d)
    print(f"\n=== paired bootstrap (n={n} common ids, {n_boot} resamples, "
          f"seed {seed}) ===", flush=True)
    print(f"  never    {A[:, 0].mean():.3f} {ci(bs[:, 0])}", flush=True)
    print(f"  balanced {A[:, 1].mean():.3f} {ci(bs[:, 1])}", flush=True)
    print(f"  delta    {delta:+.3f} [{d_ci[0]:+.3f},{d_ci[1]:+.3f}]"
          f"{' *' if d_ci[0] > 0 else ' n.s.'}", flush=True)

    # threshold transfer: sdqa eot distribution vs the calib one the
    # thresholds were quantiled on (eot_gate_report)
    art = json.load(open(GATE_ART))
    thr = art["eot_thresholds"]
    eot_sdqa = (df.sort_values("tier").drop_duplicates(subset="id")
                ["eot_score"].astype(float).to_numpy())
    cal = pd.concat([pd.read_parquet(s) for s in
                     sorted(_glob.glob(f"{EOT_SCORES}.shard*.parquet"))],
                    ignore_index=True).drop_duplicates(subset="id",
                                                       keep="last")
    eot_cal = cal["eot_score"].dropna().astype(float).to_numpy()
    q3 = lambda v: [float(np.percentile(v, p)) for p in (25, 50, 75)]
    print(f"\n=== eot-score threshold transfer ===", flush=True)
    print(f"  calib (thresholds' source, n={len(eot_cal)}): "
          f"P25/50/75 {q3(eot_cal)}", flush=True)
    print(f"  sdqa  (n={len(eot_sdqa)}):                    "
          f"P25/50/75 {q3(eot_sdqa)}", flush=True)
    for t in ("conservative", "balanced", "aggressive"):
        print(f"  fire@{t} thr={thr[t]:.3f}: calib "
              f"{(eot_cal >= thr[t]).mean():.2f} -> sdqa "
              f"{(eot_sdqa >= thr[t]).mean():.2f}", flush=True)

    out = {"n_common": n, "n_boot": n_boot, "seed": seed,
           "tiers": stats,
           "delta_balanced_minus_never": delta,
           "delta_ci": d_ci,
           "never_ci": ci(bs[:, 0]), "balanced_ci": ci(bs[:, 1]),
           "eot_calib_p25_50_75": q3(eot_cal),
           "eot_sdqa_p25_50_75": q3(eot_sdqa),
           "fire_rates": {t: {"calib": float((eot_cal >= thr[t]).mean()),
                              "sdqa": float((eot_sdqa >= thr[t]).mean())}
                          for t in thr}}
    with open(f"{DATA}/sdqa_live.json", "w") as fh:
        json.dump(out, fh, indent=1)
    df.to_parquet(f"{DATA}/sdqa_traces.parquet")
    gate_data.commit()
    print(f"\n>>> wrote {DATA}/sdqa_live.json + sdqa_traces.parquet",
          flush=True)
