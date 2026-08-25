"""Probe v4 — real-time-data awareness (user request 2026-08-24).

The gate had never seen a "needs live data" query: the whole v3 train set
is static public-benchmark knowledge, so "what is NVDA trading at" sits AT
the balanced threshold (voice turn .624 < .680, typed render .693 >= .680)
— boundary behavior, not a decision. Fix with more public-benchmark data,
same recipe as v2/v3 ([[no-selfmade-datasets]] respected):

  FreshQA (freshllms/freshqa, sheet of 2026-04-21), 600 questions with
  fact_type labels. fresh_fast = fast-changing & !false_premise (153):
  the current value is unknowable at the model's cutoff, so escalate=1
  A PRIORI — the one deliberate deviation from measured labels, justified
  because a judge against a stale gold would only add noise to a label
  that is certain by construction. fresh_never = never-changing &
  !false_premise (150): measured labels via the standard answer+judge
  path, as in-family controls so the probe can't just learn "FreshQA
  phrasing = fire". 30 of each held out, never trained on.

Feature config is FROZEN to v3's winner (L22, eot_last+eot_mean+user_mean)
so the deployed demo reader works unchanged; only w/b and the OOF-quantile
tier thresholds are refit. v3 artifact and every measured curve untouched;
new artifact midlayer_gate_audio_v4.json.

Guards (pre-registered): (1) frozen-test in-mix AUC must not regress vs
v3; (2) fresh_fast heldout fire-rate at balanced must be high under v4
while fresh_never heldout stays low.

Stages (run in order; queries_fresh.jsonl already on the volume):
  modal run modal_fresh.py::run_tts_fresh
  modal run modal_fresh.py::run_answer_fresh
  modal run modal_fresh.py::judge_fresh
  modal run modal_fresh.py::run_eoth_fresh
  modal run modal_fresh.py::refit4
"""
import json
import os
import sys

import modal

from modal_app import (app, gen_app, GPU_VOL, gate_data, DATA, OPENAI,
                       API_REGION, _read_jsonl)
from modal_train import TTS_VOICE
from modal_train2 import (image_st2, util_st2, EOTH2, ART_V3, LAYERS2,
                          K_EOT, _load2, _feat, _train_xy)

HERE = os.path.dirname(os.path.abspath(__file__))
_T2_PY = os.path.join(HERE, "modal_train2.py")
image_st3 = image_st2.add_local_file(_T2_PY, "/root/modal_train2.py")
util_st3 = util_st2.add_local_file(_T2_PY, "/root/modal_train2.py")

FQ = f"{DATA}/queries_fresh.jsonl"
FAUDIO = f"{DATA}/audio_fresh"
FLABELS = f"{DATA}/fresh_labels.parquet"
ART_V4 = f"{DATA}/midlayer_gate_audio_v4.json"


@app.function(image=util_st3, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_fresh() -> list:
    return _read_jsonl(FQ)


@gen_app.function(image=util_st3, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 50, region=API_REGION)
def run_tts_fresh(limit: int = 0, concurrency: int = 8):
    """tts-1/alloy, same engine as every other pool."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    qs = _read_jsonl(FQ)[:limit or None]
    os.makedirs(FAUDIO, exist_ok=True)
    client = OpenAI()

    def render(q):
        out = f"{FAUDIO}/{q['id']}.wav"
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
    print(f">>> tts fresh: {res.count('done')} rendered, "
          f"{res.count('cached')} cached / {len(qs)}", flush=True)


@app.function(image=image_st3, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 2)
def answer_fresh_shard(shard: list, shard_id: int) -> int:
    """MiniCPM answers fresh_never queries from AUDIO (6a style)."""
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
        au, _ = librosa.load(f"{FAUDIO}/{q['id']}.wav", sr=16000, mono=True)
        ans = model.chat(msgs=[{"role": "user", "content": [au]}],
                         tokenizer=tok, max_new_tokens=512, sampling=False,
                         use_tts_template=False, generate_audio=False)
        rows.append({"id": q["id"], "answer": str(ans).strip()})
        if k < 3 or k % 40 == 0:
            print(f"  [{k}] {q['id']} :: {rows[-1]['answer'][:60]!r}",
                  flush=True)
    pd.DataFrame(rows).to_parquet(
        f"{DATA}/fresh_answers.shard{shard_id}.parquet")
    gate_data.commit()
    return len(rows)


@app.local_entrypoint()
def run_answer_fresh(workers: int = 2, limit: int = 0):
    qs = [q for q in _read_fresh.remote() if q["pool"] == "fresh_never"]
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> answering {len(qs)} fresh_never / {workers} workers")
    total = sum(answer_fresh_shard.starmap(
        [(shards[i], i) for i in range(workers)]))
    print(f">>> answered {total}")


@gen_app.function(image=util_st3, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 60, region=API_REGION)
def judge_fresh():
    """fresh_never: measured labels (judge). fresh_fast: escalate=1 a
    priori (see module docstring) — written into the same parquet."""
    import glob as _glob
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    byid = {q["id"]: q for q in _read_jsonl(FQ)}
    answers = pd.concat(
        [pd.read_parquet(s) for s in
         sorted(_glob.glob(f"{DATA}/fresh_answers.shard*.parquet"))],
        ignore_index=True).drop_duplicates(subset="id", keep="last")
    rows = [{"id": a["id"], "query": byid[a["id"]]["query"],
             "reference_answer": byid[a["id"]]["reference_answer"],
             "answer": a["answer"]} for _, a in answers.iterrows()]
    judged = asyncio.run(escalate.judge_many(rows, concurrency=8))
    df = pd.DataFrame(judged)
    fast = pd.DataFrame([{"id": q["id"], "escalate_label": 1}
                         for q in byid.values()
                         if q["pool"] == "fresh_fast"])
    df = pd.concat([df, fast], ignore_index=True)
    df["pool"] = [byid[i]["pool"] for i in df["id"]]
    df["split"] = [byid[i]["split"] for i in df["id"]]
    df.to_parquet(FLABELS)
    gate_data.commit()
    ok = df[df["escalate_label"].notna()]
    for (p, s), g in ok.groupby(["pool", "split"]):
        print(f"  {p:12s} {s:8s} n={len(g):3d} "
              f"fail-rate={g['escalate_label'].mean():.2f}", flush=True)


@app.function(image=image_st3, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 2, max_containers=8)
def eoth_fresh_shard(shard: list, shard_id: int) -> int:
    """eoth2_shard clone, tag=fresh (writes eoth2_fresh.shard*.npz so
    modal_train2._load2("fresh") reads them)."""
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

    state = {"accum": False, "tail": {}, "sum": {}, "cnt": 0}

    def mk_hook(L, count_here):
        def hook(_m, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            h = hs[0].detach().float()
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

    ids, E, M, ELEN = [], [], [], []
    try:
        for k, q in enumerate(shard):
            wav = f"{FAUDIO}/{q['id']}.wav"
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

    np.savez_compressed(f"{EOTH2}_fresh.shard{shard_id}.npz",
                        ids=np.array(ids), H_eot=np.stack(E),
                        H_mean=np.stack(M),
                        eot_len=np.array(ELEN, dtype=np.int16),
                        layers=np.array(LAYERS2))
    gate_data.commit()
    print(f">>> wrote eoth2_fresh shard {shard_id} ({len(ids)})", flush=True)
    return len(ids)


@app.local_entrypoint()
def run_eoth_fresh(workers: int = 2, limit: int = 0):
    qs = _read_fresh.remote()
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> eoth fresh: {len(qs)} wavs / {workers} workers")
    total = sum(eoth_fresh_shard.starmap(
        [(shards[i], i if not limit else 99) for i in range(workers)]))
    print(f">>> captured {total}")


@app.function(image=util_st3, volumes={DATA: gate_data}, timeout=60 * 120,
              cpu=16, memory=32768)
def refit4(c_sweep: str = "0.0001,0.0003,0.001,0.003"):
    """v4 = v3 recipe + fresh TRAIN rows; feature config frozen to v3's
    winner. Prints both pre-registered guards; writes ART_V4."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, "/workspace/gate")
    import gate as gate_mod

    v3 = json.load(open(ART_V3))
    layers, modes = v3["layer_set"], v3["modes"]

    IDS, E, M, ELEN, y = _train_xy(with_y=True)
    n0 = len(y)          # v3 calib mix — tier budgets stay defined on it
    fl = pd.read_parquet(FLABELS)
    fl = fl[fl["escalate_label"].notna()]
    lab_f = dict(zip(fl["id"], fl["escalate_label"].astype(int)))
    split_f = dict(zip(fl["id"], fl["split"]))
    ids_fr, E_fr, M_fr, ELEN_fr = _load2("fresh")
    keep = [j for j, i in enumerate(ids_fr)
            if i in lab_f and split_f.get(i) == "train"]
    print(f">>> fresh train rows: {len(keep)}", flush=True)

    E = np.concatenate([E, E_fr[keep]])
    M = np.concatenate([M, M_fr[keep]])
    ELEN = np.concatenate([ELEN, ELEN_fr[keep]])
    y = np.concatenate([y, [lab_f[ids_fr[j]] for j in keep]])
    IDS = IDS + [ids_fr[j] for j in keep]

    X = _feat(E, M, ELEN, layers, modes)
    print(f">>> train n={len(y)} d={X.shape[1]} "
          f"fail-rate {y.mean():.2f}", flush=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    best = None
    for C in (float(c) for c in c_sweep.split(",")):
        oof = cross_val_predict(
            LogisticRegression(C=C, max_iter=5000), X, y, cv=cv,
            method="predict_proba")[:, 1]
        a = roc_auc_score(y, oof)
        print(f"  C={C}: OOF AUC={a:.3f}", flush=True)
        if best is None or a > best[1]:
            best = (C, a, oof)
    C, auc, oof = best
    clf = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    # budgets are quantiles of the DEPLOYMENT-mix scores: the fresh rows
    # oversample real-time queries relative to live traffic, so they
    # train the direction but must not inflate the budget quantiles
    thr = {t: float(np.quantile(oof[:n0], 1 - b))
           for t, b in (("conservative", 0.15), ("balanced", 0.30),
                        ("aggressive", 0.50))}

    art = dict(v3)
    art.update(version=4, n_calib=int(len(y)), C=C, oof_auc=float(auc),
               eot_thresholds=thr, thresholds=thr,
               w=clf.coef_[0].astype(float).tolist(),
               b=float(clf.intercept_[0]),
               fresh_note="v3 + FreshQA fast(a-priori 1)/never(judged); "
                          "30+30 heldout")
    with open(ART_V4, "w") as fh:
        json.dump(art, fh)
    gate_data.commit()
    print(f">>> v4: n={len(y)} C={C} OOF AUC={auc:.3f} "
          f"thr={ {k: round(v, 3) for k, v in thr.items()} }", flush=True)

    # ---- guard 1: frozen-test in-mix must not regress vs v3 ----------
    feats = pd.read_parquet(f"{DATA}/features_minicpm-o45-audio.parquet")[
        ["id", "split", "escalate_label"]]
    tst = feats[(feats["split"] == "test") & feats["escalate_label"].notna()]
    lab_t = dict(zip(tst["id"], tst["escalate_label"].astype(int)))
    ids_z, E_z, M_z, ELEN_z = _load2("frozen")
    kz = [j for j, i in enumerate(ids_z) if i in lab_t]
    Xz = _feat(E_z[kz], M_z[kz], ELEN_z[kz], layers, modes)
    yz = np.array([lab_t[ids_z[j]] for j in kz])
    P3 = gate_mod.Probe(v3["w"], v3["b"])
    P4 = gate_mod.Probe(art["w"], art["b"])
    s3 = np.array([P3.score(x) for x in Xz])
    s4 = np.array([P4.score(x) for x in Xz])
    print(f">>> GUARD frozen-test AUC: v3={roc_auc_score(yz, s3):.3f} "
          f"v4={roc_auc_score(yz, s4):.3f} (n={len(yz)})", flush=True)

    # ---- guard 2: heldout fire-rates at each tier --------------------
    kh = [j for j, i in enumerate(ids_fr) if split_f.get(i) == "heldout"]
    Xh = _feat(E_fr[kh], M_fr[kh], ELEN_fr[kh], layers, modes)
    pools = [fl.set_index("id")["pool"].get(ids_fr[j]) for j in kh]
    sh4 = np.array([P4.score(x) for x in Xh])
    sh3 = np.array([P3.score(x) for x in Xh])
    for pool in ("fresh_fast", "fresh_never"):
        pk = [i for i, p in enumerate(pools) if p == pool]
        for tag, sc, th in (("v3", sh3, v3["eot_thresholds"]),
                            ("v4", sh4, thr)):
            fr = {t: float((sc[pk] >= v).mean()) for t, v in th.items()}
            print(f">>> GUARD heldout {pool} {tag} fire-rate "
                  f"{ {k: round(v, 2) for k, v in fr.items()} } "
                  f"(n={len(pk)})", flush=True)
