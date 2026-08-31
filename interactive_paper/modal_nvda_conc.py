"""NVDA VoiceChat under interleave — the second-family concurrent test (8bd).

8as/8at established for MiniCPM: the concurrent-arrival state (target
audio entering while the model is mid-answer) is the one transfer-ladder
shift that breaks the frozen probe. Nemotron runs CACHELESS (full prefix
per 80 ms frame), so the overlap state is constructed offline: input
audio = [fixed warmup question ++ 0.3 s gap ++ target query] and the
model is (natively, no teacher forcing) answering the warmup in its
agent text channel when the target arrives. Probe reads at target EOT:
same eoth2 recipe, eot window = last 8 frames of the TARGET, user_mean
over TARGET frames only (mirrors the MiniCPM concurrent arm, which
scoped accumulation to the target prefills).

Honest scope: no teacher forcing — if VoiceChat natively yields the
floor when the user speaks over it, that yield is part of this model's
concurrent-arrival state; raw agent text (with timing markers) is kept
so the overlap can be quantified.

Order (cwd=interactive_paper, PYTHONUTF8=1):
  modal run modal_nvda_conc.py::make_warmup
  modal run modal_nvda_conc.py::run_conc --pools frozen --limit 3   # smoke
  modal run modal_nvda_conc.py::run_conc --workers 4
  modal run modal_nvda_conc.py::conc_eval
"""
import json
import os

import modal

from modal_nvda import (app, nemo_image, fit_image, VOLS, gate_data, DATA,
                        K_EOT, NVDA_LAYERS, TAIL_SIL_S, SYS_PROMPT, POOLS,
                        NVDA_H, _load_model, _read_q)

HERE = os.path.dirname(os.path.abspath(__file__))
image_nc = nemo_image.add_local_file(os.path.join(HERE, "modal_nvda.py"),
                                     "/root/modal_nvda.py")
fit_nc = fit_image.add_local_file(os.path.join(HERE, "modal_nvda.py"),
                                  "/root/modal_nvda.py")

WARMUP_WAV = f"{DATA}/nvda_warmup.wav"
WARMUP_TEXT = ("Please tell me a nice long story about a dragon who "
               "learns how to bake bread in a small village.")
GAP_S = 0.3
NVDA_HC = f"{DATA}/nvda_h_conc"              # + _{tag}.shard{i}.npz
NVDA_ANSC = f"{DATA}/nvda_answers_conc"      # + _{tag}.shard{i}.jsonl
CPOOLS = ["frozen", "striviaqa", "swebq", "sllama", "sdqa"]  # en only

util_tts = (modal.Image.debian_slim(python_version="3.11")
            .pip_install("openai", "librosa", "soundfile"))


@app.function(image=util_tts, volumes={DATA: gate_data},
              secrets=[modal.Secret.from_name("openai")], timeout=300)
def make_warmup():
    """tts-1/alloy warmup turn (same engine as every pool wav)."""
    import io
    import librosa
    import soundfile as sf
    from openai import OpenAI
    r = OpenAI().audio.speech.create(model="tts-1", voice="alloy",
                                     input=WARMUP_TEXT,
                                     response_format="wav")
    au, sr = librosa.load(io.BytesIO(r.content), sr=16000, mono=True)
    sf.write(WARMUP_WAV, au, 16000)
    gate_data.commit()
    print(f">>> warmup wav: {len(au)/16000:.1f}s -> {WARMUP_WAV}")


@app.function(image=image_nc, gpu="H100", volumes=VOLS,
              timeout=60 * 60 * 4)
def conc_shard(tag: str, shard: list, shard_id: int) -> int:
    """Overlap-arm inference: warmup++gap++target in one stream,
    capture at target EOT. Mirrors modal_nvda.answer_shard."""
    import re
    import time
    import numpy as np
    import librosa
    import torch
    import sys
    sys.path.insert(0, "/opt/nemo-speech")
    from nemo.collections.speechlm2.inference.utils.offline_voicechat \
        import encode_system_prompt, run_offline_inference

    model = _load_model()
    warm, _ = librosa.load(WARMUP_WAV, sr=16000, mono=True)
    pre = np.concatenate([warm, np.zeros(int(GAP_S * 16000),
                                         dtype=warm.dtype)])
    adir, _ = POOLS[tag]
    items = [(q, f"{adir}/{q['id']}.wav") for q in shard]
    items = [(q, w) for q, w in items if os.path.exists(w)]
    items.sort(key=lambda t: os.path.getsize(t[1]))
    BUDGET = 6 * 1024 * 1024      # tighter than clean: inputs are longer
    batches, cur, cur_max = [], [], 0
    for it in items:
        size = os.path.getsize(it[1])
        if cur and (len(cur) + 1) * max(size, cur_max) > BUDGET:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(it)
        cur_max = max(cur_max, size)
        if len(cur) == 8:
            batches.append(cur)
            cur, cur_max = [], 0
    if cur:
        batches.append(cur)

    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    # frame boundary of the pre segment (warmup+gap), computed once
    with torch.no_grad(), amp:
        p_sig = torch.tensor(pre, device="cuda")[None, :]
        p_len = torch.tensor([len(pre)], dtype=torch.long, device="cuda")
        n_pre = int(model.stt_model.perception(
            input_signal=p_sig, input_signal_length=p_len)[1][0])
    print(f">>> [{tag}] warmup+gap = {len(pre)/16000:.1f}s = {n_pre} "
          f"frames; {len(items)} targets", flush=True)

    ids, E, M, rows = [], [], [], []
    k = 0
    for chunk in batches:
        B = len(chunk)
        aus = []
        for _, w in chunk:
            au, _ = librosa.load(w, sr=16000, mono=True)
            aus.append(np.concatenate([pre, au]))
        qlens = [len(a) for a in aus]
        tail = int(TAIL_SIL_S * 16000)
        full = max(qlens) + tail
        sig = torch.zeros(B, full)
        for b, a in enumerate(aus):
            sig[b, :len(a)] = torch.tensor(a)
        sig = sig.cuda()
        lens = torch.full((B,), full, dtype=torch.long, device="cuda")
        try:
            with torch.no_grad(), amp:
                q_sig = torch.zeros(B, max(qlens), device="cuda")
                for b, a in enumerate(aus):
                    q_sig[b, :len(a)] = torch.tensor(a, device="cuda")
                q_len = torch.tensor(qlens, dtype=torch.long,
                                     device="cuda")
                n_full = [int(x) for x in model.stt_model.perception(
                    input_signal=q_sig, input_signal_length=q_len)[1]]

            prompt_tokens, prompt_token_lens = encode_system_prompt(
                model, SYS_PROMPT, device="cuda")
            if prompt_tokens.shape[0] == 1 and B > 1:
                prompt_tokens = prompt_tokens.expand(B, -1).contiguous()
                prompt_token_lens = prompt_token_lens.expand(B).contiguous()
            prompt_len = int(prompt_token_lens[0].item())

            store = {}

            def mk(L):
                def hook(_m, _i, out):
                    hs = out[0] if isinstance(out, (tuple, list)) else out
                    store[L] = hs.detach()
                return hook
            handles = [model.stt_model.llm.layers[L].register_forward_hook(
                mk(L)) for L in NVDA_LAYERS]
            t0 = time.time()
            try:
                with amp:
                    result = run_offline_inference(
                        model, input_signal=sig, input_signal_lens=lens,
                        prompt_tokens=prompt_tokens,
                        prompt_token_lens=prompt_token_lens)
            finally:
                for h in handles:
                    h.remove()
            secs = time.time() - t0
            texts = result.get("text", [""] * B)
        except Exception as e:
            print(f"  !! batch@{k}: {type(e).__name__}: {str(e)[:150]}",
                  flush=True)
            k += B
            torch.cuda.empty_cache()
            continue

        for b, (q, _) in enumerate(chunk):
            t_start = prompt_len + n_pre           # target begins here
            t_end = prompt_len + n_full[b]         # target EOT
            d = store[NVDA_LAYERS[0]].shape[-1]
            eot = np.zeros((len(NVDA_LAYERS), K_EOT, d), dtype=np.float16)
            mean = np.zeros((len(NVDA_LAYERS), d), dtype=np.float16)
            for j, L in enumerate(NVDA_LAYERS):
                h = store[L][b].float()
                hi = min(t_end, h.shape[0])
                lo = max(t_start, hi - K_EOT)
                w = h[lo:hi].cpu().numpy().astype(np.float16)
                eot[j, K_EOT - w.shape[0]:] = w
                mean[j] = (h[t_start:hi].mean(0).cpu().numpy()
                           .astype(np.float16))
            raw = texts[b] if b < len(texts) else ""
            clean = re.sub(r"<[^>]{0,24}>", " ", raw)
            clean = re.sub(r"  +", " ", clean).strip()
            ids.append(q["id"])
            E.append(eot)
            M.append(mean)
            rows.append({"id": q["id"], "answer": clean,
                         "answer_raw": raw, "secs": round(secs / B, 2),
                         "n_pre": n_pre, "t_start": t_start,
                         "t_end": t_end})
        store.clear()
        if k % 32 == 0 and rows:
            r0 = rows[-len(chunk)]
            print(f"  [{k}/{len(items)}] B={B} {r0['secs']:.1f}s/q "
                  f"{repr(r0['answer'])[:70]}", flush=True)
        k += B

    np.savez_compressed(f"{NVDA_HC}_{tag}.shard{shard_id}.npz",
                        ids=np.array(ids), H_eot=np.stack(E),
                        H_mean=np.stack(M),
                        layers=np.array(NVDA_LAYERS))
    with open(f"{NVDA_ANSC}_{tag}.shard{shard_id}.jsonl", "w",
              encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    gate_data.commit()
    print(f">>> wrote nvda_conc_{tag} shard {shard_id} ({len(ids)})",
          flush=True)
    return len(ids)


@app.local_entrypoint()
def run_conc(pools: str = "", workers: int = 4, limit: int = 0):
    for tag in (pools.split(",") if pools else CPOOLS):
        tag = tag.strip()
        qs = _read_q.remote(POOLS[tag][1])
        if limit:
            qs = qs[:limit]
        w = 1 if limit else workers
        shards = [qs[i::w] for i in range(w)]
        print(f">>> conc [{tag}]: {len(qs)} queries, {w} workers")
        done = list(conc_shard.starmap(
            [(tag, shards[i], i if not limit else 99) for i in range(w)]))
        print(f">>> [{tag}] complete: {sum(done)}")


@app.function(image=fit_nc, volumes={DATA: gate_data}, timeout=60 * 30,
              memory=16384)
def conc_eval() -> str:
    """Frozen NVDA probe (rebuilt to the 8ac recipe from the CLEAN
    shards; reproduction asserted) scored on clean vs overlap features;
    internal paired via shared CV folds; in-regime overlap refit."""
    import glob
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    J = NVDA_LAYERS.index(34)

    def load_tag(prefix, tag):
        shards = sorted(glob.glob(f"{prefix}_{tag}.shard*.npz"))
        ids, E, M = [], [], []
        for sh in shards:
            z = np.load(sh, allow_pickle=True)
            ids += [str(x) for x in z["ids"]]
            E.append(z["H_eot"])
            M.append(z["H_mean"])
        if not ids:
            return None
        E, M = np.concatenate(E), np.concatenate(M)
        df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
        df = df.drop_duplicates("id", keep="last")
        return list(df["id"]), E[df["row"].to_numpy()], M[df["row"].to_numpy()]

    def feats(E, M):
        return np.concatenate([E[:, J, -1].astype(np.float32),
                               E[:, J].mean(1).astype(np.float32),
                               M[:, J].astype(np.float32)], axis=1)

    def clf():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1e-4, max_iter=4000))

    rng = np.random.default_rng(42)

    def boot(y, s, n=10000):
        vals = []
        while len(vals) < n:
            b = rng.choice(len(y), len(y))
            if len(np.unique(y[b])) < 2:
                continue
            vals.append(roc_auc_score(y[b], s[b]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return (round(float(roc_auc_score(y, s)), 3),
                round(float(lo), 3), round(float(hi), 3))

    def pdelta(y, sn, so, n=10000):
        vals = []
        while len(vals) < n:
            b = rng.choice(len(y), len(y))
            if len(np.unique(y[b])) < 2:
                continue
            vals.append(roc_auc_score(y[b], sn[b])
                        - roc_auc_score(y[b], so[b]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return (round(float(np.mean(vals)), 3),
                round(float(lo), 3), round(float(hi), 3))

    lab = {t: pd.read_parquet(f"{DATA}/nvda_{t}.parquet")
              .set_index("id")["escalate_label"] for t in CPOOLS}

    # aligned clean/overlap frozen matrices
    cids, cE, cM = load_tag(NVDA_H, "frozen")
    oids, oE, oM = load_tag(NVDA_HC, "frozen")
    common = [i for i in cids if i in set(oids)
              and i in lab["frozen"].index
              and pd.notna(lab["frozen"].get(i))]
    ci = {i: j for j, i in enumerate(cids)}
    oi = {i: j for j, i in enumerate(oids)}
    Xc = feats(cE[[ci[i] for i in common]], cM[[ci[i] for i in common]])
    Xo = feats(oE[[oi[i] for i in common]], oM[[oi[i] for i in common]])
    y = np.array([int(lab["frozen"][i]) for i in common])
    print(f"frozen aligned n={len(y)}, fail rate {y.mean():.3f}",
          flush=True)

    kf = StratifiedKFold(5, shuffle=True, random_state=42)
    p_clean = np.zeros(len(y))
    p_over = np.zeros(len(y))
    p_inreg = np.zeros(len(y))
    for tr, te in kf.split(Xc, y):
        m = clf().fit(Xc[tr], y[tr])
        p_clean[te] = m.predict_proba(Xc[te])[:, 1]
        p_over[te] = m.predict_proba(Xo[te])[:, 1]
        m2 = clf().fit(Xo[tr], y[tr])
        p_inreg[te] = m2.predict_proba(Xo[te])[:, 1]
    out = {"n_frozen": int(len(y)), "internal": {
        "oof_clean": boot(y, p_clean),
        "oof_frozenprobe_on_overlap": boot(y, p_over),
        "delta": pdelta(y, p_over, p_clean),
        "oof_inregime_refit": boot(y, p_inreg)}}
    print("internal:", out["internal"], flush=True)

    frozen_probe = clf().fit(Xc, y)      # 8ac deployed analog
    inreg_probe = clf().fit(Xo, y)
    out["pools"] = {}
    for t in [p for p in CPOOLS if p != "frozen"]:
        c = load_tag(NVDA_H, t)
        o = load_tag(NVDA_HC, t)
        if c is None or o is None:
            continue
        common_t = [i for i in c[0] if i in set(o[0])
                    and i in lab[t].index and pd.notna(lab[t].get(i))]
        cit = {i: j for j, i in enumerate(c[0])}
        oit = {i: j for j, i in enumerate(o[0])}
        Xct = feats(c[1][[cit[i] for i in common_t]],
                    c[2][[cit[i] for i in common_t]])
        Xot = feats(o[1][[oit[i] for i in common_t]],
                    o[2][[oit[i] for i in common_t]])
        yt = np.array([int(lab[t][i]) for i in common_t])
        sc = frozen_probe.predict_proba(Xct)[:, 1]
        so = frozen_probe.predict_proba(Xot)[:, 1]
        si = inreg_probe.predict_proba(Xot)[:, 1]
        r = {"n": int(len(yt)), "fail_rate": round(float(yt.mean()), 3),
             "clean": boot(yt, sc), "overlap": boot(yt, so),
             "delta": pdelta(yt, so, sc), "overlap_inregime": boot(yt, si)}
        out["pools"][t] = r
        print(f"{t:<10} n={r['n']} clean {r['clean'][0]:.3f} -> overlap "
              f"{r['overlap'][0]:.3f} (d {r['delta'][0]:+.3f} "
              f"[{r['delta'][1]:+.3f},{r['delta'][2]:+.3f}]) "
              f"inreg {r['overlap_inregime'][0]:.3f}", flush=True)

    with open(f"{DATA}/nvda_conc_eval.json", "w") as f:
        json.dump(out, f, indent=1)
    gate_data.commit()
    return json.dumps(out)


@app.local_entrypoint()
def run_eval():
    print(conc_eval.remote()[:3000])
