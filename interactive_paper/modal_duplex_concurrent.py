"""Concurrent-prefill probe validation (the tier the paper leaves open).

While the talker is mid-answer (streaming_generate with the TTS head on,
generator being drained), the target query's audio is prefilled 1 s chunk
by 1 s chunk into the SAME session between generation chunks — listen and
speak interleaved in one KV stream, time-division multiplexed exactly like
native duplex inference. After the last target chunk, the standard
end-of-turn read (assistant-space prefill, L22) scores the query in that
concurrent state.

Feasibility basis (modeling_minicpmo.py): streaming_prefill and
chunk_generate share self.llm_past_key_values; each generation chunk
re-reads the cache and derives positions from cache length, so an
interleaved prefill is picked up with self-consistent positions.

Arm design mirrors app:duplexval's overlap arm, minus the barge-in: one
fixed easy warmup question puts the talker mid-answer (>8 s spoken), the
target audio is fed concurrently; if generation finishes before the target
audio does, remaining chunks prefill non-concurrently and the row is
flagged (n_concurrent / gen_active_at_eot).

Run:  modal run modal_duplex_concurrent.py::run_concurrent --limit 3
Full: modal run modal_duplex_concurrent.py::run_concurrent --workers 4
"""
import json
import os
import sys
import time

import modal

import os as _os

from modal_app import (gen_app, GPU_VOL, gate_data, DATA, image,
                       util_image, _read_jsonl, OPENAI)

_HERE = _os.path.dirname(_os.path.abspath(__file__))
image_cc = image.add_local_file(_os.path.join(_HERE, "modal_app.py"),
                                "/root/modal_app.py")
util_cc = util_image.add_local_file(_os.path.join(_HERE, "modal_app.py"),
                                    "/root/modal_app.py")
LAYER = 22
AUDIO_DIR = f"{DATA}/audio_pool"

ART = f"{DATA}/gate_v3_frozen.json"
# easy-chat warmup, locally answered, long spoken answer (from v3tts traces)
WARMUP_Q = "Please tell me a story."
# teacher-forced spoken text: sampling would hit EOS the moment user audio
# appears in context (trained turn-yielding); forcing keeps generation
# alive for the full overlap window with identical content per query
FORCE_TEXT = " ".join([
    "Once upon a time a small dragon lived in a quiet valley beside a",
    "slow green river. Every morning the dragon walked to the village",
    "bakery and watched the baker knead the dough with steady hands.",
    "The dragon wanted to learn, so it practiced folding and pressing",
    "the dough late into the evening, night after night, week after",
    "week. Slowly the loaves improved, first dense and burnt, then",
    "lighter, then golden and warm. The villagers began to line up at",
    "sunrise, and the dragon greeted every one of them by name. In the",
    "winter the oven kept the whole square warm, and travelers came",
    "from distant towns to taste the famous dragon bread. The baker",
    "grew old and proud, and the dragon carried the flour sacks and",
    "kept the fire even and low. Seasons turned, the river rose and",
    "fell, and the little bakery stayed open through every storm.",
] * 3)
OUT = f"{DATA}/frozen_concurrent_traces.jsonl"  # original test-scores run
OUT2 = f"{DATA}/frozen_conc"  # feature runs: {OUT2}_{tag}_traces.jsonl / _feats


@gen_app.function(image=image_cc, gpu="H100", volumes=GPU_VOL,
                  timeout=60 * 60 * 3)
def concurrent_shard(shard: list, shard_id: int = -1,
                     tag: str = "", audio_dir: str = AUDIO_DIR) -> list:
    import glob as _glob
    import shutil
    import inspect
    import threading

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
        init_vision=False, init_audio=True, init_tts=True,
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model.init_tts(model_dir=f"{MODEL_DIR}/assets/token2wav")
    ref, _ = librosa.load(f"{MODEL_DIR}/assets/system_ref_audio.wav",
                          sr=16000, mono=True)
    model.init_token2wav_cache(ref)

    art = json.load(open(ART))
    probe = gate_mod.Probe(art["w"], art["b"])
    print(f">>> concurrent-prefill arm: L{art.get('layer')} probe v3, "
          f"{len(shard)} queries", flush=True)

    def call_def(fn, /, **kw):
        params = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in params})

    # ---- v3 probe read state (bench_live semantics) ----------------------
    K3 = art.get("k_eot", 8)
    st3 = {"accum": False, "tail": None, "sum": None, "cnt": 0}

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        h = hs[0].detach().float()
        t = h[-K3:].cpu()
        st3["tail"] = (t if st3["tail"] is None
                       else torch.cat([st3["tail"], t])[-K3:])
        if st3["accum"]:
            s = h.sum(0).cpu()
            st3["sum"] = s if st3["sum"] is None else st3["sum"] + s
            st3["cnt"] += h.shape[0]

    def score_now():
        parts = []
        for m in art["modes"]:
            if m == "eot_last":
                parts.append(st3["tail"][-1])
            elif m == "eot_mean":
                parts.append(st3["tail"].mean(0))
            elif m == "user_mean":
                parts.append(st3["sum"] / max(1, st3["cnt"]))
        vec = torch.cat(parts).numpy()
        return float(probe.score(vec)), vec

    traces = []
    feat_ids, feat_X = [], []
    for qi, q in enumerate(shard):
        au, _ = librosa.load(f"{audio_dir}/{q['id']}.wav",
                             sr=16000, mono=True)
        chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        chunks = [np.pad(c, (0, 16000 - len(c))) if len(c) < 16000 else c
                  for c in chunks]

        call_def(model.reset_session, reset_token2wav_cache=False)
        sys_msg = call_def(model.get_sys_prompt, mode="omni", language="en")
        call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
                 tokenizer=tok)

        h = model.llm.model.layers[LAYER].register_forward_hook(hook)
        st3.update(tail=None, sum=None, cnt=0, accum=False)
        try:
            # warmup turn: text in (keeps its forwards out of user_mean;
            # accum only wraps the TARGET audio prefills)
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "user", "content": [WARMUP_Q]}],
                     tokenizer=tok, is_last_chunk=True)
            gen = call_def(model.streaming_generate, tokenizer=tok,
                           temperature=0.1, generate_audio=True,
                           use_tts_template=True, max_new_tokens=1400,
                           teacher_forcing=True,
                           teacher_forcing_text=FORCE_TEXT,
                           session_id="s1")

            # interleave: drain ~1 s of spoken warmup answer, then prefill
            # one 1 s chunk of the target query into the same cache
            n_concurrent = 0
            gen_active = True
            samp_budget = 0
            for ci, ch in enumerate(chunks):
                if gen_active:
                    while samp_budget < 24000:  # ~1 s of talker speech
                        try:
                            item = next(gen)
                        except StopIteration:
                            gen_active = False
                            break
                        wf = item[0] if isinstance(item, tuple) else None
                        if wf is not None:
                            samp_budget += int(
                                wf.float().cpu().numpy().reshape(-1).shape[0])
                    samp_budget = max(0, samp_budget - 24000)
                st3["accum"] = True
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[{"role": "user",
                                "content": [ch.astype(np.float32)]}],
                         tokenizer=tok,
                         is_last_chunk=(ci == len(chunks) - 1))
                st3["accum"] = False
                if gen_active:
                    n_concurrent += 1

            gen_active_at_eot = gen_active
            t0 = time.time()
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "assistant", "content": [" "]}],
                     tokenizer=tok, is_last_chunk=True)
            eot_score, feat_vec = score_now()
            eot_ms = int((time.time() - t0) * 1000)
            gen.close()
        finally:
            h.remove()

        feat_ids.append(q["id"])
        feat_X.append(feat_vec.astype(np.float32))
        traces.append({"id": q["id"], "pool": q.get("pool", tag or "?"),
                       "n_chunks": len(chunks),
                       "n_concurrent": n_concurrent,
                       "gen_active_at_eot": gen_active_at_eot,
                       "eot_score": round(eot_score, 4),
                       "eot_read_ms": eot_ms})
        print(f"  [{qi}] {q['id']} eot={eot_score:.3f} "
              f"conc={n_concurrent}/{len(chunks)} "
              f"gen_at_eot={gen_active_at_eot}", flush=True)

    base = OUT if not tag else f"{OUT2}_{tag}_traces.jsonl"
    out = f"{base}.smoke" if shard_id < 0 else f"{base}.shard{shard_id}"
    with open(out, "a", encoding="utf-8") as fh:
        for r in traces:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if tag:
        np.savez_compressed(
            f"{OUT2}_{tag}_feats.shard{max(shard_id, 0)}.npz",
            ids=np.array(feat_ids), X=np.stack(feat_X))
    gate_data.commit()
    print(f">>> appended {len(traces)} traces to {out}", flush=True)
    return [r["id"] for r in traces]


@gen_app.function(image=util_cc, volumes={DATA: gate_data},
                  timeout=60 * 5)
def _read_frozen(split: str = "test") -> list:
    return [q for q in _read_jsonl(f"{DATA}/queries.jsonl")
            if q.get("split") == split]


# feature-run pools beyond the frozen split (8bb full-scale recalibration):
# the whole 2310-row train mix + the external eval pools, same wav naming
FEAT_POOLS = {
    "frozen":     (f"{DATA}/queries.jsonl",            f"{DATA}/audio_pool"),
    "expansion":  (f"{DATA}/queries_expansion.jsonl",  f"{DATA}/audio_expansion"),
    "expansion2": (f"{DATA}/queries_expansion2.jsonl", f"{DATA}/audio_expansion2"),
    "striviaqa":  (f"{DATA}/queries_striviaqa.jsonl",  f"{DATA}/bench_audio"),
    "swebq":      (f"{DATA}/queries_swebq.jsonl",      f"{DATA}/bench_audio"),
    "sllama":     (f"{DATA}/queries_sllama.jsonl",     f"{DATA}/bench_audio"),
    "sdqa":       (f"{DATA}/queries_sdqa.jsonl",       f"{DATA}/sdqa_audio"),
    "sreason":    (f"{DATA}/queries_sreason.jsonl",    f"{DATA}/bench_audio"),
}


@gen_app.function(image=util_cc, volumes={DATA: gate_data},
                  timeout=60 * 5)
def _read_qfile(qfile: str, split: str = "") -> list:
    qs = _read_jsonl(qfile)
    return [q for q in qs if q.get("split") == split] if split else qs


@gen_app.local_entrypoint()
def run_concurrent(workers: int = 4, limit: int = 0, split: str = "test",
                   tag: str = "", pool: str = "frozen"):
    qfile, audio_dir = FEAT_POOLS[pool]
    split_qs = _read_qfile.remote(qfile, split if pool == "frozen" else "")
    if limit:
        split_qs = split_qs[:limit]
        workers = 1
    shards = [split_qs[i::workers] for i in range(workers)]
    print(f">>> concurrent arm [{pool}]: {len(split_qs)} queries, "
          f"{workers} workers")
    done = list(concurrent_shard.starmap(
        [(shards[i], i if not limit else -1, tag, audio_dir)
         for i in range(workers)]))
    print(f">>> complete: {sum(len(d) for d in done)} traces")


# ---- full escalation loop in the concurrent regime (8at) -----------------
# Gate = the in-regime refit probe (gate_conc_frozen.json), thresholds =
# label-free quantiles of concurrent calib scores at the deployed internal
# fire rates. Carrier/interleave mechanics identical to concurrent_shard;
# after the EOT read the carrier is closed (turn yield) and the answer or
# stall->expert->relay chain runs text-out, mirroring bench_live so that
# modal_bench.py::report judges the traces unchanged (suffix _conclive).
CPOOLS = {
    "frozen":    {"audio": f"{DATA}/audio_pool",
                  "q": f"{DATA}/queries.jsonl", "split": "test",
                  "asr": f"{DATA}/asr_minicpm-o45-audio"},
    "striviaqa": {"audio": f"{DATA}/bench_audio",
                  "q": f"{DATA}/queries_striviaqa.jsonl", "split": None,
                  "asr": f"{DATA}/striviaqa_transcripts"},
    "swebq":     {"audio": f"{DATA}/bench_audio",
                  "q": f"{DATA}/queries_swebq.jsonl", "split": None,
                  "asr": f"{DATA}/swebq_transcripts"},
    "sllama":    {"audio": f"{DATA}/bench_audio",
                  "q": f"{DATA}/queries_sllama.jsonl", "split": None,
                  "asr": f"{DATA}/sllama_transcripts"},
    "sdqa":      {"audio": f"{DATA}/sdqa_audio",
                  "q": f"{DATA}/queries_sdqa.jsonl", "split": None,
                  "asr": f"{DATA}/sdqa_transcripts"},
    "sreason":   {"audio": f"{DATA}/bench_audio",
                  "q": f"{DATA}/queries_sreason.jsonl", "split": None,
                  "asr": f"{DATA}/sreason_transcripts"},
    "valpaca":   {"audio": f"{DATA}/bench_audio",
                  "q": f"{DATA}/queries_valpaca.jsonl", "split": None,
                  "asr": f"{DATA}/valpaca_transcripts"},
}
ART_CONC = f"{DATA}/gate_conc_frozen.json"
STALL = "Hmm, that's a good question — give me a moment to check."
RELAY_TMPL = ("[SYSTEM NOTE] A trusted expert system has already verified "
              "the answer to the user's last spoken question.\n"
              "Expert answer: {ans}\n"
              "Relay this answer to the user in your own words, concisely "
              "and naturally, as a continuation of the conversation. Do not "
              "contradict the expert answer.")


@gen_app.function(image=image_cc, gpu="H100", volumes=GPU_VOL,
                  secrets=[OPENAI], timeout=60 * 60 * 3)
def conclive_shard(shard: list, tier: str = "balanced",
                   shard_id: int = -1, bench: str = "frozen",
                   thr_override: float = 0.0,
                   tts_answers: int = 0) -> list:
    import glob as _glob
    import shutil
    import inspect
    import threading

    import numpy as np
    import torch
    import librosa
    import pandas as pd
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import escalate

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
        init_vision=False, init_audio=True, init_tts=True,
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model.init_tts(model_dir=f"{MODEL_DIR}/assets/token2wav")
    ref, _ = librosa.load(f"{MODEL_DIR}/assets/system_ref_audio.wav",
                          sr=16000, mono=True)
    model.init_token2wav_cache(ref)

    art = json.load(open(ART_CONC))
    w = np.array(art["w"], dtype=np.float32)
    b = float(art["b"])
    thr = {"never": 1e9, "always": -1e9}.get(tier)
    if thr is None:
        thr = thr_override or art["eot_thresholds"][tier]
    print(f">>> conclive[{bench}]: in-regime gate, tier={tier} "
          f"thr={thr:.3f}", flush=True)

    asr = pd.concat([pd.read_parquet(x) for x in sorted(
        _glob.glob(f"{CPOOLS[bench]['asr']}.shard*.parquet"))],
        ignore_index=True).drop_duplicates(subset="id", keep="last")
    transcript = dict(zip(asr["id"], asr["transcript"]))

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

    def gen_speak2(t_ref, **kw):
        kw.setdefault("max_new_tokens", 512)
        res = call_def(model.streaming_generate, tokenizer=tok,
                       temperature=0.1, generate_audio=True,
                       use_tts_template=True, **kw)
        texts, first_ms, n_samp = [], None, 0
        for item in res:
            wf, txt = (item if isinstance(item, tuple)
                       else (None, getattr(item, "text", None)))
            if wf is not None:
                if first_ms is None:
                    first_ms = int((time.time() - t_ref) * 1000)
                n_samp += len(wf.float().cpu().numpy().reshape(-1))
            if isinstance(txt, str):
                texts.append(txt)
        return "".join(texts).strip(), first_ms, n_samp

    stall_pcm_s = None
    if tts_answers:
        call_def(model.reset_session, reset_token2wav_cache=False)
        sys0 = call_def(model.get_sys_prompt, mode="omni", language="en")
        call_def(model.streaming_prefill, session_id="s1", msgs=[sys0],
                 tokenizer=tok)
        call_def(model.streaming_prefill, session_id="s1",
                 msgs=[{"role": "user",
                        "content": [np.zeros(16000, dtype="float32")]}],
                 tokenizer=tok, is_last_chunk=True)
        _, _, s_samp = gen_speak2(time.time(), session_id="s1",
                                  teacher_forcing=True,
                                  teacher_forcing_text=STALL,
                                  max_new_tokens=64)
        stall_pcm_s = round(s_samp / 24000, 2)
        print(f">>> canned stall: {stall_pcm_s}s", flush=True)

    K3 = art.get("k_eot", 8)
    st3 = {"accum": False, "tail": None, "sum": None, "cnt": 0}

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        h = hs[0].detach().float()
        t = h[-K3:].cpu()
        st3["tail"] = (t if st3["tail"] is None
                       else torch.cat([st3["tail"], t])[-K3:])
        if st3["accum"]:
            sm = h.sum(0).cpu()
            st3["sum"] = sm if st3["sum"] is None else st3["sum"] + sm
            st3["cnt"] += h.shape[0]

    def score_now():
        parts = [st3["tail"][-1], st3["tail"].mean(0),
                 st3["sum"] / max(1, st3["cnt"])]
        vec = torch.cat(parts).numpy()
        return float(1.0 / (1.0 + np.exp(-(float(vec @ w) + b))))

    traces = []
    for qi, q in enumerate(shard):
        if q["id"] not in transcript:
            continue
        au, _ = librosa.load(f"{CPOOLS[bench]['audio']}/{q['id']}.wav",
                             sr=16000, mono=True)
        chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        chunks = [np.pad(c, (0, 16000 - len(c))) if len(c) < 16000 else c
                  for c in chunks]

        call_def(model.reset_session, reset_token2wav_cache=False)
        sys_msg = call_def(model.get_sys_prompt, mode="omni", language="en")
        call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
                 tokenizer=tok)

        h = model.llm.model.layers[LAYER].register_forward_hook(hook)
        st3.update(tail=None, sum=None, cnt=0, accum=False)
        try:
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "user", "content": [WARMUP_Q]}],
                     tokenizer=tok, is_last_chunk=True)
            gen = call_def(model.streaming_generate, tokenizer=tok,
                           temperature=0.1, generate_audio=True,
                           use_tts_template=True, max_new_tokens=1400,
                           teacher_forcing=True,
                           teacher_forcing_text=FORCE_TEXT,
                           session_id="s1")
            n_concurrent = 0
            gen_active = True
            samp_budget = 0
            for ci, ch in enumerate(chunks):
                if gen_active:
                    while samp_budget < 24000:
                        try:
                            item = next(gen)
                        except StopIteration:
                            gen_active = False
                            break
                        wf = item[0] if isinstance(item, tuple) else None
                        if wf is not None:
                            samp_budget += int(
                                wf.float().cpu().numpy().reshape(-1).shape[0])
                    samp_budget = max(0, samp_budget - 24000)
                st3["accum"] = True
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[{"role": "user",
                                "content": [ch.astype(np.float32)]}],
                         tokenizer=tok,
                         is_last_chunk=(ci == len(chunks) - 1))
                st3["accum"] = False
                if gen_active:
                    n_concurrent += 1

            gen_active_at_eot = gen_active
            t0 = time.time()
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[{"role": "assistant", "content": [" "]}],
                     tokenizer=tok, is_last_chunk=True)
            eot_score = score_now()
            eot_ms = int((time.time() - t0) * 1000)
            gen.close()          # turn yield: the carrier stops speaking
        finally:
            h.remove()

        fired = eot_score >= thr
        t_eot = time.time()
        row = {"id": q["id"], "pool": q["pool"], "tier": tier,
               "audio_s": round(len(au) / 16000, 2),
               "n_chunks": len(chunks), "n_concurrent": n_concurrent,
               "gen_active_at_eot": gen_active_at_eot,
               "eot_score": round(eot_score, 4), "eot_read_ms": eot_ms,
               "reference_answer": q.get("reference_answer"),
               "query": q["query"], "transcript": transcript[q["id"]]}
        if not fired:
            if tts_answers:
                ans, fa_ms, n_samp = gen_speak2(t_eot, session_id="s1")
                row.update(first_audio_ms=fa_ms,
                           spoken_s=round(n_samp / 24000, 2))
            else:
                ans = gen_text(session_id="s1")
            row.update(mode="local", answer=ans,
                       answer_ms=int((time.time() - t_eot) * 1000))
        else:
            expert = {}

            def expert_call(query_text):
                t1 = time.time()
                r = escalate.ask_expert(query_text, effort="low",
                                        cache_dir=EXPERT_CACHE)
                expert["answer"] = (r.get("answer")
                                    or f"[error: {r.get('error')}]")
                expert["cached_latency_s"] = r.get("latency_s")
                expert["wall_s"] = time.time() - t1

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
            if tts_answers:
                relay, r_fa, r_samp = gen_speak2(t_expert, session_id="s1")
                row.update(first_audio_ms=int((t_stall - t_eot) * 1000),
                           stall_pcm_s=stall_pcm_s,
                           relay_first_audio_ms=r_fa,
                           spoken_s=round(r_samp / 24000, 2))
            else:
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
              f"{'ESC' if fired else 'local':>5s} conc={n_concurrent}"
              f"/{len(chunks)} | "
              f"{(row.get('relay') or row.get('answer', ''))[:55]!r}",
              flush=True)

    base = (f"{DATA}/{bench}_conclivetts_traces.jsonl" if tts_answers
            else f"{DATA}/{bench}_conclive_traces.jsonl")
    out = (f"{base}.{tier}.smoke" if shard_id < 0
           else f"{base}.{tier}.shard{shard_id}")
    with open(out, "a", encoding="utf-8") as fh:
        for r in traces:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> appended {len(traces)} traces to {out}", flush=True)
    return [r["id"] for r in traces]


@gen_app.function(image=util_cc, volumes={DATA: gate_data},
                  timeout=60 * 5)
def _read_pool(bench: str) -> list:
    qs = _read_jsonl(CPOOLS[bench]["q"])
    sp = CPOOLS[bench]["split"]
    return [q for q in qs if sp is None or q.get("split") == sp]


@gen_app.local_entrypoint()
def run_conclive(tier: str = "balanced", workers: int = 4, limit: int = 0,
                 bench: str = "frozen", thr_override: float = 0.0,
                 tts_answers: int = 0):
    split_qs = _read_pool.remote(bench)
    if limit:
        split_qs = split_qs[:limit]
        workers = 1
    shards = [split_qs[i::workers] for i in range(workers)]
    print(f">>> conclive tier={tier}: {len(split_qs)} queries, "
          f"{workers} workers")
    done = list(conclive_shard.starmap(
        [(shards[i], tier, i if not limit else -1, bench, thr_override,
          tts_answers) for i in range(workers)]))
    print(f">>> tier {tier} complete: {sum(len(d) for d in done)} traces")
