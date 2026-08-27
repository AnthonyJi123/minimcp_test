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
                       util_image, _read_jsonl)

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
                     tag: str = "") -> list:
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
        au, _ = librosa.load(f"{AUDIO_DIR}/{q['id']}.wav",
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
        traces.append({"id": q["id"], "pool": q["pool"],
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


@gen_app.local_entrypoint()
def run_concurrent(workers: int = 4, limit: int = 0, split: str = "test",
                   tag: str = ""):
    split_qs = _read_frozen.remote(split)
    if limit:
        split_qs = split_qs[:limit]
        workers = 1
    shards = [split_qs[i::workers] for i in range(workers)]
    print(f">>> concurrent arm: {len(split_qs)} queries, {workers} workers")
    done = list(concurrent_shard.starmap(
        [(shards[i], i if not limit else -1, tag)
         for i in range(workers)]))
    print(f">>> complete: {sum(len(d) for d in done)} traces")
