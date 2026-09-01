"""Repeat-then-judge p(True) on the EXTERNAL audio pools (8bl GPU half).

5b/6a on record: verbalized self-eval collapses under audio input on the
deployed backbone (trap p_yes .055->.556), and repeat-then-judge on the
model's OWN transcript restores it (collect_asr — the internal 600 live
in asr_minicpm-o45-audio.shard*). This collects the same signal for the
five external pools, so probe (+) ptrue fusion can be scored where the
probe is weakest (native external mean .709).

Per query: wav -> verbatim transcript (ASR_INSTR) -> TEXT ptrue_pre on
the transcript -> p_yes (the deployable signal), plus ptrue_pre on the
ORIGINAL query text (text-ceiling diagnostic, one extra forward).

Outputs: /data/rtj_{pool}.shard{i}.parquet
         (id, transcript, p_yes_rtj, mass_rtj, p_yes_textq, mass_textq)

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run modal_rtj.py::run_rtj --pool striviaqa --limit 3   # smoke
  modal run modal_rtj.py::run_rtj --pool striviaqa --workers 2
  ... swebq / sllama / sdqa / sreason
"""
import json
import os
import sys

from modal_audio import (app, image_au, util_image_au, GPU_VOL, gate_data,
                         DATA, ASR_INSTR, _load_model_audio)

HERE = os.path.dirname(os.path.abspath(__file__))
_AUDIO_PY = os.path.join(HERE, "modal_audio.py")
image_rtj = image_au.add_local_file(_AUDIO_PY, "/root/modal_audio.py")
util_rtj = util_image_au.add_local_file(_AUDIO_PY, "/root/modal_audio.py")

POOLS = {
    "striviaqa": (f"{DATA}/queries_striviaqa.jsonl", f"{DATA}/bench_audio"),
    "swebq":     (f"{DATA}/queries_swebq.jsonl",     f"{DATA}/bench_audio"),
    "sllama":    (f"{DATA}/queries_sllama.jsonl",    f"{DATA}/bench_audio"),
    "sdqa":      (f"{DATA}/queries_sdqa.jsonl",      f"{DATA}/sdqa_audio"),
    "sreason":   (f"{DATA}/queries_sreason.jsonl",   f"{DATA}/bench_audio"),
}


@app.function(image=util_rtj, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_pool(pool: str) -> list:
    qfile, _ = POOLS[pool]
    return [json.loads(x) for x in open(qfile, encoding="utf-8")
            if x.strip()]


@app.function(image=image_rtj, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 2)
def collect_rtj(shard: list, shard_id: int, pool: str,
                audio_dir: str) -> int:
    """collect_asr (6a) verbatim, generalized over pools, + the
    original-text arm."""
    import librosa
    import pandas as pd
    import torch
    sys.path.insert(0, "/workspace/gate")
    import decode
    from modal_app import PTRUE_PRE

    model, tok = _load_model_audio()
    kw = decode._chat_kwargs(model, tok)

    def tok_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        return sorted(ids)

    YES = tok_ids(["Yes", "yes", "YES", " Yes", " yes", "是", "能", "对", "会"])
    NO = tok_ids(["No", "no", "NO", " No", " no", "否", "不", "错"])

    def p_yes(text):
        logits = decode.first_token_logits(model, tok,
                                           PTRUE_PRE.format(q=text))
        p = torch.softmax(logits, dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        return (py / (py + pn) if (py + pn) > 0 else 0.5), py + pn

    print(f">>> rtj[{pool}] shard {shard_id}: {len(shard)} queries",
          flush=True)
    rows = []
    for k, q in enumerate(shard):
        wav_p = f"{audio_dir}/{q['id']}.wav"
        if not os.path.exists(wav_p):
            continue
        au, _ = librosa.load(wav_p, sr=16000, mono=True)
        asr = model.chat(msgs=[{"role": "user", "content": [au, ASR_INSTR]}],
                         max_new_tokens=512, **kw)
        asr = asr.strip() if isinstance(asr, str) else str(asr)
        p_rtj, m_rtj = p_yes(asr)
        p_txt, m_txt = p_yes(str(q["query"]))
        rows.append({"id": q["id"], "transcript": asr,
                     "p_yes_rtj": p_rtj, "mass_rtj": m_rtj,
                     "p_yes_textq": p_txt, "mass_textq": m_txt})
        if k < 3 or k % 50 == 0:
            print(f"  [{k}] rtj={p_rtj:.3f} txt={p_txt:.3f} "
                  f":: {asr[:70]!r}", flush=True)
    pd.DataFrame(rows).to_parquet(
        f"{DATA}/rtj_{pool}.shard{shard_id}.parquet")
    gate_data.commit()
    print(f">>> wrote rtj_{pool} shard {shard_id} ({len(rows)})",
          flush=True)
    return len(rows)


@app.local_entrypoint()
def run_rtj(pool: str, workers: int = 2, limit: int = 0):
    qfile, audio_dir = POOLS[pool]
    qs = _read_pool.remote(pool)
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> rtj [{pool}]: {len(qs)} queries / {workers} workers")
    total = sum(collect_rtj.starmap(
        [(shards[i], i, pool, audio_dir) for i in range(workers)]))
    print(f">>> collected repeat-then-judge ptrue for {total} queries")
