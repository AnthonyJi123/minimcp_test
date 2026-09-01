"""Native floor-control sweep (8bf): is the talker's own barge-in vs
backchannel discrimination intact under the escalation machinery?

The 8ba question re-asked on the REAL duplex head. In the native design
the gate is a read-only probe — the only interventions are the stall
note and the relay unit — so the risk surface is exactly the escalated
phases. Cells:

  arm ans   -- carrier = non-firing long-answer query; stim overlaps the
               talker's LOCAL answer speech. Native baseline: bcs/bcl
               (backchannels) should HOLD the floor, stop/bq should
               YIELD it. Gate reads but never fires: this cell is
               definitionally gate-independent (read-only hook).
  arm esc   -- carrier = firing query (native probe >= balanced thr).
               Stall note + fake thinker (deterministic 6-chunk wait,
               canned answer) + relay unit, mirroring demo_duplex.py.
               stim phase 'wait'  = during the thinker-wait silence
               stim phase 'relay' = overlapping the relay speech
               bcs should not derail the wait/relay; stop/bq should
               yield the relay floor like any speech.

Per trial: chunk pattern, stim onset, yield latency (chunks from stim
onset to turn_eos), texts, relay_done. Restartable jsonl.

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run modal_native_floor.py::run_floor --arm ans --limit 2
  modal run modal_native_floor.py::run_floor --arm ans --workers 4
  modal run modal_native_floor.py::run_floor --arm esc --workers 4
"""
import json
import os
import time

import modal

app = modal.App("native-floor")
gate_data = modal.Volume.from_name("gate-data")
weights = modal.Volume.from_name("minicpm-o45-weights")
DATA = "/data"
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"
PROMPT_WAV = f"{MODEL_DIR}/assets/system_ref_audio.wav"
LAYER = 22
K3 = 8
STIM_DIR = f"{DATA}/floor_sweep/stim"
OUT_DIR = f"{DATA}/native_floor2"   # 8bl: official-config rerun
ART = f"{DATA}/gate_native.json"     # written by scripts/22
WAIT_CHUNKS = 6                      # fake-thinker wait (deterministic)
YIELD_K = 6                          # yield window after stim onset
FAKE_ANSWER = ("The verified answer is: the correct value is "
               "forty-two, according to the official record.")

RELAY_TMPL = ("A verified answer came back: {ans}\n"
              "Relay it to the user in one or two spoken sentences.")
RELAY_NUDGE = "Say the verified answer aloud to the user now."
STALL = "Hmm, let me double-check that — one moment."
STALL_NOTE = ("[SYSTEM NOTE] Your answer so far is likely wrong. You "
              "just told the user: \"" + STALL + "\" A verified answer "
              "will arrive in a moment.")

from modal_app import OPENAI  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(_HERE, "modal_app.py")

util_img = (modal.Image.debian_slim(python_version="3.11")
            .pip_install("pandas", "pyarrow")
            .add_local_file(_APP_PY, "/root/modal_app.py"))
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsndfile1")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install(
        "minicpmo-utils[all]",
        "transformers==4.51.0",
        "accelerate==1.12.0",
        "setuptools<81",
        "pydantic>=2.11",
        "PyYAML",
        "soundfile",
        "opencv-python-headless",
        "huggingface_hub[hf_transfer]",
        "scikit-learn",
        "pandas",
        "pyarrow",
        "openai",
        "sentencepiece",
        "fastapi[standard]",   # layer-hash parity with demo images
    )
    .add_local_dir(os.path.join(_HERE, "src"), "/workspace/gate")
    .add_local_file(_APP_PY, "/root/modal_app.py"))


@app.function(image=gpu_image, gpu="H100",
              volumes={"/workspace/models": weights, DATA: gate_data},
              secrets=[OPENAI], timeout=60 * 60 * 4)
def floor_shard(trials: list, shard_id: int = -1) -> list:
    import glob as _glob
    import shutil

    import librosa
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

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
    _ = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    duplex = model.as_duplex(generate_audio=False)
    duplex.force_listen_count = 3          # 8bl official config
    ref, _sr = librosa.load(PROMPT_WAV, sr=16000, mono=True)

    art = json.load(open(ART))
    w = np.array(art["w"], dtype=np.float32)
    b = float(art["b"])
    thr = float(art["eot_thresholds"]["balanced"])

    st3 = {"accum": False, "tail": None, "sum": None, "cnt": 0}

    def hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        h = hs[0].detach().float()
        t = h[-K3:].cpu()
        st3["tail"] = (t if st3["tail"] is None
                       else torch.cat([st3["tail"], t])[-K3:])
        if st3["accum"]:
            sm = h.sum(0).cpu()
            st3["sum"] = sm if st3["sum"] is None else st3["sum"] + sm
            st3["cnt"] += h.shape[0]
    hh = model.llm.model.layers[LAYER].register_forward_hook(hook)

    def score_now():
        parts = [st3["tail"][-1], st3["tail"].mean(0),
                 st3["sum"] / max(1, st3["cnt"])]
        v = torch.cat(parts).numpy()
        return float(1.0 / (1.0 + np.exp(-(float(v @ w) + b))))

    rng = np.random.default_rng(5)

    def sil():
        return rng.normal(0, 0.003, 16000).astype(np.float32)

    def load16(path):
        au, _s = librosa.load(path, sr=16000, mono=True)
        return au.astype(np.float32)

    def to_chunks(au):
        cs = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        return [np.pad(c, (0, 16000 - len(c)))
                if len(c) < 16000 else c for c in cs]

    def gen_unit(text=None):
        if text is not None:
            duplex.streaming_prefill(text_list=[text])
        return duplex.streaming_generate(top_k=20)

    results = []
    for ti, t in enumerate(trials):
        carrier = to_chunks(load16(f"{DATA}/audio_pool/{t['qid']}.wav"))
        stim = to_chunks(load16(f"{STIM_DIR}/{t['stim']}.wav")
                         if not t["stim"].startswith("q")
                         else load16(f"{DATA}/audio_pool/{t['stim']}.wav"))

        duplex.prepare(
            prefix_system_prompt="You are a friendly assistant.",
            ref_audio=ref, prompt_wav_path=None)
        st3.update(tail=None, sum=None, cnt=0, accum=False)

        rec = dict(t, pattern="", onset=None, score=None, fired=None,
                   stim_at=None, yield_at=None, relay_done=False,
                   texts=[], outcome="?")
        feed = list(carrier)
        prev_listen = True
        speak_seen = 0
        relay_active = False
        stall_done, waited, relay_started = False, 0, False
        ci = -1
        try:
            while ci < 120:
                ci += 1
                ch = feed.pop(0) if feed else sil()
                st3["accum"] = True
                ok = duplex.streaming_prefill(audio_waveform=ch)
                st3["accum"] = False
                if not ok.get("success"):
                    rec["pattern"] += "x"
                    continue
                r = duplex.streaming_generate(top_k=20)
                rec["pattern"] += "L" if r["is_listen"] else "S"
                if r.get("text"):
                    rec["texts"].append(r["text"])

                onset_now = prev_listen and not r["is_listen"]
                if onset_now and rec["onset"] is None:
                    rec["onset"] = ci
                    sc = score_now()
                    rec["score"] = round(sc, 4)
                    rec["fired"] = bool(t["arm"] == "esc" and sc >= thr)

                if not r["is_listen"]:
                    speak_seen += 1

                # --- ans arm: stim over the local answer speech --------
                if (t["arm"] == "ans" and rec["stim_at"] is None
                        and speak_seen == 2):
                    feed = stim + feed
                    rec["stim_at"] = ci + 1

                # --- esc arm: stall -> wait -> relay -------------------
                if t["arm"] == "esc" and rec["fired"] and not stall_done:
                    stall_done = True
                    r2 = gen_unit(STALL_NOTE)
                    rec["pattern"] += "l" if r2["is_listen"] else "s"
                    if r2.get("text"):
                        rec["texts"].append(r2["text"])
                    prev_listen = r2["is_listen"]
                    continue
                if t["arm"] == "esc" and stall_done and not relay_started:
                    if r["is_listen"]:
                        waited += 1
                    if (t["phase"] == "wait" and rec["stim_at"] is None
                            and waited == 2):
                        feed = stim + feed
                        rec["stim_at"] = ci + 1
                    if waited >= WAIT_CHUNKS:
                        relay_started = True
                        r2 = gen_unit(RELAY_TMPL.format(ans=FAKE_ANSWER))
                        if not r2.get("text"):
                            r2 = gen_unit(RELAY_NUDGE)
                        rec["pattern"] += "l" if r2["is_listen"] else "s"
                        if r2.get("text"):
                            rec["texts"].append(r2["text"])
                        relay_active = not r2["is_listen"]
                        if (t["phase"] == "relay"
                                and rec["stim_at"] is None):
                            feed = stim + feed
                            rec["stim_at"] = ci + 1
                        prev_listen = r2["is_listen"]
                        continue

                if r.get("end_of_turn"):
                    if relay_started:
                        rec["relay_done"] = True
                    if (rec["stim_at"] is not None
                            and rec["yield_at"] is None
                            and ci >= rec["stim_at"]):
                        rec["yield_at"] = ci
                    if rec["stim_at"] is not None or (
                            t["arm"] == "ans" and speak_seen > 0):
                        if (relay_started or t["arm"] == "ans"
                                or t["phase"] == "wait"):
                            # run a short tail then stop
                            if not feed:
                                break
                if (rec["stim_at"] is not None
                        and ci - rec["stim_at"] > YIELD_K + 8
                        and not feed):
                    break
                prev_listen = r["is_listen"]
        except Exception as e:
            rec["outcome"] = f"error:{str(e)[:80]}"
        finally:
            pass

        if not rec["outcome"].startswith("error"):
            if rec["stim_at"] is None:
                rec["outcome"] = "no_stim"
            elif t["stim"].startswith(("bcs", "bcl")):
                held = (rec["yield_at"] is None
                        or rec["yield_at"] - rec["stim_at"] > YIELD_K)
                rec["outcome"] = "held" if held else "false_stop"
            else:
                yielded = (rec["yield_at"] is not None
                           and rec["yield_at"] - rec["stim_at"] <= YIELD_K)
                rec["outcome"] = ("yielded" if yielded else "held_floor")
        rec["texts"] = "".join(rec["texts"])[-400:]
        results.append(rec)
        print(f"  [{ti}] {t['pair_id']} onset={rec['onset']} "
              f"score={rec['score']} fired={rec['fired']} "
              f"stim@{rec['stim_at']} yield@{rec['yield_at']} "
              f"-> {rec['outcome']}", flush=True)

    hh.remove()
    os.makedirs(OUT_DIR, exist_ok=True)
    sfx = "smoke" if shard_id < 0 else f"shard{max(shard_id, 0)}"
    with open(f"{OUT_DIR}/floor.jsonl.{sfx}", "a", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    return [r["pair_id"] for r in results]


@app.function(image=util_img, volumes={DATA: gate_data}, timeout=60 * 5)
def build_trials(arm: str, limit: int = 0) -> list:
    import glob as _glob

    import numpy as np
    import pandas as pd
    tr = pd.read_parquet(f"{DATA}/frozen_v3_traces.parquet")
    loc = (tr[tr["mode"] == "local"].groupby("id")
           .agg(score=("eot_score", "mean"), ok=("heard_ok", "mean"),
                aud=("audio_s", "mean"), ans_ms=("answer_ms", "mean"))
           .reset_index())
    have = {os.path.basename(p)[:-4] for p in
            _glob.glob(f"{DATA}/audio_pool/*.wav")}
    long_local = loc[(loc["score"] < 0.2) & (loc["ok"] > 0.5)
                     & (loc["ans_ms"] > 4000)]
    long_local = [i for i in long_local.sort_values("score")["id"]
                  if i in have]
    esc = (tr[tr["mode"] == "escalated"].groupby("id")
           .agg(score=("eot_score", "mean")).reset_index()
           .sort_values("score", ascending=False))
    firing = [i for i in esc["id"] if i in have]
    bq = [i for i in loc.sort_values("aud")["id"] if i in have][:20]

    stims = {"bcs": [f"bcs{i}" for i in range(5)],
             "bcl": [f"bcl{i}" for i in range(4)],
             "stop": [f"stop{i}" for i in range(3)],
             "bq": bq}
    rng = np.random.default_rng(17)
    trials = []
    if arm == "ans":
        n = limit or 12
        for kind in ("bcs", "bcl", "stop", "bq"):
            for j in range(n):
                qid = long_local[j % len(long_local)]
                stim = stims[kind][int(rng.integers(len(stims[kind])))]
                trials.append({"arm": "ans", "phase": "ans",
                               "qid": qid, "stim": stim, "kind": kind,
                               "pair_id": f"ans:{kind}:{qid}:{stim}"})
    else:
        n = limit or 10
        for phase in ("wait", "relay"):
            for kind in ("bcs", "stop", "bq"):
                for j in range(n):
                    qid = firing[j % len(firing)]
                    stim = stims[kind][int(rng.integers(
                        len(stims[kind])))]
                    trials.append({"arm": "esc", "phase": phase,
                                   "qid": qid, "stim": stim,
                                   "kind": kind,
                                   "pair_id":
                                   f"esc:{phase}:{kind}:{qid}:{stim}"})
    done = set()
    for p in _glob.glob(f"{OUT_DIR}/floor.jsonl.shard*"):
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                done.add(json.loads(ln)["pair_id"])
    trials = [t for t in trials if t["pair_id"] not in done]
    return trials


@app.local_entrypoint()
def run_floor(arm: str = "ans", workers: int = 4, limit: int = 0):
    trials = build_trials.remote(arm, limit)
    if limit:
        trials = trials[:limit * (4 if arm == "ans" else 6)]
        workers = 1
    shards = [trials[i::workers] for i in range(workers)]
    print(f">>> native floor [{arm}]: {len(trials)} trials, "
          f"{workers} workers")
    done = list(floor_shard.starmap(
        [(shards[i], i if not limit else -1) for i in range(workers)]))
    print(f">>> complete: {sum(len(d) for d in done)} trials")
