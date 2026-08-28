"""Duplex-validation sweep (reviewer #1 / todo P2): drive the DEPLOYED
gate-demo voice loop -- talker head ON, spoken output, soft barge-in --
with the frozen live-sweep pool (the 240 ids in frozen_v3_traces).

Arm 'clean':   one query per turn, talker answers, abort after first
               audible audio. Probe score + eot_read_ms + first_audio_ms
               under the native-talker loop.
Arm 'overlap': easy warmup question first; while the talker SPEAKS the
               answer, the target query is spoken OVER it -> duck ->
               interrupt -> the barge-in speech seeds the next turn
               (demo_app.py's duplex regime) -> EOT probe read.

probe_on=0: the eot score is computed and logged per turn but nothing
escalates (no expert RTT; pure gate-validity sweep).

Arm 'escalate' (P2(c) latency stats): probe_on=1 at the given tier;
fired turns run the real stall -> expert -> relay path and the client
records EOT->stall-audio and EOT->relay-audio wall clocks; unfired
turns record EOT->local-audio. This is the reviewer's
time-to-first-audible table.

Results append to /data/duplex_sweep/{arm}.jsonl (restartable: done ids
are skipped on rerun).

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _ws_duplex.py::sweep --arm clean --limit 3     # smoke
  modal run _ws_duplex.py::sweep --arm overlap --limit 2   # smoke
  modal run _ws_duplex.py::sweep --arm clean               # full
  modal run _ws_duplex.py::sweep --arm overlap             # full
"""
import json

import modal

app = modal.App("ws-duplex-sweep")
vol = modal.Volume.from_name("gate-data")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy",
                    "pandas", "pyarrow"))

BASE = "https://rhe9527--gate-demo-voice.modal.run/62dc5cd9"
WS = "wss://rhe9527--gate-demo-voice.modal.run/62dc5cd9/ws"
OUT_DIR = "/data/duplex_sweep"


@app.function(image=img, volumes={"/data": vol}, timeout=60 * 60 * 5)
async def sweep(arm: str = "clean", limit: int = 0,
                tier: str = "balanced"):
    import asyncio
    import os
    import time as _time
    import urllib.request

    import librosa
    import numpy as np
    import pandas as pd
    import websockets

    assert arm in ("clean", "overlap", "escalate"), arm
    probe_on = 1 if arm == "escalate" else 0
    url = f"{WS}?tier={tier}&probe_on={probe_on}"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{arm}.jsonl"
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            done = {json.loads(x)["id"] for x in fh if x.strip()}

    tr = pd.read_parquet("/data/frozen_v3_traces.parquet")
    ids = sorted(tr["id"].unique())
    ids = [i for i in ids
           if os.path.exists(f"/data/audio_pool/{i}.wav")]
    # warmups (overlap arm): low score, answered locally + correctly,
    # LONG spoken answer (so there is something to barge into)
    loc = tr[tr["mode"] == "local"]
    agg = loc.groupby("id").agg(score=("eot_score", "mean"),
                                ans_ms=("answer_ms", "mean"),
                                ok=("heard_ok", "mean"),
                                aud=("audio_s", "mean")).reset_index()
    warm = agg[(agg["score"] < 0.25) & (agg["ans_ms"] > 5000)
               & (agg["ok"] > 0.5) & (agg["aud"] < 10)]
    warm = warm.sort_values("score")["id"].tolist()[:8]
    assert warm, "no warmup candidates"
    print(f">>> {len(ids)} pool ids, {len(done)} already done, "
          f"warmups={warm}")
    pending = [i for i in ids if i not in done]
    if limit:
        pending = pending[:limit]

    t0 = _time.time()
    r = None
    while _time.time() - t0 < 480:
        try:
            r = json.load(urllib.request.urlopen(f"{BASE}/ready",
                                                 timeout=25))
            break
        except Exception as e:
            print(f"  warm poll: {type(e).__name__}")
            await asyncio.sleep(4)
    assert r and r.get("ready"), "GPU never became ready"
    print(f"ready: {r}")

    rng = np.random.default_rng(11)
    FR = 2048

    def frames(au, gain=1.0):
        au = au * gain + rng.normal(0, 0.008, len(au))
        i16 = (au * 32767).clip(-32767, 32767).astype(np.int16)
        return [i16[i:i + FR].tobytes() for i in range(0, len(i16), FR)]

    def noise_frame():
        return ((rng.normal(0, 0.008, FR) * 32767)
                .clip(-32767, 32767).astype(np.int16).tobytes())

    def load(qid):
        au, _ = librosa.load(f"/data/audio_pool/{qid}.wav", sr=16000,
                             mono=True)
        return au

    TAIL = np.zeros(int(1.5 * 16000))
    n_written = 0

    def reset_state(st):
        st.update(eots=[], turns=[], interrupts=[], resumes=[],
                  err=None, audio=0, audio_mark=None,
                  audio_walls=[], phases=[])

    async def run_batch(batch, base_done):
        nonlocal n_written
        async with websockets.connect(url, max_size=None,
                                      open_timeout=120) as ws:
            hello = json.loads(await ws.recv())
            if hello.get("type") == "error":
                raise RuntimeError(f"busy: {hello}")
            st = {}
            reset_state(st)

            async def rx():
                while True:
                    try:
                        m = json.loads(await ws.recv())
                    except Exception:
                        return
                    t = m.get("type")
                    if t == "audio":
                        st["audio"] += 1
                        st["audio_walls"].append(_time.time())
                        if st["audio_mark"] is None:
                            st["audio_mark"] = _time.time()
                        continue
                    if t == "phase":
                        st["phases"].append((m.get("v"), _time.time()))
                        continue
                    if t == "eot":
                        st["eots"].append((m, _time.time()))
                    elif t == "turn":
                        st["turns"].append(m)
                    elif t == "interrupt":
                        st["interrupts"].append(m)
                    elif t == "resume":
                        st["resumes"].append(m)
                    elif t == "error":
                        st["err"] = m

            rxt = asyncio.create_task(rx())

            async def stream(au, dt, gain=1.0):
                for f in frames(au, gain):
                    await ws.send(f)
                    await asyncio.sleep(dt)

            async def idle(cond, tmax):
                t1 = _time.time()
                while not cond() and _time.time() - t1 < tmax:
                    await ws.send(noise_frame())
                    await asyncio.sleep(0.128)
                return bool(cond())

            def save(rec):
                nonlocal n_written
                with open(out_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                n_written += 1
                if n_written % 8 == 0:
                    vol.commit()

            try:
                for k, qid in enumerate(batch):
                    t_q = _time.time()
                    if arm == "clean":
                        reset_state(st)
                        au = load(qid)
                        await stream(au, 0.03)
                        await stream(TAIL, 0.03)
                        if not await idle(lambda: st["eots"], 90):
                            print(f"  !! {qid}: no eot, skipping")
                            continue
                        eot, t_eot = st["eots"][-1]
                        st["audio_mark"] = None
                        got_audio = await idle(
                            lambda: st["audio_mark"], 25)
                        if not st["turns"]:
                            await ws.send(json.dumps({"type": "eot"}))
                        if not await idle(lambda: st["turns"], 90):
                            print(f"  !! {qid}: no turn record")
                            continue
                        turn = st["turns"][-1]
                        fa_ms = (int((st["audio_mark"] - t_eot) * 1000)
                                 if st["audio_mark"] else None)
                        rec = dict(
                            id=qid, arm=arm,
                            eot_score=eot["score"],
                            eot_read_ms=eot["ms"],
                            first_audio_client_ms=fa_ms,
                            scores=turn.get("scores"),
                            audio_s=turn.get("audio_s"),
                            first_audio_ms=turn.get("first_audio_ms"),
                            got_audio=bool(got_audio),
                            interrupted=turn.get("interrupted"),
                            wall_s=round(_time.time() - t_q, 1))
                        save(rec)
                        print(f"[{base_done + k + 1}] {qid} "
                              f"score={eot['score']:.3f} "
                              f"read={eot['ms']}ms "
                              f"first_audio={turn.get('first_audio_ms')}"
                              f" wall={rec['wall_s']}s", flush=True)
                    else:
                        wid = warm[k % len(warm)]
                        if wid == qid:
                            wid = warm[(k + 1) % len(warm)]
                        reset_state(st)
                        await stream(load(wid), 0.03)
                        await stream(TAIL, 0.03)
                        if not await idle(lambda: st["eots"], 90):
                            print(f"  !! {qid}: warmup no eot")
                            continue
                        st["audio_mark"] = None
                        if not await idle(lambda: st["audio_mark"], 40):
                            print(f"  !! {qid}: talker never spoke")
                            continue
                        await asyncio.sleep(0.3)
                        n_e, n_t = len(st["eots"]), len(st["turns"])
                        au = load(qid)
                        await stream(au, 0.06, gain=1.6)
                        await stream(TAIL, 0.06)
                        if not await idle(
                                lambda: len(st["eots"]) > n_e, 120):
                            print(f"  !! {qid}: no target eot")
                            continue
                        eot, t_eot = st["eots"][-1]
                        barged = len(st["interrupts"]) > 0
                        st["audio_mark"] = None
                        got_audio = await idle(
                            lambda: st["audio_mark"], 25)
                        if len(st["turns"]) < n_t + 2:
                            await ws.send(json.dumps({"type": "eot"}))
                        if not await idle(
                                lambda: len(st["turns"]) >= n_t + 2,
                                90):
                            print(f"  !! {qid}: no target turn")
                            continue
                        turn = st["turns"][-1]
                        fa_ms = (int((st["audio_mark"] - t_eot) * 1000)
                                 if st["audio_mark"] else None)
                        rec = dict(
                            id=qid, arm=arm, warmup=wid,
                            barged=barged,
                            n_resumes=len(st["resumes"]),
                            eot_score=eot["score"],
                            eot_read_ms=eot["ms"],
                            first_audio_client_ms=fa_ms,
                            scores=turn.get("scores"),
                            audio_s=turn.get("audio_s"),
                            first_audio_ms=turn.get("first_audio_ms"),
                            got_audio=bool(got_audio),
                            interrupted=turn.get("interrupted"),
                            wall_s=round(_time.time() - t_q, 1))
                        save(rec)
                        print(f"[{base_done + k + 1}] {qid} "
                              f"score={eot['score']:.3f} "
                              f"barged={barged} "
                              f"read={eot['ms']}ms "
                              f"wall={rec['wall_s']}s", flush=True)
            finally:
                rxt.cancel()

    B = 12
    i = 0
    while i < len(pending):
        batch = pending[i:i + B]
        try:
            await run_batch(batch, i)
            i += B
        except Exception as e:
            print(f"  !! batch error ({type(e).__name__}: "
                  f"{str(e)[:120]}), reconnecting in 8s", flush=True)
            with open(out_path) as fh:
                got = {json.loads(x)["id"] for x in fh if x.strip()}
            batch_left = [q for q in batch if q not in got]
            pending = pending[:i] + batch_left + pending[i + B:]
            await asyncio.sleep(8)
    vol.commit()
    with open(out_path) as fh:
        n = sum(1 for x in fh if x.strip())
    print(f">>> DONE arm={arm}: {n} records in {out_path}")
