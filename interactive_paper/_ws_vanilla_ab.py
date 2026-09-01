"""Scripted A/B on the VANILLA demo (8bl): counting carrier + "stop"
mid-answer, cfg=official vs cfg=ours, n each. Measures chunks from the
stop's first frame to turn end — the same scenario the user hit.

Run: modal run _ws_vanilla_ab.py::ab --n 3
"""
import json

import modal

from modal_app import OPENAI

app = modal.App("ws-vanilla-ab")
gate_data = modal.Volume.from_name("gate-data")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy",
                    "openai")
       .add_local_dir("src", "/workspace/gate")
       .add_local_file("modal_app.py", "/root/modal_app.py"))

WS = "wss://rhe9527--vanilla-duplex-voice.modal.run/62dc5cd9/ws"
BASE = "https://rhe9527--vanilla-duplex-voice.modal.run/62dc5cd9"


@app.function(image=img, volumes={"/data": gate_data},
              secrets=[OPENAI], timeout=60 * 40)
async def ab(n: int = 3, stims: str = "stop"):
    import asyncio
    import sys
    import time as _t
    import urllib.request

    import librosa
    import numpy as np
    import websockets
    sys.path.insert(0, "/workspace/gate")
    import escalate

    r = escalate._client().audio.speech.create(
        model="tts-1", voice="alloy",
        input="Please count slowly from one to thirty.",
        response_format="wav")
    open("/tmp/c.wav", "wb").write(r.content)
    car, _ = librosa.load("/tmp/c.wav", sr=16000, mono=True)
    STIM_PATHS = {"stop": "/data/floor_sweep/stim/stop1.wav",
                  "bc": "/data/floor_sweep/stim/bcs0.wav",
                  "bq": "/data/audio_pool/q0461.wav"}
    stim_bank = {}
    for k in stims.split(","):
        au_, _ = librosa.load(STIM_PATHS[k], sr=16000, mono=True)
        stim_bank[k] = au_

    t0 = _t.time()
    ready = None
    while _t.time() - t0 < 480:
        try:
            ready = json.load(urllib.request.urlopen(f"{BASE}/ready",
                                                     timeout=25))
            break
        except Exception:
            await asyncio.sleep(4)
    assert ready and ready.get("ready")

    FR = 2048
    results = []
    for cfg in ("official",):
      for sk, stop_au in stim_bank.items():
        for trial in range(n):
            ev = {"first_speak": None, "stop_sent": None,
                  "turn_end": None, "text": []}
            async with websockets.connect(
                    f"{WS}?cfg={cfg}", max_size=2 ** 24,
                    open_timeout=90) as sock:
                t0s = _t.time()

                async def rd():
                    async for m in sock:
                        e = json.loads(m)
                        t = _t.time() - t0s
                        if e.get("type") == "text":
                            ev["text"].append(e["v"])
                            if ev["first_speak"] is None:
                                ev["first_speak"] = t
                        if e.get("type") == "chunk" and e.get("eot") \
                                and ev["first_speak"] is not None \
                                and ev["turn_end"] is None:
                            ev["turn_end"] = t
                rt = asyncio.create_task(rd())

                async def send_au(au):
                    i16 = (au * 32767).clip(-32767, 32767).astype(
                        np.int16)
                    for i in range(0, len(i16), FR):
                        await sock.send(i16[i:i + FR].tobytes())
                        await asyncio.sleep(FR / 16000)

                rng = np.random.default_rng(7)

                async def sil(sec):
                    for _ in range(int(sec * 16000 / FR)):
                        await sock.send(
                            (rng.normal(0, 0.003, FR) * 32767)
                            .clip(-32767, 32767).astype(np.int16)
                            .tobytes())
                        await asyncio.sleep(FR / 16000)

                await send_au(car)
                # wait for speech, then 2 s in, say "stop"
                t1 = _t.time()
                while ev["first_speak"] is None and \
                        _t.time() - t1 < 20:
                    await sil(0.5)
                await sil(2.0)
                ev["stop_sent"] = _t.time() - t0s
                await send_au(stop_au)
                t2 = _t.time()
                while ev["turn_end"] is None and _t.time() - t2 < 40:
                    await sil(0.5)
                try:
                    await sock.send(json.dumps({"type": "stop"}))
                except Exception:
                    pass
                await asyncio.sleep(0.5)
                rt.cancel()
            post = (ev["turn_end"] - ev["stop_sent"]
                    if ev["turn_end"] and ev["stop_sent"] else None)
            txt = "".join(ev["text"])
            results.append({"cfg": cfg, "stim": sk, "trial": trial,
                            "post_stop_s": post,
                            "n_words": len(txt.split()),
                            "tail": txt[-80:]})
            print(f"[{cfg}:{sk} #{trial}] stop@{ev['stop_sent']:.1f}s "
                  f"end@{ev['turn_end'] and round(ev['turn_end'], 1)} "
                  f"post={post and round(post, 1)}s "
                  f"tail={txt[-60:]!r}", flush=True)

    print("\n===== VANILLA A/B =====")
    import numpy as np
    for sk in stim_bank:
        ps = [r["post_stop_s"] for r in results
              if r["stim"] == sk and r["post_stop_s"] is not None]
        nheld = sum(1 for r in results
                    if r["stim"] == sk and r["post_stop_s"] is None)
        print(f"official:{sk}: post-stim latency "
              f"{[round(p, 1) for p in ps]} "
              f"med={round(float(np.median(ps)), 1) if ps else None} "
              f"no-end-within-40s={nheld}")
    return results
