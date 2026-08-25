"""Barge-in regression test (mic-shaped, like _ws_test.py): stream a
question, wait for the talker to start SPEAKING (audio events), then talk
over it. Expect: interrupt event -> turn marked interrupted -> the
interrupting speech becomes the next turn and gets answered.

Run: modal run _ws_barge.py::t
"""
import json

import modal

app = modal.App("ws-barge-test")
vol = modal.Volume.from_name("gate-data")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy"))


@app.function(image=img, volumes={"/data": vol}, timeout=60 * 20)
async def t():
    import asyncio
    import time as _time
    import urllib.request

    import librosa
    import numpy as np
    import websockets

    url = ("wss://rhe9527--gate-demo-voice.modal.run/62dc5cd9/ws"
           "?tier=balanced&probe_on=1")
    W1 = "/data/sdqa_audio/sdqa0003.wav"   # short clean question
    W2 = "/data/sdqa_audio/sdqa0199.wav"   # different speech = the barge-in

    t0 = _time.time()
    r = None
    while _time.time() - t0 < 480:
        try:
            r = json.load(urllib.request.urlopen(
                "https://rhe9527--gate-demo-voice.modal.run/62dc5cd9/ready",
                timeout=25))
            break
        except Exception as e:
            print(f"  warm poll ({_time.time()-t0:.0f}s): "
                  f"{type(e).__name__}")
            await asyncio.sleep(4)
    assert r and r.get("ready"), "GPU never became ready"
    print(f"ready: {r}")

    rng = np.random.default_rng(7)
    FR = 2048

    def frames(au, gain=1.0):
        au = au * gain + rng.normal(0, 0.008, len(au))
        i16 = (au * 32767).clip(-32767, 32767).astype(np.int16)
        return [i16[i:i + FR].tobytes() for i in range(0, len(i16), FR)]

    def noise_frame():
        return ((rng.normal(0, 0.008, FR) * 32767)
                .clip(-32767, 32767).astype(np.int16).tobytes())

    au1, _ = librosa.load(W1, sr=16000, mono=True)
    au2, _ = librosa.load(W2, sr=16000, mono=True)

    async with websockets.connect(url, max_size=None,
                                  open_timeout=120) as ws:
        print("hello:", json.loads(await ws.recv()))
        msgs = []
        state = {"audio": 0, "interrupt": None, "turns": []}

        async def rx():
            while True:
                try:
                    m = json.loads(await ws.recv())
                except Exception:
                    return
                msgs.append(m)
                if m["type"] == "audio":
                    state["audio"] += 1
                    if state["audio"] % 3 == 1:
                        print(f"  << audio chunk #{state['audio']}")
                    continue
                if m["type"] not in ("vu", "score"):
                    print(f"  << {m['type']}: {json.dumps(m)[:140]}")
                if m["type"] == "interrupt":
                    state["interrupt"] = m
                if m["type"] == "turn":
                    state["turns"].append(m)

        rxt = asyncio.create_task(rx())

        print(">> turn 1: streaming the question")
        for f in frames(au1):
            await ws.send(f)
            await asyncio.sleep(0.06)
        for _ in range(14):                      # 1.8 s tail silence
            await ws.send(noise_frame())
            await asyncio.sleep(0.128)

        # keep the mic running (noise) until the talker starts speaking
        t1 = _time.time()
        while state["audio"] < 2 and _time.time() - t1 < 120:
            await ws.send(noise_frame())
            await asyncio.sleep(0.128)
        assert state["audio"] >= 2, "talker never started speaking"

        print(">> BARGE-IN: talking over the talker")
        for f in frames(au2, gain=2.0):          # loud, like a real barge-in
            await ws.send(f)
            await asyncio.sleep(0.06)
            if state["interrupt"] and state["turns"]:
                break
        for _ in range(14):
            await ws.send(noise_frame())
            await asyncio.sleep(0.128)

        # wait for the second turn (the barge-in speech answered)
        t2 = _time.time()
        while len(state["turns"]) < 2 and _time.time() - t2 < 150:
            await ws.send(noise_frame())
            await asyncio.sleep(0.128)

        rxt.cancel()

    assert state["interrupt"] is not None, "no interrupt event"
    assert state["turns"], "no turn at all"
    t1r = state["turns"][0]
    print(f"\n== turn 1: mode={t1r.get('mode')} "
          f"interrupted={t1r.get('interrupted')} "
          f"answer[:80]={str(t1r.get('answer'))[:80]!r}")
    assert t1r.get("interrupted"), "turn 1 not marked interrupted"
    assert len(state["turns"]) >= 2, "barge-in speech never became turn 2"
    t2r = state["turns"][1]
    print(f"== turn 2: mode={t2r.get('mode')} fired={t2r.get('fired')} "
          f"score={t2r.get('eot_score')} "
          f"answer[:100]={str(t2r.get('answer'))[:100]!r}")
    print("\nBARGE-IN TEST PASSED")
