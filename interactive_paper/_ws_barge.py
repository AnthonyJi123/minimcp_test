"""Barge-in + backchannel regression (mic-shaped, like _ws_test.py).

Scenario A: while the talker SPEAKS, say "Okay." -> duck then RESUME
            (backchannel keeps the floor), answer completes uninterrupted.
Scenario B: next turn, while the talker speaks, say "Stop!" -> duck then
            INTERRUPT (semantic commit), turn marked interrupted.

Run: modal run _ws_barge.py::t
"""
import json

import modal

from modal_app import OPENAI

app = modal.App("ws-barge-test")
vol = modal.Volume.from_name("gate-data")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy",
                    "openai")
       .add_local_file("modal_app.py", "/root/modal_app.py"))


@app.function(image=img, volumes={"/data": vol}, secrets=[OPENAI],
              timeout=60 * 25)
async def t():
    import asyncio
    import io
    import time as _time
    import urllib.request

    import librosa
    import numpy as np
    import websockets
    from openai import OpenAI

    url = ("wss://rhe9527--gate-demo-voice.modal.run/62dc5cd9/ws"
           "?tier=balanced&probe_on=1")
    W1 = "/data/sdqa_audio/sdqa0003.wav"   # question 1
    W2 = "/data/sdqa_audio/sdqa0199.wav"   # question 2

    cl = OpenAI()

    def say(text):
        r = cl.audio.speech.create(model="tts-1", voice="onyx",
                                   input=text, response_format="wav")
        au, _ = librosa.load(io.BytesIO(r.content), sr=16000, mono=True)
        return au

    ok_au = say("Okay.")
    stop_au = say("Stop!")
    print(f"backchannel burst {len(ok_au)/16000:.2f}s, "
          f"stop burst {len(stop_au)/16000:.2f}s")

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
        state = {"audio": 0, "ducks": [], "resumes": [], "interrupts": [],
                 "turns": []}

        async def rx():
            while True:
                try:
                    m = json.loads(await ws.recv())
                except Exception:
                    return
                if m["type"] == "audio":
                    state["audio"] += 1
                    continue
                if m["type"] not in ("vu", "score"):
                    print(f"  << {m['type']}: {json.dumps(m)[:130]}")
                if m["type"] == "duck":
                    state["ducks"].append(m)
                if m["type"] == "resume":
                    state["resumes"].append(m)
                if m["type"] == "interrupt":
                    state["interrupts"].append(m)
                if m["type"] == "turn":
                    state["turns"].append(m)

        rxt = asyncio.create_task(rx())

        async def stream(au, gain=1.0):
            for f in frames(au, gain):
                await ws.send(f)
                await asyncio.sleep(0.06)

        async def idle(cond, tmax):
            t1 = _time.time()
            while not cond() and _time.time() - t1 < tmax:
                await ws.send(noise_frame())
                await asyncio.sleep(0.128)

        # ---------------- scenario A: backchannel ----------------
        print(">> turn 1: question, then 'Okay.' while it talks")
        await stream(au1)
        await idle(lambda: False, 1.8)
        base_audio = state["audio"]
        await idle(lambda: state["audio"] >= base_audio + 2, 120)
        assert state["audio"] >= base_audio + 2, "talker never spoke (A)"
        await stream(ok_au, gain=1.8)
        await idle(lambda: state["resumes"] or state["interrupts"], 12)
        assert state["ducks"], "no duck event for the backchannel"
        assert state["resumes"], (
            f"backchannel did not resume: {state['interrupts']}")
        assert not state["interrupts"], "backchannel wrongly interrupted!"
        await idle(lambda: state["turns"], 90)
        assert state["turns"] and not state["turns"][0].get(
            "interrupted"), "turn 1 should complete uninterrupted"
        print(f"== A PASS: duck -> resume "
              f"(heard {state['resumes'][0].get('heard')!r}), turn "
              f"completed, answer[:60]="
              f"{str(state['turns'][0].get('answer'))[:60]!r}")

        # ---------------- scenario B: STOP ----------------
        print(">> turn 2: question, then 'Stop!' while it talks")
        await stream(au2)
        await idle(lambda: False, 1.8)
        base_audio = state["audio"]
        await idle(lambda: state["audio"] >= base_audio + 2, 150)
        assert state["audio"] >= base_audio + 2, "talker never spoke (B)"
        await stream(stop_au, gain=1.8)
        await idle(lambda: state["interrupts"], 12)
        assert state["interrupts"], "'Stop!' did not interrupt"
        await idle(lambda: len(state["turns"]) >= 2, 30)
        assert len(state["turns"]) >= 2 and \
            state["turns"][1].get("interrupted"), \
            "turn 2 not marked interrupted"
        print(f"== B PASS: interrupt "
              f"(heard {state['interrupts'][0].get('heard')!r}), "
              f"turn 2 interrupted")

        rxt.cancel()

    print("\nBACKCHANNEL + BARGE-IN TEST PASSED")
