"""End-to-end test of the voice WebSocket: stream real human speech as
int16 PCM frames (browser-shaped), plus trailing silence to trigger the
VAD end-of-turn. Two turns in one session to verify the reset loop."""
import json

import modal

app = modal.App("ws-test")
vol = modal.Volume.from_name("gate-data")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy"))


@app.function(image=img, volumes={"/data": vol}, timeout=60 * 20)
async def t():
    import asyncio
    import numpy as np
    import librosa
    import websockets

    url = ("wss://rhe9527--gate-demo-voice.modal.run/62dc5cd9/ws"
           "?tier=balanced&probe_on=1")
    wavs = ["/data/sdqa_audio/sdqa0003.wav",      # easy human speech
            "/data/audio_pool/q0225.wav"]         # hard MCQ -> escalates

    # same order as the browser: /ready first (HTTP survives the cold
    # start), only then the WS upgrade — against a warm container
    import time as _time
    import urllib.request
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
    print(f"ready: {r} (waited {_time.time()-t0:.0f}s)")
    async with websockets.connect(url, max_size=None,
                                  open_timeout=120) as ws:
        hello = json.loads(await ws.recv())
        print("hello:", hello)

        async def drain(timeout):
            msgs = []
            try:
                while True:
                    m = json.loads(await asyncio.wait_for(ws.recv(),
                                                          timeout))
                    msgs.append(m)
                    tag = {"score": f"score {m.get('v')}",
                           "turn": "TURN RESULT"}.get(m["type"], m["type"])
                    if m["type"] != "score" or m.get("i", 0) % 10 == 1:
                        print(f"  << {tag}: "
                              f"{json.dumps(m)[:160]}")
                    if m["type"] == "phase" and m["v"] == "listening" \
                            and any(x["type"] == "turn" for x in msgs):
                        break
            except asyncio.TimeoutError:
                pass
            return msgs

        rng0 = np.random.default_rng(3)
        for wi, w in enumerate(wavs):
            au, _ = librosa.load(w, sr=16000, mono=True)
            au = np.concatenate([au, np.zeros(int(1.4 * 16000))])
            # a real mic adds steady noise to EVERYTHING, including the
            # speech and the pauses inside it — test the same signal
            au = au + rng0.normal(0, 0.008, len(au))
            i16 = (au * 32767).clip(-32767, 32767).astype(np.int16)
            print(f">> streaming {w} ({len(au)/16000:.1f}s incl. tail "
                  f"silence)")
            FR = 2048
            for i in range(0, len(i16), FR):
                await ws.send(i16[i:i + FR].tobytes())
                # ~realtime-ish but 4x faster; VAD counts samples not wall
                await asyncio.sleep(0.03)

            # a real mic never stops sending — keep streaming silence
            # until the turn completes (this is exactly what the browser
            # client does; stopping cold was a test artifact that could
            # strand the VAD one frame short of the EOT threshold)
            stop = asyncio.Event()

            rng = np.random.default_rng(7)

            def noise_frame():
                # steady mic/room noise, rms ~0.008 — above the old
                # fixed 0.010/…-ish floor territory, the case that kept
                # real sessions "listening" forever
                return (rng.normal(0, 0.008, FR) * 32767).astype(
                    np.int16).tobytes()

            async def mic_silence():
                while not stop.is_set():
                    try:
                        await ws.send(noise_frame())
                    except Exception:
                        return
                    await asyncio.sleep(0.06)

            sil_task = asyncio.create_task(mic_silence())
            try:
                msgs = await drain(200)
            finally:
                stop.set()
                await sil_task
            turns = [m for m in msgs if m["type"] == "turn"]
            assert turns, f"no turn result for {w}"
            d = turns[0]
            print(f"== turn {wi}: fired={d['fired']} "
                  f"eot={d['eot_score']} mode={d['mode']} "
                  f"audio_s={d['audio_s']}")
            print(f"   answer: {(d.get('answer') or '')[:140]}")
            if d.get("uplink_text"):
                print(f"   uplink heard: {d['uplink_text'][:120]}")
        # turn 3: speech buried in the same noise, VAD may or may not
        # fire — then the manual "I'm done" message must end the turn
        au, _ = librosa.load(wavs[0], sr=16000, mono=True)
        n = rng.normal(0, 0.008, len(au))
        i16 = ((au + n) * 32767).clip(-32767, 32767).astype(np.int16)
        print(">> turn 3: speech + steady noise, then manual eot")
        for i in range(0, len(i16), FR):
            await ws.send(i16[i:i + FR].tobytes())
            await asyncio.sleep(0.03)
        await ws.send(noise_frame())
        await ws.send(json.dumps({"type": "eot"}))
        stop3 = asyncio.Event()

        async def mic3():
            while not stop3.is_set():
                try:
                    await ws.send(noise_frame())
                except Exception:
                    return
                await asyncio.sleep(0.06)
        t3 = asyncio.create_task(mic3())
        try:
            msgs = await drain(200)
        finally:
            stop3.set()
            await t3
        turns = [m for m in msgs if m["type"] == "turn"]
        assert turns, "manual eot produced no turn"
        print(f"== turn 3 (manual eot): fired={turns[0]['fired']} "
              f"mode={turns[0]['mode']} "
              f"answer: {(turns[0].get('answer') or '')[:100]}")
    print("SESSION OK — three turns incl. noisy mic + manual eot")
