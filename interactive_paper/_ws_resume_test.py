"""Post-interrupt resume test (8bn): user reports that after barging in
(model yields), their NEXT utterance sometimes gets no response at all.

Scenario, scripted (n trials, both endpoints):
  1. ask a long-answer prompt ("count slowly from one to thirty")
  2. +2 s into the answer, interrupt with a NEW question wav
  3. wait for yield; 3 s silence
  4. ask "What is the capital of France?"
  5. PASS iff a speak-commit with text happens within 15 s of (4)

Run:
  modal run _ws_resume_test.py::t --target demo --n 4
  modal run _ws_resume_test.py::t --target vanilla --n 4
"""
import json

import modal

from modal_app import OPENAI

app = modal.App("ws-resume-test")
gate_data = modal.Volume.from_name("gate-data")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy",
                    "openai")
       .add_local_dir("src", "/workspace/gate")
       .add_local_file("modal_app.py", "/root/modal_app.py"))

URLS = {
    "demo": ("wss://rhe9527--gate-duplex-voice.modal.run/62dc5cd9/ws"
             "?tier=balanced&probe_on=1",
             "https://rhe9527--gate-duplex-voice.modal.run/62dc5cd9"),
    "vanilla": ("wss://rhe9527--vanilla-duplex-voice.modal.run/62dc5cd9"
                "/ws?cfg=official",
                "https://rhe9527--vanilla-duplex-voice.modal.run/62dc5cd9"),
}


@app.function(image=img, volumes={"/data": gate_data},
              secrets=[OPENAI], timeout=60 * 40)
async def t(target: str = "demo", n: int = 4):
    import asyncio
    import sys
    import time as _t
    import urllib.request

    import librosa
    import numpy as np
    import websockets
    sys.path.insert(0, "/workspace/gate")
    import escalate

    ws_url, base = URLS[target]
    cli = escalate._client()
    wavs = {}
    for key, txt in [("carrier", "Please count slowly from one to thirty."),
                     ("intr", "Wait, what is the tallest mountain?"),
                     ("resume", "What is the capital of France?")]:
        r = cli.audio.speech.create(model="tts-1", voice="alloy",
                                    input=txt, response_format="wav")
        open(f"/tmp/{key}.wav", "wb").write(r.content)
        au, _ = librosa.load(f"/tmp/{key}.wav", sr=16000, mono=True)
        wavs[key] = au.astype(np.float32)

    t0 = _t.time()
    ready = None
    while _t.time() - t0 < 480:
        try:
            ready = json.load(urllib.request.urlopen(f"{base}/ready",
                                                     timeout=25))
            break
        except Exception:
            await asyncio.sleep(4)
    assert ready and ready.get("ready")

    FR = 2048
    results = []
    for trial in range(n):
        ev = {"texts": [], "t_first": None, "t_resume_sent": None,
              "resume_reply": None, "log": []}
        async with websockets.connect(ws_url, max_size=2 ** 24,
                                      open_timeout=90) as sock:
            t0s = _t.time()

            async def rd():
                async for m in sock:
                    e = json.loads(m)
                    tt = _t.time() - t0s
                    if e.get("type") == "text":
                        ev["texts"].append((tt, e["v"]))
                        if ev["t_first"] is None:
                            ev["t_first"] = tt
                        if (ev["t_resume_sent"] is not None
                                and tt > ev["t_resume_sent"]
                                and ev["resume_reply"] is None):
                            ev["resume_reply"] = tt
                    if e.get("type") in ("gate", "log"):
                        ev["log"].append(
                            (round(tt, 1),
                             e.get("msg") or f"gate fired={e.get('fired')} "
                                             f"info={e.get('is_info')}"))
            rt = asyncio.create_task(rd())

            async def send_au(au):
                i16 = (au * 32767).clip(-32767, 32767).astype(np.int16)
                for i in range(0, len(i16), FR):
                    await sock.send(i16[i:i + FR].tobytes())
                    await asyncio.sleep(FR / 16000)

            rng = np.random.default_rng(3)

            async def sil(sec):
                for _ in range(int(sec * 16000 / FR)):
                    await sock.send((rng.normal(0, 0.003, FR) * 32767)
                                    .clip(-32767, 32767)
                                    .astype(np.int16).tobytes())
                    await asyncio.sleep(FR / 16000)

            await send_au(wavs["carrier"])
            t1 = _t.time()
            while ev["t_first"] is None and _t.time() - t1 < 20:
                await sil(0.5)
            await sil(2.0)
            await send_au(wavs["intr"])       # barge-in w/ new question
            await sil(6.0)                    # let it yield / react
            ev["t_resume_sent"] = _t.time() - t0s
            await send_au(wavs["resume"])     # the utterance that "dies"
            t2 = _t.time()
            while ev["resume_reply"] is None and _t.time() - t2 < 18:
                await sil(0.5)
            try:
                await sock.send(json.dumps({"type": "stop"}))
            except Exception:
                pass
            await asyncio.sleep(0.5)
            rt.cancel()
        ok = ev["resume_reply"] is not None
        lat = (ev["resume_reply"] - ev["t_resume_sent"]) if ok else None
        tail = "".join(v for _, v in ev["texts"][-6:])
        results.append({"trial": trial, "resume_ok": ok,
                        "latency": lat and round(lat, 1)})
        print(f"[{target} #{trial}] resume_ok={ok} "
              f"lat={lat and round(lat, 1)}s tail={tail[-70:]!r}",
              flush=True)
        for tt, lg in ev["log"][-6:]:
            print(f"    {tt}s {lg[:90]}", flush=True)

    okn = sum(1 for r in results if r["resume_ok"])
    print(f"\n===== {target}: resume responded {okn}/{n} =====")
    return results
