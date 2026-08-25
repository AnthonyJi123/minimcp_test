"""Interactive demo of the escalation gate — probe ON/OFF, live metrics,
event log.  `modal deploy demo_app.py`

Two modes, both faithful to the deployed pipeline:

REPLAY ($0, always on).  Every one of the 4773 measured live sessions is
on the volume, and because the three gated tiers are perfectly NESTED
and carry bit-identical probe scores (RESULTS 8ad), flipping the probe
OFF for a query is not a simulation: it is the never-arm's MEASURED
outcome for that same query. So the toggle shows two real recordings of
the same question, with the real probe score, the real threshold
comparison, the real answers, the real judge verdicts and the real
wall-clock timings.

LIVE (opt-in, spins one H100).  Type any question -> OpenAI tts-1
renders it with the same voice as the frozen pool -> MiniCPM-o 4.5
streams it in 1 s chunks -> the v3 probe reads L22 at end-of-turn
(rolling last-8 tail + running user-audio mean, exactly
modal_bench.py::bench_live) -> the frozen per-tier threshold decides ->
either the talker answers or gpt-5.5 does and the talker relays it.
Ask "what is NVDA trading at" with the probe on and off.
"""
import json
import os
import time

import modal

TOKEN = "62dc5cd9"

app = modal.App("gate-demo")
gate_data = modal.Volume.from_name("gate-data")
weights = modal.Volume.from_name("minicpm-o45-weights")
DATA = "/data"
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"
LAYER = 22
POOLS = ("frozen", "striviaqa", "swebq", "sllama", "sreason", "sdqa")
TIERS = ("conservative", "balanced", "aggressive")

# The GPU image replicates modal_app's PROVEN MiniCPM spec verbatim
# (torch 2.8 / transformers 4.51.0 pin — 4.52+ breaks the Resampler)
# plus fastapi for the in-container ASGI app. It cannot be derived from
# modal_app.image directly: that one ends with add_local_dir, and Modal
# forbids stacking build layers on top of local files (the first deploy
# tried and the Voice container crash-looped on "No module named
# fastapi" while /ready timed out for 8 minutes).
from modal_app import OPENAI  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(_HERE, "modal_app.py")

web_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("fastapi[standard]", "pandas", "pyarrow")
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
        "fastapi[standard]",          # the one addition: in-container ASGI
    )
    .add_local_dir(os.path.join(_HERE, "src"), "/workspace/gate")
    .add_local_file(_APP_PY, "/root/modal_app.py"))
STALL = "Let me check that for you."
RELAY_TMPL = ("A verified answer came back: {ans}\n"
              "Relay it to the user in one or two spoken sentences.")


# ----------------------------------------------------------------- live ---
# A resident GPU class: the model loads ONCE at container start, the
# browser talks to it over a WebSocket on the same container, and the
# page's mic button stays disabled until /ready returns — which by
# construction cannot happen before the model is loaded (@enter).
#
# Continuous voice, no record button: the page streams 16 kHz int16 PCM
# frames; the server feeds 1 s chunks into the duplex loop exactly like
# bench_live, scores the probe per chunk, and a ~0.9 s silence after
# speech ends the turn (energy VAD — logged, since the benchmarks used
# known audio ends instead). On escalation there is no gold text, so
# the wav goes through the 8ae hosted-ASR uplink to gpt-5.5.

def _call_def(fn, /, **kw):
    import inspect
    p = set(inspect.signature(fn).parameters)
    return fn(**{k: v for k, v in kw.items() if k in p})


def _gen_speak(model, tok, emit_pcm, abort=None, **kw):
    """streaming_generate with the talker head ON: emit_pcm(float32 24 kHz
    chunk) per ~1 s of speech; returns (text, first_audio_ms, n_samples).
    abort (threading.Event) stops consumption at chunk granularity —
    the soft barge-in primitive."""
    kw.setdefault("max_new_tokens", 512)
    t0 = time.time()
    res = _call_def(model.streaming_generate, tokenizer=tok,
                    temperature=0.1, generate_audio=True,
                    use_tts_template=True, **kw)
    texts, first_ms, n_samp = [], None, 0
    for item in res:
        if abort is not None and abort.is_set():
            break
        wf, txt = (item if isinstance(item, tuple)
                   else (None, getattr(item, "text", None)))
        if wf is not None:
            if first_ms is None:
                first_ms = int((time.time() - t0) * 1000)
            pcm = wf.float().cpu().numpy().reshape(-1)
            n_samp += len(pcm)
            emit_pcm(pcm)
        if txt:
            texts.append(txt)
    return "".join(texts).strip(), first_ms, n_samp


@app.cls(image=gpu_image, gpu="H100",
         volumes={"/workspace/models": weights, DATA: gate_data},
         secrets=[OPENAI], timeout=60 * 60, scaledown_window=420)
@modal.concurrent(max_inputs=8)
class Voice:
    @modal.enter()
    def load(self):
        import glob as _glob
        import shutil
        import sys
        import threading
        import torch
        from transformers import AutoModel, AutoTokenizer
        sys.path.insert(0, "/workspace/gate")
        import gate as gate_mod

        t0 = time.time()
        cache = os.path.expanduser("~/.cache/huggingface/modules/"
                                   "transformers_modules/"
                                   + os.path.basename(MODEL_DIR))
        os.makedirs(cache, exist_ok=True)
        for f in _glob.glob(f"{MODEL_DIR}/*.py"):
            shutil.copy(f, cache)
        self.model = AutoModel.from_pretrained(
            MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=False, init_audio=True,
            init_tts=True).eval().cuda()
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR,
                                                 trust_remote_code=True)
        # talker audio out (TTS smoke 2026-08-24: works under the 4.51 pin;
        # +10.5 s load, VRAM peak 26.9 GB). The vocoder voice prompt is
        # vocoder-side only — the LLM context (and thus every probe read
        # against the frozen thresholds) is byte-identical to before.
        import librosa as _lb
        self.model.init_tts(model_dir=f"{MODEL_DIR}/assets/token2wav")
        ref, _ = _lb.load(f"{MODEL_DIR}/assets/system_ref_audio.wav",
                          sr=16000, mono=True)
        self.model.init_token2wav_cache(ref)
        # v4 = v3 + FreshQA real-time awareness (modal_fresh.py); tier
        # budgets stay quantiled on the v3 deployment mix. Guards:
        # frozen-test AUC .877 vs .879; fast-heldout fire .93/.80 vs
        # .80/.57 (balanced/conservative); never-heldout .17 = v3
        self.art = json.load(open(f"{DATA}/midlayer_gate_audio_v4.json"))
        self.probe = gate_mod.Probe(self.art["w"], self.art["b"])
        self.K3 = self.art.get("k_eot", 8)
        self.modes = self.art["modes"]
        self.st3 = {"accum": False, "tail": None, "sum": None, "cnt": 0}

        import torch as _t

        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            h = hs[0].detach().float()
            t = h[-self.K3:].cpu()
            self.st3["tail"] = (t if self.st3["tail"] is None
                                else _t.cat([self.st3["tail"], t])[-self.K3:])
            if self.st3["accum"]:
                sm = h.sum(0).cpu()
                self.st3["sum"] = (sm if self.st3["sum"] is None
                                   else self.st3["sum"] + sm)
                self.st3["cnt"] += h.shape[0]
        self.model.llm.model.layers[LAYER].register_forward_hook(hook)
        self.lock = threading.Lock()
        # canned stall filler in the talker's OWN voice (deployment plays
        # tts_filler, RESULTS "stall via assistant-role prefill"); teacher
        # forcing guarantees the audio matches the STALL text exactly
        self.stall_pcm = None
        try:
            import numpy as _np
            self._turn_reset()
            _call_def(self.model.streaming_prefill, session_id="s1",
                      msgs=[{"role": "user",
                             "content": [_np.zeros(16000,
                                                   dtype="float32")]}],
                      tokenizer=self.tok, is_last_chunk=True)
            parts = []
            _gen_speak(self.model, self.tok, parts.append,
                       session_id="s1", teacher_forcing=True,
                       teacher_forcing_text=STALL, max_new_tokens=64)
            if parts:
                self.stall_pcm = _np.concatenate(parts)
                print(f">>> stall filler synthesized: "
                      f"{len(self.stall_pcm) / 24000:.2f}s", flush=True)
        except Exception as e:
            print(f">>> stall filler synth failed (text-only stall): {e}",
                  flush=True)
        self.load_s = round(time.time() - t0, 1)
        print(f">>> Voice ready in {self.load_s}s", flush=True)

    # ---- turn primitives (shared by the WS loop and /say) ----------------
    def _score_now(self):
        import torch
        parts = []
        for m in self.modes:
            if m == "eot_last":
                parts.append(self.st3["tail"][-1])
            elif m == "eot_mean":
                parts.append(self.st3["tail"].mean(0))
            elif m == "user_mean":
                parts.append(self.st3["sum"] / max(1, self.st3["cnt"]))
        return float(self.probe.score(torch.cat(parts).numpy()))

    def _turn_reset(self):
        self.model.reset_session(reset_token2wav_cache=False)
        self.st3.update(tail=None, sum=None, cnt=0, accum=True)
        sys_msg = _call_def(self.model.get_sys_prompt, mode="omni",
                            language="en")
        _call_def(self.model.streaming_prefill, session_id="s1",
                  msgs=[sys_msg], tokenizer=self.tok)

    def _feed(self, ch, last):
        import numpy as np
        if len(ch) < 16000:
            ch = np.pad(ch, (0, 16000 - len(ch)))
        _call_def(self.model.streaming_prefill, session_id="s1",
                  msgs=[{"role": "user",
                         "content": [ch.astype("float32")]}],
                  tokenizer=self.tok, is_last_chunk=bool(last))
        return round(self._score_now(), 4)

    def _eot_read(self):
        self.st3["accum"] = False
        t0 = time.time()
        _call_def(self.model.streaming_prefill, session_id="s1",
                  msgs=[{"role": "assistant", "content": [" "]}],
                  tokenizer=self.tok, is_last_chunk=True)
        return self._score_now(), int((time.time() - t0) * 1000)

    def _answer(self, fired, uplink_text=None, wav_f32=None, emit=None,
                abort=None):
        """Local answer or stall+expert+relay. emit(dict) streams events.
        abort (threading.Event) = soft barge-in: stop generating/waiting
        as soon as it is set; partial results are returned marked
        interrupted."""
        import sys
        import threading
        sys.path.insert(0, "/workspace/gate")
        import escalate

        import base64

        import numpy as np

        emit = emit or (lambda *_: None)

        def emit_audio(pcm):
            i16 = (np.clip(pcm, -1, 1) * 32767).astype("<i2")
            emit({"type": "audio", "sr": 24000,
                  "pcm": base64.b64encode(i16.tobytes()).decode()})

        t_eot = time.time()
        out = {}
        if not fired:
            emit({"type": "phase", "v": "answering"})
            ans, first_ms, n_samp = _gen_speak(self.model, self.tok,
                                               emit_audio, abort=abort,
                                               session_id="s1")
            out.update(mode="local", answer=ans,
                       answer_ms=int((time.time() - t_eot) * 1000),
                       first_audio_ms=first_ms,
                       speech_out_s=round(n_samp / 24000, 2),
                       interrupted=bool(abort and abort.is_set()))
            return out
        exp = {}

        def expert_call():
            t0 = time.time()
            up = uplink_text
            if up is None and wav_f32 is not None:
                import soundfile as sf
                sf.write("/tmp/turn.wav", wav_f32, 16000)
                emit({"type": "log",
                      "msg": "uplink: transcribing your audio with the "
                             "hosted ASR (8ae path — no gold text exists "
                             "for real speech)"})
                with open("/tmp/turn.wav", "rb") as fh:
                    tr = escalate._client().audio.transcriptions.create(
                        model="gpt-transcribe", file=fh,
                        response_format="text")
                up = tr if isinstance(tr, str) else getattr(tr, "text",
                                                            str(tr))
                exp["uplink_text"] = up
                exp["asr_s"] = round(time.time() - t0, 2)
                emit({"type": "log",
                      "msg": f"ASR heard: “{up[:140]}” "
                             f"({exp['asr_s']} s)"})
            r = escalate.ask_expert_web(up, effort="low")
            if r.get("error"):
                emit({"type": "log",
                      "msg": f"web expert failed ({str(r['error'])[:80]}); "
                             "falling back to the no-tools expert"})
                r = escalate.ask_expert(up, effort="low")
            exp["answer"] = r.get("answer") or f"[error: {r.get('error')}]"
            exp["wall_s"] = time.time() - t0

        emit({"type": "phase", "v": "escalating"})
        emit({"type": "log", "msg": f"escalating to gpt-5.5 (web search ON); talker "
                                    f"stalls: “{STALL}”"})
        th = threading.Thread(target=expert_call, daemon=True)
        th.start()
        _call_def(self.model.streaming_prefill, session_id="s1",
                  msgs=[{"role": "assistant", "content": [STALL]}],
                  tokenizer=self.tok, is_last_chunk=True)
        t_stall = time.time()
        if getattr(self, "stall_pcm", None) is not None:
            emit_audio(self.stall_pcm)   # canned filler, talker's own voice
        for _ in range(1500):
            th.join(timeout=0.1)
            if not th.is_alive() or (abort is not None and abort.is_set()):
                break
        if abort is not None and abort.is_set():
            out.update(mode="escalated", answer="", interrupted=True,
                       expert_answer=exp.get("answer", ""),
                       stall_ms=int((t_stall - t_eot) * 1000))
            return out
        t_expert = time.time()
        emit({"type": "log",
              "msg": f"expert answered in {exp.get('wall_s', -1):.1f} s"})
        emit({"type": "phase", "v": "relaying"})
        _call_def(self.model.streaming_prefill, session_id="s1",
                  msgs=[{"role": "user",
                         "content": [RELAY_TMPL.format(
                             ans=exp.get("answer", ""))]}],
                  tokenizer=self.tok, is_last_chunk=True)
        relay, first_ms, n_samp = _gen_speak(self.model, self.tok,
                                             emit_audio, abort=abort,
                                             session_id="s1")
        out.update(mode="escalated", answer=relay,
                   interrupted=bool(abort and abort.is_set()),
                   first_audio_ms=first_ms,
                   speech_out_s=round(n_samp / 24000, 2),
                   expert_answer=exp.get("answer", ""),
                   uplink_text=exp.get("uplink_text"),
                   asr_s=exp.get("asr_s"),
                   expert_latency_s=round(exp.get("wall_s", -1), 2),
                   stall_ms=int((t_stall - t_eot) * 1000),
                   relay_ms=int((time.time() - t_expert) * 1000))
        return out

    # ---- web -------------------------------------------------------------
    @modal.asgi_app(label="gate-demo-voice")
    def ws_app(self):
        import asyncio
        import numpy as np
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel

        wapp = FastAPI()
        wapp.add_middleware(CORSMiddleware, allow_origins=["*"],
                            allow_methods=["*"], allow_headers=["*"])

        @wapp.get(f"/{TOKEN}/ready")
        def ready():
            # this endpoint existing at all means @enter finished, i.e.
            # the model IS loaded — that is the readiness gate
            return JSONResponse({"ready": True, "load_s": self.load_s,
                                 "busy": self.lock.locked()})

        class SayReq(BaseModel):
            question: str
            tier: str = "balanced"
            probe_on: bool = True

        @wapp.post(f"/{TOKEN}/say")
        def say(req: SayReq):
            import sys
            sys.path.insert(0, "/workspace/gate")
            import escalate
            if not self.lock.acquire(timeout=5):
                return JSONResponse({"error": "busy — a voice session "
                                     "holds the model"}, status_code=409)
            try:
                thr = self.art["eot_thresholds"][req.tier]
                r = escalate._client().audio.speech.create(
                    model="tts-1", voice="alloy", input=req.question,
                    response_format="wav")
                open("/tmp/say.wav", "wb").write(r.content)
                import librosa
                au, _ = librosa.load("/tmp/say.wav", sr=16000, mono=True)
                self._turn_reset()
                scores = []
                n = max(1, (len(au) + 15999) // 16000)
                for i in range(n):
                    scores.append(self._feed(au[i * 16000:(i + 1) * 16000],
                                             i == n - 1))
                eot, eot_ms = self._eot_read()
                fired = bool(req.probe_on and eot >= thr)
                pcm_b64 = []
                out = self._answer(fired, uplink_text=req.question,
                                   emit=lambda m: (
                                       pcm_b64.append(m["pcm"])
                                       if m.get("type") == "audio" else None))
                if pcm_b64:
                    import base64
                    import io

                    import numpy as np
                    import soundfile as sf
                    raw = b"".join(base64.b64decode(p) for p in pcm_b64)
                    pcm = (np.frombuffer(raw, dtype="<i2")
                           .astype("float32") / 32767)
                    bio = io.BytesIO()
                    sf.write(bio, pcm, 24000, format="WAV")
                    out["answer_wav"] = base64.b64encode(
                        bio.getvalue()).decode()
                out.update(question=req.question, tier=req.tier,
                           probe_on=req.probe_on, fired=fired,
                           eot_score=round(eot, 4), threshold=round(thr, 4),
                           scores=scores, eot_read_ms=eot_ms,
                           audio_s=round(len(au) / 16000, 2),
                           total_ms=int(sum(filter(None, [
                               out.get("answer_ms"),
                               out.get("stall_ms"),
                               int(1000 * (out.get("expert_latency_s") or 0)),
                               out.get("relay_ms")])) + eot_ms))
                return JSONResponse(out)
            finally:
                self.lock.release()

        @wapp.websocket(f"/{TOKEN}/ws")
        async def ws(sock: WebSocket):
            import threading as _th
            await sock.accept()
            tier = sock.query_params.get("tier", "balanced")
            probe_on = sock.query_params.get("probe_on", "1") == "1"
            thr = self.art["eot_thresholds"][tier]
            if not self.lock.acquire(timeout=3):
                await sock.send_json({"type": "error",
                                      "msg": "model busy — try again"})
                await sock.close()
                return
            try:
                await sock.send_json({"type": "hello",
                                      "thr": round(thr, 4), "tier": tier,
                                      "probe_on": probe_on,
                                      "vad": "speech ≥0.2 s, then "
                                             "1.25 s silence = end of turn"})
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._turn_reset)

                TH, SIL_EOT, SP_MIN, CAP = 0.010, 1.25, 0.2, 60.0
                buf = np.zeros(0, dtype=np.float32)
                wav = []
                scores = []
                speech, sp, sil, fed = False, 0.0, 0.0, 0
                floor, nvu = 0.006, 0
                await sock.send_json({"type": "phase", "v": "listening"})

                async def do_eot():
                    nonlocal buf, wav, scores, speech, sp, sil, fed
                    rest = buf
                    if len(rest) >= 1600 or fed == 0:
                        s = await loop.run_in_executor(
                            None, self._feed, rest, True)
                        scores.append(s)
                        await sock.send_json({"type": "score",
                                              "i": len(scores), "v": s})
                    eot, eot_ms = await loop.run_in_executor(
                        None, self._eot_read)
                    fired = bool(probe_on and eot >= thr)
                    await sock.send_json(
                        {"type": "eot", "score": round(eot, 4),
                         "ms": eot_ms, "thr": round(thr, 4),
                         "fired": fired, "probe_on": probe_on})
                    full = np.concatenate(wav) if wav else np.zeros(1600)
                    abort = _th.Event()
                    fut = loop.run_in_executor(
                        None, lambda: self._answer(
                            fired, None, full,
                            lambda m: asyncio.run_coroutine_threadsafe(
                                sock.send_json(m), loop), abort))
                    # soft barge-in: keep reading the mic WHILE the model
                    # answers. Trigger = sustained loud speech (>=0.4 s
                    # above 5x floor) or the manual stop button —
                    # deliberately stricter than the turn VAD so speaker
                    # echo through imperfect AEC cannot self-interrupt.
                    # The interrupting speech seeds the next turn.
                    int_sp, int_buf, gone = 0.0, [], False
                    while not fut.done():
                        try:
                            m2 = await asyncio.wait_for(sock.receive(),
                                                        timeout=0.1)
                        except asyncio.TimeoutError:
                            continue
                        except RuntimeError:
                            gone = True
                            abort.set()
                            break
                        if m2.get("type") == "websocket.disconnect":
                            gone = True
                            abort.set()
                            break
                        t2 = m2.get("text")
                        if t2:
                            try:
                                manual = (json.loads(t2).get("type")
                                          == "eot")
                            except Exception:
                                manual = False
                            if manual and not abort.is_set():
                                abort.set()
                                await sock.send_json({"type": "interrupt",
                                                      "why": "manual"})
                            continue
                        b2 = m2.get("bytes")
                        if not b2:
                            continue
                        f2 = (np.frombuffer(b2, dtype=np.int16)
                              .astype(np.float32) / 32768.0)
                        rms2 = float(np.sqrt((f2 * f2).mean() + 1e-12))
                        if abort.is_set():
                            int_buf.append(f2)    # keep the utterance
                        elif rms2 > max(0.008, floor * 5.0):
                            int_sp += len(f2) / 16000.0
                            int_buf.append(f2)
                            if int_sp >= 0.4:
                                abort.set()
                                await sock.send_json({"type": "interrupt",
                                                      "why": "speech"})
                        else:
                            int_sp, int_buf = 0.0, []
                    out = await fut
                    if gone:
                        return
                    out.update(fired=fired, eot_score=round(eot, 4),
                               threshold=round(thr, 4), scores=scores,
                               eot_read_ms=eot_ms, probe_on=probe_on,
                               audio_s=round(sum(len(w) for w in wav)
                                             / 16000, 2))
                    await sock.send_json({"type": "turn", **{
                        k: v for k, v in out.items()
                        if isinstance(v, (str, int, float, bool, list,
                                          type(None)))}})
                    buf = np.zeros(0, dtype=np.float32)
                    wav, scores = [], []
                    speech, sp, sil, fed = False, 0.0, 0.0, 0
                    # floor intentionally kept: same room, same mic
                    await loop.run_in_executor(None, self._turn_reset)
                    if abort.is_set() and int_buf:
                        # barge-in speech becomes the start of this turn
                        seed = np.concatenate(int_buf)[-16000 * 8:]
                        buf = seed
                        wav = [seed.copy()]
                        speech, sp = True, int_sp
                        await sock.send_json({"type": "speech", "on": True})
                    await sock.send_json({"type": "phase",
                                          "v": "listening"})

                while True:
                    try:
                        msg = await sock.receive()
                    except RuntimeError:
                        break     # disconnect already consumed by a drain
                    if msg.get("type") == "websocket.disconnect":
                        break
                    txt = msg.get("text")
                    force_eot = False
                    if txt:
                        try:
                            force_eot = (json.loads(txt).get("type")
                                         == "eot")
                        except Exception:
                            pass
                        if force_eot and not speech:
                            await sock.send_json(
                                {"type": "log",
                                 "msg": "manual end-of-turn, but no "
                                        "speech was detected yet — "
                                        "check the level readout"})
                            continue
                        if not force_eot:
                            continue
                    b = msg.get("bytes")
                    if not b and not force_eot:
                        continue
                    rms, dur = 0.0, 0.0
                    if b:
                        f = (np.frombuffer(b, dtype=np.int16)
                             .astype(np.float32) / 32768.0)
                        dur = len(f) / 16000.0
                        rms = float(np.sqrt((f * f).mean() + 1e-12))
                        # ADAPTIVE floor: real mics have real noise and
                        # browser AGC pumps quiet passages, so a fixed
                        # absolute threshold either never sees silence
                        # or never sees speech (user report 2026-08-21).
                        # Floor tracks down instantly, up slowly, and
                        # only adapts up while NOT in speech so long
                        # utterances don't erode their own threshold.
                        # EMA both ways: instant-min tracking collapses
                        # on a single digitally-silent frame and then
                        # steady mic noise reads as "speech" forever
                        if rms < floor:
                            floor += (rms - floor) * 0.10
                        elif not speech:
                            floor += (rms - floor) * 0.02
                        thr_sp = max(0.005, floor * 3.5)
                        nvu += 1
                        if nvu % 4 == 0:
                            await sock.send_json(
                                {"type": "vu", "rms": round(rms, 4),
                                 "thr": round(thr_sp, 4),
                                 "speech": speech,
                                 "sil": round(sil, 2)})
                        if rms > thr_sp:
                            sp += dur
                            sil = 0.0
                            if not speech and sp >= SP_MIN:
                                speech = True
                                await sock.send_json({"type": "speech",
                                                      "on": True})
                        else:
                            sil += dur
                            if not speech:
                                sp = 0.0
                        if not speech:
                            # keep a short pre-speech tail only
                            buf = np.concatenate([buf, f])[-8000:]
                            continue
                        wav.append(f)
                        buf = np.concatenate([buf, f])
                        while len(buf) >= 16000:
                            ch, buf = buf[:16000], buf[16000:]
                            s = await loop.run_in_executor(
                                None, self._feed, ch, False)
                            scores.append(s)
                            fed += 1
                            await sock.send_json({"type": "score",
                                                  "i": len(scores),
                                                  "v": s})
                    if (speech and sil >= SIL_EOT) or sp >= CAP \
                            or (force_eot and speech):
                        if force_eot:
                            await sock.send_json(
                                {"type": "log",
                                 "msg": "manual end-of-turn"})
                        await sock.send_json({"type": "speech",
                                              "on": False})
                        await do_eot()
                        # no post-answer drain: the barge-in watch loop
                        # inside do_eot already consumed (and either used
                        # or discarded) everything sent while answering
            except WebSocketDisconnect:
                pass
            finally:
                self.lock.release()
        return wapp


# ---------------------------------------------------------------- replay --
def _load():
    import pandas as pd
    store, agg, thr = {}, {}, {}
    for p in POOLS:
        f = f"{DATA}/gate_v3_{p}.json"
        if os.path.exists(f):
            a = json.load(open(f))
            t = a.get("eot_thresholds") or a.get("thresholds") or a
            thr[p] = {k: float(v) for k, v in t.items()
                      if k in TIERS and isinstance(v, (int, float))}
    for p in POOLS:
        f = f"{DATA}/{p}_v3_traces.parquet"
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        col = "oab_ok" if ("oab_ok" in df.columns
                           and df["oab_ok"].notna().any()) else "heard_ok"
        df["_ok"] = df[col]
        df["expert_ms"] = df["expert_latency_s"].fillna(0) * 1000
        e = df["mode"] == "escalated"
        df["total_ms"] = (
            df["eot_read_ms"]
            + e * (df[["stall_ms", "expert_ms"]].max(axis=1).fillna(0)
                   + df["relay_ms"].fillna(0))
            + (~e) * df["answer_ms"].fillna(0))
        keep = [c for c in ("id", "pool", "tier", "mode", "query",
                            "transcript", "reference_answer", "eot_score",
                            "eot_read_ms", "scores", "answer", "relay",
                            "expert_answer", "expert_latency_s", "stall_ms",
                            "relay_ms", "answer_ms", "audio_s", "_ok",
                            "total_ms") if c in df.columns]
        d = df[keep].copy()
        store[p] = {t: g.set_index("id").to_dict("index")
                    for t, g in d.groupby("tier")}
        arms = {}
        for t in ("never",) + TIERS:
            g = d[d["tier"] == t]
            if len(g):
                arms[t] = {"n": int(len(g)), "acc": float(g["_ok"].mean()),
                           "esc": float((g["mode"] == "escalated").mean()),
                           "p50_ms": float(g["total_ms"].median()),
                           "judge": col}
        agg[p] = arms
    return store, agg, thr




# hand-picked exemplars for the /cases page: pool, id, category
CASES_PICK = [
    ("sllama", "sllama0164", "winfast", "拐弯的来源:升级反而更快"),
    ("striviaqa", "striviaqa0192", "winfast", "拐弯的来源:升级反而更快"),
    ("striviaqa", "striviaqa0074", "bothright",
     "本地也对,但绕得慢——升级更快"),
    ("sllama", "sllama0024", "tradeoff", "典型形态:多花几秒,换回正确"),
    ("sllama", "sllama0000", "kept", "探针分很低 → 正确地留在本地"),
    ("striviaqa", "striviaqa0137", "missed",
     "自信错:15% 预算漏网,50% 档被抓回"),
]
_HEDGE_RE = ("there is no|no widely|not a real|i'?m not sure|not sure|"
             "might be|likely|however|unfortunately|i don'?t know|"
             "as an ai|apolog|unclear|difficult to|depends on")


def _cases():
    import re
    import pandas as pd
    thr = {}
    for pool in ("striviaqa", "sllama"):
        a = json.load(open(f"{DATA}/gate_v3_{pool}.json"))
        t = a.get("eot_thresholds") or a
        thr[pool] = {k: float(v) for k, v in t.items()
                     if isinstance(v, (int, float))}
    hedge = re.compile(_HEDGE_RE, re.I)

    def mark(txt):
        return hedge.sub(lambda m: "<mark>" + m.group(0) + "</mark>",
                         str(txt or ""))

    out = []
    for pool, qid, cat, cat_label in CASES_PICK:
        df = pd.read_parquet(f"{DATA}/{pool}_v3_traces.parquet")
        nev = df[(df["tier"] == "never") & (df["id"] == qid)].iloc[0]
        esc_row, esc_tier = None, None
        for t in ("conservative", "balanced", "aggressive"):
            r = df[(df["tier"] == t) & (df["id"] == qid)]
            if len(r) and r.iloc[0]["mode"] == "escalated":
                esc_row, esc_tier = r.iloc[0], t
                break
        cons = df[(df["tier"] == "conservative") & (df["id"] == qid)]
        cons = cons.iloc[0] if len(cons) else nev
        case = {"id": qid, "pool": pool, "cat": cat,
                "cat_label": cat_label, "q": str(nev["query"]),
                "ref": str(nev["reference_answer"])[:160],
                "audio_s": float(nev["audio_s"]),
                "off": {"ans": mark(nev["answer"]),
                        "ms": float(nev["answer_ms"]),
                        "ok": int(nev["oab_ok"]),
                        "chars": len(str(nev["answer"]))}}
        if esc_row is not None:
            e = esc_row
            tot = (e["eot_read_ms"]
                   + max(e["stall_ms"] or 0,
                         (e["expert_latency_s"] or 0) * 1000)
                   + (e["relay_ms"] or 0))
            case["on"] = {"mode": "escalated", "tier": esc_tier,
                          "score": float(e["eot_score"]),
                          "thr": thr[pool].get(esc_tier),
                          "eot_ms": float(e["eot_read_ms"]),
                          "stall_ms": float(e["stall_ms"] or 0),
                          "expert_s": float(e["expert_latency_s"] or 0),
                          "relay_ms": float(e["relay_ms"] or 0),
                          "total_ms": float(tot),
                          "expert": str(e["expert_answer"]),
                          "relay": str(e["relay"]),
                          "ok": int(e["oab_ok"])}
        else:
            case["on"] = {"mode": "local",
                          "score": float(cons["eot_score"]),
                          "thr": thr[pool].get("conservative"),
                          "ans": mark(cons["answer"]),
                          "ms": float(cons["answer_ms"]),
                          "ok": int(cons["oab_ok"])}
        out.append(case)
    return out


@app.function(image=web_image, volumes={DATA: gate_data}, timeout=60 * 10,
              min_containers=0)
@modal.asgi_app()
def web():
    import random
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel

    STORE, AGG, THR = _load()
    api = FastAPI()

    @api.get(f"/{TOKEN}", response_class=HTMLResponse)
    def index():
        return PAGE.replace("__TOKEN__", TOKEN).replace(
            "__AGG__", json.dumps(AGG)).replace("__THR__", json.dumps(THR))

    CASES = _cases()

    @api.get(f"/{TOKEN}/cases", response_class=HTMLResponse)
    def cases_page():
        return CASES_PAGE.replace("__TOKEN__", TOKEN).replace(
            "__CASES__", json.dumps(CASES, ensure_ascii=False))

    @api.get(f"/{TOKEN}/api/pick")
    def pick(pool: str = "frozen", tier: str = "balanced", qid: str = ""):
        if pool not in STORE:
            raise HTTPException(404, "unknown pool")
        tiers = STORE[pool]
        if tier not in tiers or "never" not in tiers:
            raise HTTPException(404, "arm missing")
        ids = sorted(set(tiers[tier]) & set(tiers["never"]))
        if not ids:
            raise HTTPException(404, "no shared ids")
        qid = qid if qid in ids else random.choice(ids)
        on, off = tiers[tier][qid], tiers["never"][qid]

        def clean(d):
            out = {}
            for k, v in d.items():
                if hasattr(v, "tolist"):
                    v = v.tolist()
                if v is not None and not isinstance(
                        v, (str, int, float, bool, list, dict)):
                    v = str(v)
                if isinstance(v, float) and v != v:
                    v = None
                out[k] = v
            return out

        return JSONResponse({"id": qid, "pool": pool, "tier": tier,
                             "n_ids": len(ids), "on": clean(on),
                             "off": clean(off)})

    @api.get(f"/{TOKEN}/api/ids")
    def ids(pool: str = "frozen", tier: str = "balanced", q: str = ""):
        t = STORE.get(pool, {})
        if tier not in t:
            return JSONResponse([])
        rows = [{"id": k, "query": (v.get("query") or "")[:110]}
                for k, v in sorted(t[tier].items())]
        if q:
            ql = q.lower()
            rows = [r for r in rows
                    if ql in r["id"].lower() or ql in r["query"].lower()]
        return JSONResponse(rows[:400])

    return api


PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Escalation gate — live demo</title><style>
:root{--blue:#2a78d6;--green:#1e9e50;--red:#b00;--ink:#0b0b0b;
--mut:#52514e;--grid:#e6e4de;--bg:#faf9f7;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
header{background:var(--card);border-bottom:1px solid var(--grid);
padding:.7rem 1rem;display:flex;gap:1rem;align-items:center;flex-wrap:wrap;
position:sticky;top:0;z-index:9}
h1{font-size:1rem;margin:0;font-weight:640}
.sub{color:var(--mut);font-size:.78rem}
.wrap{display:grid;grid-template-columns:270px 1fr 340px;gap:1rem;
padding:1rem;align-items:start}
@media(max-width:1100px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;
padding:.85rem}
.card h2{font-size:.8rem;margin:0 0 .6rem;text-transform:uppercase;
letter-spacing:.04em;color:var(--mut);font-weight:640}
button{font:inherit;border:1px solid var(--grid);background:#fff;
border-radius:7px;padding:.4rem .7rem;cursor:pointer}
button:hover{border-color:var(--blue)}
button.primary{background:var(--blue);color:#fff;border-color:var(--blue)}
button.primary:disabled{opacity:.5;cursor:not-allowed}
select,input{font:inherit;border:1px solid var(--grid);border-radius:7px;
padding:.38rem .5rem;background:#fff;width:100%}
.toggle{display:flex;align-items:center;gap:.55rem;padding:.35rem .6rem;
border:1px solid var(--grid);border-radius:99px;background:#fff}
.sw{width:44px;height:24px;border-radius:99px;background:#cfcdc8;
position:relative;transition:.16s;cursor:pointer;flex:none}
.sw.on{background:var(--blue)}
.sw::after{content:"";position:absolute;top:3px;left:3px;width:18px;
height:18px;border-radius:50%;background:#fff;transition:.16s}
.sw.on::after{left:23px}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}
.tile{border:1px solid var(--grid);border-radius:8px;padding:.5rem .6rem}
.tile .k{font-size:.68rem;color:var(--mut);text-transform:uppercase;
letter-spacing:.03em}
.tile .v{font-size:1.25rem;font-weight:650;font-variant-numeric:tabular-nums}
.tile .d{font-size:.7rem;color:var(--mut)}
.log{font:11.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
background:#fbfbfa;border:1px solid var(--grid);border-radius:8px;
padding:.55rem;height:330px;overflow:auto;white-space:pre-wrap}
.log b{color:var(--blue)}.log .esc{color:var(--green);font-weight:700}
.log .off{color:var(--mut)}
.ans{border-left:3px solid var(--grid);padding:.15rem 0 .15rem .7rem;
margin:.45rem 0;white-space:pre-wrap}
.ans.local{border-color:var(--blue)}.ans.esc{border-color:var(--green)}
.pill{display:inline-block;font-size:.7rem;padding:.1rem .5rem;
border-radius:99px;color:#fff;vertical-align:middle}
.pill.ok{background:var(--green)}.pill.no{background:var(--red)}
.pill.g{background:var(--blue)}.pill.m{background:var(--mut)}
.row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
.q{font-size:.95rem;margin:.2rem 0 .6rem}
.qlist{max-height:340px;overflow:auto;margin-top:.5rem}
.qlist div{padding:.3rem .4rem;border-radius:6px;cursor:pointer;
font-size:.78rem;border:1px solid transparent}
.qlist div:hover{background:#f3f6fb;border-color:var(--grid)}
.qlist div.sel{background:#eaf1fb;border-color:var(--blue)}
.spark{width:100%;height:54px;display:block}
.muted{color:var(--mut);font-size:.76rem}
table{width:100%;border-collapse:collapse;font-size:.76rem}
td,th{text-align:right;padding:.22rem .3rem;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
.warn{background:#fff8e6;border:1px solid #f0e0b0;border-radius:8px;
padding:.5rem .6rem;font-size:.76rem;color:#6b5a1e;margin-top:.5rem}
</style></head><body>
<header>
  <h1>Escalation gate</h1>
  <span class=sub>MiniCPM-o 4.5 duplex talker · L22 probe v4 · gpt-5.5 expert</span>
  <div class=toggle><div id=sw class="sw on"></div>
    <b id=swlab>PROBE ON</b></div>
  <select id=tier style="width:auto">
    <option>conservative</option><option selected>balanced</option>
    <option>aggressive</option></select>
  <select id=mode style="width:auto">
    <option value=replay>replay — measured sessions ($0)</option>
    <option value=live>live — real GPU turn</option></select>
  <a href="/__TOKEN__/cases" style="font-size:.8rem">案例对比 →</a>
  <span id=state class=sub></span>
</header>
<div class=wrap>
  <div>
    <div class=card id=picker>
      <h2>Query</h2>
      <select id=pool></select>
      <input id=search placeholder="filter by id or text" style="margin-top:.45rem">
      <div class=qlist id=qlist></div>
      <div class=row style="margin-top:.5rem">
        <button id=rand>Random</button><button id=run class=primary>Run</button>
      </div>
    </div>
    <div class=card id=livebox style="display:none;margin-top:1rem">
      <h2>Voice session</h2>
      <div id=gstate class=warn>GPU offline — it starts when you switch
        to live mode; the mic stays disabled until the model is loaded.</div>
      <div class=row style="margin-top:.5rem">
        <button id=talk class=primary disabled
          style="flex:1;font-size:.95rem">Waiting for GPU…</button>
        <button id=done disabled title="force end-of-turn if the silence
          detector misses your pause">I'm done talking</button>
      </div>
      <div id=vad class=muted style="margin-top:.3rem;font-variant-numeric:
        tabular-nums">vad: —</div>
      <div class=row style="margin-top:.45rem">
        <div style="flex:1;height:8px;background:#eee;border-radius:99px;
          overflow:hidden"><div id=vu
          style="height:100%;width:0%;background:#2a78d6"></div></div>
        <span id=vstate class=muted>mic off</span>
      </div>
      <div class=muted style="margin-top:.4rem">Just talk — no buttons to
        hold. Pause ~1.3 s and the turn ends: the probe reads L22, the gate
        decides, and either the talker answers or gpt-5.5 does (your audio
        goes through the hosted-ASR uplink — no gold text exists for real
        speech). The talker SPEAKS its answer — native TTS head, 24 kHz.
        Barge-in: talk over it (or hit the button) and it stops and
        listens to you instead. Then it listens again.</div>
      <h2 style="margin-top:.9rem">…or type instead</h2>
      <input id=q placeholder="e.g. what is NVDA trading at right now?">
      <div class=row style="margin-top:.4rem" id=chips></div>
      <button id=runlive class=primary disabled
        style="margin-top:.5rem;width:100%">Send typed question</button>
    </div>
  </div>
  <div class=card>
    <h2>Turn</h2>
    <div id=turn class=muted>Pick a query and press Run.</div>
  </div>
  <div>
    <div class=card>
      <h2>Session metrics</h2>
      <div class=tiles id=tiles></div>
      <div class=muted style="margin-top:.5rem">Accumulated over the turns
        you ran in this browser session.</div>
      <h2 style="margin-top:.9rem">Reference (full sweep)</h2>
      <table id=ref></table>
    </div>
    <div class=card style="margin-top:1rem">
      <h2>Event log</h2><div class=log id=log></div>
    </div>
  </div>
</div>
<script>
const T="__TOKEN__", AGG=__AGG__, THR=__THR__;
const $=s=>document.querySelector(s);
let probeOn=true, sel=null, S={n:0,ok:0,esc:0,ms:0,onN:0,onOk:0,offN:0,offOk:0};

let playCtx=null,playT=0,playSrcs=[];
function stopPlayback(){
  playSrcs.forEach(s=>{try{s.stop()}catch(_){}});playSrcs=[];playT=0;}
function playPCM(b64,sr){
  if(!playCtx)playCtx=new (window.AudioContext||window.webkitAudioContext)();
  if(playCtx.state==="suspended")playCtx.resume();
  const raw=atob(b64),n=raw.length>>1,f=new Float32Array(n);
  for(let i=0;i<n;i++){let v=raw.charCodeAt(2*i)|(raw.charCodeAt(2*i+1)<<8);
    if(v>=32768)v-=65536;f[i]=v/32768;}
  const buf=playCtx.createBuffer(1,n,sr);
  buf.getChannelData(0).set(f);
  const src=playCtx.createBufferSource();src.buffer=buf;
  src.connect(playCtx.destination);
  playSrcs.push(src);
  src.onended=()=>{const i=playSrcs.indexOf(src);
    if(i>=0)playSrcs.splice(i,1)};
  playT=Math.max(playT,playCtx.currentTime);
  src.start(playT);playT+=buf.duration;
}
const pool=$("#pool");
Object.keys(AGG).forEach(p=>{const o=document.createElement("option");
  o.value=p;o.textContent=p;pool.appendChild(o)});

$("#sw").onclick=()=>{probeOn=!probeOn;
  $("#sw").classList.toggle("on",probeOn);
  $("#swlab").textContent=probeOn?"PROBE ON":"PROBE OFF";
  log(probeOn?"probe switched ON — the gate decides per query"
             :"probe switched OFF — every query stays local (never arm)","off");};
$("#mode").onchange=()=>{const live=$("#mode").value==="live";
  $("#livebox").style.display=live?"":"none";
  $("#picker").style.display=live?"none":"";
  if(live)warmGPU();};
$("#tier").onchange=()=>{loadIds();refTable()};
pool.onchange=()=>{loadIds();refTable()};
$("#search").oninput=()=>loadIds();
$("#rand").onclick=()=>{sel=null;run()};
$("#run").onclick=()=>run();
$("#runlive").onclick=()=>runLive();

function log(msg,cls){const l=$("#log");const t=new Date().toLocaleTimeString();
  l.innerHTML+=`<div class="${cls||''}"><b>${t}</b> ${msg}</div>`;
  l.scrollTop=l.scrollHeight;}
function fmt(x,d=3){return x==null?"—":(+x).toFixed(d)}
function esc(s){return (s==null?"":String(s)).replace(/[<>&]/g,
  c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]))}

async function loadIds(){
  if($("#mode").value==="live")return;
  const r=await fetch(`/${T}/api/ids?pool=${pool.value}&tier=${$("#tier").value}`
    +`&q=${encodeURIComponent($("#search").value)}`);
  const rows=await r.json();
  $("#qlist").innerHTML=rows.map(x=>
    `<div data-id="${x.id}" class="${x.id===sel?'sel':''}">
      <b>${x.id}</b> ${esc(x.query)}</div>`).join("");
  [...document.querySelectorAll("#qlist div")].forEach(d=>
    d.onclick=()=>{sel=d.dataset.id;loadIds();run()});
}
function refTable(){
  const a=AGG[pool.value]||{};
  const rows=Object.entries(a).map(([k,v])=>
    `<tr><td>${k}</td><td>${(v.esc*100).toFixed(0)}%</td>
     <td>${fmt(v.acc,3)}</td><td>${(v.p50_ms/1000).toFixed(2)}s</td></tr>`);
  $("#ref").innerHTML=`<tr><th>arm</th><th>esc</th><th>acc</th><th>P50</th></tr>`
    +rows.join("");
}
function tiles(){
  const t=[["turns",S.n,""],
    ["accuracy",S.n?fmt(S.ok/S.n,3):"—","judge verdict"],
    ["escalated",S.n?Math.round(100*S.esc/S.n)+"%":"—","of your turns"],
    ["P50-ish",S.n?(S.ms/S.n/1000).toFixed(2)+"s":"—","mean total"],
    ["probe ON",S.onN?fmt(S.onOk/S.onN,3):"—",`${S.onN} turns`],
    ["probe OFF",S.offN?fmt(S.offOk/S.offN,3):"—",`${S.offN} turns`]];
  $("#tiles").innerHTML=t.map(([k,v,d])=>
    `<div class=tile><div class=k>${k}</div><div class=v>${v}</div>
     <div class=d>${d}</div></div>`).join("");
}
function spark(scores,thr,eot){
  if(!scores||!scores.length)return"";
  const w=300,h=54,n=scores.length,pad=4;
  const X=i=>pad+i*(w-2*pad)/Math.max(n-1,1), Y=v=>h-pad-v*(h-2*pad);
  const d=scores.map((v,i)=>`${i?"L":"M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  return `<svg class=spark viewBox="0 0 ${w} ${h}">
   <line x1=0 y1=${Y(thr).toFixed(1)} x2=${w} y2=${Y(thr).toFixed(1)}
    stroke="#b00" stroke-dasharray="3 3" stroke-width=1/>
   <path d="${d}" fill=none stroke="#2a78d6" stroke-width=2/>
   <circle cx=${X(n-1).toFixed(1)} cy=${Y(scores[n-1]).toFixed(1)} r=3
    fill="#2a78d6"/>
   <text x=${w-2} y=${(Y(thr)-3).toFixed(1)} font-size=8 fill="#b00"
    text-anchor=end>threshold ${thr.toFixed(2)}</text></svg>`;
}
function verdict(ok){return ok===1||ok===true
  ?'<span class="pill ok">judge: correct</span>'
  :(ok===0||ok===false?'<span class="pill no">judge: wrong</span>'
   :'<span class="pill m">unjudged</span>')}

async function run(){
  if($("#mode").value==="live")return runLive();
  $("#state").textContent="loading…";
  const r=await fetch(`/${T}/api/pick?pool=${pool.value}`
    +`&tier=${$("#tier").value}&qid=${sel||""}`);
  if(!r.ok){$("#state").textContent="error";return}
  const d=await r.json(); sel=d.id; $("#state").textContent=
    `${d.n_ids} measured queries in ${d.pool}`;
  const row=probeOn?d.on:d.off;
  const fired=probeOn&&row.mode==="escalated";
  const thr=probeOn?impliedThr(d):null;
  log(`— replay ${d.id} (${d.pool}), probe ${probeOn?"ON":"OFF"}, `
    +`tier ${d.tier} —`);
  (row.scores||[]).forEach((s,i)=>log(
    `chunk ${i+1}/${row.scores.length} — running P(fail)=${fmt(s)}`));
  log(`END OF TURN — probe read L22 in ${row.eot_read_ms??"?"} ms: `
    +`P(fail)=<b>${fmt(row.eot_score)}</b>`);
  if(!probeOn) log("PROBE OFF — gate bypassed, answering locally","off");
  else log(`gate → <span class="${fired?'esc':''}">`
    +`${fired?"ESCALATE":"keep local"}</span>`);
  if(fired){log(`expert answered in ${fmt(row.expert_latency_s,1)}s; `
    +`talker stalled ${row.stall_ms??"?"} ms, relayed in ${row.relay_ms??"?"} ms`);}
  else log(`talker answered locally in ${row.answer_ms??"?"} ms`);

  const ans=fired?(row.relay||row.expert_answer):row.answer;
  $("#turn").innerHTML=`
   <div class=row><span class="pill g">${d.pool}</span>
     <span class="pill m">${d.id}</span>
     <span class="pill ${fired?'ok':'m'}">${fired?"ESCALATED":"LOCAL"}</span>
     ${verdict(row._ok)}</div>
   <div class=q><b>Q.</b> ${esc(row.query)}</div>
   <div class=muted>what the talker actually heard (its own transcript):
     <i>${esc(row.transcript||"—")}</i></div>
   ${spark(row.scores,thr??1.01,row.eot_score)}
   <div class=muted>P(fail) as the question streams in; red = the
     ${d.tier} threshold${probeOn?"":" (probe is OFF — not applied)"}</div>
   <div class="ans ${fired?'esc':'local'}"><b>${fired?"Relay (talker voicing the expert)":"Talker, alone"}:</b>
     ${esc(ans)}</div>
   ${fired?`<div class=muted><b>expert (gpt-5.5) said:</b>
     ${esc(row.expert_answer)}</div>`:""}
   <div class=muted style="margin-top:.5rem">reference answer:
     ${esc(row.reference_answer)}</div>
   <table style="margin-top:.6rem">
    <tr><th>metric</th><th>value</th></tr>
    <tr><td>probe score (P fail)</td><td>${fmt(row.eot_score)}</td></tr>
    <tr><td>probe read latency</td><td>${row.eot_read_ms??"—"} ms</td></tr>
    <tr><td>total response</td><td>${(row.total_ms/1000).toFixed(2)} s</td></tr>
    <tr><td>audio length</td><td>${fmt(row.audio_s,1)} s</td></tr>
   </table>`;
  S.n++;S.ok+=(row._ok===1||row._ok===true)?1:0;S.esc+=fired?1:0;
  S.ms+=row.total_ms||0;
  if(probeOn){S.onN++;S.onOk+=(row._ok===1||row._ok===true)?1:0}
  else{S.offN++;S.offOk+=(row._ok===1||row._ok===true)?1:0}
  tiles();
}
function impliedThr(d){
  // the real per-domain quantile threshold the sweep deployed
  return ((THR[d.pool]||{})[d.tier]);
}

const VOICE="https://rhe9527--gate-demo-voice.modal.run";
let gpuReady=false, warming=false, ws=null, ac=null, micStream=null,
    proc=null, talking=false, liveScores=[], liveThr=null;

async function warmGPU(){
  if(gpuReady||warming)return; warming=true;
  const t0=Date.now();
  const tick=setInterval(()=>{ if(!gpuReady)$("#gstate").textContent=
    `GPU starting + loading MiniCPM… ${((Date.now()-t0)/1000|0)}s `
    +`(cold start can take ~2 min — the mic unlocks by itself)`;},1000);
  let j=null;
  for(let i=0;i<90&&!j;i++){
    try{
      const r=await fetch(`${VOICE}/${T}/ready`,
        {signal:AbortSignal.timeout(30000)});
      if(r.ok)j=await r.json();
    }catch(_){/* cold start: redirect chains / timeouts — keep polling */}
    if(!j)await new Promise(s=>setTimeout(s,4000));
  }
  try{
    if(j&&j.ready){gpuReady=true;
      $("#gstate").textContent=`GPU ready — model loaded in ${j.load_s}s. `
        +`Click the button and just speak.`;
      $("#gstate").className="muted";
      $("#talk").disabled=false;$("#talk").textContent="Start voice session";
      $("#runlive").disabled=false;
      log("GPU ready — model resident, further turns have no load cost");}
    else $("#gstate").textContent=
      "GPU start timed out — switch modes to retry.";
  }catch(e){
    $("#gstate").textContent="GPU start failed — switch modes to retry. "
      +"("+e+")";
  }finally{clearInterval(tick);warming=false;}
}

function stopTalk(){
  talking=false;
  try{if(proc)proc.disconnect()}catch(_){}
  try{if(ac)ac.close()}catch(_){}
  try{if(micStream)micStream.getTracks().forEach(t=>t.stop())}catch(_){}
  try{if(ws&&ws.readyState<2)ws.close()}catch(_){}
  ws=null;ac=null;proc=null;micStream=null;
  $("#talk").textContent="Start voice session";
  $("#talk").classList.add("primary");
  $("#done").disabled=true;$("#vad").textContent="vad: —";
  $("#vstate").textContent="mic off";$("#vu").style.width="0%";
}

async function startTalk(){
  if(!gpuReady){log("GPU not ready yet","off");return}
  try{
    micStream=await navigator.mediaDevices.getUserMedia({audio:{
      channelCount:1,echoCancellation:true,noiseSuppression:true,
      autoGainControl:true}});
  }catch(e){log("microphone permission denied: "+e,"off");return}
  ac=new AudioContext();
  const src=ac.createMediaStreamSource(micStream);
  proc=ac.createScriptProcessor(2048,1,1);
  const ratio=ac.sampleRate/16000;
  ws=new WebSocket(`${VOICE.replace("https","wss")}/${T}/ws`
    +`?tier=${$("#tier").value}&probe_on=${probeOn?1:0}`);
  ws.onmessage=ev=>handleVoice(JSON.parse(ev.data));
  ws.onclose=()=>{if(talking){log("voice session closed","off");stopTalk()}};
  ws.onerror=()=>{log("websocket error","off");stopTalk()};
  ws.onopen=()=>{
    src.connect(proc);proc.connect(ac.destination);
    talking=true;
    $("#done").disabled=false;
    $("#talk").textContent="■ End voice session";
    $("#vstate").textContent="listening";
    log(`voice session open (tier ${$("#tier").value}, probe `
      +`${probeOn?"ON":"OFF"}) — speak whenever you like`);
  };
  proc.onaudioprocess=e=>{
    const f=e.inputBuffer.getChannelData(0);
    let ss=0;for(let i=0;i<f.length;i++)ss+=f[i]*f[i];
    const rms=Math.sqrt(ss/f.length);
    $("#vu").style.width=Math.min(100,rms*700)+"%";
    if(!ws||ws.readyState!==1)return;
    const n=Math.floor(f.length/ratio);
    const out=new Int16Array(n);
    for(let i=0;i<n;i++){
      const v=f[Math.floor(i*ratio)];
      out[i]=Math.max(-32768,Math.min(32767,v*32767));}
    ws.send(out.buffer);
  };
}
$("#talk").onclick=()=>{talking?stopTalk():startTalk()};
$("#done").onclick=()=>{if(ws&&ws.readyState===1){
  ws.send(JSON.stringify({type:"eot"}));
  log("manual end-of-turn sent","off");}};

function handleVoice(m){
  if(m.type==="hello"){liveThr=m.thr;
    log(`session config: threshold ${fmt(m.thr)} (${m.tier}), `
      +`probe ${m.probe_on?"ON":"OFF"}; VAD: ${m.vad}`);}
  else if(m.type==="vu"){
    $("#vad").textContent=`vad: level ${m.rms.toFixed(3)} / threshold `
      +`${m.thr.toFixed(3)} · ${m.speech?"speech":"quiet"}`
      +(m.speech?` · silence ${m.sil.toFixed(1)}/1.25s`:"");}
  else if(m.type==="speech"){
    $("#vstate").textContent=m.on?"hearing you…":"turn ended";
    if(m.on){liveScores=[];log("speech detected — streaming into the "
      +"duplex talker")}}
  else if(m.type==="score"){liveScores.push(m.v);
    log(`chunk ${m.i} — running P(fail)=${fmt(m.v)}`);
    $("#turn").innerHTML=`<div class=q><b>listening…</b></div>`
      +spark(liveScores,liveThr??1.01,m.v)
      +`<div class=muted>P(fail) while you speak; red = threshold</div>`;}
  else if(m.type==="eot"){
    log(`END OF TURN — probe read L22 in ${m.ms} ms: `
      +`P(fail)=<b>${fmt(m.score)}</b>`);
    if(!m.probe_on)log("PROBE OFF — gate bypassed, answering locally","off");
    else log(`gate: ${fmt(m.score)} ${m.fired?"≥":"<"} ${fmt(m.thr)} → `
      +`<span class="${m.fired?'esc':''}">`
      +`${m.fired?"ESCALATE":"keep local"}</span>`);}
  else if(m.type==="phase"){$("#vstate").textContent=
    {listening:"listening",answering:"talker answering…",
     escalating:"expert thinking…",relaying:"relaying…"}[m.v]||m.v;}
  else if(m.type==="audio"){playPCM(m.pcm,m.sr||24000);}
  else if(m.type==="interrupt"){stopPlayback();
    log(`barge-in (${m.why==="manual"?"stop button":"you spoke"}) — `
      +`generation and playback stopped`,"off");
    $("#vstate").textContent="interrupted";}
  else if(m.type==="log"){log(esc(m.msg));}
  else if(m.type==="error"){log("server: "+esc(m.msg),"off");}
  else if(m.type==="turn"){renderTurn(m,true);}
}

function renderTurn(d,voice){
  const fired=d.fired;
  $("#turn").innerHTML=`
   <div class=row><span class="pill g">${voice?"voice":"typed"}</span>
     <span class="pill ${fired?'ok':'m'}">${fired?"ESCALATED":"LOCAL"}</span>
     ${d.interrupted?'<span class="pill m">INTERRUPTED</span>':''}</div>
   <div class=q><b>Q.</b> ${esc(d.question)||"<i>(your voice)</i>"}</div>
   ${d.uplink_text?`<div class=muted>hosted ASR heard (what the expert
     read): <i>${esc(d.uplink_text)}</i></div>`:""}
   ${spark(d.scores,d.threshold,d.eot_score)}
   <div class=muted>P(fail) as the audio streamed; red = threshold</div>
   <div class="ans ${fired?'esc':'local'}">
     <b>${fired?"Relay (talker voicing the expert)":"Talker, alone"}:</b>
     ${esc(d.answer)}</div>
   ${d.answer_wav?`<audio controls autoplay style="width:100%;margin-top:.4rem"
     src="data:audio/wav;base64,${d.answer_wav}"></audio>`:""}
   ${fired?`<div class=muted><b>expert (gpt-5.5) said:</b>
     ${esc(d.expert_answer)}</div>`:""}
   <table style="margin-top:.6rem">
    <tr><th>metric</th><th>value</th></tr>
    <tr><td>probe score (P fail)</td><td>${fmt(d.eot_score)}</td></tr>
    <tr><td>threshold</td><td>${fmt(d.threshold)}</td></tr>
    <tr><td>probe read latency</td><td>${d.eot_read_ms} ms</td></tr>
    ${d.asr_s?`<tr><td>ASR uplink</td><td>${d.asr_s} s</td></tr>`:""}
    ${d.expert_latency_s?`<tr><td>expert</td>
      <td>${d.expert_latency_s} s</td></tr>`:""}
    <tr><td>speech length</td><td>${fmt(d.audio_s,1)} s</td></tr>
    ${d.first_audio_ms?`<tr><td>first answer audio</td>
      <td>${d.first_audio_ms} ms</td></tr>`:""}
    ${d.speech_out_s?`<tr><td>spoken answer</td>
      <td>${d.speech_out_s} s</td></tr>`:""}
   </table>`;
  S.n++;S.esc+=fired?1:0;S.ms+=d.total_ms||0;
  if(d.probe_on===false)S.offN++;else S.onN++;
  tiles();
}

async function runLive(){
  const q=$("#q").value.trim();
  if(!q){log("type a question first","off");return}
  if(!gpuReady){log("GPU not ready yet","off");return}
  $("#runlive").disabled=true;
  $("#state").textContent="typed turn running on the warm GPU…";
  log(`— typed turn, probe ${probeOn?"ON":"OFF"}, tier `
    +`${$("#tier").value} —`);
  try{
    const r=await fetch(`${VOICE}/${T}/say`,{method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({question:q,tier:$("#tier").value,
        probe_on:probeOn})});
    if(!r.ok){log("error: "+await r.text(),"off");return}
    const d=await r.json();
    (d.scores||[]).forEach((v,i)=>log(`chunk ${i+1} — P(fail)=${fmt(v)}`));
    log(`END OF TURN — P(fail)=<b>${fmt(d.eot_score)}</b> `
      +`${d.fired?"≥":"<"} ${fmt(d.threshold)} → `
      +`${d.fired?"ESCALATE":"local"}`);
    renderTurn(d,false);
  }catch(e){log("error: "+e,"off")}
  finally{$("#runlive").disabled=false;$("#state").textContent="";}
}
loadIds();refTable();tiles();
log("ready — replay mode reads the 4773 measured sessions; "
   +"flipping the probe shows the same query's other measured arm.");
</script></body></html>"""


CASES_PAGE = '<!doctype html><html><head><meta charset=utf-8>\n<meta name=viewport content="width=device-width,initial-scale=1">\n<title>升级前后对比 — 真实案例</title><style>\n:root{--blue:#2a78d6;--green:#1e9e50;--red:#b00;--amber:#b8860b;\n--ink:#0b0b0b;--mut:#52514e;--grid:#e6e4de;--bg:#faf9f7;--card:#fff}\n*{box-sizing:border-box}\nbody{margin:0;background:var(--bg);color:var(--ink);\nfont:14px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}\nheader{background:var(--card);border-bottom:1px solid var(--grid);\npadding:.8rem 1.2rem;position:sticky;top:0;z-index:9}\nh1{font-size:1.05rem;margin:0;font-weight:650}\n.sub{color:var(--mut);font-size:.8rem;margin-top:.15rem}\n.wrap{max-width:1080px;margin:0 auto;padding:1rem}\n.case{background:var(--card);border:1px solid var(--grid);\nborder-radius:12px;margin-bottom:1.4rem;overflow:hidden}\n.chead{padding:.8rem 1rem;border-bottom:1px solid var(--grid)}\n.cat{display:inline-block;font-size:.72rem;padding:.14rem .6rem;\nborder-radius:99px;color:#fff;margin-right:.5rem;vertical-align:middle}\n.cat.winfast{background:var(--green)}.cat.bothright{background:var(--blue)}\n.cat.tradeoff{background:var(--amber)}.cat.kept{background:var(--blue)}\n.cat.missed{background:var(--red)}\n.q{font-size:1rem;font-weight:600;margin:.4rem 0 .1rem}\n.ref{color:var(--mut);font-size:.8rem}\n.cols{display:grid;grid-template-columns:1fr 1fr;gap:0}\n@media(max-width:800px){.cols{grid-template-columns:1fr}}\n.col{padding:.85rem 1rem}\n.col.left{border-right:1px solid var(--grid);background:#fcfcfb}\n.col h3{margin:0 0 .5rem;font-size:.78rem;text-transform:uppercase;\nletter-spacing:.04em;color:var(--mut)}\n.lat{font-size:1.7rem;font-weight:700;font-variant-numeric:tabular-nums}\n.lat small{font-size:.85rem;font-weight:400;color:var(--mut)}\n.pill{display:inline-block;font-size:.74rem;padding:.12rem .55rem;\nborder-radius:99px;color:#fff;vertical-align:middle;margin-left:.5rem}\n.pill.ok{background:var(--green)}.pill.no{background:var(--red)}\n.ans{margin-top:.55rem;font-size:.86rem;background:#fff;\nborder:1px solid var(--grid);border-radius:8px;padding:.6rem .7rem;\nmax-height:170px;overflow:auto;white-space:pre-wrap}\nmark{background:#ffe9a8;padding:0 .15rem;border-radius:3px}\n.gate{font:12px/1.5 ui-monospace,Menlo,monospace;background:#f4f7fb;\nborder:1px solid #dbe6f5;border-radius:8px;padding:.45rem .6rem;\nmargin-bottom:.55rem;color:var(--ink)}\n.gate b{color:var(--blue)}\n.bar{display:flex;height:14px;border-radius:6px;overflow:hidden;\nmargin:.45rem 0 .2rem;border:1px solid var(--grid)}\n.bar div{height:100%}\n.seg{font-size:.72rem;color:var(--mut);margin-bottom:.3rem}\n.delta{padding:.55rem 1rem;font-weight:650;font-size:.9rem;\nborder-top:1px solid var(--grid)}\n.delta.win{color:var(--green);background:#f0f9f2}\n.delta.trade{color:var(--amber);background:#fdf8ec}\n.delta.miss{color:var(--red);background:#fdf0f0}\n.delta.keep{color:var(--blue);background:#f0f5fd}\n.note{color:var(--mut);font-size:.8rem;max-width:1080px;\nmargin:0 auto;padding:0 1rem 2rem}\na{color:var(--blue)}\n</style></head><body>\n<header><h1>升级前后对比 — 六个真实案例</h1>\n<div class=sub>左 = 探针关(never 臂实测) · 右 = 探针开(该题实际被升级/保留的那一档实测) ·\n所有回答、毫秒、判分均来自 live sweep 的 trace,无任何改写 ·\n<a href="/__TOKEN__">← 回主 demo</a></div></header>\n<div class=wrap id=root></div>\n<div class=note>黄色高亮 = 回答文本里的 hedging 措辞。判分 = OpenAudioBench\n官方判分器。延迟口径:用户话音落下 → 回答文本完成。</div>\n<script>\nconst CASES=__CASES__;\nconst $=(s)=>document.querySelector(s);\nconst fmt=(ms)=>(ms/1000).toFixed(2);\nconst verdict=(ok)=>`<span class="pill ${ok?\'ok\':\'no\'}">${ok?\'判分:对\':\'判分:错\'}</span>`;\nfunction offCol(c){\n  return `<div class="col left"><h3>探针关 · 只靠小模型</h3>\n   <span class=lat>${fmt(c.off.ms)}<small> s</small></span>${verdict(c.off.ok)}\n   <div class=seg>本地解码 ${c.off.chars} 字符</div>\n   <div class=ans>${c.off.ans}</div></div>`;\n}\nfunction onCol(c){\n  const o=c.on;\n  if(o.mode==="escalated"){\n    const tot=o.total_ms;\n    const segs=[[o.eot_ms,"#0b0b0b","句尾读数"],\n      [Math.max(o.stall_ms,o.expert_s*1000),"#1e9e50","gpt-5.5"],\n      [o.relay_ms,"#8fcda4","转述"]];\n    const bar=segs.map(s=>`<div style="width:${100*s[0]/tot}%;background:${s[1]}"\n      title="${s[2]}"></div>`).join("");\n    return `<div class=col><h3>探针开 · ${o.tier} 档,实际被升级</h3>\n     <span class=lat>${fmt(tot)}<small> s</small></span>${verdict(o.ok)}\n     <div class=gate>探针读数 <b>${o.score.toFixed(3)}</b> ≥ 阈值\n       ${o.thr.toFixed(3)} → <b>升级</b>(读数仅 ${o.eot_ms|0} ms,\n       此刻本地回答一个字都未生成)</div>\n     <div class=bar>${bar}</div>\n     <div class=seg>读数 ${o.eot_ms|0}ms → gpt-5.5 ${o.expert_s.toFixed(2)}s\n       (stall 掩护 ${o.stall_ms|0}ms)→ 转述 ${(o.relay_ms/1000).toFixed(2)}s</div>\n     <div class=ans><b>talker 转述:</b>${o.relay}<br><br><b>专家原文:</b>${o.expert}</div></div>`;\n  }\n  return `<div class=col><h3>探针开 · 判定无需升级</h3>\n   <span class=lat>${fmt(o.ms)}<small> s</small></span>${verdict(o.ok)}\n   <div class=gate>探针读数 <b>${o.score.toFixed(3)}</b> &lt; 阈值\n     ${o.thr.toFixed(3)} → <b>留在本地</b>(省一次专家调用)</div>\n   <div class=ans>${o.ans}</div></div>`;\n}\nfunction delta(c){\n  const o=c.on;\n  if(c.cat==="kept")\n    return `<div class="delta keep">✔ 正确保留:本地又快又对,专家预算留给真正难的题</div>`;\n  if(c.cat==="missed")\n    return `<div class="delta miss">✘ 15% 预算下漏网的"自信错"——回答毫无 hedging、探针分 ${o.score.toFixed(3)} 在 conservative 阈值之下;在 50% 档(aggressive)被抓回。这类错是升级率往上提时曲线继续上涨的原因</div>`;\n  const d=(c.off.ms-o.total_ms)/1000;\n  const acc=c.off.ok? "两边都对" : (o.ok? "错 → 对":"仍错");\n  if(d>0) return `<div class="delta win">▲ 升级反而快 ${d.toFixed(2)} s,且${acc}——这类题就是延迟拐弯的来源</div>`;\n  return `<div class="delta trade">▼ 多花 ${(-d).toFixed(2)} s,换来${acc}——升级的典型形态(用延迟买准确率)</div>`;\n}\n$("#root").innerHTML=CASES.map(c=>`\n <div class=case>\n  <div class=chead><span class="cat ${c.cat}">${c.cat_label}</span>\n   <span class=ref>${c.pool} · ${c.id} · 音频 ${c.audio_s.toFixed(1)}s</span>\n   <div class=q>${c.q}</div>\n   <div class=ref>参考答案:${c.ref}</div></div>\n  <div class=cols>${offCol(c)}${onCol(c)}</div>\n  ${delta(c)}\n </div>`).join("");\n</script></body></html>\n'
