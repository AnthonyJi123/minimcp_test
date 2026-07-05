"""Modal deployment of the MiniCPM-o 4.5 full-duplex eval harness.

Replaces the RunPod-over-SSH workflow: instead of scp'ing the harness to a pod
and running it inside a venv, this bakes the same /workspace layout into a Modal
image and runs the unchanged harness on a serverless H100.

Layout reproduced inside the container (matches the harness's hard-coded paths):
    /workspace/MiniCPM-o-Demo          duplex API (UnifiedProcessor -> DuplexView)
    /workspace/models/MiniCPM-o-4_5    weights  (persisted on a Volume)
    /workspace/minicpm-o-eval/harness  this repo's harness/ (mounted from local)
    /workspace/minicpm-o-eval/results  traces + report.md (persisted on a Volume)

Usage:
    modal run modal_app.py::download_weights      # one-time, ~19 GB -> Volume
    modal run modal_app.py::smoke                 # H100 smoke test
    modal run modal_app.py::run_eval              # all 9 probes on H100
    modal run modal_app.py::run_eval --probes "p2 p3" --repeat 5
    modal run modal_app.py::fetch_report          # print report.md from the Volume
"""
import os
import subprocess
import sys

import modal

MODEL_REPO = "openbmb/MiniCPM-o-4_5"
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"
DEMO_DIR = "/workspace/MiniCPM-o-Demo"
HARNESS_DIR = "/workspace/minicpm-o-eval/harness"
RESULTS_DIR = "/workspace/minicpm-o-eval/results"

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- image: torch 2.8 + this harness's validated stack (SDPA, no flash-attn) -
# Recipe mirrors the project README, NOT the demo's requirements.txt: every
# `minicpmo-utils` release pins librosa 0.9.x, so forcing librosa>=0.10.2 is an
# unsolvable resolve. Instead let `minicpmo-utils[all]` bring librosa 0.9.0 and
# keep `setuptools<81` so librosa 0.9's `pkg_resources` import still works.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsndfile1")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install(
        "minicpmo-utils[all]",       # brings librosa 0.9.0 + audio stack
        "transformers==4.51.0",      # 4.52+ breaks MiniCPM's Resampler weight-init
        "accelerate==1.12.0",
        "setuptools<81",             # librosa 0.9 imports pkg_resources (gone in 81+)
        "pydantic>=2.11",
        "PyYAML",
        "soundfile",
        "edge-tts",
        "opencv-python-headless",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",         # public test endpoint (DuplexServer)
    )
    # the duplex API the harness imports as `core.processors.unified`
    .run_commands(
        f"git clone --depth 1 https://github.com/OpenBMB/MiniCPM-o-Demo {DEMO_DIR}"
    )
    # the unchanged harness from this repo, mounted at the path it expects
    .add_local_dir(os.path.join(HERE, "harness"), HARNESS_DIR)
    # the GUI page (mounted, so editing it only needs a fast redeploy, no rebuild)
    .add_local_dir(os.path.join(HERE, "web"), "/web")
)

# Lightweight image for the one-time weight pull — decoupled from the model
# image so rebuilding the heavy stack never forces a 19 GB re-download.
download_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("minicpm-o45-eval", image=image)

weights = modal.Volume.from_name("minicpm-o45-weights", create_if_missing=True)
results = modal.Volume.from_name("minicpm-o45-results", create_if_missing=True)

VOLUMES = {"/workspace/models": weights, RESULTS_DIR: results}


@app.function(image=download_image, volumes={"/workspace/models": weights}, timeout=60 * 60)
def download_weights(force: bool = False):
    """Populate the weights Volume once (CPU box — keeps H100 time for inference)."""
    from huggingface_hub import snapshot_download

    if os.path.exists(os.path.join(MODEL_DIR, "config.json")) and not force:
        print(f">>> weights already present at {MODEL_DIR}; skipping", flush=True)
        return
    print(f">>> downloading {MODEL_REPO} -> {MODEL_DIR}", flush=True)
    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR)
    weights.commit()
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(MODEL_DIR) for f in fs
    )
    print(f">>> done, {total / 1e9:.1f} GB on the Volume", flush=True)


@app.function(timeout=60 * 10)
def verify_image():
    """CPU-only import gate: surface any dependency breakage in the heavy image
    (e.g. librosa 0.9 vs numpy, transformers pin, the duplex API) before paying
    for an H100."""
    import importlib
    import numpy, librosa, transformers, torch  # noqa: F401

    print("numpy", numpy.__version__, "| librosa", librosa.__version__,
          "| transformers", transformers.__version__, "| torch", torch.__version__,
          flush=True)
    sys.path.insert(0, DEMO_DIR)
    importlib.import_module("core.processors.unified")
    importlib.import_module("core.schemas.duplex")
    # harness modules import cleanly (stimuli pulls edge_tts/soundfile/PIL)
    sys.path.insert(0, HARNESS_DIR)
    sys.path.insert(0, os.path.join(HARNESS_DIR, "probes"))
    for m in ("timeline", "stimuli", "judge", "probe_base", "session"):
        importlib.import_module(m)
    print(">>> all imports OK", flush=True)


@app.function(timeout=60 * 10)
def verify_audio():
    """CPU-only audio-path gate. librosa 0.9 must work under numpy 2, and edge-tts
    must actually synthesize — the harness silently substitutes SILENCE on edge-tts
    failure (stimuli.py), which would invalidate the spoken-stimulus probes without
    crashing. Catch that here, free, before the H100 run."""
    sys.path.insert(0, HARNESS_DIR)
    import numpy as np
    import stimuli

    wav = os.path.join(DEMO_DIR, "tests/cases/common/user_audio/000_user_audio0.wav")
    w = stimuli.load_wav(wav)  # exercises librosa 0.9 + numpy 2
    print(f">>> librosa.load OK: {w.shape} {w.dtype} peak={float(np.abs(w).max()):.3f}",
          flush=True)

    chunks = stimuli.utterance_chunks("switch to English now", voice="en-US-AriaNeural")
    energy = float(np.concatenate(chunks).std())
    print(f">>> edge-tts OK: {len(chunks)} chunk(s), energy={energy:.4f}", flush=True)
    if energy < 1e-4:
        raise RuntimeError("edge-tts produced silence — spoken-stimulus probes would be invalid")
    print(">>> audio path OK", flush=True)


@app.function(gpu="H100", volumes=VOLUMES, timeout=60 * 20)
def probe_stream_audio():
    """Feed REAL speech (the demo wav) through the duplex loop and inspect the
    model's audio_data on speaking chunks — settles float32-vs-int16 and the rate,
    and confirms the streaming path actually produces speech."""
    import base64

    import numpy as np

    sys.path.insert(0, HARNESS_DIR)
    sys.path.insert(0, os.path.join(HARNESS_DIR, "probes"))
    from session import DuplexSession
    import librosa
    import stimuli

    sess = DuplexSession()
    sess.prepare("你是一个实时视频助手。请用中文回应用户，并主动解说你看到的画面。")
    wav = os.path.join(DEMO_DIR, "tests/cases/common/user_audio/000_user_audio0.wav")
    user, _ = librosa.load(wav, sr=16000, mono=True)
    chunks = []
    for i in range(0, len(user), 16000):
        c = user[i:i + 16000]
        chunks.append((np.pad(c, (0, 16000 - len(c))) if len(c) < 16000 else c).astype(np.float32))
    while len(chunks) < 16:
        chunks.append(np.zeros(16000, np.float32))

    frame = stimuli.card("CAT", shape="circle")
    d, spoke = sess.duplex, 0
    for i, a in enumerate(chunks[:16]):
        d.prefill(audio_waveform=a, frame_list=[frame])
        r = d.generate()
        d.finalize()
        if not r.is_listen:
            spoke += 1
        ad = getattr(r, "audio_data", None)
        info = ""
        if isinstance(ad, str) and not r.is_listen:
            raw = base64.b64decode(ad)
            f32 = np.frombuffer(raw, dtype="<f4")
            i16 = np.frombuffer(raw, dtype="<i2")
            info = (f" | bytes={len(raw)} f32[n={f32.size} min={f32.min():.3f} max={f32.max():.3f}]"
                    f" i16[n={i16.size} min={i16.min()} max={i16.max()}]")
        print(f"[{i:02d}] listen={int(r.is_listen)} eot={int(r.end_of_turn)} "
              f"text={r.text!r}{info}", flush=True)
    print(f">>> spoke={spoke}/16", flush=True)
    sess.stop()


def _run_script(script: str, args=None, env=None):
    """Run a harness script as a subprocess so the harness stays unmodified."""
    cmd = [sys.executable, script] + (args or [])
    e = dict(os.environ)
    e.setdefault("PYTHONUNBUFFERED", "1")
    if env:
        e.update(env)
    print(f">>> $ {' '.join(cmd)}  (cwd={HARNESS_DIR})", flush=True)
    p = subprocess.run(cmd, cwd=HARNESS_DIR, env=e)
    if p.returncode != 0:
        raise RuntimeError(f"{script} exited {p.returncode}")


@app.function(gpu="H100", volumes=VOLUMES, timeout=60 * 30)
def smoke():
    """Phase-1 gate: model loads and the duplex loop produces >=1 speaking chunk."""
    import torch

    print(">>> GPU:", torch.cuda.get_device_name(0), flush=True)
    assert os.path.exists(os.path.join(MODEL_DIR, "config.json")), (
        "weights missing — run download_weights first"
    )
    _run_script("smoke.py")


@app.function(gpu="H100", volumes=VOLUMES, timeout=60 * 60 * 2)
def run_eval(probes: str = "", repeat: int = 1):
    """Run probes (default: all 9) on an H100, persisting traces to the Volume."""
    import torch

    print(">>> GPU:", torch.cuda.get_device_name(0), flush=True)
    assert os.path.exists(os.path.join(MODEL_DIR, "config.json")), (
        "weights missing — run download_weights first"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    args = probes.split() if probes.strip() else []
    _run_script("run_eval.py", args, env={"REPEAT": str(repeat)})
    results.commit()

    report = os.path.join(RESULTS_DIR, "report.md")
    if os.path.exists(report):
        print("\n" + "=" * 70 + "\n>>> report.md\n" + "=" * 70, flush=True)
        with open(report, encoding="utf-8") as f:
            print(f.read(), flush=True)


@app.function(volumes={RESULTS_DIR: results})
def fetch_report():
    """Print the persisted report.md (and list result files) without a GPU."""
    report = os.path.join(RESULTS_DIR, "report.md")
    if not os.path.exists(report):
        print("no report.md yet — run run_eval first")
        return
    print("\n".join(sorted(os.listdir(RESULTS_DIR))))
    print("=" * 70)
    with open(report, encoding="utf-8") as f:
        print(f.read())


# ---- public test endpoint ----------------------------------------------------
# A warm-on-H100 FastAPI service: upload an image + a question, and the duplex
# model (the proven path) answers via its per-second speak/listen loop. Scales to
# zero after `scaledown_window` so an idle endpoint stops billing.
# A bare "answer the user" prompt leaves this model in listen mode (speaking_chunks=0).
# The probes show it reliably SPEAKS when told to narrate the scene continuously, so
# the default mirrors that and folds in answering the user's question.
DEFAULT_SYSTEM = (
    "你是一个实时视频助手。请持续、详细地解说你当前看到的画面，不要长时间停顿；"
    "画面变化时立刻描述新画面。同时听取并回答用户的问题，"
    "用户用什么语言就用什么语言回答。"
)

INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>MiniCPM-o 4.5 — duplex test</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:32px auto;padding:0 16px;color:#111}
 h1{font-size:20px} .muted{color:#666;font-size:13px}
 textarea,input,button{font:inherit} textarea{width:100%;box-sizing:border-box;padding:8px}
 .row{margin:14px 0} button{padding:9px 18px;border:0;border-radius:8px;background:#10a37f;color:#fff;cursor:pointer}
 button:disabled{opacity:.5;cursor:wait} #out{white-space:pre-wrap;background:#f6f7f8;border-radius:8px;padding:14px;min-height:40px}
 img#prev{max-width:220px;border-radius:8px;display:none;margin-top:8px}
</style></head><body>
<h1>MiniCPM-o 4.5 · full-duplex test</h1>
<p class="muted">Upload an image and ask a question. The duplex model runs its 1&nbsp;Hz
speak/listen loop and returns what it says. First call after idle cold-starts the H100 (~50&nbsp;s).</p>
<div class="row"><label>Question<br><textarea id="prompt" rows="2"
 placeholder="What is in this picture? / 画面里有什么？"></textarea></label></div>
<div class="row"><input type="file" id="image" accept="image/*">
 <img id="prev"></div>
<div class="row"><button id="go">Ask</button>
 <span id="status" class="muted"></span></div>
<div class="row"><b>Response</b><div id="out"></div></div>
<script>
 const img=document.getElementById('image'),prev=document.getElementById('prev');
 img.onchange=()=>{if(img.files[0]){prev.src=URL.createObjectURL(img.files[0]);prev.style.display='block';}};
 document.getElementById('go').onclick=async()=>{
   const b=document.getElementById('go'),st=document.getElementById('status'),out=document.getElementById('out');
   const fd=new FormData();
   fd.append('prompt',document.getElementById('prompt').value);
   if(img.files[0])fd.append('image',img.files[0]);
   b.disabled=true;st.textContent='running…';out.textContent='';
   const t=Date.now();
   try{
     const r=await fetch('infer',{method:'POST',body:fd});
     const j=await r.json();
     out.textContent=j.error?('ERROR: '+j.error):(j.response||'(model stayed silent)');
     st.textContent=`${((Date.now()-t)/1000).toFixed(1)}s · ${j.speaking_chunks??0} speaking / ${j.chunks??0} chunks`;
   }catch(e){out.textContent='ERROR: '+e;st.textContent='';}
   b.disabled=false;
 };
</script></body></html>"""


@app.cls(image=image, gpu="H100", volumes=VOLUMES,
         scaledown_window=300, timeout=600, max_containers=1)
@modal.concurrent(max_inputs=8)  # serve the page during inference; GPU work is lock-serialized
class DuplexServer:
    @modal.enter()
    def _load(self):
        # Keep container start-up cheap so GET / (the GUI page) is served fast.
        # The 20GB model loads lazily on the first /infer — that wait is contextual
        # (the page's cold-start UI handles it) instead of blocking the page itself.
        import threading

        sys.path.insert(0, HARNESS_DIR)
        sys.path.insert(0, os.path.join(HARNESS_DIR, "probes"))
        self.sess = None
        self.lock = threading.Lock()   # one GPU user at a time (/infer or a stream)
        self.out_sr = 24000            # model TTS output rate (confirmed by probe)

    def _ensure(self):
        if self.sess is None:
            assert os.path.exists(os.path.join(MODEL_DIR, "config.json")), (
                "weights missing — run download_weights first"
            )
            from session import DuplexSession

            self.sess = DuplexSession()
            print(f">>> model warm in {self.sess.load_s:.1f}s", flush=True)

    def _infer(self, prompt: str, image_bytes, system_prompt: str, max_chunks: int):
        import io
        import time

        from PIL import Image
        import stimuli

        if image_bytes:
            frame = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((448, 448))
        else:
            frame = stimuli.card("(no image)")

        prompt = (prompt or "").strip()
        if prompt:
            voice = ("zh-CN-XiaoxiaoNeural"
                     if any("一" <= c <= "鿿" for c in prompt)
                     else "en-US-AriaNeural")
            audio = stimuli.utterance_chunks(prompt, voice=voice)
        else:
            voice, audio = None, [stimuli.silence_chunk()]
        n_speech = len(audio)  # real utterance length, before silence padding
        n = min(max(max_chunks, n_speech + 18), 48)
        audio += [stimuli.silence_chunk() for _ in range(n - n_speech)]

        with self.lock:
            self._ensure()  # lazy model load on first request
            self.sess.prepare(system_prompt or DEFAULT_SYSTEM)
            rows, spoke = [], False
            t0 = time.time()
            try:
                for i in range(n):
                    rec = self.sess.step(i, audio[i], frame, loop_t0=t0)
                    rows.append(rec)
                    if rec.speaking:
                        spoke = True
                    # once the user's speech is consumed and the model has taken
                    # and finished a turn, stop (don't burn GPU on trailing silence)
                    if rec.end_of_turn and spoke and i >= n_speech:
                        break
            finally:
                self.sess.stop()

        return {
            "response": "".join(r.text for r in rows).strip(),
            "chunks": len(rows),
            "speaking_chunks": sum(1 for r in rows if r.speaking),
            "voice": voice,
        }

    # -- live streaming (WebSocket) helpers -----------------------------------
    def _decode_tick(self, msg):
        """One client tick -> (16k mono float32 audio [1s], PIL frame or None)."""
        import base64
        import io

        import numpy as np
        from PIL import Image

        a = np.frombuffer(base64.b64decode(msg["audio"]), dtype="<i2")
        a = a.astype(np.float32) / 32768.0
        a = a[:16000] if a.size >= 16000 else np.pad(a, (0, 16000 - a.size))
        frame = None
        if msg.get("image"):
            frame = Image.open(io.BytesIO(base64.b64decode(msg["image"]))).convert("RGB").resize((448, 448))
        return a, frame

    def _encode_audio(self, audio_data):
        """Model audio_data -> base64 int16 PCM (mono) for browser playback.
        The duplex result's audio_data is already a base64 string; numpy/tensor/bytes
        are handled too for safety."""
        import base64

        import numpy as np

        if audio_data is None:
            return None
        if isinstance(audio_data, str):
            return audio_data or None          # model already returns base64
        if isinstance(audio_data, (bytes, bytearray)):
            return base64.b64encode(bytes(audio_data)).decode("ascii")
        a = audio_data
        try:
            import torch
            if isinstance(a, torch.Tensor):
                a = a.detach().float().cpu().numpy()
        except Exception:
            pass
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.size == 0:
            return None
        pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        return base64.b64encode(pcm).decode("ascii")

    def _stream_tick(self, audio, frame):
        """Drive the duplex directly (not sess.step) to capture r.audio_data too."""
        d = self.sess.duplex
        d.prefill(audio_waveform=audio, frame_list=[frame] if frame is not None else None)
        try:
            r = d.generate()
        finally:
            d.finalize()
        return r

    @modal.asgi_app()
    def web(self):
        import asyncio

        from fastapi import FastAPI, File, Form, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, JSONResponse

        api = FastAPI()

        @api.get("/", response_class=HTMLResponse)
        def index():
            try:
                with open("/web/index.html", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                return INDEX_HTML  # built-in fallback

        # NOTE: sync handler on purpose. edge-tts (in stimuli.tts) calls
        # asyncio.run(), which raises inside an async handler's running event loop
        # — the harness then swallows it and substitutes SILENCE, so the model
        # never hears the prompt. A sync def runs in FastAPI's threadpool (no
        # running loop), so asyncio.run() works. `image: bytes` avoids an await.
        @api.post("/infer")
        def infer(prompt: str = Form(""), max_chunks: int = Form(24),
                  system_prompt: str = Form(""), image: bytes = File(None)):
            try:
                return JSONResponse(self._infer(prompt, image or None,
                                                system_prompt, max_chunks))
            except Exception as e:  # surface errors to the caller instead of a 500 page
                import traceback
                traceback.print_exc()
                return JSONResponse({"error": str(e)}, status_code=500)

        @api.websocket("/ws")
        async def stream(ws: WebSocket):
            """Real-time full-duplex: client streams 1s (audio, frame) ticks; we run
            the duplex loop per chunk and stream back text + the model's own audio."""
            await ws.accept()
            if not self.lock.acquire(blocking=False):   # one GPU session at a time
                await ws.send_json({"type": "busy"})
                await ws.close()
                return
            try:
                first = await ws.receive_json()
                sysp = (first.get("system_prompt") or "").strip() or DEFAULT_SYSTEM
                await ws.send_json({"type": "loading"})     # may cold-start the H100
                await asyncio.to_thread(self._ensure)
                await asyncio.to_thread(self.sess.prepare, sysp)
                await ws.send_json({"type": "ready", "out_sr": self.out_sr})

                probed = False
                while True:
                    msg = await ws.receive_json()
                    if msg.get("type") == "stop":
                        break
                    if msg.get("type") != "tick":
                        continue
                    audio, frame = self._decode_tick(msg)
                    r = await asyncio.to_thread(self._stream_tick, audio, frame)
                    ad = getattr(r, "audio_data", None)
                    if not probed and ad is not None:
                        import base64
                        info = {"pytype": type(ad).__name__}
                        if isinstance(ad, str):
                            try:
                                raw = base64.b64decode(ad)
                                info["bytes"] = len(raw)      # /2 = int16 samples
                                info["head"] = raw[:12].hex()  # 'RIFF' => wav
                            except Exception as ex:
                                info["b64_err"] = str(ex)
                        print(">>> audio_data probe:", info, flush=True)
                        probed = True
                    out = {"type": "step",
                           "text": r.text or "",
                           "is_listen": bool(r.is_listen),
                           "eot": bool(r.end_of_turn)}
                    # only ship audio while the model is speaking (float32 @ 24kHz);
                    # listen chunks are silence — skip them to save bandwidth
                    if not bool(r.is_listen):
                        aud = self._encode_audio(ad)
                        if aud:
                            out["audio"] = aud
                    await ws.send_json(out)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    await ws.send_json({"type": "error", "message": str(e)})
                except Exception:
                    pass
            finally:
                try:
                    await asyncio.to_thread(self.sess.stop)
                except Exception:
                    pass
                self.lock.release()

        return api
