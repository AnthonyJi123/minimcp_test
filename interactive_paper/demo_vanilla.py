"""VANILLA MiniCPM-o 4.5 duplex demo (8bl) — the control arm.

User request 2026-09-01: "give me a stock MiniCPM-o 4.5 demo with none
of our mechanisms, so we can compare side by side — I suspect your
demo's probe-off is not equal to stock."

This app is the bare loop and nothing else: mic 16k PCM -> 1 s
streaming_prefill -> streaming_generate -> audio/text back. No L22
hook, no probe, no act gate, no stall, no relay, no history, no gate
events. The only knob is ?cfg=:

  cfg=official  — the official pytorch-simple-demo serving config:
                  top_k=20, force_listen_count=3 (startup guard),
                  system prompt "You are a friendly assistant."
  cfg=ours      — the exact config our gate demo runs (as_duplex class
                  defaults: top_k=100, force_listen_count=0,
                  "Streaming Omni Conversation.")

Known config deltas vs official (verified against
OpenBMB/minicpm-o-4_5-pytorch-simple-demo @ main): top_k 20 vs 100,
force_listen_count 3 vs 0, system prompt text. The duplex decode code
itself (incl. the mid-turn listen suppression at modeling l.3100) is
byte-identical between the official repo and our checkpoint.

Deploy: modal deploy demo_vanilla.py
Page:   https://rhe9527--vanilla-duplex.modal.run/62dc5cd9/
"""
import json
import os
import time

import modal

TOKEN = "62dc5cd9"

app = modal.App("minicpm-vanilla")
weights = modal.Volume.from_name("minicpm-o45-weights")
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"
PROMPT_WAV = f"{MODEL_DIR}/assets/system_ref_audio.wav"

from modal_app import OPENAI  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(_HERE, "modal_app.py")

web_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("fastapi[standard]")
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
        "fastapi[standard]",
    )
    .add_local_file(_APP_PY, "/root/modal_app.py"))

CFGS = {
    # OpenBMB/minicpm-o-4_5-pytorch-simple-demo DuplexConfig defaults
    "official": {"top_k": 20, "force_listen_count": 3,
                 "sys": "You are a friendly assistant."},
    # what our gate demo effectively runs (as_duplex class defaults)
    "ours": {"top_k": 100, "force_listen_count": 0,
             "sys": "Streaming Omni Conversation."},
}


@app.cls(image=gpu_image, gpu="H100",
         volumes={"/workspace/models": weights},
         secrets=[OPENAI], timeout=60 * 60, scaledown_window=420)
@modal.concurrent(max_inputs=4)
class Vanilla:

    @modal.enter()
    def load(self):
        import glob as _glob
        import shutil
        import threading

        import librosa
        import torch
        from transformers import AutoModel, AutoTokenizer

        t0 = time.time()
        cache = os.path.expanduser("~/.cache/huggingface/modules/"
                                   "transformers_modules/"
                                   + os.path.basename(MODEL_DIR))
        os.makedirs(cache, exist_ok=True)
        for f in _glob.glob(f"{MODEL_DIR}/*.py"):
            shutil.copy(f, cache)      # STOCK checkpoint sources only
        self.model = AutoModel.from_pretrained(
            MODEL_DIR, trust_remote_code=True,
            attn_implementation="sdpa", torch_dtype=torch.bfloat16,
            init_vision=False, init_audio=True,
            init_tts=True).eval().cuda()
        _ = AutoTokenizer.from_pretrained(MODEL_DIR,
                                          trust_remote_code=True)
        self.duplex = self.model.as_duplex()
        self.ref, _sr = librosa.load(PROMPT_WAV, sr=16000, mono=True)
        self.lock = threading.Lock()
        self.load_s = round(time.time() - t0, 1)
        print(f">>> Vanilla ready in {self.load_s}s", flush=True)

    @modal.asgi_app(label="vanilla-duplex-voice")
    def ws_app(self):
        import asyncio
        import base64
        import threading as _th

        import numpy as np
        from fastapi import FastAPI, WebSocket
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse

        wapp = FastAPI()
        wapp.add_middleware(CORSMiddleware, allow_origins=["*"],
                            allow_methods=["*"], allow_headers=["*"])

        @wapp.get(f"/{TOKEN}/ready")
        def ready():
            return JSONResponse({"ready": True, "load_s": self.load_s,
                                 "busy": self.lock.locked()})

        @wapp.websocket(f"/{TOKEN}/ws")
        async def ws(sock: WebSocket):
            await sock.accept()
            cfg = CFGS.get(sock.query_params.get("cfg", "official"),
                           CFGS["official"])
            if not self.lock.acquire(timeout=3):
                await sock.send_json({"type": "error",
                                      "msg": "model busy"})
                await sock.close()
                return
            loop = asyncio.get_event_loop()
            stop = _th.Event()
            inbox, ilock = [], _th.Lock()

            def emit(m):
                asyncio.run_coroutine_threadsafe(sock.send_json(m), loop)

            def prep():
                self.duplex.force_listen_count = cfg[
                    "force_listen_count"]
                self.duplex.prepare(
                    prefix_system_prompt=cfg["sys"],
                    ref_audio=self.ref, prompt_wav_path=PROMPT_WAV)

            def chunk_loop():
                pend = np.zeros(0, dtype=np.float32)
                while not stop.is_set():
                    with ilock:
                        got, inbox[:] = inbox[:], []
                    if got:
                        pend = np.concatenate([pend] + got)
                    if len(pend) < 16000:
                        time.sleep(0.02)
                        continue
                    ch, pend = pend[:16000], pend[16000:]
                    ok = self.duplex.streaming_prefill(audio_waveform=ch)
                    if not ok.get("success"):
                        continue
                    r = self.duplex.streaming_generate(
                        prompt_wav_path=PROMPT_WAV, top_k=cfg["top_k"])
                    wf = r.get("audio_waveform")
                    if not r["is_listen"] and wf is not None and len(wf):
                        i16 = (np.clip(np.asarray(wf, dtype=np.float32),
                                       -1, 1) * 32767).astype("<i2")
                        emit({"type": "audio", "sr": 24000,
                              "pcm": base64.b64encode(
                                  i16.tobytes()).decode()})
                    if not r["is_listen"] and r.get("text"):
                        emit({"type": "text", "v": r["text"]})
                    emit({"type": "chunk",
                          "listen": bool(r["is_listen"]),
                          "eot": bool(r.get("end_of_turn"))})

            try:
                await sock.send_json(
                    {"type": "hello", "cfg": cfg,
                     "mode": "VANILLA stock MiniCPM-o 4.5 duplex — "
                             "no probe, no gate, no stall/relay, "
                             "no hooks; bare prefill/generate loop"})
                await loop.run_in_executor(None, prep)
                await sock.send_json({"type": "phase", "v": "listening"})
                worker = _th.Thread(target=chunk_loop, daemon=True)
                worker.start()
                while True:
                    msg = await sock.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    txt = msg.get("text")
                    if txt:
                        try:
                            if json.loads(txt).get("type") == "stop":
                                break
                        except Exception:
                            pass
                        continue
                    b = msg.get("bytes")
                    if b:
                        f = (np.frombuffer(b, dtype=np.int16)
                             .astype(np.float32) / 32768.0)
                        with ilock:
                            inbox.append(f)
            except RuntimeError:
                pass
            finally:
                stop.set()
                self.duplex.set_session_stop()
                await loop.run_in_executor(None, worker.join)
                self.duplex.clear_session_stop()
                self.lock.release()

        return wapp


@app.function(image=web_image)
@modal.asgi_app(label="vanilla-duplex")
def page():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    wapp = FastAPI()

    @wapp.get(f"/{TOKEN}/")
    def root():
        return HTMLResponse(HTML.replace("__TOKEN__", TOKEN))
    return wapp


HTML = r"""<!doctype html><html><head><meta charset=utf-8>
<title>vanilla MiniCPM-o 4.5 duplex</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f5f6f8;color:#1c2733}
.wrap{max-width:880px;margin:0 auto;padding:1.2rem}
h1{font-size:1.25rem}.sub{color:#5b6b7c;font-size:.85rem}
.card{background:#fff;border:1px solid #dfe5ec;border-radius:10px;
 padding:1rem;margin:.8rem 0}
button{font:inherit;padding:.5rem 1rem;border-radius:8px;
 border:1px solid #c8d2dd;background:#fff;cursor:pointer}
button.primary{background:#7a3fd1;color:#fff;border-color:#7a3fd1}
button:disabled{opacity:.45}
#vu{height:6px;background:#7a3fd1;width:0%;border-radius:3px}
#log{font:12px/1.55 ui-monospace,monospace;background:#0e1621;color:#cfe3f7;
 border-radius:8px;padding:.7rem;height:280px;overflow-y:auto;
 white-space:pre-wrap}
.pill{display:inline-block;padding:.1rem .55rem;border-radius:999px;
 font-size:.75rem;background:#e8edf3;margin-right:.4rem}
</style></head><body><div class=wrap>
<h1>VANILLA MiniCPM-o 4.5 · stock duplex（对照组）</h1>
<div class=sub>裸 prefill/generate 循环：无 probe、无 gate、无垫话、无 relay、
无 hook。cfg=official 完整复刻官方 pytorch-simple-demo 的 serving 参数
（top_k 20 / force_listen 3 / friendly-assistant prompt）。戴耳机。</div>
<div class=card>
 cfg <select id=cfg><option selected>official</option><option>ours</option></select>
 <button id=talk class=primary disabled>GPU starting…</button>
 <div style="margin-top:.6rem"><div id=vu></div></div>
 <div class=sub id=state>—</div>
</div>
<div class=card><b>talker</b> <span id=phase class=pill>idle</span>
 <div id=text class=sub style="min-height:2.2rem"></div></div>
<div class=card><div id=log></div></div>
</div><script>
const T="__TOKEN__";
const VOICE="https://rhe9527--vanilla-duplex-voice.modal.run";
const $=s=>document.querySelector(s);
let ws=null,ac=null,micStream=null,proc=null,talking=false,
    playCtx=null,playT=0,gpuReady=false,turnText="";
function log(m){const l=$("#log");
 l.innerHTML+=`<div><b>${new Date().toLocaleTimeString()}</b> ${m}</div>`;
 l.scrollTop=l.scrollHeight;}
function playPCM(b64,sr){
 if(!playCtx)playCtx=new (window.AudioContext||window.webkitAudioContext)();
 if(playCtx.state==="suspended")playCtx.resume();
 const raw=atob(b64),n=raw.length>>1,f=new Float32Array(n);
 for(let i=0;i<n;i++){let v=raw.charCodeAt(2*i)|(raw.charCodeAt(2*i+1)<<8);
  if(v>=32768)v-=65536;f[i]=v/32768;}
 const buf=playCtx.createBuffer(1,n,sr);buf.getChannelData(0).set(f);
 const src=playCtx.createBufferSource();src.buffer=buf;
 src.connect(playCtx.destination);
 playT=Math.max(playT,playCtx.currentTime);
 src.start(playT);playT+=buf.duration;}
async function warm(){
 const t0=Date.now();let j=null;
 const tick=setInterval(()=>{if(!gpuReady)$("#state").textContent=
  `GPU starting… ${((Date.now()-t0)/1000|0)}s`;},1000);
 for(let i=0;i<90&&!j;i++){
  try{const r=await fetch(`${VOICE}/${T}/ready`,
   {signal:AbortSignal.timeout(30000)});if(r.ok)j=await r.json();}
  catch(_){}
  if(!j)await new Promise(s=>setTimeout(s,4000));}
 clearInterval(tick);
 if(j&&j.ready){gpuReady=true;$("#talk").disabled=false;
  $("#talk").textContent="Start vanilla session";
  $("#state").textContent=`GPU ready (${j.load_s}s)`;}
 else $("#state").textContent="GPU timeout — reload.";}
warm();
function stopTalk(){talking=false;
 try{if(ws&&ws.readyState<2){ws.send(JSON.stringify({type:"stop"}));ws.close();}}catch(_){}
 try{if(proc)proc.disconnect()}catch(_){}
 try{if(ac)ac.close()}catch(_){}
 try{if(micStream)micStream.getTracks().forEach(t=>t.stop())}catch(_){}
 ws=ac=proc=micStream=null;
 $("#talk").textContent="Start vanilla session";$("#phase").textContent="idle";}
async function startTalk(){
 try{micStream=await navigator.mediaDevices.getUserMedia({audio:{
  channelCount:1,echoCancellation:true,noiseSuppression:true,
  autoGainControl:true}});}catch(e){log("mic denied: "+e);return;}
 ac=new AudioContext();
 const src=ac.createMediaStreamSource(micStream);
 proc=ac.createScriptProcessor(2048,1,1);
 const ratio=ac.sampleRate/16000;
 ws=new WebSocket(`${VOICE.replace("https","wss")}/${T}/ws`
  +`?cfg=${$("#cfg").value}`);
 ws.onmessage=ev=>handle(JSON.parse(ev.data));
 ws.onclose=()=>{if(talking){log("session closed");stopTalk();}};
 ws.onopen=()=>{src.connect(proc);proc.connect(ac.destination);
  talking=true;$("#talk").textContent="■ End session";
  log(`vanilla session open (cfg=${$("#cfg").value})`);};
 proc.onaudioprocess=e=>{
  const f=e.inputBuffer.getChannelData(0);
  let ss=0;for(let i=0;i<f.length;i++)ss+=f[i]*f[i];
  $("#vu").style.width=Math.min(100,Math.sqrt(ss/f.length)*700)+"%";
  if(!ws||ws.readyState!==1)return;
  const n=Math.floor(f.length/ratio),out=new Int16Array(n);
  for(let i=0;i<n;i++){const v=f[Math.floor(i*ratio)];
   out[i]=Math.max(-32768,Math.min(32767,v*32767));}
  ws.send(out.buffer);};}
$("#talk").onclick=()=>{talking?stopTalk():startTalk()};
function handle(m){
 if(m.type==="hello")log("config: "+JSON.stringify(m.cfg));
 else if(m.type==="phase")$("#phase").textContent=m.v;
 else if(m.type==="audio")playPCM(m.pcm,m.sr);
 else if(m.type==="text"){turnText+=m.v;
  $("#text").textContent=turnText.slice(-300);log("TEXT: "+m.v);}
 else if(m.type==="chunk"&&m.eot){turnText="";
  log("— turn ended —");$("#phase").textContent="listening";}
 else if(m.type==="error")log("ERROR: "+m.msg);
}
</script></body></html>"""
