"""Phase-1 smoke test: load MiniCPM-o 4.5 and run a short duplex loop.

Gate: model loads, duplex prepare/prefill/generate/finalize loop runs, and we
observe per-chunk {is_listen, text, end_of_turn, cost} with at least one
speaking (is_listen=False) chunk.
"""
import sys, time, os
import numpy as np

DEMO = "/workspace/MiniCPM-o-Demo"
MODEL = "/workspace/models/MiniCPM-o-4_5"
REF = "/workspace/MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav"
USER_WAV = "/workspace/MiniCPM-o-Demo/tests/cases/common/user_audio/000_user_audio0.wav"

sys.path.insert(0, DEMO)
from core.processors.unified import UnifiedProcessor
from core.schemas.duplex import DuplexConfig
from PIL import Image, ImageDraw
import librosa

SR = 16000
CHUNK = SR  # 1 second


def make_frame(label, color=(40, 90, 160)):
    img = Image.new("RGB", (448, 448), color)
    d = ImageDraw.Draw(img)
    d.rectangle([120, 120, 328, 328], fill=(230, 60, 60))
    d.text((150, 210), label, fill=(255, 255, 255))
    return img


def chunkify(wave, n):
    out = []
    for i in range(0, max(len(wave), 1), n):
        c = wave[i:i + n]
        if len(c) < n:
            c = np.pad(c, (0, n - len(c)))
        out.append(c.astype(np.float32))
    return out


def main():
    t0 = time.time()
    print(">>> loading model ...", flush=True)
    proc = UnifiedProcessor(
        model_path=MODEL,
        ref_audio_path=REF,
        device="cuda",
        attn_implementation="sdpa",
        compile=False,
    )
    print(f">>> model loaded in {time.time()-t0:.1f}s", flush=True)

    duplex = proc.set_duplex_mode()
    duplex.prepare(
        system_prompt_text="你是一个实时视频助手，请用中文简短回应用户，并在适当时主动解说你看到的画面。",
        ref_audio_path=REF,
    )

    # user audio (a real sample utterance), chunked to 1s; pad timeline to 16 chunks
    user, _ = librosa.load(USER_WAV, sr=SR, mono=True)
    audio_chunks = chunkify(user, CHUNK)
    n = max(16, len(audio_chunks))
    while len(audio_chunks) < n:
        audio_chunks.append(np.zeros(CHUNK, dtype=np.float32))

    frame = make_frame("CAT")
    speaking = 0
    print(">>> running duplex loop", flush=True)
    for i in range(n):
        t = time.time()
        duplex.prefill(audio_waveform=audio_chunks[i], frame_list=[frame])
        r = duplex.generate()
        duplex.finalize()
        kv = getattr(proc, "kv_cache_length", None)
        if not r.is_listen:
            speaking += 1
        print(f"[{i:02d}] listen={r.is_listen} eot={r.end_of_turn} "
              f"kv={kv} cost={ (time.time()-t)*1000:.0f}ms text={r.text!r}", flush=True)
        if r.end_of_turn and i > 6:
            break
    duplex.stop()
    print(f">>> DONE speaking_chunks={speaking} total_wall={time.time()-t0:.1f}s")
    print(">>> SMOKE_PASS" if speaking >= 1 else ">>> SMOKE_NO_SPEECH (loop ok, model stayed silent)")


if __name__ == "__main__":
    main()
