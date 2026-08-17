"""One-off audit: scan every wav in gate-data/audio_pool for silence /
near-silence (broken TTS renders). Stdlib only. ~$0.
Run: modal run scan_wavs.py
"""
import json
import modal

app = modal.App("scan-wavs")
vol = modal.Volume.from_name("gate-data")


@app.function(image=modal.Image.debian_slim(), volumes={"/data": vol},
              timeout=60 * 20)
def scan():
    import os
    import struct

    out = []
    pool = "/data/audio_pool"
    for fn in sorted(os.listdir(pool)):
        if not fn.endswith(".wav"):
            continue
        raw = open(os.path.join(pool, fn), "rb").read()
        data = raw[44:]
        n = len(data) // 2
        vals = struct.unpack(f"<{n}h", data[: n * 2])
        peak = max(abs(v) for v in vals) if n else 0
        # leading silence in seconds (sr 24000)
        sr = 24000
        lead = 0
        for i in range(0, n, sr):
            if max((abs(v) for v in vals[i:i + sr]), default=0) > 100:
                break
            lead += 1
        out.append({"id": fn[:-4], "dur_s": round(n / sr, 1),
                    "peak": peak, "lead_silence_s": lead})
    return out


@app.local_entrypoint()
def main():
    rows = scan.remote()
    bad = [r for r in rows if r["peak"] < 500]
    quiet = [r for r in rows if 500 <= r["peak"] < 3000]
    long_lead = [r for r in rows if r["peak"] >= 500 and r["lead_silence_s"] >= 3]
    print(f"total={len(rows)} silent/broken(peak<500)={len(bad)} "
          f"quiet(peak<3000)={len(quiet)} long_lead_silence={len(long_lead)}")
    for r in bad:
        print("BAD  ", r)
    for r in quiet:
        print("QUIET", r)
    for r in long_lead:
        print("LEAD ", r)
    with open("figures/wav_audit.json", "w") as f:
        json.dump(rows, f)
