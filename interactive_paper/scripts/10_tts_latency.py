"""P2(c): EOT -> first-audible percentiles from the TTS-on balanced-arm
re-run (frozen pool, suffix _v3tts).

Definitions (t=0 = gate decision, i.e. end of the EOT read):
  local rows      first-audible = first PCM chunk of the talker's own answer
                  (first_audio_ms, measured inside gen_speak from t_eot)
  escalated rows  first-audible = canned-stall onset = stall prefill complete
                  (first_audio_ms = t_stall - t_eot; the pre-synthesized
                  buffer plays immediately, stall_pcm_s long)
                  expert-content first-audible = expert_latency_s (true
                  round-trip, cache-corrected) + relay_first_audio_ms
                  (expert return -> first relay PCM)

Usage (from interactive_paper/): .venv_boot\\Scripts\\python.exe
scripts\\10_tts_latency.py [path-to-jsonl ...]
"""
import json
import sys
from pathlib import Path

import numpy as np


def pct(a, ps=(50, 95, 99)):
    a = np.asarray(sorted(a), dtype=float)
    return {f"p{p}": round(float(np.percentile(a, p)), 3) for p in ps}


def main(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    loc = [r for r in rows if r.get("mode") == "local"]
    esc = [r for r in rows if r.get("mode") == "escalated"]
    print(f"n={len(rows)} rows: {len(loc)} local, {len(esc)} escalated")

    fa_loc = [r["first_audio_ms"] / 1000 for r in loc
              if r.get("first_audio_ms") is not None]
    fa_esc = [r["first_audio_ms"] / 1000 for r in esc
              if r.get("first_audio_ms") is not None]
    eot = [r["eot_read_ms"] / 1000 for r in rows]
    print(f"\nEOT read (gate decision), all rows      {pct(eot)}")
    print(f"local:   EOT -> first audible (n={len(fa_loc)})  {pct(fa_loc)}")
    print(f"escal.:  EOT -> first audible (stall onset, n={len(fa_esc)}) "
          f"{pct(fa_esc)}")

    both = sorted(fa_loc + fa_esc)
    print(f"pooled:  EOT -> first audible (n={len(both)})  {pct(both)}")

    exp_first = [r["expert_latency_s"] + r["relay_first_audio_ms"] / 1000
                 for r in esc if r.get("relay_first_audio_ms") is not None
                 and r.get("expert_latency_s") is not None]
    print(f"escal.:  EOT -> expert-content audible (true expert RTT + "
          f"relay first PCM, n={len(exp_first)})  {pct(exp_first)}")

    stall = [r.get("stall_pcm_s") for r in esc if r.get("stall_pcm_s")]
    if stall:
        print(f"canned stall covers {stall[0]}s of the expert wait")

    # dead air = gap between end of the canned stall and expert content
    # becoming audible, per escalated turn (0 if the stall outlives the wait)
    da = [max(0.0, (r["expert_latency_s"] + r["relay_first_audio_ms"] / 1000)
              - (r["first_audio_ms"] / 1000 + r["stall_pcm_s"]))
          for r in esc
          if r.get("expert_latency_s") is not None
          and r.get("relay_first_audio_ms") is not None
          and r.get("first_audio_ms") is not None
          and r.get("stall_pcm_s")]
    if da:
        covered = sum(1 for d in da if d == 0.0)
        print(f"escal.:  dead air after stall (n={len(da)}, "
              f"{covered} fully covered)  {pct(da)}")

    ans = [r["answer_ms"] / 1000 for r in loc if r.get("answer_ms")]
    if ans:
        print(f"\nreference: local completion (text decode)  {pct(ans)}")
    sp = [r.get("spoken_s") for r in rows if r.get("spoken_s")]
    if sp:
        print(f"spoken length across rows  {pct(sp)}")


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(
        str(p) for p in Path("data").glob(
            "frozen_v3tts_traces.jsonl.balanced.shard*"))
    if not args:
        sys.exit("no trace files found; pass paths explicitly")
    print("reading:", *args, sep="\n  ")
    main(args)
