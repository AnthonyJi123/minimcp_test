"""Pretty-print a probe's chunk-level JSONL trace.

Usage: python show_trace.py p2_user_interruption [p6_proactive_timing ...]
"""
import json, sys, os

RESULTS = "/workspace/minicpm-o-eval/results"


def show(name):
    path = os.path.join(RESULTS, f"{name}.jsonl")
    print(f"\n===== {name} =====")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        state = "SPK" if r["speaking"] else "lst"
        mark = f"  <<< {r['event']}" if r["event"] else ""
        lbl = (r.get("label") or "")[:10]
        print(f"[{r['chunk_idx']:02d}] {state} eot={int(r['end_of_turn'])} "
              f"kv={r['kv_cache_length']} {lbl:>10} :: {r['text']!r}{mark}")


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["p1_av_fullduplex"]):
        show(n)
