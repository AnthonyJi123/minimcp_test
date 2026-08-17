"""Profiling readout from the live-sweep traces (gated_traces_v2.parquet).

What did escalation cost in wall-clock terms? Per-arm response-latency
distributions (mean / P50 / P95 / P99) reconstructed from the per-session
timers the v2 loop recorded:
  local     : eot_read_ms + answer_ms
  escalated : eot_read_ms + max(stall_ms, expert_latency_ms) + relay_ms
              (the expert thread launches at the gate decision, in parallel
               with the stall prefill, so the wait is the max, not the sum)
expert_latency_s is the TRUE API latency (cache-corrected: for cache hits it
is the original call's latency, not the ~0s cache read) — a deployment has
no cache, so this is the honest number. All times are the text-mode
pipeline (prefill+decode); speech synthesis is not included.

Run: python interactive_paper/figures/latency_profile.py
"""
import json

import numpy as np
import pandas as pd

DATA = "interactive_paper/data"
df = pd.read_parquet(f"{DATA}/gated_traces_v2.parquet")

esc = df["mode"] == "escalated"
df["expert_ms"] = df["expert_latency_s"].fillna(0) * 1000
df["total_ms"] = np.where(
    esc,
    df["eot_read_ms"] + np.maximum(df["stall_ms"].fillna(0),
                                   df["expert_ms"]) + df["relay_ms"].fillna(0),
    df["eot_read_ms"] + df["answer_ms"].fillna(0))

def q(s, p):
    return float(np.percentile(s, p))

def stats(s):
    s = s.dropna() / 1000
    return {"n": int(len(s)), "mean": round(float(s.mean()), 2),
            "p50": round(q(s, 50), 2), "p95": round(q(s, 95), 2),
            "p99": round(q(s, 99), 2)}

out = {"component": {}, "arm": {}}

print("=== component timers (rows where the stage ran, seconds) ===")
for col in ("eot_read_ms", "stall_ms", "answer_ms", "relay_ms", "expert_ms"):
    st = stats(df.loc[esc, col] if col == "expert_ms" else df[col])
    out["component"][col] = st
    print(f"  {col:12s} n={st['n']:3d}  mean={st['mean']:6.2f}  "
          f"P50={st['p50']:6.2f}  P95={st['p95']:6.2f}  P99={st['p99']:6.2f}")

print("\n=== per-arm total response latency (query end -> answer text done, s) ===")
loc_never = df[(df["tier"] == "never")]["total_ms"] / 1000
for tier in ("never", "conservative", "balanced", "aggressive"):
    d = df[df["tier"] == tier]
    row = {"all": stats(d["total_ms"]),
           "local": stats(d[d["mode"] == "local"]["total_ms"]),
           "escalated": stats(d[d["mode"] == "escalated"]["total_ms"])
           if (d["mode"] == "escalated").any() else None}
    out["arm"][tier] = row
    a = row["all"]
    print(f"  {tier:12s} esc={100 * (d['mode'] == 'escalated').mean():4.1f}%  "
          f"mean={a['mean']:6.2f}  P50={a['p50']:6.2f}  "
          f"P95={a['p95']:6.2f}  P99={a['p99']:6.2f}")
    if row["escalated"]:
        e, l = row["escalated"], row["local"]
        print(f"    {'escalated':12s} n={e['n']:3d}  mean={e['mean']:6.2f}  "
              f"P50={e['p50']:6.2f}  P95={e['p95']:6.2f}  P99={e['p99']:6.2f}")
        print(f"    {'local':12s} n={l['n']:3d}  mean={l['mean']:6.2f}  "
              f"P50={l['p50']:6.2f}  P95={l['p95']:6.2f}  P99={l['p99']:6.2f}")

print("\n=== the loss: arm minus never-floor, same percentile (s) ===")
base = out["arm"]["never"]["all"]
for tier in ("conservative", "balanced", "aggressive"):
    a = out["arm"][tier]["all"]
    d = {k: round(a[k] - base[k], 2) for k in ("mean", "p50", "p95", "p99")}
    out["arm"][tier]["delta_vs_never"] = d
    print(f"  {tier:12s} mean{d['mean']:+6.2f}  P50{d['p50']:+6.2f}  "
          f"P95{d['p95']:+6.2f}  P99{d['p99']:+6.2f}")

print("\n=== escalated-row decomposition (escalated rows only, s) ===")
e = df[esc]
for tier in ("conservative", "balanced", "aggressive"):
    d = e[e["tier"] == tier]
    exp = d["expert_ms"] / 1000
    print(f"  {tier:12s} n={len(d):3d}  expert mean={exp.mean():5.2f} "
          f"P50={q(exp, 50):5.2f} P95={q(exp, 95):5.2f} "
          f"P99={q(exp, 99):5.2f} | relay P50="
          f"{q(d['relay_ms'] / 1000, 50):4.2f} | share of total: expert "
          f"{(np.maximum(d['stall_ms'], d['expert_ms']).sum() / d['total_ms'].sum()):.0%}")

with open("interactive_paper/figures/latency_profile.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("\nwrote interactive_paper/figures/latency_profile.json")
