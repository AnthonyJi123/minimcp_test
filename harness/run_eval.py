"""Runner: load MiniCPM-o 4.5 once, drive selected probes, write per-probe JSONL
traces + a markdown report aggregating the metrics.

Usage (on the pod, inside the venv):
    python run_eval.py                # run all probes
    python run_eval.py p1 p2          # run a subset
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "probes"))

from session import DuplexSession  # noqa: E402

RESULTS = "/workspace/minicpm-o-eval/results"
os.makedirs(RESULTS, exist_ok=True)

REPEAT = int(os.environ.get("REPEAT", "1"))


def _aggregate(metrics):
    """Aggregate N per-run metric dicts: rates for bools, mean/min/max for numbers,
    first value for strings. Behaviors at temp>0 are stochastic, so rates are the
    defensible summary."""
    agg = {"_runs": len(metrics)}
    for k in metrics[0]:
        if k.startswith("_"):
            continue
        vals = [m.get(k) for m in metrics]
        nn = [v for v in vals if v is not None]
        if nn and all(isinstance(v, bool) for v in nn):
            agg[k + "_rate"] = round(sum(1 for v in nn if v) / len(nn), 2)
        elif nn and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in nn):
            agg[k] = {"mean": round(sum(nn) / len(nn), 1), "min": min(nn),
                      "max": max(nn), "n": len(nn), "null_runs": len(vals) - len(nn)}
        else:
            agg[k] = vals[0]
    return agg

ALL_PROBES = [
    "p1_av_fullduplex",
    "p2_user_interruption",
    "p3_visual_interruption",
    "p4_crossmodal_conflict",
    "p5_backchannel",
    "p6_proactive_timing",
    "p7_online_correction",
    "p8_long_horizon",
    "p9_memory_budget",
]


def _match(token):
    """Accept a full probe name or a short prefix like 'p2'."""
    if token in ALL_PROBES:
        return token
    for full in ALL_PROBES:
        if full.startswith(token):
            return full
    raise SystemExit(f"unknown probe: {token}")


def _report_only():
    summary = []
    for full in ALL_PROBES:
        p = os.path.join(RESULTS, f"{full}.metric.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                summary.append(json.load(f))
    _write_report(summary)
    print(f">>> report regenerated at {RESULTS}/report.md")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--report-only":
        _report_only()
        return
    names = [_match(a) for a in args] if args else ALL_PROBES

    print(f">>> loading model (once) ...", flush=True)
    sess = DuplexSession()
    print(f">>> model ready in {sess.load_s:.1f}s", flush=True)

    summary = []
    for full in names:
        try:
            mod = importlib.import_module(full)
            probe = mod.PROBE
            print(f"\n================ {probe.name} :: {probe.capability} ================", flush=True)
            t0 = time.time()
            # control arms (causality baselines) — run once, shared across repeats
            controls = {}
            for cname, cbuild in (probe.controls or {}).items():
                print(f"---- control: {cname} ----", flush=True)
                csc = cbuild()
                csc.stop_on_eot = probe.stop_on_eot
                try:
                    sess.prepare(probe.system_prompt)
                    controls[cname] = sess.run(
                        csc, trace_path=os.path.join(RESULTS, f"{probe.name}.control.{cname}.jsonl"))
                finally:
                    sess.stop()
            # main arm, repeated for stochastic behaviors
            per_run = []
            for run_i in range(REPEAT):
                scenario = probe.build()
                scenario.stop_on_eot = probe.stop_on_eot
                suffix = "" if REPEAT == 1 else f".run{run_i}"
                try:
                    sess.prepare(probe.system_prompt)
                    records = sess.run(scenario, trace_path=os.path.join(RESULTS, f"{probe.name}{suffix}.jsonl"))
                finally:
                    sess.stop()
                per_run.append(probe.score(records, scenario, controls))
            metric = per_run[0] if REPEAT == 1 else _aggregate(per_run)
            metric["_probe"] = probe.name
            metric["_capability"] = probe.capability
            metric["_wall_s"] = round(time.time() - t0, 1)
        except Exception as e:
            import traceback
            traceback.print_exc()
            metric = {"_probe": full, "_capability": "(errored)", "_error": str(e)}
        with open(os.path.join(RESULTS, f"{full}.metric.json"), "w",
                  encoding="utf-8") as f:
            json.dump(metric, f, ensure_ascii=False, indent=2)
        summary.append(metric)
        print(f">>> {full} metric: {json.dumps(metric, ensure_ascii=False)}", flush=True)

    _write_report(summary)
    print(f"\n>>> report written to {RESULTS}/report.md")


def _write_report(summary):
    lines = [
        "# MiniCPM-o 4.5 — Full-Duplex Capability Probes",
        "",
        "Model: `openbmb/MiniCPM-o-4_5` (9B) on a single H100 80GB, bf16, sdpa.",
        "Each probe drives the duplex loop at 1 Hz, injects a controlled event at a",
        "known chunk **T**, and derives a metric from the model's per-second",
        "`is_listen` / `text` / `end_of_turn` / `kv_cache_length` stream.",
        "These cover the audio+vision+speech full-duplex behaviors that the paper's",
        "only quantitative full-duplex benchmark (LiveSports-3K-CC, vision-only) omits.",
        "",
        "| # | Capability | Headline result |",
        "|---|-----------|-----------------|",
    ]
    def g(m, k):
        """Fetch a metric value, transparently handling aggregated forms:
        bool -> k+'_rate'; numeric -> {'mean':..} dict."""
        if k + "_rate" in m:
            return m[k + "_rate"]
        v = m.get(k)
        if isinstance(v, dict) and "mean" in v:
            return v["mean"]
        return v

    headline = {
        "p1_av_fullduplex": lambda m: f"comply latency={g(m,'comply_latency_chunks')} chunk(s); causal rate={g(m,'valid_causal')}",
        "p2_user_interruption": lambda m: f"sustained-stop after 停: latency={g(m,'stop_latency_chunks')} chunk(s), ack rate={g(m,'acknowledged_stop')}, causal rate={g(m,'stop_is_causal')}",
        "p3_visual_interruption": lambda m: f"speaking@T rate={g(m,'precondition_speaking_at_T')}; revise latency={g(m,'revise_latency_chunks')} chunk(s); causal rate={g(m,'valid_causal')}",
        "p4_crossmodal_conflict": lambda m: f"trusts vision both directions rate={g(m,'trusts_vision_both_directions')} (main={m.get('main_vision_red_audio_green')}, mirror={m.get('mirror_vision_green_audio_red')})",
        "p5_backchannel": lambda m: f"{m.get('verdict')} (interrupted_user rate={g(m,'interrupted_user')}, seg_chars={g(m,'speaking_segment_chars')})",
        "p6_proactive_timing": lambda m: f"idle-silent rate={g(m,'stayed_silent_when_idle')}, proactive latency={g(m,'proactive_latency_chunks')} chunk(s); causal rate={g(m,'proactive_is_causal')}",
        "p7_online_correction": lambda m: f"narrate-new-count latency={g(m,'narration_latency_chunks')} chunk(s); explicit-correction rate={g(m,'explicitly_corrected')}; causal rate={g(m,'revision_is_causal')}",
        "p8_long_horizon": lambda m: f"{g(m,'completed_chunks')}/{g(m,'intended_chunks')} chunks, crashed rate={g(m,'crashed')}, listen-drift={g(m,'listen_latency_drift_ms')}ms, speak-max={g(m,'speak_latency_max_ms')}ms, RT-violations={g(m,'real_time_violation_chunks')}, kv/chunk={g(m,'kv_tokens_per_chunk')}, vram+{g(m,'vram_growth_mb')}MB",
        "p9_memory_budget": lambda m: f"recall@q1 rate={g(m,'recall_at_q1')}, recall@q2 rate={g(m,'recall_at_q2')}; eff dist q2={g(m,'effective_distance_q2')} chunk(s)",
    }
    for i, m in enumerate(summary, 1):
        p = m["_probe"]
        try:
            h = headline.get(p, lambda _m: "")(m) if "_error" not in m else f"ERROR: {m['_error']}"
        except Exception:
            h = json.dumps({k: v for k, v in m.items() if not k.startswith("_")}, ensure_ascii=False)[:160]
        lines.append(f"| {i} | {m.get('_capability','')} | {h} |")

    lines += ["", "## Per-probe detail", ""]
    for m in summary:
        lines.append(f"### {m['_probe']} — {m.get('_capability','')}")
        lines.append("```json")
        lines.append(json.dumps(m, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append(f"Full chunk-level trace: `results/{m['_probe']}.jsonl`")
        lines.append("")
    with open(os.path.join(RESULTS, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
