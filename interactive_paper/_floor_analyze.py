"""Analysis for the floor-control sweep (todo P1 2026-08-27).

Claims:
  A (orthogonality) -- ans-phase cells, paired g0 vs g1 (same qid +
    stimulus variant per pair index): backchannel floor-hold rate,
    stop/barge interrupt rate, duck/resume/interrupt latencies.
    Prediction: paired deltas ~ 0 (the gate reads once at EOT and
    never touches the floor state machine).
  B (escalated phases) -- stall/wait/relay cells (g1 only):
    backchannel must NOT cancel the pending escalation (mode stays
    'escalated', turn not interrupted); barge-in must abort and seed
    the next turn.

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _floor_analyze.py::t
"""
import modal

app = modal.App("floor-analyze")
vol = modal.Volume.from_name("gate-data")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("pandas", "numpy"))


@app.function(image=img, volumes={"/data": vol}, timeout=600)
def t():
    import json

    import numpy as np
    import pandas as pd

    rows = [json.loads(x) for x in
            open("/data/floor_sweep/floor.jsonl") if x.strip()]
    df = (pd.DataFrame(rows)
          .drop_duplicates("pair_id", keep="last"))
    print(f"{len(df)} pairs "
          f"({(df['outcome'] == 'error').sum()} errors, "
          f"{(df['outcome'] == 'wrong_fire').sum()} wrong_fire, "
          f"{(df['outcome'] == 'no_fire').sum()} no_fire, "
          f"{(df['outcome'] == 'no_phase').sum()} no_phase, "
          f"{(df['outcome'] == 'no_duck').sum()} no_duck)")
    for c in ("duck_ms", "resume_ms", "interrupt_ms"):
        if c in df:
            neg = df[c] < 0
            if neg.any():
                print(f"WARNING: {int(neg.sum())} rows with negative "
                      f"{c} -> excluded (pre-injection event)")
                df = df[~neg.fillna(False)]
    ok = df[df["outcome"].isin(["resume", "interrupt"])].copy()
    ok["hold"] = ((ok["outcome"] == "resume")
                  & (ok["turn_interrupted"] == False))  # noqa: E712
    ok["intr"] = ok["outcome"] == "interrupt"

    rng = np.random.default_rng(0)

    def rate_ci(v, n_boot=2000):
        v = np.asarray(v, dtype=float)
        if not len(v):
            return "-"
        bs = [v[rng.integers(0, len(v), len(v))].mean()
              for _ in range(n_boot)]
        return (f"{v.mean():.3f} "
                f"[{np.percentile(bs, 2.5):.3f},"
                f"{np.percentile(bs, 97.5):.3f}]")

    def med(v):
        v = [x for x in v if x == x and x is not None]
        return int(np.median(v)) if v else "-"

    # ---------------- per-cell table ----------------
    print("\n== per-cell outcomes (resolved pairs) ==")
    print(f"{'cell':<16}{'n':>4}  {'hold-rate':<22}{'intr-rate':<22}"
          f"duck/res/int ms")
    for (ph, stx, arm), g in ok.groupby(["phase", "stim", "arm"]):
        print(f"{ph}:{stx}:{arm:<6}{len(g):>4}  "
              f"{rate_ci(g['hold']):<22}{rate_ci(g['intr']):<22}"
              f"{med(g['duck_ms'])}/{med(g.get('resume_ms', []))}"
              f"/{med(g.get('interrupt_ms', []))}")

    # ---------------- claim A: paired g0 vs g1 ----------------
    print("\n== claim A: ans-phase paired deltas (g1 - g0) ==")
    a = ok[ok["phase"] == "ans"].copy()
    a["idx"] = a["pair_id"].str.split(":").str[-1]
    for stx, g in a.groupby("stim"):
        piv = g.pivot_table(index="idx", columns="arm",
                            values=["hold", "intr"],
                            aggfunc="first").dropna()
        if not len(piv):
            print(f"{stx}: no paired rows")
            continue
        for m in ("hold", "intr"):
            try:
                d = (piv[(m, "g1")].astype(float)
                     - piv[(m, "g0")].astype(float)).values
            except KeyError:
                continue
            bs = [d[rng.integers(0, len(d), len(d))].mean()
                  for _ in range(2000)]
            print(f"{stx:<5} d({m}) = {d.mean():+.3f} "
                  f"[{np.percentile(bs, 2.5):+.3f},"
                  f"{np.percentile(bs, 97.5):+.3f}]  n={len(d)}")
        for m in ("duck_ms", "resume_ms", "interrupt_ms"):
            if m not in g:
                continue
            piv2 = g.pivot_table(index="idx", columns="arm",
                                 values=m, aggfunc="first").dropna()
            if len(piv2) >= 5 and "g0" in piv2 and "g1" in piv2:
                d = (piv2["g1"] - piv2["g0"]).values
                print(f"{stx:<5} d({m}) med {np.median(d):+.0f} ms "
                      f"(n={len(d)})")

    # ---------------- claim B: escalated phases ----------------
    print("\n== claim B: escalation-only phases (g1) ==")
    e = ok[ok["phase"].isin(["stall", "wait", "relay"])]
    for (ph, stx), g in e.groupby(["phase", "stim"]):
        if stx == "bcs":
            surv = ((g["outcome"] == "resume")
                    & (g["mode"] == "escalated")
                    & (g["turn_interrupted"] == False))  # noqa: E712
            print(f"{ph}:bcs  escalation-survives {rate_ci(surv)} "
                  f"n={len(g)}  res_ms={med(g.get('resume_ms', []))}")
        else:
            ab = g["intr"]
            seed = g.get("seed_next_eot")
            seed_s = (rate_ci(seed.dropna())
                      if seed is not None else "-")
            print(f"{ph}:bq   abort {rate_ci(ab)}  seed {seed_s} "
                  f"n={len(g)}  int_ms={med(g.get('interrupt_ms', []))}")

    # ---------------- diagnosis strata ----------------
    print("\n== diagnosis: why do backchannels die? ==")
    for stx in ("bcs", "bcl"):
        g = ok[(ok["phase"] == "ans") & (ok["stim"] == stx)]
        ints = g[g["intr"]]
        why = ints["why"].value_counts().to_dict() if len(ints) else {}
        print(f"{stx}: {len(ints)}/{len(g)} interrupted, why={why}")
        for tx, gg in g.groupby("stim_text"):
            path = ("sustained (no ASR)"
                    if gg[gg["intr"]]["interrupt_ms"].median()
                    < gg["stim_s"].iloc[0] * 1000 + 400
                    else "ASR-empty fail-closed") \
                if gg["intr"].any() else "-"
            print(f"   {tx!r:<28} {gg['stim_s'].iloc[0]:.2f}s  "
                  f"die {int(gg['intr'].sum())}/{len(gg)}  "
                  f"int_med={gg[gg['intr']]['interrupt_ms'].median()}"
                  f"  [{path}]")
