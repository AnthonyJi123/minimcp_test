"""Figures 3-8: the live gated sweep on three EXTERNAL speech pools —
Speech TriviaQA + Speech Web Questions (OpenAudioBench renders) and
SD-QA (real human recordings). Two figures per pool:
  {bench}_dualview.png : accuracy vs escalation rate (heard curve, CIs)
                         + random reference + gpt-5.5 ceiling
                         + official chat-mode anchor where one exists
  {bench}_pareto.png   : P50 total latency vs accuracy
All arms ran with the FROZEN 600-pool gate + thresholds — zero
recalibration; the curves are transfer readouts.

Needs (fetch from gate-data): {bench}_traces.parquet,
{bench}_ceiling.parquet. Run from figures/:
..\\..\\.venv_ip\\Scripts\\python bench_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, GREEN = "#2a78d6", "#1e9e50"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e4de"
ARMS = ("never", "conservative", "balanced", "aggressive")
DATA = "../data"

BENCHES = {
    "striviaqa": {"title": "Speech TriviaQA (OpenAudioBench, n=250)",
                  "official": .755,
                  "official_label": "MiniCPM-o 4.5 official 75.5 "
                                    "(offline chat mode)",
                  "judge": "oab_ok", "chatmode": .712,
                  "others": [("Qwen3-Omni-30B", .629),
                             ("Kimi-Audio", .419)]},
    "swebq": {"title": "Speech Web Questions (OpenAudioBench, n=250)",
              "official": .702,
              "official_label": "MiniCPM-o 4.5 official 70.2 "
                                "(offline chat mode)",
              "judge": "oab_ok", "chatmode": .716,
              "others": [("Qwen3-Omni-30B", .749),
                         ("Kimi-Audio", .464)]},
    "sllama": {"title": "Speech Llama Questions (OpenAudioBench, n=250)",
               "official": None, "official_label": None,
               "judge": "oab_ok"},
    "sreason": {"title": "Speech Reasoning QA — Chinese "
                         "(OpenAudioBench, n=202)",
                "official": None, "official_label": None},
    # sdqa: no random reference line; the local-only floor is drawn instead
    # (user call, 2026-08-13 — the two bounds that matter are the floor the
    # gate starts from and the ceiling it is walking toward)
    "sdqa": {"title": "SD-QA — real human speech (VoiceBench, n=200)",
             "official": None, "official_label": None,
             "random_line": False, "floor_line": True},
}


def style(ax, xlab, ylab, title):
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.grid(color=GRID, lw=.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


VER = "_v3"      # "" = the original v1 sweep; "_v2" = probe v2 re-run

summary = {}
for bench, spec in BENCHES.items():
    df = pd.read_parquet(f"{DATA}/{bench}{VER}_traces.parquet")
    # OpenAudioBench pools are scored with OAB's own judge so the curves,
    # the official line and the other models' numbers share one scale
    col = spec.get("judge", "heard_ok")
    if col not in df.columns:
        col = "heard_ok"
    df = df[df["tier"].isin(ARMS) & df[col].notna()]
    heard = df.pivot(index="id", columns="tier", values=col).dropna()
    esc = (df.pivot(index="id", columns="tier", values="mode")
           .loc[heard.index] == "escalated")
    A = heard[list(ARMS)].to_numpy(float)
    E = esc[list(ARMS)].to_numpy(bool)
    n = len(heard)
    rates = E.mean(axis=0)
    hm = A.mean(axis=0)

    ceil_df = (pd.read_parquet(f"{DATA}/{bench}_ceiling.parquet")
               .set_index("id"))
    ceil_df = ceil_df[ceil_df.index.isin(heard.index)]
    ccol = "oab_ok" if (col == "oab_ok" and "oab_ok" in ceil_df.columns) \
        else "adequate"
    ceil = float(ceil_df[ccol].dropna().astype(float).mean())
    b = lambda x: 1 if x is True or x == 1 else 0
    gold = np.array([b(ceil_df[ccol].get(i)) for i in heard.index])
    B = np.where(E, gold[:, None], A)          # gold-inject counterfactual
    gm = B.mean(axis=0)

    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(10000, n))
    bA, bB = A[idx].mean(axis=1), B[idx].mean(axis=1)
    ciA = np.percentile(bA, [2.5, 97.5], axis=0)
    ciB = np.percentile(bB, [2.5, 97.5], axis=0)

    # latency: same reconstruction as the 600-pool profile
    df["expert_ms"] = df["expert_latency_s"].fillna(0) * 1000
    is_esc = df["mode"] == "escalated"
    df["total_ms"] = np.where(
        is_esc,
        df["eot_read_ms"] + np.maximum(df["stall_ms"].fillna(0),
                                       df["expert_ms"])
        + df["relay_ms"].fillna(0),
        df["eot_read_ms"] + df["answer_ms"].fillna(0))
    dfc = df[df["id"].isin(heard.index)]
    lat50 = [float(dfc[dfc["tier"] == a]["total_ms"].median() / 1000)
             for a in ARMS]

    # random-escalation reference for the pareto (Jisen note 2026-08-18):
    # acc = the dualview random line (pairs with the gold view); latency =
    # simulated P50 of a random mixture at rate r -- per-id local latency
    # from the never arm, per-id escalated latency where any arm escalated
    # that id (draw from the pool's observed escalated rows otherwise)
    loc_ms = dfc[dfc["tier"] == "never"].set_index("id")["total_ms"] \
        .reindex(heard.index)
    esc_med = dfc[dfc["mode"] == "escalated"].groupby("id")["total_ms"] \
        .median().reindex(heard.index)
    esc_obs = dfc[dfc["mode"] == "escalated"]["total_ms"].to_numpy()
    esc_med[esc_med.isna()] = rng.choice(esc_obs,
                                         size=int(esc_med.isna().sum()))
    Lm, Xm = loc_ms.to_numpy(), esc_med.to_numpy()
    rand_r = np.linspace(0, 1, 21)
    rand_lat = [float(np.median(np.where(
        rng.random((400, n)) < r, Xm, Lm), axis=1).mean() / 1000)
        for r in rand_r]
    rand_acc = [hm[0] + (ceil - hm[0]) * r for r in rand_r]

    summary[bench] = {"n": n, "esc": [float(x) for x in rates],
                      "heard": [float(x) for x in hm],
                      "heard_ci": ciA.tolist(), "ceiling": ceil,
                      "gold_inject": [float(x) for x in gm],
                      "gold_inject_ci": ciB.tolist(),
                      "p50_latency_s": lat50,
                      "random_pareto": {"esc": rand_r.tolist(),
                                        "lat": rand_lat,
                                        "acc": rand_acc}}
    print(f"{bench}: n={n} esc={np.round(rates, 2)} "
          f"heard={np.round(hm, 3)} gold-inj={np.round(gm, 3)} "
          f"ceiling={ceil:.3f} lat50={np.round(lat50, 2)}")

    # ---- dualview: acc vs escalation rate -------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xr = np.array([0, 1])
    if spec.get("random_line", True):
        ax.plot(xr, [hm[0] + (ceil - hm[0]) * r for r in xr], ls="--",
                lw=1.0, color=MUT, alpha=.8, zorder=2,
                label="random escalation")
    ax.axhline(ceil, color=GREEN, ls=":", lw=1.2, alpha=.75, zorder=1)
    ax.text(.02, ceil + .013, f"always-escalate (gpt-5.5 on gold text) "
            f"{ceil:.3f}", fontsize=7.5, color=GREEN)
    if spec.get("floor_line"):
        ax.axhline(hm[0], color=BLUE, ls=":", lw=1.2, alpha=.7, zorder=1)
        ax.text(.99, hm[0] - .016, f"local-only floor (never escalate) "
                f"{hm[0]:.3f}", fontsize=7.5, color=BLUE, ha="right",
                va="top")
    # reference lines: our own offline chat mode, the official number,
    # and the other speech models from the SAME official table
    if spec.get("chatmode"):
        ax.axhline(spec["chatmode"], color=BLUE, ls=(0, (4, 3)), lw=1.1,
                   alpha=.55, zorder=1)
        ax.text(.99, spec["chatmode"] - .014,
                f"same model, offline chat mode (ours) "
                f"{spec['chatmode']:.3f}", fontsize=7, color=BLUE,
                ha="right", va="top")
    if spec["official"]:
        ax.axhline(spec["official"], color=INK, ls=(0, (1, 3)), lw=1.2,
                   alpha=.6, zorder=1)
        ax.text(.99, spec["official"] + .012, spec["official_label"],
                fontsize=7, color=MUT, ha="right")
    for name, val in spec.get("others", []):
        ax.axhline(val, color=MUT, ls=(0, (5, 2, 1, 2)), lw=1.0,
                   alpha=.55, zorder=1)
        ax.text(.99, val + .012, f"{name} {val * 100:.1f} (official, "
                "offline)", fontsize=7, color=MUT, ha="right")
    ax.errorbar(rates, gm, yerr=[gm - ciB[0], ciB[1] - gm], fmt="--s",
                ms=5.5, color=GREEN, capsize=3, lw=1.4, alpha=.9, zorder=3,
                label="gold-inject — counterfactual: expert answers the "
                      "gold text")
    ax.errorbar(rates, hm, yerr=[hm - ciA[0], ciA[1] - hm], fmt="-o",
                ms=6, color=BLUE, capsize=3, lw=1.7, zorder=4,
                label="heard-acc — deployed" + (f" (probe {VER[1:]})" if VER else
                                                " (frozen thresholds)"))
    for j, a in enumerate(ARMS):
        ax.annotate(a, (rates[j], hm[j]), xytext=(5, -13),
                    textcoords="offset points", fontsize=7.5, color=BLUE)
    sub = (f"probe {VER[1:]}, per-domain quantile thresholds; "
           + ("scored with OpenAudioBench's own judge (gpt-4o)"
              if col == "oab_ok" else "our reference-anchored judge")
           if VER else
           "gate + thresholds frozen from the 600-query calibration pool")
    style(ax, "realized escalation rate", "heard accuracy",
          f"Accuracy vs escalation rate — {spec['title']}\n{sub}")
    ax.set_xlim(-.03, 1.03)
    refs = [v for _, v in spec.get("others", [])] + \
           [spec["official"] or 1, spec.get("chatmode") or 1, hm[0]]
    ax.set_ylim(max(0, min(refs) - .10), min(1.02, ceil + .10))
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{bench}_dualview.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)

    # ---- pareto: latency vs acc ------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(rand_lat, rand_acc, ls="--", lw=1.0, color=MUT, alpha=.8,
            zorder=2, label="random escalation (simulated P50 latency)")
    ax.axhline(ceil, color=GREEN, ls=":", lw=1.0, alpha=.55, zorder=1)
    ax.text(lat50[0] + .03, ceil - .012, f"always-escalate ceiling "
            f"{ceil:.3f} (synthesized — no live latency)", fontsize=7.5,
            color=MUT, va="top")
    ax.errorbar(lat50, gm, yerr=[gm - ciB[0], ciB[1] - gm], fmt="--s",
                ms=5.5, color=GREEN, capsize=3, lw=1.4, alpha=.9, zorder=3,
                label="gold-inject counterfactual")
    ax.errorbar(lat50, hm, yerr=[hm - ciA[0], ciA[1] - hm], fmt="-o",
                ms=6, color=BLUE, capsize=3, lw=1.7, zorder=4,
                label="gated live loop (deployed)")
    for j, a in enumerate(ARMS):
        ax.annotate(f"{a} ({rates[j]:.0%})", (lat50[j], hm[j]),
                    xytext=(4, -14), textcoords="offset points",
                    fontsize=7.5, color=BLUE)
    # no per-segment exchange labels: small escalation deltas make the
    # P50s non-monotonic on the easy pools; the line itself tells it
    style(ax, "P50 total response latency, query end → answer text done "
          "(s)", "heard accuracy",
          f"What latency buys — {spec['title']}\n(points = escalation "
          "arms, labelled by escalation rate)")
    ax.set_ylim(max(0, hm[0] - .12), min(1.0, ceil + .10))
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{bench}_pareto.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

json.dump(summary, open("bench_figures.json", "w"), indent=1)
print("wrote {bench}_dualview + {bench}_pareto x3 + bench_figures.json")
