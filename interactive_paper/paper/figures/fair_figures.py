"""Figures 1-2 of the 2026-08-12 request: the live sweep on the
SPEAKABLE (fair) subset — "assuming TTS is fixed" operationalized as the
8q input-side filter (n=218; 21 LaTeX ids + 1 broken wav removed).

Fig 1  fair_pareto_latency : P50 latency vs accuracy (both views)
Fig 2  fair_dualview       : escalation rate vs accuracy (both views
                             + random reference + always ceiling)

Everything recomputed per-row from gated_traces_v2.parquet on fair ids —
acc AND latency (dropping unfair rows moves both). Style matches the
pareto_latency.py / live_dualview family (validated blue/green pair).

Run from figures/: ..\\..\\.venv_ip\\Scripts\\python fair_figures.py
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

VER = "_v2"     # "" = original v1 sweep (gated_traces_v2), "_v2" = probe v2
df = pd.read_parquet(f"{DATA}/frozen_v2_traces.parquet" if VER
                     else f"{DATA}/gated_traces_v2.parquet")
df = df[df["tier"].isin(ARMS) & df["heard_ok"].notna()]
unfair = set(json.load(open("fair_subset_audit.json")))
df = df[~df["id"].isin(unfair)]

heard = df.pivot(index="id", columns="tier", values="heard_ok").dropna()
esc = (df.pivot(index="id", columns="tier", values="mode")
       .loc[heard.index] == "escalated")
exp = pd.read_parquet(f"{DATA}/eval_expert.parquet").set_index("id")
b = lambda x: 1 if x is True or x == 1 else 0
gold = np.array([b(exp.loc[i, "expert_adequate"]) for i in heard.index])

A = heard[list(ARMS)].to_numpy(float)
E = esc[list(ARMS)].to_numpy(bool)
B = np.where(E, gold[:, None], A)
n = len(heard)
rates = E.mean(axis=0)
ceil = gold.mean()

rng = np.random.default_rng(42)
idx = rng.integers(0, n, size=(10000, n))
bA, bB = A[idx].mean(axis=1), B[idx].mean(axis=1)
ci = lambda v: np.percentile(v, [2.5, 97.5], axis=0)
ciA, ciB = ci(bA), ci(bB)

# per-arm P50 total latency on the SAME fair rows
df["expert_ms"] = df["expert_latency_s"].fillna(0) * 1000
is_esc = df["mode"] == "escalated"
df["total_ms"] = np.where(
    is_esc,
    df["eot_read_ms"] + np.maximum(df["stall_ms"].fillna(0),
                                   df["expert_ms"]) + df["relay_ms"].fillna(0),
    df["eot_read_ms"] + df["answer_ms"].fillna(0))
df_c = df[df["id"].isin(heard.index)]
lat50 = [float(df_c[df_c["tier"] == a]["total_ms"].median() / 1000)
         for a in ARMS]

hm, gm = A.mean(axis=0), B.mean(axis=0)
print("fair n =", n, "| esc", np.round(rates, 2), "| heard",
      np.round(hm, 3), "| gold-inj", np.round(gm, 3),
      "| always", round(ceil, 3), "| P50 lat", np.round(lat50, 2))

json.dump({"n": n, "always": float(ceil),
           "arms": {a: {"esc": float(rates[j]), "heard": float(hm[j]),
                        "heard_ci": [float(ciA[0, j]), float(ciA[1, j])],
                        "gold_inject": float(gm[j]),
                        "gold_inject_ci": [float(ciB[0, j]),
                                           float(ciB[1, j])],
                        "p50_latency_s": lat50[j]}
                    for j, a in enumerate(ARMS)}},
          open("fair_figures.json", "w"), indent=1)


def style(ax, xlab, ylab, title):
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.grid(color=GRID, lw=.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


# ---- Fig 1: latency vs accuracy ------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.axhline(ceil, color=GREEN, ls=":", lw=1.0, alpha=.55, zorder=1)
ax.text(lat50[0] + .05, ceil - .012,
        f"always-escalate ceiling {ceil:.3f} (synthesized — no live "
        "latency)", fontsize=7.5, color=MUT, va="top")
ax.errorbar(lat50, gm, yerr=[gm - ciB[0], ciB[1] - gm], fmt="--s", ms=5.5,
            color=GREEN, capsize=3, lw=1.4, alpha=.9, zorder=3,
            label="gold-inject — counterfactual: expert answers the gold text")
ax.errorbar(lat50, hm, yerr=[hm - ciA[0], ciA[1] - hm], fmt="-o", ms=6,
            color=BLUE, capsize=3, lw=1.7, zorder=4,
            label="heard-acc — deployed: expert answers the talker's transcript")
for j, a in enumerate(ARMS):
    ax.annotate(f"{rates[j]:.0%}", (lat50[j], hm[j]), xytext=(2, -14),
                textcoords="offset points", fontsize=8, color=BLUE)
for i in range(3):
    dx, dy = lat50[i + 1] - lat50[i], hm[i + 1] - hm[i]
    mx, my = (lat50[i] + lat50[i + 1]) / 2, (hm[i] + hm[i + 1]) / 2
    off = ((-8, 14), (0, -30), (-14, 16))[i]
    ax.annotate(f"+{dy * 100:.1f} pts / +{dx:.1f} s", (mx, my), xytext=off,
                textcoords="offset points", fontsize=7.5, color=MUT,
                ha="right" if i in (0, 2) else "center")
ax.annotate("", xy=(lat50[3], gm[3] - .012),
            xytext=(lat50[3], hm[3] + .012),
            arrowprops=dict(arrowstyle="<->", color=MUT, lw=0.9))
ax.text(lat50[3] + .07, (hm[3] + gm[3]) / 2,
        f"speech-channel cost −{gm[3] - hm[3]:.3f}\n(speakable subset — "
        "unspeakable\ncontent already excluded)", fontsize=7.5, color=MUT,
        va="center")
style(ax, "P50 total response latency, query end → answer text done (s)",
      f"accuracy, speakable subset (n={n})",
      "What latency buys — speakable subset (“TTS-fair” view)\n"
      + ("probe v2; points = escalation arms, labelled by escalation rate"
         if VER else
         "(points = escalation arms, labelled by escalation rate)"))
ax.set_ylim(.30, 1.0)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="lower right", fontsize=7.5,
          frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"fair_pareto_latency.{ext}", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---- Fig 2: escalation rate vs accuracy ----------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
xr = np.array([0, 1])
ax.plot(xr, [hm[0] + (ceil - hm[0]) * r for r in xr], ls="--", lw=1.0,
        color=MUT, alpha=.8, zorder=2,
        label="random escalation (pairs with the gold view)")
ax.plot(1.0, ceil, marker="*", ms=11, color=GREEN, zorder=4)
ax.text(.985, ceil + .015, f"always-escalate {ceil:.3f}", fontsize=7.5,
        color=MUT, ha="right")
ax.errorbar(rates, gm, yerr=[gm - ciB[0], ciB[1] - gm], fmt="--s", ms=5.5,
            color=GREEN, capsize=3, lw=1.4, alpha=.9, zorder=3,
            label="gold-inject — counterfactual: expert answers the gold text")
ax.errorbar(rates, hm, yerr=[hm - ciA[0], ciA[1] - hm], fmt="-o", ms=6,
            color=BLUE, capsize=3, lw=1.7, zorder=4,
            label="heard-acc — deployed: expert answers the talker's transcript")
for j, a in enumerate(ARMS):
    ax.annotate(a, (rates[j], hm[j]), xytext=(4, -13),
                textcoords="offset points", fontsize=7.5, color=BLUE)
style(ax, "realized escalation rate",
      f"accuracy, speakable subset (n={n})",
      "Accuracy vs escalation rate — speakable subset (“TTS-fair” view)"
      + ("\nprobe v2; blue = deployed channel, green = channel-controlled "
         "counterfactual" if VER else
         "\nblue = deployed channel, green = channel-controlled "
         "counterfactual"))
ax.set_xlim(-.03, 1.05)
ax.set_ylim(.30, 1.0)
ax.legend(loc="lower right", fontsize=7.5, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"fair_dualview.{ext}", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote fair_pareto_latency.{png,pdf} + fair_dualview.{png,pdf} "
      "+ fair_figures.json")
