"""Figures 9-10: VoiceBench AlpacaEval — the one speech-QA row of the
official MiniCPM-o 4.5 condensed matrix (reported 4.8).

Open-ended instructions with no reference answers, so the y axis is the
VoiceBench 1-5 judge score (their gpt-4o-mini judge + their exact
meta_prompt_open, copied verbatim in modal_bench.py) rather than
accuracy. Same four arms, probe v2, per-domain quantile thresholds.

Needs data/valpaca_v2_scored.parquet + valpaca_ceiling.parquet.
Run from figures/: ..\\..\\.venv_ip\\Scripts\\python valpaca_figures.py
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
OFFICIAL = 4.8          # MiniCPM-o 4.5, VoiceBench AlpacaEval (official)

df = pd.read_parquet(f"{DATA}/valpaca_v2_scored.parquet")
df = df[df["tier"].isin(ARMS) & df["score"].notna()]
piv = df.pivot(index="id", columns="tier", values="score").dropna()
esc = (df.pivot(index="id", columns="tier", values="mode")
       .loc[piv.index] == "escalated")
S = piv[list(ARMS)].to_numpy(float)
E = esc[list(ARMS)].to_numpy(bool)
n = len(piv)
rates, sm = E.mean(axis=0), S.mean(axis=0)

ceil_df = pd.read_parquet(f"{DATA}/valpaca_ceiling.parquet").set_index("id")
ceil_df = ceil_df[ceil_df.index.isin(piv.index)]
ceil = float(ceil_df["score"].dropna().mean())
gold = np.array([float(ceil_df["score"].get(i, np.nan)) for i in piv.index])
G = np.where(E, gold[:, None], S)
gm = np.nanmean(G, axis=0)

rng = np.random.default_rng(42)
idx = rng.integers(0, n, size=(10000, n))
ciS = np.percentile(S[idx].mean(axis=1), [2.5, 97.5], axis=0)
ciG = np.percentile(np.nanmean(G[idx], axis=1), [2.5, 97.5], axis=0)

# latency, same reconstruction as the other pools
df["expert_ms"] = df["expert_latency_s"].fillna(0) * 1000
is_esc = df["mode"] == "escalated"
df["total_ms"] = np.where(
    is_esc,
    df["eot_read_ms"] + np.maximum(df["stall_ms"].fillna(0),
                                   df["expert_ms"]) + df["relay_ms"].fillna(0),
    df["eot_read_ms"] + df["answer_ms"].fillna(0))
dfc = df[df["id"].isin(piv.index)]
lat50 = [float(dfc[dfc["tier"] == a]["total_ms"].median() / 1000)
         for a in ARMS]

print(f"valpaca: n={n} esc={np.round(rates, 2)} score={np.round(sm, 2)} "
      f"gold-inj={np.round(gm, 2)} ceiling={ceil:.2f} "
      f"lat50={np.round(lat50, 2)}")
json.dump({"n": n, "esc": rates.tolist(), "score": sm.tolist(),
           "score_ci": ciS.tolist(), "gold_inject": gm.tolist(),
           "gold_inject_ci": ciG.tolist(), "ceiling": ceil,
           "official": OFFICIAL, "p50_latency_s": lat50},
          open("valpaca_figures.json", "w"), indent=1)


def style(ax, xlab, ylab, title):
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.grid(color=GRID, lw=.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_ylim(3.6, 5.05)


SUB = ("probe v2, per-domain quantile thresholds; VoiceBench's own "
       "gpt-4o-mini judge prompt (1-5)")

# ---- Fig 9: score vs escalation rate ------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
xr = np.array([0, 1])
ax.plot(xr, [sm[0] + (ceil - sm[0]) * r for r in xr], ls="--", lw=1.0,
        color=MUT, alpha=.8, zorder=2, label="random escalation")
ax.axhline(ceil, color=GREEN, ls=":", lw=1.2, alpha=.75, zorder=1)
ax.text(.02, ceil - .04, f"always-escalate (gpt-5.5 on gold text) "
        f"{ceil:.2f}", fontsize=7.5, color=GREEN, va="top")
ax.axhline(OFFICIAL, color=INK, ls=(0, (1, 3)), lw=1.2, alpha=.6, zorder=1)
ax.text(.99, OFFICIAL - .04, f"MiniCPM-o 4.5 official {OFFICIAL} "
        "(offline chat mode)", fontsize=7.5, color=MUT, ha="right",
        va="top")
ax.errorbar(rates, gm, yerr=[gm - ciG[0], ciG[1] - gm], fmt="--s", ms=5.5,
            color=GREEN, capsize=3, lw=1.4, alpha=.9, zorder=3,
            label="gold-inject — counterfactual: expert answers the gold text")
ax.errorbar(rates, sm, yerr=[sm - ciS[0], ciS[1] - sm], fmt="-o", ms=6,
            color=BLUE, capsize=3, lw=1.7, zorder=4,
            label="deployed live loop (probe v2)")
for j, a in enumerate(ARMS):
    ax.annotate(a, (rates[j], sm[j]), xytext=(5, -13),
                textcoords="offset points", fontsize=7.5, color=BLUE)
style(ax, "realized escalation rate", "VoiceBench judge score (1-5)",
      f"Score vs escalation rate — VoiceBench AlpacaEval (n={n})\n{SUB}")
ax.set_xlim(-.03, 1.03)
ax.legend(loc="lower right", fontsize=7.5, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"valpaca_dualview.{ext}", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---- Fig 10: latency vs score -------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.axhline(ceil, color=GREEN, ls=":", lw=1.0, alpha=.55, zorder=1)
ax.text(lat50[0] + .02, ceil - .04, f"always-escalate ceiling {ceil:.2f} "
        "(synthesized — no live latency)", fontsize=7.5, color=MUT,
        va="top")
ax.axhline(OFFICIAL, color=INK, ls=(0, (1, 3)), lw=1.2, alpha=.6, zorder=1)
ax.text(.99, .62, f"official {OFFICIAL}", transform=ax.transAxes,
        fontsize=7.5, color=MUT, ha="right")
ax.errorbar(lat50, gm, yerr=[gm - ciG[0], ciG[1] - gm], fmt="--s", ms=5.5,
            color=GREEN, capsize=3, lw=1.4, alpha=.9, zorder=3,
            label="gold-inject counterfactual")
ax.errorbar(lat50, sm, yerr=[sm - ciS[0], ciS[1] - sm], fmt="-o", ms=6,
            color=BLUE, capsize=3, lw=1.7, zorder=4,
            label="deployed live loop (probe v2)")
for j, a in enumerate(ARMS):
    ax.annotate(f"{a} ({rates[j]:.0%})", (lat50[j], sm[j]), xytext=(4, -14),
                textcoords="offset points", fontsize=7.5, color=BLUE)
style(ax, "P50 total response latency, query end → answer text done (s)",
      "VoiceBench judge score (1-5)",
      f"What latency buys — VoiceBench AlpacaEval (n={n})\n{SUB}")
ax.legend(loc="lower right", fontsize=7.5, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"valpaca_pareto.{ext}", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote valpaca_dualview + valpaca_pareto (+ valpaca_figures.json)")
