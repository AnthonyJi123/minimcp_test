"""NVDA escalation re-mix figure (2026-08-26) — tab:transfer's NVDA analog.

One panel per pool: accuracy (VB score for AlpacaEval) vs escalation
rate; selective = top-r by the frozen-recipe NVDA probe, random =
matched-rate expectation. Official judge per pool (OAB / ours for
SD-QA / VoiceBench 1-5). Expert = gpt-5.5 (low) on the heard
transcript, judged directly (relay-free A-protocol).

Data: nvda_scores(.parquet|_valpaca), nvda_{pool}.parquet (oab_ok),
nvda_expert_outcomes.parquet.  Output: nvda_remix.{png,pdf}.
Run from figures/: ..\\..\\.venv_ip\\Scripts\\python nvda_remix_fig.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, GREEN = "#2a78d6", "#1e9e50"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e4de"
D = "../data"

S = pd.read_parquet(f"{D}/nvda_scores.parquet")
EXP = pd.read_parquet(f"{D}/nvda_expert_outcomes.parquet")

POOLS = [("striviaqa", "Speech TriviaQA", "oab_ok", "OAB judge"),
         ("swebq", "Speech Web Questions", "oab_ok", "OAB judge"),
         ("sllama", "Llama Questions", "oab_ok", "OAB judge"),
         ("sdqa", "SD-QA", "adequate", "our judge"),
         ("valpaca", "AlpacaEval", None, "VoiceBench 1-5")]
RS = np.arange(0, 1.01, 0.05)
TIERS = (0.15, 0.30, 0.50)

fig, axes = plt.subplots(1, 5, figsize=(13.2, 2.9))
for ax, (pool, title, loc_col, judge) in zip(axes, POOLS):
    if pool == "valpaca":
        sv = pd.read_parquet(f"{D}/nvda_scores_valpaca.parquet")
        e = EXP[EXP.pool == "valpaca"].set_index("id")
        d = sv[["id", "score"]].copy()
        d["loc"] = sv["vb_score"].values
        d["exp"] = d["id"].map(e["expert_score"]).astype(float)
    else:
        nv = pd.read_parquet(f"{D}/nvda_{pool}.parquet").set_index("id")
        e = EXP[EXP.pool == pool].set_index("id")
        d = S[S.pool == pool][["id", "score"]].copy()
        d["loc"] = d["id"].map(nv[loc_col]).astype(float)
        d["exp"] = d["id"].map(e["expert_ok"]).astype(float)
    d = d.dropna().sort_values("score", ascending=False) \
        .reset_index(drop=True)
    n = len(d)

    sel = [np.concatenate([d["exp"][:int(round(r * n))],
                           d["loc"][int(round(r * n)):]]).mean()
           for r in RS]
    rnd = [(1 - r) * d["loc"].mean() + r * d["exp"].mean() for r in RS]

    ax.grid(color=GRID, lw=.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.plot(RS, rnd, ls="--", color=MUT, lw=1.3, zorder=3,
            label="random @ rate")
    ax.plot(RS, sel, "-", color=BLUE, lw=1.8, zorder=4,
            label="probe top-r")
    ax.plot(TIERS, [sel[int(np.argmin(np.abs(RS - t)))] for t in TIERS],
            "o", ms=5, color=BLUE, zorder=5)
    ax.set_title(f"{title}\n({judge}, n={n})", fontsize=8.5, loc="left")
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("escalation rate", fontsize=8)
    if pool == "valpaca":
        ax.set_ylim(3.3, 5.05)
    else:
        ax.set_ylim(0, 1.0)
axes[0].set_ylabel("live-floor accuracy", fontsize=8)
axes[-1].set_ylabel("judge score (1-5)", fontsize=8)
axes[0].legend(loc="lower right", fontsize=7, frameon=False)
fig.suptitle("Frozen-recipe NVDA probe, offline re-mix: selective vs "
             "matched-rate random escalation (all five pools "
             "p≤.0003, permutation)", fontsize=9.5, y=1.04)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"nvda_remix.{ext}", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote nvda_remix.png/pdf")
