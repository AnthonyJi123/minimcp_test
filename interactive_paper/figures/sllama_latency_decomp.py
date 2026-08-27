"""Jisen note 2026-08-18, Q5: why does the sllama pareto fold left?
Two-panel answer: (left) the fold lives only in the P50 — the mean is
monotonic; (right) the mixture decomposition — the local-only median
falls as the probe strips the slowest decodes, while the escalated rows
pay a flat ~3 s expert round-trip. Style matches bench_figures.py.

Run from figures/: ..\\..\\.venv_ip\\Scripts\\python sllama_latency_decomp.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, GREEN = "#2a78d6", "#1baf7a"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e4de"
ARMS = ("never", "conservative", "balanced", "aggressive")

df = pd.read_parquet("../data/sllama_v3_traces.parquet")
df = df[df["tier"].isin(ARMS) & df["oab_ok"].notna()]
df["expert_ms"] = df["expert_latency_s"].fillna(0) * 1000
is_esc = df["mode"] == "escalated"
df["total_ms"] = np.where(
    is_esc,
    df["eot_read_ms"] + np.maximum(df["stall_ms"].fillna(0),
                                   df["expert_ms"]) + df["relay_ms"].fillna(0),
    df["eot_read_ms"] + df["answer_ms"].fillna(0))

acc, p50, mean, esc_rate, loc50, esc50, esc_n = [], [], [], [], [], [], []
for a in ARMS:
    sub = df[df["tier"] == a]
    acc.append(sub["oab_ok"].mean())
    p50.append(sub["total_ms"].median() / 1000)
    mean.append(sub["total_ms"].mean() / 1000)
    e = sub[sub["mode"] == "escalated"]
    l = sub[sub["mode"] != "escalated"]
    esc_rate.append(len(e) / len(sub))
    loc50.append(l["total_ms"].median() / 1000)
    esc50.append(e["total_ms"].median() / 1000 if len(e) else np.nan)
    esc_n.append(len(e))

fig, ax2 = plt.subplots(figsize=(6.4, 4.0))
ax2.grid(color=GRID, lw=.7, zorder=0)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(labelsize=8)

# mixture decomposition per arm (single panel per 2026-08-26 feedback;
# the median-vs-mean fold fact lives in prose)
x = np.arange(len(ARMS))
ax2.plot(x, loc50, "-o", ms=6, color=BLUE, lw=1.5, zorder=4,
         label="kept-local rows, P50 decode")
ax2.plot(x[1:], esc50[1:], "-s", ms=6, color=GREEN, lw=1.5, zorder=4,
         label="escalated rows, P50 expert round-trip")
ax2.plot(x, p50, "--", color=MUT, lw=1.2, alpha=.8, zorder=3,
         label="arm P50 (the mixture)")
for j in range(len(ARMS)):
    ax2.annotate(f"{loc50[j]:.2f}", (x[j], loc50[j]), xytext=(0, -14),
                 textcoords="offset points", fontsize=7.5, color=BLUE,
                 ha="center")
    if esc_n[j]:
        ax2.annotate(f"{esc50[j]:.2f}", (x[j], esc50[j]), xytext=(0, 7),
                     textcoords="offset points", fontsize=7.5,
                     color=GREEN, ha="center")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{a}\n({esc_rate[j]:.0%} esc)"
                     for j, a in enumerate(ARMS)], fontsize=8)
ax2.set_ylabel("P50 latency of each sub-population (s)", fontsize=9)
ax2.set_title("Why: the probe escalates the slowest local decodes —\n"
              "the kept-local median falls faster than the expert path "
              "enters", fontsize=9.5, loc="left")
ax2.legend(loc="center right", fontsize=7.5, frameon=False)
ax2.set_ylim(0, 4.3)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"sllama_latency_decomp.{ext}", dpi=220,
                bbox_inches="tight")
plt.close(fig)
print("wrote sllama_latency_decomp.{png,pdf}",
      "| P50", np.round(p50, 2), "| mean", np.round(mean, 2),
      "| local P50", np.round(loc50, 2), "| esc P50", np.round(esc50, 2))
