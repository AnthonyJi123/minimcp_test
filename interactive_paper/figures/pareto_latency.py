"""Figure: accuracy-latency Pareto tradeoff of the live sweep.

Joins the two frozen readouts of the same 4x240 live sessions:
  y : accuracy + 95% paired-bootstrap CIs  (figures/live_dualview.json, 8h)
  x : per-arm P50 total response latency   (figures/latency_profile.json, 8i)
Blue = honest heard-acc; green = gold-inject (channel-controlled) view of
the SAME sessions -- rescoring changes outcomes, not timing, so both views
share their x positions. Segment labels give the exchange rate ("+X pts /
+Y s at the median"). The always-escalate ceiling (synthesized, 8h) has no
live latency and is drawn as an asymptote. The expert P99 tail is priced
in the latency table, not here (median view; see tab:latency).

Run from repo root: python interactive_paper/figures/pareto_latency.py
"""
import json

import matplotlib.pyplot as plt

BLUE, GREEN = "#2a78d6", "#1e9e50"           # validated pair (deutan dE 24)
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e4de"
ARMS = ("never", "conservative", "balanced", "aggressive")

FIG = "interactive_paper/figures"
acc = json.load(open(f"{FIG}/live_dualview.json"))["arms"]
lat = json.load(open(f"{FIG}/latency_profile.json"))["arm"]

x = [lat[a]["all"]["p50"] for a in ARMS]
heard = [acc[a]["heard"] for a in ARMS]
h_lo = [acc[a]["heard"] - acc[a]["heard_ci"][0] for a in ARMS]
h_hi = [acc[a]["heard_ci"][1] - acc[a]["heard"] for a in ARMS]
gold = [acc[a]["gold_inject"] for a in ARMS]
g_lo = [acc[a]["gold_inject"] - acc[a]["gold_inject_ci"][0] for a in ARMS]
g_hi = [acc[a]["gold_inject_ci"][1] - acc[a]["gold_inject"] for a in ARMS]
ceil = json.load(open(f"{FIG}/live_dualview.json"))["gold_big"]

fig, ax = plt.subplots(figsize=(6.4, 4.2))

ax.axhline(ceil, color=GREEN, ls=":", lw=1.0, alpha=.55, zorder=1)
ax.text(1.62, ceil - .012, f"always-escalate ceiling {ceil:.3f} "
        "(synthesized — no live latency)", fontsize=7.5, color=MUT,
        va="top")

ax.errorbar(x, gold, yerr=[g_lo, g_hi], fmt="--s", ms=5.5, color=GREEN,
            capsize=3, lw=1.4, alpha=.9, zorder=3,
            label="gold-inject — counterfactual: expert answers the gold text")
ax.errorbar(x, heard, yerr=[h_lo, h_hi], fmt="-o", ms=6, color=BLUE,
            capsize=3, lw=1.7, zorder=4,
            label="heard-acc — deployed: expert answers the talker's transcript")

for a, xi, yi in zip(ARMS, x, heard):
    esc = acc[a]["esc"]
    ax.annotate(f"{esc:.0%}", (xi, yi), xytext=(2, -14),
                textcoords="offset points", fontsize=8, color=BLUE)

# exchange rate per segment, on the honest curve (first above, rest below
# the blue line where the plane is empty)
for i in range(3):
    dx, dy = x[i + 1] - x[i], heard[i + 1] - heard[i]
    mx, my = (x[i] + x[i + 1]) / 2, (heard[i] + heard[i + 1]) / 2
    off = (-8, 14) if i == 0 else (0, -24)
    ax.annotate(f"+{dy * 100:.1f} pts / +{dx:.1f} s", (mx, my),
                xytext=off, textcoords="offset points", fontsize=7.5,
                color=MUT, ha="right" if i == 0 else "center")

# channel cost at the aggressive arm: same sessions, gold vs heard outcomes
ax.annotate("", xy=(x[3], gold[3] - .012), xytext=(x[3], heard[3] + .012),
            arrowprops=dict(arrowstyle="<->", color=MUT, lw=0.9))
ax.text(x[3] + .07, (heard[3] + gold[3]) / 2,
        f"speech-channel cost −{gold[3] - heard[3]:.3f}\n(95% upstream of the"
        " relay:\ncorrupted transcripts)", fontsize=7.5,
        color=MUT, va="center")

ax.text(.985, .02, "median view; the tail is priced separately — P99 "
        "17.8 s (never) → 30.4 s (balanced),\nall of it the expert "
        "reasoning tail (see the latency table)", transform=ax.transAxes,
        fontsize=7, color=MUT, ha="right", va="bottom", style="italic")

ax.set_xlabel("P50 total response latency, query end → answer text "
              "done (s)", fontsize=9)
ax.set_ylabel("accuracy on the live test set (n=240)", fontsize=9)
ax.set_title("What latency buys: the live accuracy–latency tradeoff\n"
             "(points = escalation arms, labelled by escalation rate)",
             fontsize=9.5, loc="left")
ax.set_xlim(1.5, 5.75)
ax.set_ylim(.30, 1.0)
ax.grid(color=GRID, lw=.7, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper left", fontsize=7.5,
          frameon=False)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG}/pareto_latency.{ext}", dpi=220, bbox_inches="tight")
print("wrote pareto_latency.png/.pdf")
