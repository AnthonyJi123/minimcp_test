"""Figure: escalation timelines — the fork, and who occupies the audio channel.

Panel (a): millisecond zoom — the gate decision (L22 probe) precedes the first
output token; the cloud call launches while prefill is still running.
Panel (b): second-scale scenario comparison — pre-answer routing (dead air)
vs the gated design on three pools (full overlap / short stall / worst case).

All numbers are measured medians (text arm, greedy, H100):
  latency_bench.parquet  (6c; n=47 non-warmup)  — local draft durations,
      L22 decision 20 ms, TTFT 36 ms
  eval_expert.parquet    (Phase 5; n=240)       — per-pool gpt-5.5 latency
Run from repo root: python interactive_paper/figures/timeline_scenarios.py
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1-2
GRAY, INK, MUT = "#b8b6af", "#0b0b0b", "#52514e"

GATE_S, TTFT_S = 0.020, 0.036                # L22 decision / first token
DRAFT = {"hard-math": 3.53, "easy-fact": 0.51, "trap": 0.82}   # local P50 (s)
EXPERT = {"hard-math": 2.72, "easy-fact": 2.27, "trap": 8.22}  # gpt-5.5 P50 (s)
RELAY = 2.0                                  # talker re-voices expert result

fig, (ax0, ax1) = plt.subplots(
    2, 1, figsize=(9.0, 5.2), gridspec_kw={"height_ratios": [1, 3.6]})
plt.rcParams["font.size"] = 9

# ---------------- (a) millisecond zoom: the fork --------------------------
ax0.barh(0, 36, left=0, height=0.42, color=GRAY, edgecolor="none")
ax0.axvline(20, color=BLUE, lw=1.4)
ax0.axvline(36, color=INK, lw=1.0, ls=":")
ax0.annotate("L22 hidden → probe decision (20 ms, 57% of prefill)\n"
             "→ cloud call launched", xy=(20, 0.21), xytext=(21.2, 0.62),
             fontsize=8, color=BLUE,
             arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
ax0.annotate("first output token (36 ms)", xy=(36, 0.21), xytext=(30.5, -0.82),
             fontsize=8, color=MUT,
             arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8))
ax0.text(1.0, 0.0, "prefill (36 layers)", va="center", fontsize=8,
         color=INK)
ax0.set_xlim(0, 43)
ax0.set_ylim(-1.05, 1.05)
ax0.set_yticks([])
ax0.set_xlabel("time after user query ends (ms)", fontsize=8, labelpad=1)
ax0.set_title("(a)  The fork: the gate decides before the first token "
              "(measured medians, text input)", fontsize=9, loc="left")

# ---------------- (b) second-scale scenarios ------------------------------
H = 0.34                                     # lane bar height
lanes = [                                    # bottom-up y positions
    ("trap", 0), ("easy-fact", 1), ("hard-math", 2), ("baseline", 3)]
ypos = {name: i * 1.05 for name, i in lanes}

def seg(y, x0, x1, color, hatch=None, label_in=None):
    ax1.barh(y, x1 - x0, left=x0, height=H, color=color, hatch=hatch,
             edgecolor="white" if hatch is None else color,
             lw=0.5 if hatch is None else 0.0,
             fill=hatch is None, zorder=3)
    if label_in:
        ax1.text((x0 + x1) / 2, y, label_in, ha="center", va="center",
                 fontsize=7.5, color=INK, zorder=4)

def cloud(y, t_ready):                       # in-flight line + arrival marker
    ax1.plot([GATE_S, t_ready], [y + H * 0.95] * 2, color=ORANGE, lw=1.0,
             ls="--", zorder=2)
    ax1.plot(t_ready, y + H * 0.95, marker="v", color=ORANGE, ms=5, zorder=4)

# baseline: pre-answer routing --------------------------------------------
y = ypos["baseline"]
seg(y, GATE_S, EXPERT["hard-math"], GRAY, hatch="..",
    label_in="dead air 2.7 s")
seg(y, EXPERT["hard-math"], EXPERT["hard-math"] + RELAY + 1.5, ORANGE,
    label_in="speaks expert result")
cloud(y, EXPERT["hard-math"])

# gated: hard-math — full overlap ------------------------------------------
y = ypos["hard-math"]
seg(y, TTFT_S, DRAFT["hard-math"], BLUE, label_in="draft answer 3.5 s")
seg(y, DRAFT["hard-math"], DRAFT["hard-math"] + RELAY, ORANGE,
    label_in="refined result")
cloud(y, GATE_S + EXPERT["hard-math"])
ax1.annotate("expert ready before draft ends — zero silence",
             xy=(GATE_S + EXPERT["hard-math"], y + H), xytext=(4.9, y + 0.52),
             fontsize=7.5, color=MUT,
             arrowprops=dict(arrowstyle="-", color=MUT, lw=0.7))

# gated: easy-fact — short stall -------------------------------------------
y = ypos["easy-fact"]
seg(y, TTFT_S, DRAFT["easy-fact"], BLUE, label_in="draft")
seg(y, DRAFT["easy-fact"], GATE_S + EXPERT["easy-fact"], BLUE, hatch="//",
    label_in="stall 1.8 s")
seg(y, GATE_S + EXPERT["easy-fact"], GATE_S + EXPERT["easy-fact"] + RELAY,
    ORANGE, label_in="expert result")
cloud(y, GATE_S + EXPERT["easy-fact"])

# gated: trap — worst case -------------------------------------------------
y = ypos["trap"]
seg(y, TTFT_S, DRAFT["trap"], BLUE, label_in="draft")
seg(y, DRAFT["trap"], GATE_S + EXPERT["trap"], BLUE, hatch="//")
seg(y, GATE_S + EXPERT["trap"], GATE_S + EXPERT["trap"] + RELAY, ORANGE,
    label_in="expert result")
cloud(y, GATE_S + EXPERT["trap"])
ax1.text(4.55, y + 0.50, "P50 gap 7.4 s (P90 54 s): stalling alone can't "
         "bridge → defer / fast-expert tier / streamed partials (step 2)",
         fontsize=7.5, color=MUT)

ax1.axvline(0, color=INK, lw=0.8)
ax1.text(0.06, ypos["baseline"] + 0.72,
         "gate fires + cloud call at t = 20 ms (panel a)", fontsize=7.5,
         color=MUT, style="italic")

ax1.set_yticks([ypos[n] for n, _ in lanes])
ax1.set_yticklabels(["gated · trap\n(worst case)",
                     "gated · easy-fact\n(short stall)",
                     "gated · hard-math\n(full overlap)",
                     "pre-answer routing\n(RouteLLM-style)"], fontsize=8)
ax1.set_xlim(-0.05, 11.0)
ax1.set_ylim(-0.38, ypos["baseline"] + 1.0)
ax1.set_xlabel("time after user query ends (s)", fontsize=8)
ax1.set_title("(b)  Who occupies the audio channel (per-pool measured "
              "medians: draft = local answer, ▾ = expert result ready)",
              fontsize=9, loc="left")

legend = [Patch(facecolor=BLUE, label="talker: draft answer"),
          Patch(facecolor="white", edgecolor=BLUE, hatch="///",
                label="talker: stall phrase"),
          Patch(facecolor=ORANGE, label="talker: expert result"),
          Line2D([], [], color=ORANGE, ls="--", marker="v", ms=5,
                 label="cloud round-trip (gpt-5.5)"),
          Patch(facecolor=GRAY, hatch="...", edgecolor=GRAY,
                label="dead air")]
ax1.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.14),
           fontsize=7.5, frameon=False, ncol=5, columnspacing=1.2,
           handlelength=1.6)

for ax in (ax0, ax1):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)

fig.tight_layout(h_pad=1.6)
for ext in ("png", "pdf"):
    fig.savefig(f"interactive_paper/figures/timeline_scenarios.{ext}",
                dpi=220, bbox_inches="tight")
print("wrote timeline_scenarios.png/.pdf")
