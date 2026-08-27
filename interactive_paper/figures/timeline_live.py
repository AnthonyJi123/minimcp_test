"""Figure: three REAL live sessions from the balanced arm, replotted as
timestamped timelines (gated_traces_v2.parquet — every event time is a
recorded timer from the live loop, not a median reconstruction).

  (a) q0254  escalated, expert ready at 1.9 s while the talker is still
      voicing the stall phrase -> seamless handoff, zero dead air
  (b) q0593  escalated (trap), expert takes 10.3 s -> the stall ends first,
      5.9 s of dead air before the relayed answer
  (c) q0388  gate stays closed -> local answer, fully decoded at 0.4 s

Measured: chunked prefill cadence, eot read, gate fire, stall prefill,
expert round-trip, relay prefill+decode, local answer decode. Estimated
(hatched): speech occupancy at 150 wpm — the live loop outputs text
(RESULTS 8e), so voicing time is words/2.5 s.

Run from repo root: python interactive_paper/figures/timeline_live.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1-2
GRAY, INK, MUT = "#b8b6af", "#0b0b0b", "#52514e"
WPS = 2.5                                    # est. speech rate (150 wpm)
LEAD = 5.5                                   # user-audio lead-in shown (s)
STALL_W = 11                                 # words in the stall phrase

df = pd.read_parquet("interactive_paper/data/gated_traces_v2.parquet")
df = df[df["tier"] == "balanced"].set_index("id")
S = {qid: df.loc[qid] for qid in ("q0254", "q0593", "q0388")}

fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.6), sharex=True)
plt.rcParams["font.size"] = 9

YT, YE = 0.0, 1.0                            # talker / expert lane y
H = 0.36

def lane_base(ax, r, fired):
    """User lead-in bar + chunk ticks + the gate event at t~0."""
    ax.barh(YT, LEAD, left=-LEAD, height=H, color=GRAY, zorder=2)
    for t in range(1, int(LEAD)):            # 1 s chunk boundaries
        ax.plot([-t, -t], [YT - H / 2, YT + H / 2], color="white",
                lw=0.8, zorder=3)
    ax.text(-LEAD + 0.12, YT, f"user speaks {r['audio_s']:.1f}s"
            + (" (trunc.)" if r["audio_s"] > LEAD else ""),
            va="center", fontsize=7.5, color=INK, zorder=4)
    ax.axvline(0, color=INK, lw=0.8)
    t_gate = r["eot_read_ms"] / 1000
    ax.plot(t_gate, YT + H * 0.85, marker="v", color=BLUE, ms=5, zorder=5)
    return t_gate

def speech(ax, x0, x1, color, label=None):
    ax.barh(YT, x1 - x0, left=x0, height=H, fill=False, hatch="///",
            edgecolor=color, lw=0.0, zorder=3)
    if label:
        ax.text((x0 + x1) / 2, YT, label, ha="center", va="center",
                fontsize=7.5, color=INK, zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))

def decode(ax, x0, x1, color):
    ax.barh(YT - H * 0.78, x1 - x0, left=x0, height=H * 0.34, color=color,
            zorder=4)

def note(ax, x, y, text, tx, ty, color=MUT):
    ax.annotate(text, xy=(x, y), xytext=(tx, ty), fontsize=7.5, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7))

# ---------- (a) q0254: expert beats the stall — seamless ------------------
ax, r = axes[0], S["q0254"]
t_gate = lane_base(ax, r, True)
note(ax, t_gate, YT + H * 0.85,
     f"gate fires t={r['eot_read_ms']:.0f} ms → expert launched; "
     f"stall prefilled +{r['stall_ms']:.0f} ms", 0.7, YT - 1.12, BLUE)
t_stall_end = 0.06 + STALL_W / WPS
speech(ax, 0.06, t_stall_end, BLUE, "stall phrase")
t_exp = 0.02 + r["expert_latency_s"]
ax.plot([0.02, t_exp], [YE] * 2, color=ORANGE, lw=1.0, ls="--", zorder=2)
ax.plot(t_exp, YE, marker="v", color=ORANGE, ms=6, zorder=4)
t_inj = t_exp + r["relay_ms"] / 1000
decode(ax, t_exp, t_inj, ORANGE)
note(ax, t_exp, YE, f"expert ready t={t_exp:.1f} s; injected + relay "
     f"decoded in {r['relay_ms']:.0f} ms —\ntalker still mid-stall → "
     "handoff with zero dead air", t_exp + 3.0, YE + 0.28, ORANGE)
ax.plot([t_inj, t_stall_end], [YE - 0.18, YT + H * 0.7], color=ORANGE,
        lw=0.7, ls=":", zorder=2)
n_w = len(str(r["relay"]).split())
speech(ax, t_stall_end, t_stall_end + n_w / WPS, ORANGE,
       "relayed expert answer")
ax.set_title("(a)  q0254 · hard-knowledge · escalated — expert (1.9 s) beats "
             "the stall: seamless handoff", fontsize=9, loc="left")

# ---------- (b) q0593: expert slower than the stall — dead air ------------
ax, r = axes[1], S["q0593"]
t_gate = lane_base(ax, r, True)
t_stall_end = 0.06 + STALL_W / WPS
speech(ax, 0.06, t_stall_end, BLUE, "stall phrase")
t_exp = 0.02 + r["expert_latency_s"]
ax.plot([0.02, t_exp], [YE] * 2, color=ORANGE, lw=1.0, ls="--", zorder=2)
ax.plot(t_exp, YE, marker="v", color=ORANGE, ms=6, zorder=4)
ax.barh(YT, t_exp - t_stall_end, left=t_stall_end, height=H, color=GRAY,
        hatch="...", edgecolor=GRAY, fill=False, zorder=3)
ax.text((t_stall_end + t_exp) / 2, YT,
        f"dead air {t_exp - t_stall_end:.1f} s", ha="center", va="center",
        fontsize=7.5, color=MUT, zorder=5)
t_inj = t_exp + r["relay_ms"] / 1000
decode(ax, t_exp, t_inj, ORANGE)
note(ax, t_exp, YE, f"expert ready t={t_exp:.1f} s (P95 of expert "
     "round-trips is 20 s) —\nstall alone can't bridge it",
     t_exp - 6.4, YE + 0.32, ORANGE)
n_w = len(str(r["relay"]).split())
speech(ax, t_inj, t_inj + n_w / WPS, ORANGE, "relayed expert answer")
ax.set_title("(b)  q0593 · trap · escalated — expert (10.3 s) outlives the "
             "stall: 5.9 s dead air", fontsize=9, loc="left")

# ---------- (c) q0388: gate closed — local answer -------------------------
ax, r = axes[2], S["q0388"]
t_gate = lane_base(ax, r, False)
note(ax, t_gate, YT + H * 0.85,
     f"gate reads end-of-turn state t={r['eot_read_ms']:.0f} ms → "
     "below threshold, no expert call", 0.55, YT + 0.62, BLUE)
t_dec = t_gate + r["answer_ms"] / 1000
decode(ax, t_gate, t_dec, BLUE)
note(ax, t_dec, YT - H * 0.78, f"answer decoded in {r['answer_ms']:.0f} ms",
     1.6, YT - 1.05)
n_w = len(str(r["answer"]).split())
speech(ax, t_dec, t_dec + n_w / WPS, BLUE, "local answer  (est. speech)")
ax.set_title("(c)  q0388 · easy-fact · not escalated — local answer "
             "fully decoded at 0.4 s", fontsize=9, loc="left")
ax.set_xlabel("time relative to end of user query (s)", fontsize=8)

for ax in axes:
    ax.set_xlim(-LEAD - 0.4, 18.2)
    ax.set_ylim(-1.45, 1.75)
    ax.set_yticks([YT, YE])
    ax.set_yticklabels(["talker\n(MiniCPM-o)", "expert\n(gpt-5.5 low)"],
                       fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)

legend = [Patch(facecolor=GRAY, label="user audio (1 s chunks → prefill)"),
          Patch(facecolor="white", edgecolor=BLUE, hatch="///",
                label="talker speech, own content (est. 150 wpm)"),
          Patch(facecolor="white", edgecolor=ORANGE, hatch="///",
                label="talker speech, expert content (est.)"),
          Patch(facecolor=BLUE, label="measured decode/prefill"),
          Line2D([], [], color=ORANGE, ls="--", marker="v", ms=5,
                 label="expert round-trip (measured)"),
          Patch(facecolor="white", hatch="...", edgecolor=GRAY,
                label="dead air")]
axes[2].legend(handles=legend, loc="upper center",
               bbox_to_anchor=(0.5, -0.34), fontsize=7.5, frameon=False,
               ncol=3, columnspacing=1.2, handlelength=1.6)

fig.suptitle("Three live sessions, timestamped (balanced arm; every event "
             "time is a recorded live-loop timer)", fontsize=9.5, x=0.02,
             ha="left")
fig.tight_layout(rect=(0, 0.02, 1, 0.97), h_pad=1.4)
for ext in ("png", "pdf"):
    fig.savefig(f"interactive_paper/figures/timeline_live.{ext}", dpi=220,
                bbox_inches="tight")
print("wrote timeline_live.png/.pdf")
