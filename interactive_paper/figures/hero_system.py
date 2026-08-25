# -*- coding: utf-8 -*-
"""Hero system figure for the intro (2026-08-25): speech -> talker ->
L22 gate -> {local, cloud->inject->speak}, with headline numbers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, GREEN, ORANGE, INK, MUT = "#2a78d6", "#1e9e50", "#E4572E", "#0b0b0b", "#52514e"

fig, ax = plt.subplots(figsize=(10.5, 3.1))
ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.1); ax.axis("off")

def box(x, y, w, h, text, fc="#ffffff", ec=INK, fs=10.5, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, color=tc or INK)

def arrow(x1, y1, x2, y2, label=None, ly=0.16):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.3, color=INK))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + ly, label, ha="center",
                fontsize=8.5, color=MUT)

box(0.15, 1.55, 1.25, 0.95, "speech\nin / out")
box(2.05, 1.55, 2.05, 0.95, "9B duplex talker\n(MiniCPM-o 4.5)")
box(4.95, 1.55, 2.0, 0.95, "competence gate\n$w{\cdot}h_{L22}+b$,  ~30 ms",
    fc="#eef3fb", ec=BLUE)
box(7.85, 2.05, 2.45, 0.7, "answer locally", ec=MUT)
box(7.85, 0.85, 2.45, 0.95,
    "cloud expert\nanswer injected; same talker speaks it", ec=GREEN, fs=9.5)

arrow(1.40, 2.02, 2.05, 2.02)
arrow(4.10, 2.02, 4.95, 2.02, "$h_{L22}$ at end of turn", 0.30)
arrow(6.95, 2.18, 7.85, 2.38)
arrow(6.95, 1.86, 7.85, 1.42)

ax.text(5.25, 0.28,
        "final-layer probe, held-out math: .37    mid-layer (L22): .93    "
        "live end-of-turn gate: AUC .887, pre-first-token    "
        "Speech TriviaQA: .664 \u2192 .860 live",
        ha="center", fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f7f6f3", ec="#e6e4de"))

fig.tight_layout()
fig.savefig("hero_system.pdf", bbox_inches="tight")
fig.savefig("hero_system.png", dpi=150, bbox_inches="tight")
print("saved")
