# -*- coding: utf-8 -*-
"""Paper teaser (Figure 1), v2 2026-08-25: pictographic, minimal text,
with an explicit decision timeline (gate fires before the first token).
Vector output: PDF (paper) + SVG (editable). Palette: dataviz reference
slots, validated."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Polygon, Arc, Rectangle)

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
RED, GREEN = "#e34948", "#008300"
INK, MUT = "#0b0b0b", "#52514e"
ORANGE_T, AQUA_T, BLUE_T = "#fbe4da", "#dcf3ea", "#e2edfa"

fig, ax = plt.subplots(figsize=(13.2, 3.0))
ax.set_xlim(0, 13.2); ax.set_ylim(0, 3.0); ax.axis("off")

def pill(x, y, w, h, text, fc, ec, fs=9, tc=INK, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal")

def robot(cx, cy, s=0.34, color=INK):
    ax.add_patch(FancyBboxPatch((cx - s/2, cy - s/2), s, s*0.82,
                                boxstyle="round,pad=0.04", fc="white",
                                ec=color, lw=1.5))
    for dx in (-s/5, s/5):
        ax.add_patch(Circle((cx + dx, cy + s*0.05), s*0.085, fc=color))
    ax.plot([cx, cx], [cy + s*0.36, cy + s*0.56], color=color, lw=1.5)
    ax.add_patch(Circle((cx, cy + s*0.6), s*0.07, fc=color))

def soundwave(cx, cy, color=INK):
    for i, r in enumerate((0.10, 0.19, 0.28)):
        ax.add_patch(Arc((cx, cy), r*2, r*2, angle=0, theta1=-42,
                         theta2=42, color=color, lw=1.6))

def bolt(cx, cy, s=0.16, color=ORANGE):
    pts = np.array([[0.15, 1], [-0.35, 0.05], [0.02, 0.05], [-0.15, -1],
                    [0.42, 0.18], [0.05, 0.18]]) * s
    ax.add_patch(Polygon(pts + [cx, cy], closed=True, fc=color, ec="none"))

def cloud(cx, cy, s=0.30, fc="white", ec=INK):
    for dx, dy, r in [(-0.5, 0, 0.42), (0.05, 0.26, 0.55), (0.62, 0, 0.42)]:
        ax.add_patch(Circle((cx + dx*s, cy + dy*s), r*s, fc=fc, ec="none"))
    ax.add_patch(Rectangle((cx - 0.85*s, cy - 0.42*s), 1.85*s, 0.42*s,
                           fc=fc, ec="none"))
    for dx, dy, r in [(-0.5, 0, 0.42), (0.05, 0.26, 0.55), (0.62, 0, 0.42)]:
        ax.add_patch(Arc((cx + dx*s, cy + dy*s), 2*r*s, 2*r*s,
                         color=ec, lw=1.5))
    ax.plot([cx - 0.9*s, cx + 1.02*s], [cy - 0.42*s, cy - 0.42*s],
            color=ec, lw=1.5)

def layerstack(x, y, w, h, n=12, hot=7, hotcolor=BLUE):
    dh = h / n
    for i in range(n):
        c = hotcolor if i == hot else "#d9d7d2"
        ax.add_patch(FancyBboxPatch((x, y + i*dh), w, dh*0.62,
                                    boxstyle="round,pad=0.008",
                                    fc=c, ec="none"))

def cross(x, y, s=0.075, color=RED):
    ax.plot([x-s, x+s], [y-s, y+s], color=color, lw=2.6,
            solid_capstyle="round")
    ax.plot([x-s, x+s], [y+s, y-s], color=color, lw=2.6,
            solid_capstyle="round")

def sep(x):
    ax.plot([x, x], [0.10, 2.92], ls=(0, (4, 3)), color=INK, lw=1.5)

# ================= Panel 1 · problem =================
robot(0.75, 2.15, 0.44)
soundwave(0.30, 2.15)
ax.text(1.12, 2.42, "?", fontsize=17, color=ORANGE, fontweight="bold")

# three broken readouts, icon-first
# (a) last-layer tap
layerstack(2.05, 1.72, 0.34, 0.82, n=8, hot=7, hotcolor=ORANGE)
cross(2.60, 2.42)
ax.text(2.22, 1.57, "last layer", fontsize=8, color=MUT, ha="center",
        va="top")
# (b) ask itself (speech bubble)
ax.add_patch(FancyBboxPatch((2.62, 1.00), 0.62, 0.44,
                            boxstyle="round,pad=0.05", fc="white",
                            ec=ORANGE, lw=1.4))
ax.text(2.93, 1.22, "OK?", fontsize=9, color=ORANGE, ha="center",
        va="center", fontweight="bold")
ax.add_patch(Polygon([[2.76, 0.98], [2.90, 0.98], [2.72, 0.81]],
                     fc="white", ec=ORANGE, lw=1.2))
cross(3.42, 1.22)
ax.text(2.95, 0.70, "ask itself", fontsize=8, color=MUT, ha="center",
        va="top")
# (c) entropy squiggle
t = np.linspace(0, 1, 80)
ax.plot(0.55 + 0.9*t, 1.10 + 0.09*np.sin(16*t) * (0.4 + t), color=ORANGE,
        lw=1.8)
cross(1.62, 1.15)
ax.text(1.0, 0.92, "decode entropy", fontsize=8, color=MUT, ha="center",
        va="top")

ax.text(1.85, 0.28, "when to hand off?", ha="center", fontsize=11.5,
        fontweight="bold", color=INK)

sep(3.72)

# ================= Panel 2 · mechanism: fast, mid-network =================
# layer stack with L22 highlighted, probe needle out
layerstack(4.02, 0.95, 0.5, 1.7, n=14, hot=8, hotcolor=BLUE)
ax.text(4.27, 2.78, "36 layers", fontsize=8, color=MUT, ha="center")
hot_y = 0.95 + 8*(1.7/14) + 0.05
ax.add_patch(FancyArrowPatch((4.56, hot_y), (5.05, hot_y),
                             arrowstyle="-|>", mutation_scale=12, lw=1.8,
                             color=BLUE))
ax.text(4.82, hot_y + 0.12, "L22", fontsize=9.5, color=BLUE,
        fontweight="bold", ha="center")
gx, gy = 5.38, hot_y
ax.add_patch(Circle((gx, gy), 0.26, fc=BLUE_T, ec=BLUE, lw=1.7))
bolt(gx, gy, 0.15, BLUE)

# ---- the timeline ----
t0, t1 = 5.95, 8.35
ty = 1.15
ax.add_patch(FancyArrowPatch((t0, ty), (t1 + 0.25, ty), arrowstyle="-|>",
                             mutation_scale=12, lw=1.6, color=INK))
def tick(x, label, color=MUT, dy=-0.13):
    ax.plot([x, x], [ty - 0.05, ty + 0.05], color=color, lw=1.8)
    ax.text(x, ty + dy, label, fontsize=8, color=color, ha="center",
            va="top")
# end of user turn
tick(t0, "0", MUT)
# gate decision at 30ms
xg = t0 + 0.55
bolt(xg, ty + 0.28, 0.14, BLUE)
tick(xg, "30 ms", BLUE)
ax.text(xg, ty + 0.52, "gate", fontsize=9, color=BLUE, ha="center",
        fontweight="bold")
# first token at 68ms
xt = t0 + 1.35
ax.add_patch(Circle((xt, ty + 0.28), 0.075, fc=MUT))
tick(xt, "68 ms", MUT)
ax.text(xt, ty + 0.52, "1st token", fontsize=8.5, color=MUT, ha="center")
# expert call already flying (green arrow from gate onward, above)
ay = ty + 0.95
cloud(t1 - 0.15, ay + 0.02, 0.24, fc=AQUA_T, ec=AQUA)
ax.add_patch(FancyArrowPatch((xg, ay), (t1 - 0.55, ay), arrowstyle="-|>",
                             mutation_scale=11, lw=1.6, color=AQUA,
                             linestyle=(0, (5, 2))))
ax.plot([xg, xg], [ty + 0.06, ay], color=AQUA, lw=1.2,
        ls=(0, (2, 2)))
ax.text((xg + t1 - 0.6)/2, ay + 0.12, "expert already called",
        fontsize=8, color=AQUA, ha="center", fontweight="bold")
ax.text((t0 + t1)/2 + 0.1, 0.62,
        "decides before the first token", fontsize=10.5, color=INK,
        ha="center", fontweight="bold")
ax.text((t0 + t1)/2 + 0.1, 0.36, "LOPO .93 · text→speech .86",
        fontsize=8.5, color=MUT, ha="center")

sep(8.72)

# ================= Panel 3 · system =================
robot(9.25, 2.2, 0.4)
soundwave(8.95, 2.2)
gx2, gy2 = 9.95, 1.65
ax.add_patch(FancyArrowPatch((9.4, 2.0), (gx2 - 0.18, gy2 + 0.18),
                             arrowstyle="-|>", mutation_scale=11, lw=1.4,
                             color=INK))
ax.add_patch(Circle((gx2, gy2), 0.26, fc=BLUE_T, ec=BLUE, lw=1.7))
bolt(gx2, gy2, 0.15, BLUE)
# local branch
ax.add_patch(FancyArrowPatch((gx2 + 0.2, gy2 + 0.2), (10.7, 2.15),
                             arrowstyle="-|>", mutation_scale=11, lw=1.5,
                             color=AQUA))
robot(10.98, 2.2, 0.34, color=AQUA)
soundwave(11.35, 2.2, AQUA)
ax.text(11.15, 1.85, "local", fontsize=8.5, color=MUT, ha="center")
# cloud branch: gate -> cloud -> back to talker -> speaks
ax.add_patch(FancyArrowPatch((gx2 + 0.2, gy2 - 0.2), (10.65, 1.15),
                             arrowstyle="-|>", mutation_scale=11, lw=1.5,
                             color=AQUA))
cloud(11.0, 1.15, 0.3, fc=AQUA_T, ec=AQUA)
ax.add_patch(FancyArrowPatch((11.45, 1.15), (12.1, 1.15),
                             arrowstyle="-|>", mutation_scale=11, lw=1.5,
                             color=AQUA))
robot(12.4, 1.15, 0.34, color=AQUA)
soundwave(12.78, 1.15, AQUA)
ax.text(12.15, 0.78, "same talker speaks", fontsize=8.5, color=MUT,
        ha="center")
# result pills
pill(9.1, 0.22, 1.75, 0.4, "live  .42 → .63", "white", GREEN, fs=9.5,
     tc=GREEN)
pill(11.05, 0.22, 2.0, 0.4, "frozen  .66 → .86", "white", GREEN, fs=9.5,
     tc=GREEN)

fig.tight_layout(pad=0.3)
fig.savefig("teaser.pdf", bbox_inches="tight")
fig.savefig("teaser.svg", bbox_inches="tight")
fig.savefig("teaser.png", dpi=150, bbox_inches="tight")
print("saved")
