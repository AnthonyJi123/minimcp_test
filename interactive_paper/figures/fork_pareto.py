"""Clean redraw of the fork-depth Pareto (meeting feedback 2026-08-26:
no zigzag connect-the-dots). Scatter every layer colored by depth, draw
only the Pareto frontier (max AUC, min time), annotate the landmarks.
Data: _fork_data.txt = stdout of modal_audio.py::fork_report.

Run from figures/: ..\\..\\.venv_ip\\Scripts\\python fork_pareto.py
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, INK, MUT, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e4de"

arms = {}
arm = None
for line in open("_fork_data.txt"):
    m = re.match(r"=== (\w+)", line)
    if m:
        arm = m.group(1)
        arms[arm] = []
        continue
    m = re.match(r"\s+L(\d+)\s+([\d.]+) ms \(\s*([\d.]+)%\) oof=([\d.]+)",
                 line)
    if m and arm:
        arms[arm].append((int(m.group(1)), float(m.group(2)),
                          float(m.group(4))))

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
for ax, name in zip(axes, ("text", "audio")):
    pts = np.array(arms[name])
    layer, ms, oof = pts[:, 0], pts[:, 1], pts[:, 2]
    ax.grid(color=GRID, lw=.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    sc = ax.scatter(ms, oof, c=layer, cmap="BuPu", s=26, zorder=3,
                    edgecolors="white", linewidths=.4,
                    vmin=-6, vmax=layer.max())
    # Pareto frontier: sweep by time, keep points that raise the best AUC
    order = np.argsort(ms)
    fx, fy, best = [], [], -np.inf
    for i in order:
        if oof[i] > best:
            fx.append(ms[i])
            fy.append(oof[i])
            best = oof[i]
    ax.step(fx, fy, where="post", color=BLUE, lw=1.6, zorder=2,
            alpha=.85, label="Pareto frontier")
    for li, dy in ((11, -13), (16, -13), (22, 8)):
        r = pts[layer == li][0]
        ax.annotate(f"L{int(li)}", (r[1], r[2]), xytext=(2, dy),
                    textcoords="offset points", fontsize=8.5,
                    color=INK, zorder=5)
    r = pts[layer == layer.max()][0]
    ax.annotate(f"L{int(layer.max())} (final)", (r[1], r[2]),
                xytext=(-4, -13), textcoords="offset points",
                fontsize=8, color=MUT, zorder=5, ha="right")
    ax.set_title(f"{name} input", fontsize=10, loc="left")
    ax.set_xlabel("time-to-layer during prefill (ms, median)", fontsize=9)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("calibration OOF AUC\n(last-token probe)", fontsize=9)
axes[0].legend(loc="lower right", fontsize=8, frameon=False)
cb = fig.colorbar(sc, ax=axes, fraction=.025, pad=.02)
cb.set_label("layer depth", fontsize=8)
cb.ax.tick_params(labelsize=7)
for ext in ("png", "pdf"):
    fig.savefig(f"fork_pareto.{ext}", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote fork_pareto.{png,pdf}")
