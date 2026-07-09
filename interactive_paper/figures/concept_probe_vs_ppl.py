# -*- coding: utf-8 -*-
"""Conceptual figure: probe vs density-check vs perplexity (for discussion,
2026-07-08). Left: hidden-space view — supervised probe hyperplane vs
unsupervised density contours. Right: output-distribution view — what
entropy/PPL can and cannot see. Synthetic data, illustration only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GRAY, INK = "#4269D0", "#E4572E", "#9CA3AF", "#111827"
rng = np.random.default_rng(42)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6))

# ---------------- Panel 1: hidden-state space ----------------
n = 120
ok = rng.normal([-1.1, -0.55], [0.85, 0.75], (n, 2))      # 答对的题的 h
bad = rng.normal([1.1, 0.75], [0.9, 0.8], (n, 2))         # 答错的题的 h
ax1.scatter(*ok.T, s=26, c=BLUE, alpha=.55, lw=0, label="答对的题的 h")
ax1.scatter(*bad.T, marker="x", s=30, c=ORANGE, alpha=.65, lw=1.4,
            label="答错的题的 h")

# density contours of the POOLED cloud (unsupervised — labels ignored)
allpts = np.vstack([ok, bad])
mu, cov = allpts.mean(0), np.cov(allpts.T)
evals, evecs = np.linalg.eigh(cov)
ang = np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1]))
for k, lab in [(1.6, None), (2.6, "密度检查:只问「h 典型吗?」\n(无监督,不看对错标签)")]:
    e = Ellipse(mu, 2*k*np.sqrt(evals[-1]), 2*k*np.sqrt(evals[0]), angle=ang,
                fc="none", ec=GRAY, ls="--", lw=1.4)
    ax1.add_patch(e)
    if lab:
        ax1.annotate(lab, xy=(mu[0]-2.1*np.sqrt(evals[-1])*.72,
                              mu[1]+2.6*np.sqrt(evals[0])*.62),
                     fontsize=9.5, color="#6B7280", ha="right")

# probe hyperplane: perpendicular bisector of class means (supervised)
m0, m1 = ok.mean(0), bad.mean(0)
w = m1 - m0; mid = (m0 + m1) / 2
xs = np.linspace(-4.2, 4.4, 2)
ys = mid[1] - (w[0]/w[1]) * (xs - mid[0])
ax1.plot(xs, ys, c=INK, lw=2)
ax1.annotate("probe 超平面 w·h+b=0\n(有监督:方向由「对错」标签学出)",
             xy=(2.55, ys[-1]*0.25+0.3), xytext=(1.15, -2.55), fontsize=9.5,
             color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=1))

# the killer example: perfectly typical point on the fail side
star = np.array([0.62, 0.42])
ax1.scatter(*star, marker="*", s=340, c=ORANGE, ec=INK, lw=1.2, zorder=5)
ax1.annotate("这道题的 h 非常「典型」(在密度圈内)\n→ 密度检查放行;但落在答错一侧\n→ probe 拦截。两者根本不是一件事",
             xy=star, xytext=(-4.0, 2.35), fontsize=9.5, color=INK,
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))

ax1.set_title("视角一:隐状态空间 h(4096 维,示意画成 2 维)", fontsize=12)
ax1.legend(loc="lower right", fontsize=9, frameon=False)
ax1.set_xlim(-4.4, 4.6); ax1.set_ylim(-3.4, 3.6)
ax1.set_xticks([]); ax1.set_yticks([])
for s in ("top", "right"): ax1.spines[s].set_visible(False)

# ---------------- Panel 2: output-distribution view ----------------
toks = ["巴", "黎", "伦", "敦", "柏", "林", "罗", "马"]
conf = np.array([.84, .04, .03, .02, .02, .02, .02, .01])
hesi = np.array([.20, .16, .15, .13, .11, .10, .08, .07])
x = np.arange(8)
ax2.bar(x, conf, width=.72, color=BLUE, label="低熵(嘴上很自信)")
ax2.bar(x + 9.5, hesi, width=.72, color=GRAY, label="高熵(嘴上在犹豫)")
ax2.set_xticks(list(x) + list(x + 9.5), toks + toks, fontsize=8)
ax2.set_ylabel("下一个词的概率", fontsize=10)
ax2.set_title("视角二:输出分布(熵 / perplexity 只能看到这里)", fontsize=12)
ax2.annotate("「自信地答错」时,输出长这样——\n熵很低,perplexity 完全没有报警;\n但它的 h 在左图的橙色一侧,probe 看得见",
             xy=(0.4, .84), xytext=(3.1, .64), fontsize=9.5, color=INK,
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
ax2.legend(loc="upper right", fontsize=9, frameon=False)
ax2.set_ylim(0, 1.0)
for s in ("top", "right"): ax2.spines[s].set_visible(False)

fig.suptitle("同一个 h,三种「读法」:密度检查(h 典型吗?)/ probe(h 在会答错的一侧吗?)/ 熵(输出分布平不平?)",
             fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig("interactive_paper/figures/concept_probe_vs_ppl.png",
            dpi=130, bbox_inches="tight")
print("saved")
