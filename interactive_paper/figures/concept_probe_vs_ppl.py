# -*- coding: utf-8 -*-
"""Conceptual figure: probe vs density-check vs perplexity (for discussion,
2026-07-08; relabeled in English 2026-08-25 for the paper, M2a).
Left: hidden-space view - supervised probe hyperplane vs unsupervised
density contours. Right: output-distribution view - what entropy/PPL can
and cannot see. Synthetic data, illustration only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

BLUE, ORANGE, GRAY, INK = "#4269D0", "#E4572E", "#9CA3AF", "#111827"
rng = np.random.default_rng(42)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6))

# ---------------- Panel 1: hidden-state space ----------------
n = 120
ok = rng.normal([-1.1, -0.55], [0.85, 0.75], (n, 2))      # h of passes
bad = rng.normal([1.1, 0.75], [0.9, 0.8], (n, 2))         # h of fails
ax1.scatter(*ok.T, s=26, c=BLUE, alpha=.55, lw=0, label="h of answered-correctly queries")
ax1.scatter(*bad.T, marker="x", s=30, c=ORANGE, alpha=.65, lw=1.4,
            label="h of failed queries")

# density contours of the POOLED cloud (unsupervised - labels ignored)
allpts = np.vstack([ok, bad])
mu, cov = allpts.mean(0), np.cov(allpts.T)
evals, evecs = np.linalg.eigh(cov)
ang = np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1]))
for k, lab in [(1.6, None), (2.6, "density check: only asks \"is $h$ typical?\"\n(unsupervised; ignores fail labels)")]:
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
ax1.annotate("probe hyperplane $w{\\cdot}h+b=0$\n(supervised: direction learned\nfrom fail labels)",
             xy=(2.55, ys[-1]*0.25+0.3), xytext=(1.15, -2.75), fontsize=9.5,
             color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=1))

# the killer example: perfectly typical point on the fail side
star = np.array([0.62, 0.42])
ax1.scatter(*star, marker="*", s=340, c=ORANGE, ec=INK, lw=1.2, zorder=5)
ax1.annotate("a perfectly \"typical\" $h$ (inside the contour):\ndensity check passes it --- but it sits on the\nfail side, so the probe catches it.\nThe two tests differ in kind",
             xy=star, xytext=(-4.0, 2.25), fontsize=9.5, color=INK,
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))

ax1.set_title("View 1: hidden-state space $h$ (4096-d, drawn as 2-d)", fontsize=12)
ax1.legend(loc="lower right", fontsize=9, frameon=False)
ax1.set_xlim(-4.4, 4.6); ax1.set_ylim(-3.4, 3.6)
ax1.set_xticks([]); ax1.set_yticks([])
for s in ("top", "right"): ax1.spines[s].set_visible(False)

# ---------------- Panel 2: output-distribution view ----------------
toks = ["Pa-", "ris", "Lon-", "don", "Ber-", "lin", "Ro-", "me"]
conf = np.array([.84, .04, .03, .02, .02, .02, .02, .01])
hesi = np.array([.20, .16, .15, .13, .11, .10, .08, .07])
x = np.arange(8)
ax2.bar(x, conf, width=.72, color=BLUE, label="low entropy (sounds confident)")
ax2.bar(x + 9.5, hesi, width=.72, color=GRAY, label="high entropy (sounds hesitant)")
ax2.set_xticks(list(x) + list(x + 9.5), toks + toks, fontsize=8)
ax2.set_ylabel("next-token probability", fontsize=10)
ax2.set_title("View 2: output distribution (all that entropy/PPL sees)", fontsize=12)
ax2.annotate("a confident wrong answer looks like this:\nlow entropy, no perplexity alarm ---\nbut its $h$ sits on the fail side at left,\nwhere the probe can see it",
             xy=(0.4, .84), xytext=(3.1, .58), fontsize=9.5, color=INK,
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
ax2.legend(loc="upper right", fontsize=9, frameon=False)
ax2.set_ylim(0, 1.0)
for s in ("top", "right"): ax2.spines[s].set_visible(False)

fig.suptitle("One hidden state, three readings: density (is $h$ typical?) / "
             "probe (is $h$ on the fail side?) / entropy (is the output flat?)",
             fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig("concept_probe_vs_ppl.png", dpi=130, bbox_inches="tight")
fig.savefig("concept_probe_vs_ppl.pdf", bbox_inches="tight")
print("saved")
