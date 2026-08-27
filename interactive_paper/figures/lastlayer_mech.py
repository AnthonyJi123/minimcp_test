# -*- coding: utf-8 -*-
"""8aw mechanism figure: what the final-layer cliff is, and is not.

Both panels are NON-DESTRUCTIVE measurements of the probe trained at each
depth on the four non-math pools (no projections applied to the model, no
labels used to define the subspace). For each layer the held-out math
score is split into the part carried by that layer's top-5 principal
directions and the residual:  s = w.P x + w.(I-P) x.

  A  share of held-out score variance carried by the dominant subspace.
     Flat and small everywhere except the duplex model's last few layers.
  B  transfer AUC of the residual component --- the distributed part of
     the read. It survives to the output in both raw backbones and decays
     to inversion over the duplex model's final layers.

Data: scripts/17_probe_reliance.py -> data/interp_reliance.json.
Run from interactive_paper/: .venv_boot/Scripts/python.exe figures/lastlayer_mech.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
OUT = ["figures", "paper/figures"]

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUT, "ytick.color": MUT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42,
})

D = json.load(open("data/interp_reliance.json"))
MODELS = [
    ("minicpm-o45", "MiniCPM-o 4.5 (duplex)", C1, "-", 3.0),
    ("qwen3-8b", "Qwen3-8B (its raw backbone)", C2, "--", 2.2),
    ("minicpm-o26", "MiniCPM-o 2.6 (duplex)", C3, "-", 2.0),
    ("qwen2.5-7b", "Qwen2.5-7B (raw)", C4, "--", 2.0),
    ("qwen2.5-omni-7b", "Qwen2.5-Omni (streaming)", C5, ":", 2.0),
]

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5))

for tag, label, col, ls, lw in MODELS:
    if tag not in D:
        continue
    r = D[tag]
    n = r["n_layers"]
    x = [(i + 1) / n for i in r["layers"]]
    axes[0].plot(x, r["var_share_dom"], ls, color=col, lw=lw, label=label,
                 solid_capstyle="round")
    axes[1].plot(x, r["auc_res"], ls, color=col, lw=lw, label=label,
                 solid_capstyle="round")

ax = axes[0]
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 0.45)
ax.set_xlabel("relative depth")
ax.set_ylabel("score variance in the dominant subspace")
ax.set_title("A. At the output the o4.5 read collapses onto\n"
             "a few directions (o2.6: same sign, much weaker)")
ax.annotate(f"{D['minicpm-o45']['var_share_dom'][-1]:.2f}", xy=(1.0, 0.393),
            xytext=(0.80, 0.40), color=C1, fontsize=12,
            arrowprops=dict(arrowstyle="->", color=C1, lw=1.2))
ax.legend(loc="upper left", framealpha=.95)

ax = axes[1]
ax.axhline(0.5, color=MUT, lw=1.1, ls=(0, (3, 2)))
ax.text(0.015, 0.515, "chance", fontsize=10.5, color=MUT)
x22 = 23 / 36
ax.axvline(x22, color=INK, lw=1.2, ls=(0, (1, 2)))
ax.text(x22 - 0.02, 0.985, "L22 (deployed)", color=INK, fontsize=10.5,
        ha="right", va="top")
ax.set_xlim(0, 1.02)
ax.set_ylim(0.2, 1.0)
ax.set_xlabel("relative depth")
ax.set_ylabel("LOPO hard-math AUC, residual component")
ax.set_title("B. The distributed part of the read decays\n"
             "to the output only in the duplex models")
ax.legend(loc="lower left", framealpha=.95)

fig.tight_layout()
for d in OUT:
    fig.savefig(os.path.join(d, "lastlayer_mech.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(d, "lastlayer_mech.png"), dpi=200,
                bbox_inches="tight")
plt.close(fig)
print("wrote lastlayer_mech.{pdf,png} ->", ", ".join(OUT))
