"""NVDA NemotronLabs-VoiceChat-11B transfer-arm figures (2026-08-19).

Fig 15 nvda_layer_sweep : probe AUC vs backbone layer (56-layer hybrid
        Mamba2/attn) — does the mid-band structure replicate on a
        second full-duplex family? (§9 pre-registered test)
Fig 16 nvda_transfer    : frozen-methodology transfer — OOF + external
        AUC per feature config, MiniCPM-o v3 reference per pool.

Data: ../data/nvda_probe_sweep.json + nvda_{tag}.parquet.
Run from figures/: ..\\..\\.venv_ip\\Scripts\\python nvda_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, GREEN = "#2a78d6", "#1baf7a"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e4de"

sweep = json.load(open("../data/nvda_probe_sweep.json"))
layers = sorted(int(k) for k in sweep["layers"])
aucs = [sweep["layers"][str(L)] for L in layers]

# MiniCPM-o 4.5 v3 references (RESULTS 8z, same recipe, our judge)
MINICPM = {"frozen-test": .879, "striviaqa": .789, "swebq": .785,
           "sdqa": .792, "sllama": .806}


def style(ax, xlab, ylab, title):
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.grid(color=GRID, lw=.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


# ---- Fig 15: layer sweep -------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(layers, aucs, "-o", ms=6, color=BLUE, lw=1.7, zorder=4)
bi = int(np.argmax(aucs))
ax.annotate(f"L{layers[bi]} = {aucs[bi]:.3f}",
            (layers[bi], aucs[bi]), xytext=(6, 8),
            textcoords="offset points", fontsize=8.5, color=BLUE)
ax.axhline(.5, color=MUT, ls=":", lw=1, alpha=.6)
ax.text(.99, .505, "chance", transform=ax.get_yaxis_transform(),
        fontsize=7.5, color=MUT, ha="right", va="bottom")
ax.axvspan(28, 40, color=BLUE, alpha=.06, zorder=1)
ax.text(34, min(aucs) - .004, "mid-band", fontsize=7.5, color=MUT,
        ha="center")
style(ax, "backbone layer (56 NemotronHBlocks, hybrid Mamba2/attention)",
      "5-fold OOF AUC, eot_last read (calib n=600)",
      "Second duplex family, same probe recipe — NemotronLabs-"
      "VoiceChat-11B\nlayer sweep replicates the mid-band structure "
      "(MiniCPM-o: L22/28)")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"nvda_layer_sweep.{ext}", dpi=220, bbox_inches="tight")
plt.close(fig)

# ---- Fig 16: transfer AUC ------------------------------------------------
combos = list(sweep["combos"].items())
pools = ["striviaqa", "swebq", "sdqa", "sllama"]
labels = ["frozen-test\n(OOF)"] + pools
x = np.arange(len(labels))
w = 0.26
fig, ax = plt.subplots(figsize=(7.4, 4.2))
shades = [.35, .65, 1.0]
for i, (name, row) in enumerate(combos):
    vals = [row["oof"]] + [row["ext"].get(p, np.nan) for p in pools]
    ax.bar(x + (i - 1) * w, vals, w, color=BLUE, alpha=shades[i],
           zorder=3, label=name)
for j, lbl in enumerate(labels):
    key = "frozen-test" if j == 0 else pools[j - 1]
    ax.plot([x[j] - 1.55 * w, x[j] + 1.55 * w], [MINICPM[key]] * 2,
            color=GREEN, lw=1.6, zorder=4)
ax.plot([], [], color=GREEN, lw=1.6,
        label="MiniCPM-o 4.5 probe v3 (n=2310 calib)")
ax.axhline(.5, color=MUT, ls=":", lw=1, alpha=.6)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(.45, .95)
style(ax, "", "escalation-probe AUC (never-arm fail label, our judge)",
      "Frozen methodology on a new backbone — NVDA VoiceChat-11B\n"
      "same three reads, logistic C=1e-4; calib n=600 (vs MiniCPM's "
      "2310)")
ax.legend(loc="upper right", fontsize=7, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"nvda_transfer.{ext}", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote nvda_layer_sweep + nvda_transfer")
