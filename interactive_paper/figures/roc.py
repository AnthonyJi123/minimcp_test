# -*- coding: utf-8 -*-
"""fig:roc regenerated locally (2026-08-25, M2b): adds the p(True) curves
to the original probe-vs-scalar ROC (calibration split, MiniCPM-o 4.5
text mode). Recipe matches modal_app.py::calibrate: 5-fold OOF LR(C=1.0)
on h_prompt; scalars straight from stored per-step entropies; p(True)
from ptrue.shard*.parquet (Phase 5b). Legend carries no AUC numbers on
purpose - Table tab:rq1 is the source of truth (local CV-fold jitter is
±.003-.013 vs the archived Modal run)."""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

BLUE, GREEN, ORANGE, GRAY, INK = "#2a78d6", "#1e9e50", "#E4572E", "#9CA3AF", "#111827"

df = pd.read_parquet("../data/calib_features.parquet")
pt = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob("../data/ptrue.shard*.parquet"))]).set_index("id")
c = df[df.split == "calib"].reset_index(drop=True)
y = c["escalate_label"].astype(int).values
H = np.stack(c["h_prompt"].values)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
probe = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), H, y,
                          cv=cv, method="predict_proba")[:, 1]
mx4 = np.array([max(list(e)[:4]) if len(e) else 0 for e in c["entropy"]])
pre = np.array([1 - pt.loc[i, "p_yes_pre"] for i in c["id"]])
post = np.array([1 - pt.loc[i, "p_yes_post"] for i in c["id"]])

plt.figure(figsize=(5.6, 5.6))
for name, sc, color, ls in [
        ("final-layer probe ($h_{\mathrm{prompt}}$, 5-fold OOF)", probe, BLUE, "-"),
        ("$p(\mathrm{True})$-post (needs a full draft)", post, GREEN, "-"),
        ("$p(\mathrm{True})$-pre (pre-answer)", pre, GREEN, "--"),
        ("max entropy@4 (best decode scalar)", mx4, ORANGE, "-")]:
    fpr, tpr, _ = roc_curve(y, sc)
    plt.plot(fpr, tpr, color=color, ls=ls, lw=1.9, label=name)
plt.plot([0, 1], [0, 1], color=GRAY, ls=":", lw=1.2)
plt.xlabel("false positive rate", fontsize=10)
plt.ylabel("true positive rate", fontsize=10)
plt.legend(loc="lower right", fontsize=8.5, frameon=False)
ax = plt.gca()
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("roc.png", dpi=150, bbox_inches="tight")
plt.savefig("roc.pdf", bbox_inches="tight")
print("saved")
