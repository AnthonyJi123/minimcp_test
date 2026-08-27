# -*- coding: utf-8 -*-
"""8aw: what does the probe LEAN ON at each depth? (non-destructive)

Projection experiments are destructive --- removing top PCs damages the raw
backbone too, so "the rescue" was never evidence of a duplex-added
subspace. This measures the probe itself instead, with no surgery.

For the probe trained at each layer on the four non-math pools, split its
score into the part carried by the layer's dominant-variance subspace and
the part carried by the residual:

    s(x) = w . x = w . P_k x  +  w . (I - P_k) x
             \_______/          \____________/
              s_dom                 s_res

and report (a) each part's own LOPO hard-math AUC and (b) the share of
score variance in s_dom. The dominant subspace is defined per layer from
the states alone, without labels or the backbone.

Claim under test: at the deployed mid layer the probe rides the residual,
which transfers; at the final layer it is dominated by the top-PC part,
which inverts. Run on the duplex model AND its raw backbone --- if the
backbone shows the same late shift, the effect is not duplex-specific.
Writes data/interp_reliance.json.
"""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D = "data"
K = 5
OUT = {}


def load_layers(tag):
    ids, hl = [], []
    for s in sorted(glob.glob(f"{D}/layers/layers_{tag}.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    return np.array(ids), np.concatenate(hl)


def frame(tag, feats):
    ids, hl = load_layers(tag)
    lab = pd.read_parquet(feats)[["id", "pool", "split", "escalate_label"]]
    lab = lab[(lab["split"] == "calib") & lab["escalate_label"].notna()]
    m = pd.DataFrame({"id": ids, "row": range(len(ids))}).merge(lab, on="id")
    return (hl, m["escalate_label"].astype(int).to_numpy(),
            m["pool"].to_numpy(), m["row"].to_numpy())


def reliance(X, y, pools, k=K):
    """Train on non-math pools; decompose the held-out math scores."""
    tr, te = pools != "hard-math", pools == "hard-math"
    Xtr = X[tr].astype(np.float64)
    mu = Xtr.mean(0)
    # dominant subspace from TRAINING rows only (no held-out leakage)
    B = np.linalg.svd(Xtr - mu, full_matrices=False)[2][:k]
    Q, _ = np.linalg.qr(B.T)
    lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    w = lr.coef_[0]
    Xte = X[te].astype(np.float64) - mu
    s_dom = (Xte @ Q) @ (Q.T @ w)
    s_res = (Xte - (Xte @ Q) @ Q.T) @ w
    tot = s_dom + s_res
    return {
        "auc_full": float(roc_auc_score(y[te], tot)),
        "auc_dom": float(roc_auc_score(y[te], s_dom)),
        "auc_res": float(roc_auc_score(y[te], s_res)),
        "var_share_dom": float(s_dom.var() / (tot.var() + 1e-12)),
        "corr_dom_tot": float(np.corrcoef(s_dom, tot)[0, 1]),
        # how much of the probe's own weight vector lies in the subspace
        "w_share": float(np.linalg.norm(Q.T @ w) / np.linalg.norm(w)),
    }


for tag, feats, label in [
    ("minicpm-o45", f"{D}/calib_features.parquet", "duplex o4.5"),
    ("qwen3-8b", f"{D}/calib_features.parquet", "raw Qwen3-8B"),
    ("minicpm-o26", f"{D}/layers/features_minicpm-o26.parquet", "duplex o2.6"),
    ("qwen2.5-7b", f"{D}/layers/features_minicpm-o26.parquet", "raw Qwen2.5-7B"),
    ("qwen2.5-omni-7b", f"{D}/layers/features_qwen2.5-omni-7b.parquet",
     "omni-streaming Qwen2.5-Omni"),
]:
    try:
        hl, y, pools, ridx = frame(tag, feats)
    except Exception as e:
        print(f"{label}: skipped ({e})")
        continue
    nL = hl.shape[1]
    rec = {"n_layers": nL, "layers": [], "auc_full": [], "auc_dom": [],
           "auc_res": [], "var_share_dom": [], "w_share": []}
    for li in range(nL):
        r = reliance(hl[ridx, li, :].astype(np.float32), y, pools)
        rec["layers"].append(li)
        for key in ("auc_full", "auc_dom", "auc_res", "var_share_dom",
                    "w_share"):
            rec[key].append(r[key])
    OUT[tag] = rec
    print(f"\n== {label} (n={len(y)}, {nL} layers) ==")
    print("  layer :", [f"{li:5d}" for li in rec["layers"][::4]])
    print("  full  :", [f"{v:5.2f}" for v in rec["auc_full"][::4]],
          f"| L{nL-1} {rec['auc_full'][-1]:.2f}")
    print("  dom   :", [f"{v:5.2f}" for v in rec["auc_dom"][::4]],
          f"| L{nL-1} {rec['auc_dom'][-1]:.2f}")
    print("  resid :", [f"{v:5.2f}" for v in rec["auc_res"][::4]],
          f"| L{nL-1} {rec['auc_res'][-1]:.2f}")
    print("  var%  :", [f"{v:5.2f}" for v in rec["var_share_dom"][::4]],
          f"| L{nL-1} {rec['var_share_dom'][-1]:.2f}")

with open(f"{D}/interp_reliance.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\n>>> wrote data/interp_reliance.json")
