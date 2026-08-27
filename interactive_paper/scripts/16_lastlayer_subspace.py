# -*- coding: utf-8 -*-
"""8aw final: the cliff is a dominant-variance subspace, by layer.

The controls in 15_* killed two candidate stories (representational drift
from the backbone does not localize the cliff once features are
standardized; the massive-activation axes and the pool-mean directions
rescue nothing) and left one that does not need the backbone at all:
removing the top few PRINCIPAL COMPONENTS of the duplex model's own
final-layer state restores leave-one-pool-out transfer.

This script makes that the measurement:
  S1  by layer: LOPO hard-math AUC of the last-token read, raw vs with
      the top-5 PCs of that layer projected out, for the duplex model and
      its raw backbone. Same for in-mix OOF (does the removal cost
      in-distribution accuracy?).
  S2  at the final layer: split the state into the top-5 PC projection
      and its residual, probe each --- which half carries the inversion,
      which carries the competence.
  S3  k-scan at the final layer with matched random / massive-activation
      / pool-mean controls (from 15_*), plus the same scan on the raw
      backbone as a null.
  S4  replication on the second duplex pair.
Writes data/interp_subspace.json.
"""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

D = "data"
K = 5
OUT = {}
CV = StratifiedKFold(5, shuffle=True, random_state=42)


def load_layers(tag):
    shards = sorted(glob.glob(f"{D}/layers/layers_{tag}.shard*.npz"))
    if not shards:
        return None
    ids, hl = [], []
    for s in shards:
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    return np.array(ids), np.concatenate(hl)


def lopo_math(X, y, pools):
    tr, te = pools != "hard-math", pools == "hard-math"
    lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], lr.predict_proba(X[te])[:, 1]))


def oof(X, y):
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=CV,
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def top_pcs(M, k):
    Mc = M.astype(np.float64) - M.astype(np.float64).mean(0)
    return np.linalg.svd(Mc, full_matrices=False)[2][:k]


def split_pc(X, k):
    """Return (projection onto top-k PCs, residual)."""
    B = top_pcs(X, k)
    Q, _ = np.linalg.qr(B.T)
    Xd = X.astype(np.float64)
    proj = (Xd @ Q) @ Q.T
    return proj.astype(np.float32), (Xd - proj).astype(np.float32)


def frame(tag, feats):
    ids, hl = load_layers(tag)
    lab = pd.read_parquet(feats)[["id", "pool", "split", "escalate_label"]]
    lab = lab[(lab["split"] == "calib") & lab["escalate_label"].notna()]
    m = pd.DataFrame({"id": ids, "row": range(len(ids))}).merge(lab, on="id")
    return (hl, m["escalate_label"].astype(int).to_numpy(),
            m["pool"].to_numpy(), m["row"].to_numpy())


hl45, y, pools, ridx = frame("minicpm-o45", f"{D}/calib_features.parquet")
nL = hl45.shape[1]
H = lambda a, li: a[ridx, li, :].astype(np.float32)
print(f"o4.5 n={len(y)} layers={nL}, removing top-{K} PCs per layer")

# ---- S1 by layer ----------------------------------------------------------
idsq, hlq = load_layers("qwen3-8b")
s1 = {"layers": [], "duplex_raw": [], "duplex_cut": [], "duplex_oof_raw": [],
      "duplex_oof_cut": [], "raw_raw": [], "raw_cut": []}
for li in range(nL):
    X = H(hl45, li)
    Xc = split_pc(X, K)[1]
    Q = hlq[ridx, li, :].astype(np.float32)
    s1["layers"].append(li)
    s1["duplex_raw"].append(lopo_math(X, y, pools))
    s1["duplex_cut"].append(lopo_math(Xc, y, pools))
    s1["duplex_oof_raw"].append(oof(X, y))
    s1["duplex_oof_cut"].append(oof(Xc, y))
    s1["raw_raw"].append(lopo_math(Q, y, pools))
    s1["raw_cut"].append(lopo_math(split_pc(Q, K)[1], y, pools))
OUT["S1"] = s1
print("S1 duplex raw :", [round(v, 2) for v in s1["duplex_raw"][-8:]])
print("   duplex cut :", [round(v, 2) for v in s1["duplex_cut"][-8:]])
print("   backbone   :", [round(v, 2) for v in s1["raw_raw"][-8:]])
print("   backbone c :", [round(v, 2) for v in s1["raw_cut"][-8:]])

# ---- S2 which half carries what ------------------------------------------
L_FIN, L_MID = nL - 1, 22
h35, h22 = H(hl45, L_FIN), H(hl45, L_MID)
proj35, res35 = split_pc(h35, K)
proj22, res22 = split_pc(h22, K)
evr = None
Mc = h35.astype(np.float64) - h35.astype(np.float64).mean(0)
sv = np.linalg.svd(Mc, compute_uv=False)
evr = float((sv[:K] ** 2).sum() / (sv ** 2).sum())
s2 = {
    "evr_top5": evr,
    "fin_full_lopo": lopo_math(h35, y, pools),
    "fin_proj_lopo": lopo_math(proj35, y, pools),
    "fin_resid_lopo": lopo_math(res35, y, pools),
    "fin_full_oof": oof(h35, y), "fin_proj_oof": oof(proj35, y),
    "fin_resid_oof": oof(res35, y),
    "mid_full_lopo": lopo_math(h22, y, pools),
    "mid_proj_lopo": lopo_math(proj22, y, pools),
    "mid_resid_lopo": lopo_math(res22, y, pools),
    "mid_full_oof": oof(h22, y), "mid_proj_oof": oof(proj22, y),
    "mid_resid_oof": oof(res22, y),
}
OUT["S2"] = s2
print(f"S2 top-{K} PCs hold {evr:.1%} of final-layer variance")
print(f"   final  full {s2['fin_full_lopo']:.3f} | top-PC part "
      f"{s2['fin_proj_lopo']:.3f} | residual {s2['fin_resid_lopo']:.3f}"
      f"   (OOF {s2['fin_full_oof']:.3f}/{s2['fin_proj_oof']:.3f}/"
      f"{s2['fin_resid_oof']:.3f})")
print(f"   L22    full {s2['mid_full_lopo']:.3f} | top-PC part "
      f"{s2['mid_proj_lopo']:.3f} | residual {s2['mid_resid_lopo']:.3f}"
      f"   (OOF {s2['mid_full_oof']:.3f}/{s2['mid_proj_oof']:.3f}/"
      f"{s2['mid_resid_oof']:.3f})")

# ---- S3 k-scan + controls -------------------------------------------------
KS = list(range(0, 13))


def cut_k(X, k):
    return X if k == 0 else split_pc(X, k)[1]


big = np.argsort(-np.abs(h35.mean(0)))[:12]
pm = np.stack([h35[pools == p].mean(0) for p in sorted(set(pools))])


def proj_out(X, B):
    Q, _ = np.linalg.qr(np.asarray(B, np.float64).T)
    Xd = X.astype(np.float64)
    return (Xd - (Xd @ Q) @ Q.T).astype(np.float32)


s3 = {"k": KS, "duplex_pc": [], "backbone_pc": [], "random": [],
      "massive_axes": [], "pool_dirs": []}
q35 = hlq[ridx, L_FIN, :].astype(np.float32)
for k in KS:
    s3["duplex_pc"].append(lopo_math(cut_k(h35, k), y, pools))
    s3["backbone_pc"].append(lopo_math(cut_k(q35, k), y, pools))
    if k == 0:
        s3["random"].append(s3["duplex_pc"][0])
        s3["massive_axes"].append(s3["duplex_pc"][0])
        s3["pool_dirs"].append(s3["duplex_pc"][0])
        continue
    s3["random"].append(float(np.mean([
        lopo_math(proj_out(h35, np.random.default_rng(s).standard_normal(
            (k, h35.shape[1]))), y, pools) for s in range(5)])))
    ax = np.zeros((k, h35.shape[1]))
    ax[np.arange(k), big[:k]] = 1.0
    s3["massive_axes"].append(lopo_math(proj_out(h35, ax), y, pools))
    s3["pool_dirs"].append(lopo_math(
        proj_out(h35, np.repeat(pm, k // len(pm) + 1, axis=0)[:k]), y, pools))
OUT["S3"] = s3
print("S3 duplex   :", [round(v, 2) for v in s3["duplex_pc"]])
print("   backbone :", [round(v, 2) for v in s3["backbone_pc"]])
print("   random   :", [round(v, 2) for v in s3["random"]])

# ---- S4 replication -------------------------------------------------------
try:
    hl26, y2, p2, r2 = frame("minicpm-o26",
                             f"{D}/layers/features_minicpm-o26.parquet")
    n2 = hl26.shape[1]
    H2 = lambda li: hl26[r2, li, :].astype(np.float32)
    sweep26 = json.load(open(f"{D}/layers/layer_sweep_minicpm-o26.json"))
    mid2 = int(np.argmax([c["last_lopo_hard-math"]
                          for c in sweep26["curves"]]))
    s4 = {"n_layers": n2, "mid": mid2, "layers": [], "raw": [], "cut": []}
    for li in range(n2):
        X = H2(li)
        s4["layers"].append(li)
        s4["raw"].append(lopo_math(X, y2, p2))
        s4["cut"].append(lopo_math(split_pc(X, K)[1], y2, p2))
    OUT["S4"] = s4
    print("S4 o2.6 raw:", [round(v, 2) for v in s4["raw"][-6:]])
    print("        cut:", [round(v, 2) for v in s4["cut"][-6:]])
except Exception as e:      # replication is optional, main result stands
    print("S4 skipped:", e)

with open(f"{D}/interp_subspace.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(">>> wrote data/interp_subspace.json")
