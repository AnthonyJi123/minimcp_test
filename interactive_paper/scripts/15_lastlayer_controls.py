# -*- coding: utf-8 -*-
"""8aw controls: is the subspace rescue real, or an outlier-dimension artifact?

Transformer residual streams carry a few "massive activation" dimensions
(|h| ~ 250 here vs ~1 typical). Raw linear CKA is dominated by them --- the
apparent last-token CKA collapse at L35 (0.47) becomes 0.785 once features
are standardized, i.e. representational drift does NOT localize the cliff.
That artifact forces the same question of the E4 rescue: does projecting out
the top-k *duplex-displacement* directions beat projecting out the top-k
directions any other criterion would pick?

Controls, all with the layer_sweep_report probe protocol:
  C1  standardized CKA by layer (last vs mean) --- the corrected E2.
  C2  rescue vs alternative subspaces at matched k: displacement PCs,
      top-variance PCs of h35 itself, top-|mean| coordinate axes
      (the massive-activation dims), random, and pool-mean directions.
  C3  the rescue under per-feature standardization of h35 (kills the
      scale story outright: if standardizing alone restores transfer,
      the cliff was never a subspace phenomenon).
  C4  E5's control coordinate re-derived from the standardized
      displacement, and its per-pool correlations.
Writes data/interp_controls.json.
"""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D = "data"
OUT = {}


def load_layers(tag):
    shards = sorted(glob.glob(f"{D}/layers/layers_{tag}.shard*.npz"))
    ids, hl, hm = [], [], []
    for s in shards:
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
        hm.append(z["h_mean"])
    return np.array(ids), np.concatenate(hl), np.concatenate(hm)


def cka(X, Y, standardize):
    X = X.astype(np.float64) - X.astype(np.float64).mean(0)
    Y = Y.astype(np.float64) - Y.astype(np.float64).mean(0)
    if standardize:
        X = X / (X.std(0) + 1e-6)
        Y = Y / (Y.std(0) + 1e-6)
    a = np.linalg.norm(Y.T @ X, "fro") ** 2
    b = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(a / b)


def lopo_math(X, y, pools):
    tr, te = pools != "hard-math", pools == "hard-math"
    lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], lr.predict_proba(X[te])[:, 1]))


def project_out(X, dirs):
    Q, _ = np.linalg.qr(np.asarray(dirs, np.float64).T)
    Xd = X.astype(np.float64)
    return (Xd - (Xd @ Q) @ Q.T).astype(np.float32)


def top_pcs(M, k):
    Mc = M.astype(np.float64) - M.astype(np.float64).mean(0)
    return np.linalg.svd(Mc, full_matrices=False)[2][:k]


ids45, hl45, hm45 = load_layers("minicpm-o45")
idsq, hlq, hmq = load_layers("qwen3-8b")
assert (ids45 == idsq).all()
lab = pd.read_parquet(f"{D}/calib_features.parquet")[
    ["id", "pool", "split", "escalate_label"]]
lab = lab[(lab["split"] == "calib") & lab["escalate_label"].notna()]
m = pd.DataFrame({"id": ids45, "row": range(len(ids45))}).merge(lab, on="id")
y = m["escalate_label"].astype(int).to_numpy()
pools = m["pool"].to_numpy()
ridx = m["row"].to_numpy()
nL = hl45.shape[1]
L_MID, L_FIN = 22, nL - 1
H = lambda a, li: a[ridx, li, :].astype(np.float32)
h22, h35 = H(hl45, L_MID), H(hl45, L_FIN)
q35 = H(hlq, L_FIN)

# ---- C1 standardized CKA --------------------------------------------------
c1 = {"layers": list(range(nL)), "last_raw": [], "last_std": [],
      "mean_raw": [], "mean_std": []}
for li in range(nL):
    c1["last_raw"].append(cka(H(hl45, li), H(hlq, li), False))
    c1["last_std"].append(cka(H(hl45, li), H(hlq, li), True))
    c1["mean_raw"].append(cka(H(hm45, li), H(hmq, li), False))
    c1["mean_std"].append(cka(H(hm45, li), H(hmq, li), True))
OUT["C1"] = c1
print(f"C1 CKA last  raw L22 {c1['last_raw'][L_MID]:.3f} -> L35 "
      f"{c1['last_raw'][L_FIN]:.3f} | std L22 {c1['last_std'][L_MID]:.3f} "
      f"-> L35 {c1['last_std'][L_FIN]:.3f}")
print(f"   CKA mean  std L22 {c1['mean_std'][L_MID]:.3f} -> L35 "
      f"{c1['mean_std'][L_FIN]:.3f}")

# ---- C2 rescue vs alternative subspaces at matched k ----------------------
disp = h35 - q35
KS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
KMAX = max(KS)
bases = {
    "displacement_pc": top_pcs(disp, KMAX),
    "h35_variance_pc": top_pcs(h35, KMAX),
    "duplex_only_pc": top_pcs(h35 - h35.mean(0), KMAX),
}
# massive-activation coordinate axes: the largest mean-|activation| dims
big = np.argsort(-np.abs(h35.mean(0)))[:KMAX]
axes_basis = np.zeros((KMAX, h35.shape[1]))
axes_basis[np.arange(KMAX), big] = 1.0
bases["massive_activation_axes"] = axes_basis
# pool-mean directions (the query-type shortcut, at most 4 useful)
pm = np.stack([h35[pools == p].mean(0) for p in sorted(set(pools))])
bases["pool_mean_dirs"] = np.repeat(pm, KMAX // len(pm) + 1, axis=0)[:KMAX]

c2 = {"k": KS, "curves": {}}
for name, B in bases.items():
    c2["curves"][name] = [lopo_math(project_out(h35, B[:k]), y, pools)
                          for k in KS]
c2["curves"]["random"] = [
    float(np.mean([lopo_math(project_out(
        h35, np.random.default_rng(s).standard_normal((k, h35.shape[1]))),
        y, pools) for s in range(5)])) for k in KS]
OUT["C2"] = c2
print("C2 rescue by subspace (k =", KS, "):")
for name, v in c2["curves"].items():
    print(f"   {name:24s}", [round(x, 3) for x in v])

# ---- C3 does standardization alone rescue? -------------------------------
mu, sd = h35.mean(0), h35.std(0) + 1e-6
h35z = ((h35 - mu) / sd).astype(np.float32)
h22z = ((h22 - h22.mean(0)) / (h22.std(0) + 1e-6)).astype(np.float32)
c3 = {
    "h35_raw": lopo_math(h35, y, pools),
    "h35_standardized": lopo_math(h35z, y, pools),
    "h22_raw": lopo_math(h22, y, pools),
    "h22_standardized": lopo_math(h22z, y, pools),
    "h35_drop_massive_dims": lopo_math(
        np.delete(h35, big[:10], axis=1), y, pools),
    "h35z_disp_pc_k6": lopo_math(
        project_out(h35z, top_pcs(
            (h35 - q35) / sd, 6)), y, pools),
}
OUT["C3"] = c3
print("C3:", {k: round(v, 3) for k, v in c3.items()})

# ---- C4 control coordinate, standardized ---------------------------------
disp_z = ((h35 - q35) / sd).astype(np.float32)
pc1z = top_pcs(disp_z, 1)[0]
lr22 = LogisticRegression(max_iter=2000).fit(
    h22[pools != "hard-math"], y[pools != "hard-math"])
z = h22 @ lr22.coef_[0]
c_raw = h35 @ top_pcs(disp, 1)[0]
c_std = h35z @ pc1z
c4 = {}
for p in sorted(set(pools)):
    msk = pools == p
    if len(set(y[msk])) < 2:
        continue
    c4[p] = {
        "corr_z": float(np.corrcoef(z[msk], y[msk])[0, 1]),
        "corr_c_raw": float(np.corrcoef(c_raw[msk], y[msk])[0, 1]),
        "corr_c_std": float(np.corrcoef(c_std[msk], y[msk])[0, 1]),
    }
OUT["C4"] = c4
print("C4 per-pool corr:")
for p, v in c4.items():
    print(f"   {p:15s} z={v['corr_z']:+.3f} c_raw={v['corr_c_raw']:+.3f} "
          f"c_std={v['corr_c_std']:+.3f}")

with open(f"{D}/interp_controls.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(">>> wrote data/interp_controls.json")
