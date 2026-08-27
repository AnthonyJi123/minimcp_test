# -*- coding: utf-8 -*-
"""8aw: why does the final layer invert? (advisor request 8-27)

Small-experiment battery on the Phase-5d all-layer dumps (CPU only),
following the GPT-5.5 consult in askgpt_interp.md. All probes mirror the
layer_sweep_report protocol exactly: raw float32 states, calib split,
LogisticRegression(max_iter=2000), LOPO trains on pools != held-out.

E0  sanity: reproduce the published L22/L35 LOPO hard-math AUCs.
E1  late-update decomposition: probes on h22, h35, D=h35-h22, [h22;D];
    the L35 probe's score applied to the h22 and D components.
E2  duplex-minus-backbone displacement (o4.5 vs raw Qwen3-8B, same
    queries): per-layer linear CKA + displacement norm; PCA of the
    final-layer displacement; alignment of PC1 with the (inverting) L35
    probe direction vs the (transferring) L22 direction.
E4  control-subspace removal: project top-k displacement PCs out of h35,
    re-run the LOPO-math probe; random-subspace + L22 controls.
E5  pool-wise sign mediation: per-pool corr(fail, score) for the L22
    competence score vs the displacement-PC1 control score.
E6  modality axis (if audio dumps present): audio-vs-text direction per
    layer, cosine with the L35/L22 probe directions.
Replication on the second duplex pair (o2.6 vs Qwen2.5-7B) where dumps
allow. Writes data/interp_lastlayer.json for the figure script.
"""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D = "data"
RNG = np.random.default_rng(14)
OUT = {}


def load_layers(tag):
    shards = sorted(glob.glob(f"{D}/layers/layers_{tag}.shard*.npz"))
    if not shards:
        return None
    ids, hl, hm = [], [], []
    for s in shards:
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
        hm.append(z["h_mean"])
    return np.array(ids), np.concatenate(hl), np.concatenate(hm)


def calib_frame(feats_path):
    df = pd.read_parquet(feats_path)[["id", "pool", "split",
                                      "escalate_label"]]
    return df[(df["split"] == "calib") & df["escalate_label"].notna()]


def lopo_math_auc(X, y, pools, return_w=False):
    tr, te = pools != "hard-math", pools == "hard-math"
    lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    auc = roc_auc_score(y[te], lr.predict_proba(X[te])[:, 1])
    return (auc, lr) if return_w else auc


def cka(X, Y):
    """Linear CKA, feature-centered, float64 accumulation."""
    Xc = (X - X.mean(0)).astype(np.float64)
    Yc = (Y - Y.mean(0)).astype(np.float64)
    a = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    b = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    return float(a / b)


def cos(a, b):
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---- load o4.5 + labels ---------------------------------------------------
ids45, hl45, hm45 = load_layers("minicpm-o45")
lab = calib_frame(f"{D}/calib_features.parquet")
m = pd.DataFrame({"id": ids45, "row": range(len(ids45))}).merge(lab, on="id")
y = m["escalate_label"].astype(int).to_numpy()
pools = m["pool"].to_numpy()
ridx = m["row"].to_numpy()
n_layers = hl45.shape[1]
L_MID, L_FIN = 22, n_layers - 1
print(f"o4.5 calib n={len(m)} layers={n_layers} mid=L{L_MID} fin=L{L_FIN}")

H = lambda arr, li: arr[ridx, li, :].astype(np.float32)
h22, h35 = H(hl45, L_MID), H(hl45, L_FIN)

# ---- E0 sanity ------------------------------------------------------------
ref = {c["layer"]: c["last_lopo_hard-math"]
       for c in json.load(open(f"{D}/layers/layer_sweep_minicpm-o45.json"))
       ["curves"]}
a22, lr22 = lopo_math_auc(h22, y, pools, return_w=True)
a35, lr35 = lopo_math_auc(h35, y, pools, return_w=True)
print(f"E0 sanity: L{L_MID} {a22:.3f} (ref {ref[L_MID]:.3f}) | "
      f"L{L_FIN} {a35:.3f} (ref {ref[L_FIN]:.3f})")
# L22 reproduces exactly; L35 sits in the non-converged near-chance zone
# where lbfgs is sklearn-version sensitive — 0.02 tolerance there.
assert abs(a22 - ref[L_MID]) < 5e-3 and abs(a35 - ref[L_FIN]) < 2e-2, \
    "protocol mismatch vs published sweep"
OUT["E0"] = {"L22": a22, "L35": a35, "ref_L22": ref[L_MID],
             "ref_L35": ref[L_FIN]}
w22 = lr22.coef_[0] / np.linalg.norm(lr22.coef_[0])
w35 = lr35.coef_[0] / np.linalg.norm(lr35.coef_[0])

# ---- E1 late-update decomposition ----------------------------------------
delta = h35 - h22
res = {
    "h22": a22, "h35": a35,
    "delta": lopo_math_auc(delta, y, pools),
    "concat": lopo_math_auc(np.hstack([h22, delta]), y, pools),
}
# the trained L35 probe's score, applied to each additive component
te = pools == "hard-math"
res["w35_on_h22"] = roc_auc_score(y[te], h22[te] @ lr35.coef_[0])
res["w35_on_delta"] = roc_auc_score(y[te], delta[te] @ lr35.coef_[0])
res["cos_w22_w35"] = cos(w22, w35)
OUT["E1"] = res
print("E1 late-update:", {k: round(v, 3) for k, v in res.items()})

# ---- E2 displacement vs raw backbone -------------------------------------
idsq, hlq, hmq = load_layers("qwen3-8b")
assert (idsq == ids45).all()
cka_last = [cka(H(hl45, li), H(hlq, li)) for li in range(n_layers)]
cka_mean = [cka(H(hm45, li), H(hmq, li)) for li in range(n_layers)]
# per-layer displacement norm, scale-normalized
disp_rel = []
for li in range(n_layers):
    a, b = H(hl45, li), H(hlq, li)
    d = a - b
    disp_rel.append(float(np.linalg.norm(d, axis=1).mean()
                          / np.sqrt(np.linalg.norm(a, axis=1).mean()
                                    * np.linalg.norm(b, axis=1).mean())))
d35 = h35 - H(hlq, L_FIN)
d35c = d35 - d35.mean(0)
U, S, Vt = np.linalg.svd(d35c, full_matrices=False)
pc = Vt[:10]
evr = (S[:10] ** 2 / (S ** 2).sum()).tolist()
OUT["E2"] = {
    "cka_last": cka_last, "cka_mean": cka_mean, "disp_rel": disp_rel,
    "pc1_evr": evr,
    "cos_pc1_w35": cos(pc[0], w35), "cos_pc1_w22": cos(pc[0], w22),
    "cos_pc2_w35": cos(pc[1], w35), "cos_pc2_w22": cos(pc[1], w22),
}
print(f"E2 CKA last: L{L_MID} {cka_last[L_MID]:.3f} -> L{L_FIN} "
      f"{cka_last[L_FIN]:.3f} | mean: {cka_mean[L_MID]:.3f} -> "
      f"{cka_mean[L_FIN]:.3f}")
print(f"   disp PC1 EVR {evr[0]:.2f}; cos(PC1,w35)={OUT['E2']['cos_pc1_w35']:.3f} "
      f"cos(PC1,w22)={OUT['E2']['cos_pc1_w22']:.3f}")

# ---- E3 pool identity by layer (the shortcut, localized) -----------------
from sklearn.model_selection import StratifiedKFold, cross_val_predict

cv5 = StratifiedKFold(5, shuffle=True, random_state=42)


def pool_acc(X):
    pred = cross_val_predict(LogisticRegression(max_iter=2000), X, pools,
                             cv=cv5)
    return float((pred == pools).mean())


def fail_oof(X):
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cv5,
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


e3 = {"layers": [], "pool_duplex": [], "pool_raw": [], "pool_duplex_mean": [],
      "fail_oof_duplex": []}
for li in range(0, n_layers, 2):
    e3["layers"].append(li)
    e3["pool_duplex"].append(pool_acc(H(hl45, li)))
    e3["pool_raw"].append(pool_acc(H(hlq, li)))
    e3["pool_duplex_mean"].append(pool_acc(H(hm45, li)))
    e3["fail_oof_duplex"].append(fail_oof(H(hl45, li)))
for li in (L_FIN,):
    if li not in e3["layers"]:
        e3["layers"].append(li)
        e3["pool_duplex"].append(pool_acc(H(hl45, li)))
        e3["pool_raw"].append(pool_acc(H(hlq, li)))
        e3["pool_duplex_mean"].append(pool_acc(H(hm45, li)))
        e3["fail_oof_duplex"].append(fail_oof(H(hl45, li)))
OUT["E3"] = e3
print("E3 pool-identity acc (duplex last):",
      [round(v, 3) for v in e3["pool_duplex"]])
print("   raw backbone       :", [round(v, 3) for v in e3["pool_raw"]])

# ---- E4 control-subspace removal -----------------------------------------
def project_out(X, dirs):
    Q, _ = np.linalg.qr(np.asarray(dirs, np.float64).T)
    Xd = X.astype(np.float64)
    return (Xd - (Xd @ Q) @ Q.T).astype(np.float32)

KS = list(range(1, 21))
U2, S2, Vt2_ = np.linalg.svd(d35c, full_matrices=False)
pc_full = Vt2_[:max(KS)]
rescue, rand_ctrl, l22_ctrl, wshare, wshare_rnd = [], [], [], [], []
for k in KS:
    rescue.append(lopo_math_auc(project_out(h35, pc_full[:k]), y, pools))
    l22_ctrl.append(lopo_math_auc(project_out(h22, pc_full[:k]), y, pools))
    rs, ws = [], []
    for seed in range(5):
        rnd = np.random.default_rng(seed).standard_normal((k, h35.shape[1]))
        rs.append(lopo_math_auc(project_out(h35, rnd), y, pools))
        Qr, _ = np.linalg.qr(rnd.T)
        ws.append(float(np.linalg.norm(Qr.T @ w35)))
    rand_ctrl.append(float(np.mean(rs)))
    Q, _ = np.linalg.qr(pc_full[:k].astype(np.float64).T)
    wshare.append(float(np.linalg.norm(Q.T @ w35)))
    wshare_rnd.append(float(np.mean(ws)))
OUT["E4"] = {"k": KS, "rescue": rescue, "random": rand_ctrl,
             "l22_control": l22_ctrl, "w35_in_subspace": wshare,
             "w35_in_random": wshare_rnd}
print("E4 rescue  :", [round(v, 3) for v in rescue])
print("   random  :", [round(v, 3) for v in rand_ctrl])
print("   L22 ctrl:", [round(v, 3) for v in l22_ctrl])

# ---- E5 pool-wise sign mediation -----------------------------------------
z = h22 @ lr22.coef_[0]            # competence score (L22 probe)
c = h35 @ pc[0]                    # control score (displacement PC1)
tab = {}
for pl in sorted(set(pools)):
    msk = pools == pl
    if len(set(y[msk])) < 2:
        tab[pl] = {"n": int(msk.sum()), "fail_rate": float(y[msk].mean()),
                   "corr_z": None, "corr_c": None}
        continue
    tab[pl] = {"n": int(msk.sum()),
               "corr_z": float(np.corrcoef(z[msk], y[msk])[0, 1]),
               "corr_c": float(np.corrcoef(c[msk], y[msk])[0, 1])}
OUT["E5"] = tab
print("E5 per-pool corr (z=L22 score, c=disp PC1):")
for pl, v in tab.items():
    print(f"   {pl:15s} corr_z={v.get('corr_z')} corr_c={v.get('corr_c')}")

# ---- E6 modality axis (audio dumps) --------------------------------------
aud = load_layers("minicpm-o45-audio")
if aud is not None:
    idsa, hla, _ = aud
    common = sorted(set(idsa) & set(m["id"]))
    if common:
        pos_a = {q: i for i, q in enumerate(idsa)}
        pos_t = {q: i for i, q in enumerate(ids45)}
        ra = [pos_a[q] for q in common]
        rt = [pos_t[q] for q in common]
        tnorm, tcos35, tcos22 = [], [], []
        for li in range(n_layers):
            t = (hla[ra, li, :].astype(np.float32)
                 - hl45[rt, li, :].astype(np.float32)).mean(0)
            tnorm.append(float(np.linalg.norm(t)))
            if li == L_FIN:
                tcos35 = [cos(t, w35), cos(t, w22)]
            if li == L_MID:
                tcos22 = [cos(t, w35), cos(t, w22)]
        OUT["E6"] = {"n_common": len(common), "t_norm": tnorm,
                     "L35_cos_w35_w22": tcos35, "L22_cos_w35_w22": tcos22}
        print(f"E6 modality axis on {len(common)} common ids: "
              f"L35 cos(t,w35)={tcos35[0]:.3f} cos(t,w22)={tcos35[1]:.3f}")

# ---- replication: o2.6 vs Qwen2.5-7B -------------------------------------
r26 = load_layers("minicpm-o26")
rq25 = load_layers("qwen2.5-7b")
if r26 is not None and rq25 is not None:
    ids26, hl26, _ = r26
    idsq25, hlq25, _ = rq25
    lab26 = calib_frame(f"{D}/layers/features_minicpm-o26.parquet")
    m2 = pd.DataFrame({"id": ids26, "row": range(len(ids26))}) \
        .merge(lab26, on="id")
    y2 = m2["escalate_label"].astype(int).to_numpy()
    p2 = m2["pool"].to_numpy()
    r2 = m2["row"].to_numpy()
    nl2 = hl26.shape[1]
    common = sorted(set(ids26) & set(idsq25))
    pa = {q: i for i, q in enumerate(ids26)}
    pb = {q: i for i, q in enumerate(idsq25)}
    ra = [pa[q] for q in common]
    rb = [pb[q] for q in common]
    cka2 = [cka(hl26[ra, li, :].astype(np.float32),
                hlq25[rb, li, :].astype(np.float32)) for li in range(nl2)]
    mid2 = int(np.argmax([ref2["last_lopo_hard-math"]
                          for ref2 in json.load(
                              open(f"{D}/layers/layer_sweep_minicpm-o26.json")
                          )["curves"]]))
    fin2 = nl2 - 1
    h_mid = hl26[r2, mid2, :].astype(np.float32)
    h_fin = hl26[r2, fin2, :].astype(np.float32)
    am, lrm = lopo_math_auc(h_mid, y2, p2, return_w=True)
    af, lrf = lopo_math_auc(h_fin, y2, p2, return_w=True)
    d2 = hl26[ra, fin2, :].astype(np.float32) \
        - hlq25[rb, fin2, :].astype(np.float32)
    d2c = d2 - d2.mean(0)
    _, _, Vt2 = np.linalg.svd(d2c, full_matrices=False)
    w_f = lrf.coef_[0] / np.linalg.norm(lrf.coef_[0])
    w_m = lrm.coef_[0] / np.linalg.norm(lrm.coef_[0])
    OUT["repl_o26"] = {
        "n_layers": nl2, "mid": mid2, "auc_mid": am, "auc_fin": af,
        "cka_last": cka2,
        "cos_pc1_wfin": cos(Vt2[0], w_f), "cos_pc1_wmid": cos(Vt2[0], w_m),
        "k": KS,
        "rescue": [lopo_math_auc(project_out(h_fin, Vt2[:k]), y2, p2)
                   for k in KS],
        "random": [float(np.mean([
            lopo_math_auc(project_out(
                h_fin, np.random.default_rng(s).standard_normal(
                    (k, h_fin.shape[1]))), y2, p2) for s in range(3)]))
            for k in KS],
    }
    rr = OUT["repl_o26"]
    print(f"repl o2.6: mid L{mid2} {am:.3f} fin L{fin2} {af:.3f} | "
          f"CKA {cka2[mid2]:.3f}->{cka2[fin2]:.3f} | "
          f"cos(PC1,w_fin)={rr['cos_pc1_wfin']:.3f}")
    print("   rescue:", [round(v, 3) for v in rr["rescue"]])
    print("   random:", [round(v, 3) for v in rr["random"]])

with open(f"{D}/interp_lastlayer.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(">>> wrote data/interp_lastlayer.json")
