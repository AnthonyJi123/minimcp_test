"""Official-config native refit (8bl final): the 8be+8bj recipe on
features re-collected under the OFFICIAL serving config (top_k 20,
force_listen 3, assistant prompt — the config the demo now runs).

Train: caliboff/expoff/exp2off (2310, same escalate_label parquets) +
freshoff train rows (a-priori labels). Thresholds: core-mix OOF
quantiles (fresh excluded, v4 rule). Guards: testoff AUC vs the
interim old-config probe; freshoff-heldout fire rates.

Usage: .venv_boot\\Scripts\\python.exe scripts\\26_official_refit.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

D = Path("data")


def load_feats(tag):
    ids, X = [], []
    for p in sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += list(z["ids"])
        X.append(z["X"])
    if not X:
        raise FileNotFoundError(tag)
    X = np.concatenate(X)
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    return list(df["id"]), X[df["row"].to_numpy()]


def main():
    parts_X, parts_y = [], []
    for tag, lab_file in [("caliboff", "calib_features.parquet"),
                          ("expoff", "expansion_labels.parquet"),
                          ("exp2off", "expansion2_labels.parquet")]:
        ids, X = load_feats(tag)
        lab = pd.read_parquet(D / lab_file).set_index("id")[
            "escalate_label"]
        y = lab.reindex(ids).to_numpy().astype(float)
        keep = ~np.isnan(y)
        parts_X.append(X[keep])
        parts_y.append(y[keep].astype(int))
        print(f"{tag}: {keep.sum()} rows")
    X0 = np.concatenate(parts_X)
    y0 = np.concatenate(parts_y)
    n0 = len(y0)

    fl = pd.read_parquet(D / "fresh_labels.parquet")
    fl = fl[fl["escalate_label"].notna()]
    lab_f = dict(zip(fl["id"], fl["escalate_label"].astype(int)))
    split_f = dict(zip(fl["id"], fl["split"]))
    ids_fr, X_fr = load_feats("freshoff")
    tr_j = [j for j, i in enumerate(ids_fr)
            if i in lab_f and split_f.get(i) == "train"]
    he_j = [j for j, i in enumerate(ids_fr)
            if i in lab_f and split_f.get(i) != "train"]
    X_tr = np.concatenate([X0, X_fr[tr_j]])
    y_tr = np.concatenate([y0, [lab_f[ids_fr[j]] for j in tr_j]])
    print(f"train {len(y_tr)} (core {n0} + fresh {len(tr_j)})")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    best = None
    for C in (1e-4, 3e-4, 1e-3):
        oof = cross_val_predict(
            LogisticRegression(C=C, max_iter=5000), X_tr, y_tr, cv=cv,
            method="predict_proba")[:, 1]
        a = roc_auc_score(y_tr, oof)
        print(f"  C={C}: OOF={a:.3f}")
        if best is None or a > best[1]:
            best = (C, a, oof)
    C, auc, oof = best
    clf = LogisticRegression(C=C, max_iter=5000).fit(X_tr, y_tr)
    thr = {t: float(np.quantile(oof[:n0], 1 - b))
           for t, b in (("conservative", .15), ("balanced", .30),
                        ("aggressive", .50))}
    print("official thresholds: "
          + "  ".join(f"{t}={v:.4f}" for t, v in thr.items()))

    old = json.loads((D / "gate_native.json").read_text())
    tst_ids, Xt = load_feats("testoff")
    tr = pd.read_parquet(D / "frozen_v3_traces.parquet")
    loc = tr[tr["mode"] == "local"].groupby("id")["heard_ok"].max()
    y_t = (1 - loc.reindex(tst_ids)).to_numpy().astype(float)
    k = ~np.isnan(y_t)
    a_old = roc_auc_score(
        y_t[k].astype(int),
        Xt[k] @ np.array(old["w"]) + old["b"])
    s_new = clf.predict_proba(Xt[k])[:, 1]
    a_new = roc_auc_score(y_t[k].astype(int), s_new)
    print(f"guard1 testoff AUC: old-cfg probe {a_old:.3f} -> "
          f"official refit {a_new:.3f}")
    for t, v in thr.items():
        print(f"  test fire@{t}: {float((s_new >= v).mean()):.2f}")

    s_he = clf.predict_proba(X_fr[he_j])[:, 1]
    y_he = np.array([lab_f[ids_fr[j]] for j in he_j])
    for t, v in thr.items():
        print(f"guard2 {t:<13} fresh-heldout fire: "
              f"fast {(s_he[y_he == 1] >= v).mean():.2f} "
              f"never {(s_he[y_he == 0] >= v).mean():.2f}")

    art = dict(old)
    art.update(w=clf.coef_[0].tolist(), b=float(clf.intercept_[0]),
               C=C, train_n=int(len(y_tr)), eot_thresholds=thr,
               recipe="scripts/26 official-config native refit "
                      "(8bl final: 2310 core + fresh, official "
                      "serving params)")
    (D / "gate_native.json").write_text(json.dumps(art))
    print("wrote data/gate_native.json (official)")


if __name__ == "__main__":
    main()
