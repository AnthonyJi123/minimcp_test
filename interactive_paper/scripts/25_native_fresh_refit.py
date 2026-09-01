"""Native v2 refit (8bj): restore the v4 FreshQA real-time awareness
that the 8be native refit silently dropped (it reused the v3-era 2310
labels — "what is NVDA trading at" scored P(fail)=.29 live and the
talker delivered an empty promise).

Recipe = modal_fresh.refit4 transplanted: fresh TRAIN rows (a-priori
labels: fast-changing => escalate) appended to the 2310 native rows;
tier budgets stay quantiled on the 2310 deployment-mix OOF ONLY (the
fresh rows train the direction but must not inflate the budgets).

Guards printed: internal test-240 AUC vs the 8be probe (must hold),
fresh HELDOUT fire rates per tier (fast-changing should fire high,
never-changing low).

Usage: .venv_boot\\Scripts\\python.exe scripts\\25_native_fresh_refit.py
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
        raise FileNotFoundError(f"no native feats for tag {tag}")
    X = np.concatenate(X)
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    return list(df["id"]), X[df["row"].to_numpy()]


def main():
    parts_ids, parts_X, parts_y = [], [], []
    for tag, lab_file in [("calib", "calib_features.parquet"),
                          ("exp", "expansion_labels.parquet"),
                          ("exp2", "expansion2_labels.parquet"),
                          # optional expansion3 parts (modal_train3.py;
                          # exp3zh v2 is source-disjoint from sreason)
                          ("exp3", "expansion3_labels.parquet"),
                          ("exp3zh", "expansion3zh_labels.parquet")]:
        try:
            ids, X = load_feats(tag)
            lab = pd.read_parquet(D / lab_file).set_index("id")[
                "escalate_label"]
        except FileNotFoundError:
            if tag in ("exp3", "exp3zh"):
                print(f"core part {tag}: feats/labels missing — skipped")
                continue
            raise
        y = lab.reindex(ids).to_numpy().astype(float)
        keep = ~np.isnan(y)
        parts_ids += list(np.array(ids)[keep])
        parts_X.append(X[keep])
        parts_y.append(y[keep].astype(int))
    X0 = np.concatenate(parts_X)
    y0 = np.concatenate(parts_y)
    n0 = len(y0)
    print(f"core train: {n0} (fail {y0.mean():.3f})")

    fl = pd.read_parquet(D / "fresh_labels.parquet")
    fl = fl[fl["escalate_label"].notna()]
    print("fresh label cols:", fl.columns.tolist())
    lab_f = dict(zip(fl["id"], fl["escalate_label"].astype(int)))
    split_f = dict(zip(fl["id"], fl["split"]))
    ids_fr, X_fr = load_feats("fresh")
    tr_j = [j for j, i in enumerate(ids_fr)
            if i in lab_f and split_f.get(i) == "train"]
    he_j = [j for j, i in enumerate(ids_fr)
            if i in lab_f and split_f.get(i) != "train"]
    print(f"fresh: {len(tr_j)} train rows, {len(he_j)} heldout")

    X_tr = np.concatenate([X0, X_fr[tr_j]])
    y_tr = np.concatenate([y0, [lab_f[ids_fr[j]] for j in tr_j]])

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    best = None
    for C in (1e-4, 3e-4, 1e-3):
        oof = cross_val_predict(
            LogisticRegression(C=C, max_iter=5000), X_tr, y_tr, cv=cv,
            method="predict_proba")[:, 1]
        a = roc_auc_score(y_tr, oof)
        print(f"  C={C}: OOF AUC={a:.3f}")
        if best is None or a > best[1]:
            best = (C, a, oof)
    C, auc, oof = best
    clf = LogisticRegression(C=C, max_iter=5000).fit(X_tr, y_tr)

    # tier budgets: quantiles of the 2310 deployment-mix OOF only
    thr = {t: float(np.quantile(oof[:n0], 1 - b))
           for t, b in (("conservative", .15), ("balanced", .30),
                        ("aggressive", .50))}
    print("thresholds (core-mix quantiles): "
          + "  ".join(f"{t}={v:.4f}" for t, v in thr.items()))

    # guard 1: internal test AUC vs the 8be probe
    old = json.loads((D / "gate_native.json").read_text())
    w_old, b_old = np.array(old["w"]), old["b"]
    tst_ids, Xt = load_feats("test")
    tr = pd.read_parquet(D / "frozen_v3_traces.parquet")
    loc = tr[tr["mode"] == "local"].groupby("id")["heard_ok"].max()
    y_t = (1 - loc.reindex(tst_ids)).to_numpy().astype(float)
    k = ~np.isnan(y_t)
    a_old = roc_auc_score(y_t[k].astype(int), Xt[k] @ w_old + b_old)
    a_new = roc_auc_score(y_t[k].astype(int),
                          clf.predict_proba(Xt[k])[:, 1])
    print(f"guard1 internal test AUC: 8be={a_old:.3f} -> v2={a_new:.3f}")

    # guard 2: fresh heldout fire rates per tier, by label class
    s_he = clf.predict_proba(X_fr[he_j])[:, 1]
    y_he = np.array([lab_f[ids_fr[j]] for j in he_j])
    s_he_old = 1 / (1 + np.exp(-(X_fr[he_j] @ w_old + b_old)))
    for t, v in thr.items():
        f1 = (s_he[y_he == 1] >= v).mean() if (y_he == 1).any() else -1
        f0 = (s_he[y_he == 0] >= v).mean() if (y_he == 0).any() else -1
        o1 = (s_he_old[y_he == 1] >=
              old["eot_thresholds"][t]).mean() if (y_he == 1).any() else -1
        print(f"guard2 {t:<13} fresh-heldout fire: fast {f1:.2f} "
              f"(8be was {o1:.2f})  never {f0:.2f}")

    art = dict(old)
    art.update(w=clf.coef_[0].tolist(), b=float(clf.intercept_[0]),
               C=C, train_n=int(len(y_tr)), eot_thresholds=thr,
               recipe="scripts/25 native v2 = 8be + FreshQA train rows "
                      "(8bj; budgets on core-mix quantiles)")
    (D / "gate_native.json").write_text(json.dumps(art))
    print("wrote data/gate_native.json (v2)")


if __name__ == "__main__":
    main()
