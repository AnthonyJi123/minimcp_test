"""Training receipt for the CURRENT (native, 2310-row) probe.

The 8j/8k receipt accounting — train / 5-fold-OOF / test logloss,
acc@0.5 vs majority, AUC, plus test classification acc at the budget
thresholds (OOF-score quantiles .15/.30/.50, the gate's actual
operating mode) — was last generated for the 360-row probes
(probe_receipt.json, 2026-08-05) and has gone stale through the 8bb
fullscale, 8bc benefit, 8be native, and 8bj fresh refits. This
regenerates it for the shipped native gate, mirroring the deployed v2
recipe (scripts/25): calib+exp+exp2 native feats + the fresh TRAIN
rows, turn-based escalate_label, C from gate_native.json; budget
thresholds quantiled on the core-mix OOF only (the deployed budget
convention). Internal test-240, plus test-only rows for each external
pool whose native feats exist.

Output: data/probe_receipt_native.json — same list-of-dicts shape as
probe_receipt.json so receipt_figure-style tooling can read it.

Usage (from interactive_paper/): python scripts/27_probe_receipt_native.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

D = Path("data")
EXTERNAL = [("striviaqa", "oab_ok"), ("swebq", "oab_ok"),
            ("sllama", "oab_ok"), ("sdqa", "heard_ok"),
            ("sreason", "heard_ok")]


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
    art = json.loads((D / "gate_native.json").read_text())
    C = art["C"]

    parts = []
    for tag, lab_file in [("calib", "calib_features.parquet"),
                          ("exp", "expansion_labels.parquet"),
                          ("exp2", "expansion2_labels.parquet"),
                          # optional expansion3 parts (modal_train3.py;
                          # exp3zh v2 is source-disjoint from sreason)
                          ("exp3", "expansion3_labels.parquet"),
                          ("exp3zh", "expansion3zh_labels.parquet")]:
        try:
            ids, X = load_feats(tag)
            lab = pd.read_parquet(
                D / lab_file).set_index("id")["escalate_label"]
        except FileNotFoundError:
            if tag in ("exp3", "exp3zh"):
                print(f"recipe part {tag}: feats/labels missing — "
                      f"skipped")
                continue
            raise
        y = lab.reindex(ids).to_numpy().astype(float)
        keep = ~np.isnan(y)
        parts.append((X[keep], y[keep].astype(int)))
    Xc = np.concatenate([p[0] for p in parts])
    yc = np.concatenate([p[1] for p in parts])
    n_core = len(yc)

    # deployed v2 recipe (scripts/25): fresh TRAIN rows appended; the
    # tier budgets stay quantiled on the core mix only
    try:
        fl = pd.read_parquet(D / "fresh_labels.parquet")
        fl = fl[fl["escalate_label"].notna()]
        lab_f = dict(zip(fl["id"], fl["escalate_label"].astype(int)))
        split_f = dict(zip(fl["id"], fl["split"]))
        ids_fr, X_fr = load_feats("fresh")
        tr_j = [j for j, i in enumerate(ids_fr)
                if i in lab_f and split_f.get(i) == "train"]
        Xc = np.concatenate([Xc, X_fr[tr_j]])
        yc = np.concatenate([yc,
                             [lab_f[ids_fr[j]] for j in tr_j]])
        print(f"fresh train rows appended: {len(tr_j)}")
    except FileNotFoundError:
        print("fresh feats/labels missing — receipt covers the core "
              "mix only (pre-8bj probe)")

    lr = LogisticRegression(C=C, max_iter=5000)
    oof = cross_val_predict(LogisticRegression(C=C, max_iter=5000),
                            Xc, yc,
                            cv=StratifiedKFold(5, shuffle=True,
                                               random_state=42),
                            method="predict_proba")[:, 1]
    lr.fit(Xc, yc)
    p_tr = lr.predict_proba(Xc)[:, 1]
    r = {"name": "native L22 probe (deployed gate)", "C": C,
         "dim": int(Xc.shape[1]), "n_train": int(len(yc)),
         "escalate_rate": float(yc.mean()),
         "train_logloss": float(log_loss(yc, p_tr)),
         "train_acc": float(accuracy_score(yc, p_tr >= .5)),
         "oof_auc": float(roc_auc_score(yc, oof)),
         "oof_logloss": float(log_loss(yc, oof)),
         "oof_acc": float(accuracy_score(yc, oof >= .5)),
         "majority_calib": float(max(yc.mean(), 1 - yc.mean()))}
    print(f"=== {r['name']}: LR(C={C}) dim={r['dim']} n={r['n_train']} "
          f"(esc {r['escalate_rate']:.3f}) ===")
    print(f"  train logloss={r['train_logloss']:.3f} "
          f"acc={r['train_acc']:.3f} | OOF AUC={r['oof_auc']:.3f} "
          f"logloss={r['oof_logloss']:.3f} acc={r['oof_acc']:.3f} "
          f"(majority {r['majority_calib']:.3f})")

    tst_ids, Xt = load_feats("test")
    tr = pd.read_parquet(D / "frozen_v3_traces.parquet")
    loc = tr[tr["mode"] == "local"].groupby("id")["heard_ok"].max()
    yt = (1 - loc.astype(int)).reindex(tst_ids).to_numpy().astype(float)
    keep = ~np.isnan(yt)
    Xt, yt = Xt[keep], yt[keep].astype(int)
    s = lr.predict_proba(Xt)[:, 1]
    r.update({"n_test": int(len(yt)),
              "test_auc": float(roc_auc_score(yt, s)),
              "test_logloss": float(log_loss(yt, s)),
              "test_acc": float(accuracy_score(yt, s >= .5)),
              "majority_test": float(max(yt.mean(), 1 - yt.mean()))})
    print(f"  test n={r['n_test']} AUC={r['test_auc']:.3f} "
          f"logloss={r['test_logloss']:.3f} acc={r['test_acc']:.3f} "
          f"(majority {r['majority_test']:.3f})")
    r["budget_ops"] = {}
    for bud in (.15, .30, .50):
        thr = float(np.quantile(oof[:n_core], 1 - bud))
        fire = s >= thr
        r["budget_ops"][str(bud)] = {
            "acc": float(accuracy_score(yt, fire)),
            "realized_rate": float(fire.mean())}
        print(f"  @budget {bud:.0%}: classification acc="
              f"{accuracy_score(yt, fire):.3f} "
              f"(realized escalation {fire.mean():.3f})")
    out = [r]

    for pool, col in EXTERNAL:
        try:
            ids_p, Xp = load_feats(pool)
        except FileNotFoundError:
            print(f"external {pool}: no native feats — skipped")
            continue
        cl = pd.read_parquet(D / f"{pool}_conclive_traces.parquet")
        nv = cl[cl.tier == "never"].dropna(subset=[col])
        lab = nv.drop_duplicates("id", keep="last").set_index("id")[col]
        y = lab.reindex(ids_p).to_numpy().astype(float)
        keep = ~np.isnan(y)
        Xp, y = Xp[keep], (1 - y[keep]).astype(int)
        s = lr.predict_proba(Xp)[:, 1]
        row = {"name": f"native probe on {pool} (test-only)",
               "n_test": int(len(y)),
               "test_auc": float(roc_auc_score(y, s)),
               "test_logloss": float(log_loss(y, s)),
               "test_acc": float(accuracy_score(y, s >= .5)),
               "majority_test": float(max(y.mean(), 1 - y.mean()))}
        out.append(row)
        print(f"  {pool}: n={row['n_test']} AUC={row['test_auc']:.3f} "
              f"logloss={row['test_logloss']:.3f} "
              f"acc={row['test_acc']:.3f} "
              f"(majority {row['majority_test']:.3f})")

    (D / "probe_receipt_native.json").write_text(json.dumps(out))
    print("\nwrote data/probe_receipt_native.json")


if __name__ == "__main__":
    main()
