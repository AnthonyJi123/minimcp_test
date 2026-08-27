"""In-regime probe refit for the concurrent-prefill state.

Question the frozen-transfer result leaves open: is the -.109 AUC drop a
property of the concurrent STATE (information gone from L22) or of the
frozen PROBE (calibrated turn-based, stale in-regime)? Refit the same
12288-d linear read on calib-360 features collected in the concurrent
state and evaluate on test-240 concurrent features.

Mirrors the v3 recipe (modal_train2.py): LogisticRegression, C swept
around 1e-4, max_iter=5000, no scaler; labels = calib escalate_label /
test fail (same construction as scripts/11).

Usage (from interactive_paper/): .venv_boot\\Scripts\\python.exe
scripts\\12_concurrent_refit.py
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
    for p in sorted(D.glob(f"frozen_conc_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += list(z["ids"])
        X.append(z["X"])
    X = np.concatenate(X)
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    return list(df["id"]), X[df["row"].to_numpy()]


def main():
    cal_ids, Xc = load_feats("calib")
    tst_ids, Xt = load_feats("test")
    print(f"calib feats: {len(cal_ids)} x {Xc.shape[1]}; "
          f"test feats: {len(tst_ids)} x {Xt.shape[1]}")

    calib = pd.read_parquet(D / "calib_features.parquet")
    y_cal = (calib.set_index("id")["escalate_label"]
             .reindex(cal_ids).to_numpy().astype(int))

    tr = pd.read_parquet(D / "frozen_v3_traces.parquet")
    loc = tr[tr["mode"] == "local"].groupby("id")["heard_ok"].max()
    y_tst = (1 - loc.astype(int)).reindex(tst_ids).to_numpy()
    assert not np.isnan(y_cal).any() and not np.isnan(y_tst).any()
    print(f"labels: calib fail {y_cal.mean():.3f}, test fail "
          f"{y_tst.mean():.3f}")

    # sanity: frozen v3 probe applied to these test features must
    # reproduce the harness-scored AUC (.758)
    art = json.loads((D / "gate_v3_frozen_local.json").read_text()) \
        if (D / "gate_v3_frozen_local.json").exists() else None
    if art:
        w, b = np.array(art["w"]), art["b"]
        s_frozen_probe = Xt @ w + b
        print(f"sanity frozen-probe-on-test-feats AUC: "
              f"{roc_auc_score(y_tst, s_frozen_probe):.3f} (expect ~.758)")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    best = None
    for C in (3e-5, 1e-4, 3e-4, 1e-3):
        oof = cross_val_predict(
            LogisticRegression(C=C, max_iter=5000), Xc, y_cal, cv=cv,
            method="predict_proba")[:, 1]
        a = roc_auc_score(y_cal, oof)
        print(f"  C={C}: calib OOF AUC={a:.3f}")
        if best is None or a > best[1]:
            best = (C, a)
    C = best[0]

    clf = LogisticRegression(C=C, max_iter=5000).fit(Xc, y_cal)
    s = clf.predict_proba(Xt)[:, 1]
    a = roc_auc_score(y_tst, s)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y_tst), len(y_tst))
        if len(set(y_tst[i])) < 2:
            continue
        bs.append(roc_auc_score(y_tst[i], s[i]))
    print(f"\nREFIT (in-regime, C={C}, calib OOF {best[1]:.3f}):")
    print(f"  test-240 concurrent AUC = {a:.3f} "
          f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
    print("  reference: frozen probe on same state .758 [.693,.816]; "
          "headless .869")

    out = {"C": C, "calib_oof_auc": round(best[1], 4),
           "test_auc": round(float(a), 4),
           "test_auc_ci": [round(float(np.percentile(bs, 2.5)), 4),
                           round(float(np.percentile(bs, 97.5)), 4)],
           "n_calib": len(cal_ids), "n_test": len(tst_ids)}
    (Path("figures") / "concurrent_refit.json").write_text(
        json.dumps(out, indent=2))
    print("wrote figures/concurrent_refit.json")


if __name__ == "__main__":
    main()
