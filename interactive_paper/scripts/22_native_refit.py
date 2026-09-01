"""Native-duplex regime refit (8be): the §8bb recipe on MiniCPMODuplex
features.

Train: frozen_native_{calib,exp,exp2} feats (modal_native_dump.py) +
the SAME escalate_label parquets as scripts/21 (labels stay turn-based
by §8bb methodology; in-regime native labels are the validity-table
step, not the calibration step).

Eval: native test-240 feats + frozen_v3 local heard_ok labels
(scripts/21's internal_test convention), scored by (a) conc-frozen
360-row probe, (b) conc-fullscale 2310-row probe, (c) the new native
refit — the three-way tells us whether calibration-follows-the-regime
extends to the native schema. External pools appended once their native
dumps exist (same load path, tags = pool names).

Also writes gate_native.json (w, b, eot_thresholds from label-free
quantiles of the calib-score distribution at 15/30/50% fire rates) for
demo_duplex.py to load directly.

Usage (from interactive_paper/): .venv_boot\\Scripts\\python.exe
scripts\\22_native_refit.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

D = Path("data")
RNG = np.random.default_rng(42)
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


def boot_auc(y, s, n=10000):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx))
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], s[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(roc_auc_score(y, s)), 3), round(lo, 3), round(hi, 3)


def paired_boot_delta(y, s_new, s_old, n=10000):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx))
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], s_new[b]) -
                    roc_auc_score(y[b], s_old[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(np.mean(vals)), 3), round(lo, 3), round(hi, 3)


def main():
    parts = []
    for tag, lab_file in [("calib", "calib_features.parquet"),
                          ("exp", "expansion_labels.parquet"),
                          ("exp2", "expansion2_labels.parquet"),
                          # optional expansion3 parts (modal_train3.py);
                          # exp3zh v2 = MGSM+XCOPA zh, source-disjoint
                          # from sreason, which stays fully external
                          ("exp3", "expansion3_labels.parquet"),
                          ("exp3zh", "expansion3zh_labels.parquet")]:
        try:
            ids, X = load_feats(tag)
            lab = pd.read_parquet(
                D / lab_file).set_index("id")["escalate_label"]
        except FileNotFoundError:
            if tag in ("exp3", "exp3zh"):
                print(f"train part {tag}: feats/labels missing — skipped")
                continue
            raise
        y = lab.reindex(ids).to_numpy().astype(float)
        keep = ~np.isnan(y)
        parts.append((np.array(ids)[keep], X[keep], y[keep].astype(int)))
        print(f"train part {tag}: {keep.sum()} rows "
              f"(fail rate {y[keep].mean():.3f})")
    ids_tr = np.concatenate([p[0] for p in parts])
    X_tr = np.concatenate([p[1] for p in parts])
    y_tr = np.concatenate([p[2] for p in parts])
    assert len(set(ids_tr)) == len(ids_tr)
    print(f"train total: {len(y_tr)} x {X_tr.shape[1]}, "
          f"fail {y_tr.mean():.3f}")

    evals = {}
    tst_ids, Xt = load_feats("test")
    tr = pd.read_parquet(D / "frozen_v3_traces.parquet")
    loc = tr[tr["mode"] == "local"].groupby("id")["heard_ok"].max()
    y_tst = (1 - loc.astype(int)).reindex(tst_ids).to_numpy()
    keep = ~np.isnan(y_tst.astype(float))
    evals["internal_test"] = (Xt[keep], y_tst[keep].astype(int))
    print(f"eval internal_test: {keep.sum()} rows")

    for pool, col in EXTERNAL:
        try:
            ids_p, Xp = load_feats(pool)
        except FileNotFoundError:
            print(f"eval {pool}: no native feats yet — skipped")
            continue
        cl = pd.read_parquet(D / f"{pool}_conclive_traces.parquet")
        nv = cl[cl.tier == "never"].dropna(subset=[col])
        lab = nv.drop_duplicates("id", keep="last").set_index("id")[col]
        y = lab.reindex(ids_p).to_numpy().astype(float)
        keep = ~np.isnan(y)
        evals[pool] = (Xp[keep], (1 - y[keep]).astype(int))
        print(f"eval {pool}: {keep.sum()} rows")

    art_cf = json.loads((D / "gate_conc_frozen.json").read_text())
    art_fs = json.loads((D / "gate_conc_fullscale.json").read_text())
    w_cf, b_cf = np.array(art_cf["w"]), art_cf["b"]
    w_fs, b_fs = np.array(art_fs["w"]), art_fs["b"]

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    best = None
    for C in (3e-5, 1e-4, 3e-4, 1e-3):
        oof = cross_val_predict(
            LogisticRegression(C=C, max_iter=5000), X_tr, y_tr, cv=cv,
            method="predict_proba")[:, 1]
        a = roc_auc_score(y_tr, oof)
        print(f"  C={C}: train OOF AUC={a:.3f}")
        if best is None or a > best[1]:
            best = (C, a)
    C, oof_auc = best
    print(f"chosen C={C} (OOF {oof_auc:.3f})")
    clf = LogisticRegression(C=C, max_iter=5000).fit(X_tr, y_tr)

    out = {"train_n": int(len(y_tr)), "C": C,
           "train_oof_auc": round(float(oof_auc), 3), "pools": {}}
    print(f"\n{'pool':<14} {'n':>5} {'conc360':>18} {'conc2310':>18} "
          f"{'native':>18} {'d(nat-2310)':>18}")
    for name, (X, y) in evals.items():
        s_cf = X @ w_cf + b_cf
        s_fs = X @ w_fs + b_fs
        s_nat = clf.predict_proba(X)[:, 1]
        a_cf, a_fs, a_nat = boot_auc(y, s_cf), boot_auc(y, s_fs), \
            boot_auc(y, s_nat)
        d = paired_boot_delta(y, s_nat, s_fs)
        out["pools"][name] = {
            "n": int(len(y)), "fail_rate": round(float(y.mean()), 3),
            "auc_conc360": a_cf, "auc_conc2310": a_fs,
            "auc_native": a_nat, "delta_native_vs_2310": d}
        print(f"{name:<14} {len(y):>5} "
              f"{a_cf[0]:.3f} [{a_cf[1]:.3f},{a_cf[2]:.3f}] "
              f"{a_fs[0]:.3f} [{a_fs[1]:.3f},{a_fs[2]:.3f}] "
              f"{a_nat[0]:.3f} [{a_nat[1]:.3f},{a_nat[2]:.3f}] "
              f"{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")

    # label-free tier thresholds: quantiles of the native calib-score
    # distribution at the deployed nominal fire rates (scripts/13 recipe)
    s_cal = clf.predict_proba(X_tr)[:, 1]
    thr = {t: float(np.quantile(s_cal, 1 - r))
           for t, r in [("conservative", .15), ("balanced", .30),
                        ("aggressive", .50)]}
    print("\nthresholds (calib quantiles): "
          + "  ".join(f"{t}={v:.4f}" for t, v in thr.items()))

    art = {"w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]),
           "layer": 22, "k_eot": art_cf.get("k_eot", 8),
           "modes": art_cf.get("modes"), "C": C,
           "train_n": int(len(y_tr)), "eot_thresholds": thr,
           "recipe": "scripts/22 native-duplex in-regime refit (8be)"}
    (D / "gate_native.json").write_text(json.dumps(art))
    # carry the scaling curve forward and append this run's full-mix
    # point (the gallery figure reads it)
    rf = Path("figures/native_refit.json")
    curve = []
    if rf.exists():
        curve = json.loads(rf.read_text()).get("scaling_curve", [])
    ext = [v["auc_native"][0] for k, v in out["pools"].items()
           if k != "internal_test"]
    curve = [p for p in curve if p["n"] != out["train_n"]]
    curve.append({
        "n": out["train_n"],
        "auc_internal": out["pools"]["internal_test"]["auc_native"][0],
        "auc_external_mean": round(sum(ext) / len(ext), 3)})
    out["scaling_curve"] = sorted(curve, key=lambda p: p["n"])
    rf.write_text(json.dumps(out, indent=1))
    print("\nwrote data/gate_native.json + figures/native_refit.json")


if __name__ == "__main__":
    main()
