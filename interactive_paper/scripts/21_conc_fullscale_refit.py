"""Full-scale (2310-row) in-regime recalibration for the concurrent state.

8at left the external gap open: the in-regime probe was refit on only the
360-row calib slice (quarter-scale) and scored .57-.67 external AUC; the
paper attributes the gap to calibration scale without testing it. Here the
whole v3 train mix (calib 360 + expansion 800 + expansion2 1150 = 2310) is
re-collected in the concurrent-prefill state and the same 12288-d linear
read is refit at full scale, then scored on the concurrent test-240 and the
five binary external pools (features collected in the same state).

Recipe mirrors scripts/12 (LogisticRegression, C swept, 5-fold OOF, no
scaler). External labels mirror scripts/14 auc_never (never-tier conclive
outcome, pool's own judge). Also fits a data-scaling curve (n = 360..2310)
to test the scale attribution directly.

Usage (from interactive_paper/): .venv_boot\\Scripts\\python.exe
scripts\\21_conc_fullscale_refit.py
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
    for p in sorted(D.glob(f"frozen_conc_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += list(z["ids"])
        X.append(z["X"])
    if not X:
        raise FileNotFoundError(f"no feats for tag {tag}")
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
    # ---- train matrix: 2310 concurrent-state rows -----------------------
    parts = []
    for tag, lab_file in [("calib", "calib_features.parquet"),
                          ("exp", "expansion_labels.parquet"),
                          ("exp2", "expansion2_labels.parquet")]:
        ids, X = load_feats(tag)
        lab = pd.read_parquet(D / lab_file).set_index("id")["escalate_label"]
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

    # ---- eval sets ------------------------------------------------------
    evals = {}
    tst_ids, Xt = load_feats("test")
    tr = pd.read_parquet(D / "frozen_v3_traces.parquet")
    loc = tr[tr["mode"] == "local"].groupby("id")["heard_ok"].max()
    y_tst = (1 - loc.astype(int)).reindex(tst_ids).to_numpy()
    assert not np.isnan(y_tst.astype(float)).any()
    evals["internal_test"] = (Xt, y_tst.astype(int), None)

    for pool, col in EXTERNAL:
        ids_p, Xp = load_feats(pool)
        cl = pd.read_parquet(D / f"{pool}_conclive_traces.parquet")
        nv = cl[cl.tier == "never"].dropna(subset=[col])
        lab = nv.drop_duplicates("id", keep="last").set_index("id")[col]
        y = lab.reindex(ids_p).to_numpy().astype(float)
        keep = ~np.isnan(y)
        # trace-AUC replication reference (8at number, same probe diff run)
        trace_auc = round(float(roc_auc_score(
            1 - nv[col].astype(int), nv["eot_score"])), 3)
        evals[pool] = (Xp[keep], (1 - y[keep]).astype(int), trace_auc)
        print(f"eval {pool}: {keep.sum()} rows, trace-AUC(8at)={trace_auc}")

    # ---- probes: existing 360-row in-regime + new full-scale ------------
    art360 = json.loads((D / "gate_conc_frozen.json").read_text())
    w360, b360 = np.array(art360["w"]), art360["b"]

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

    # ---- score all eval sets -------------------------------------------
    out = {"train_n": int(len(y_tr)), "C": C, "train_oof_auc":
           round(float(oof_auc), 3), "pools": {}}
    print(f"\n{'pool':<14} {'n':>5} {'p360':>18} {'p2310':>18} "
          f"{'delta':>18} {'trace8at':>8}")
    for name, (X, y, trace_auc) in evals.items():
        s_old = X @ w360 + b360
        s_new = clf.predict_proba(X)[:, 1]
        a_old = boot_auc(y, s_old)
        a_new = boot_auc(y, s_new)
        d = paired_boot_delta(y, s_new, s_old)
        out["pools"][name] = {
            "n": int(len(y)), "fail_rate": round(float(y.mean()), 3),
            "auc_360": a_old, "auc_2310": a_new, "delta_paired": d,
            "trace_auc_8at": trace_auc}
        print(f"{name:<14} {len(y):>5} "
              f"{a_old[0]:.3f} [{a_old[1]:.3f},{a_old[2]:.3f}] "
              f"{a_new[0]:.3f} [{a_new[1]:.3f},{a_new[2]:.3f}] "
              f"{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] "
              f"{trace_auc if trace_auc else '-':>8}")

    # ---- data-scaling curve: is the gap really calibration scale? -------
    print("\nscaling curve (3 seeds, stratified subsample, fixed C):")
    curve = []
    for n_sub in (360, 720, 1150, 1560, 2310):
        aucs_int, aucs_ext = [], []
        for seed in range(3):
            if n_sub == len(y_tr):
                sub = np.arange(len(y_tr))
                if seed:
                    break
            else:
                r = np.random.default_rng(100 + seed)
                pos = np.where(y_tr == 1)[0]
                neg = np.where(y_tr == 0)[0]
                npos = int(round(n_sub * y_tr.mean()))
                sub = np.concatenate([r.choice(pos, npos, replace=False),
                                      r.choice(neg, n_sub - npos,
                                               replace=False)])
            c = LogisticRegression(C=C, max_iter=5000).fit(
                X_tr[sub], y_tr[sub])
            aucs_int.append(roc_auc_score(
                evals["internal_test"][1],
                c.predict_proba(evals["internal_test"][0])[:, 1]))
            aucs_ext.append(np.mean([
                roc_auc_score(evals[p][1],
                              c.predict_proba(evals[p][0])[:, 1])
                for p, _ in EXTERNAL]))
        row = {"n": n_sub,
               "auc_internal": round(float(np.mean(aucs_int)), 3),
               "auc_external_mean": round(float(np.mean(aucs_ext)), 3)}
        curve.append(row)
        print(f"  n={n_sub:>5}: internal {row['auc_internal']:.3f}  "
              f"external-mean {row['auc_external_mean']:.3f}")
    out["scaling_curve"] = curve

    art = {"w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]),
           "layer": 22, "modes": art360.get("modes"), "C": C,
           "train_n": int(len(y_tr)),
           "recipe": "scripts/21 full-scale in-regime refit (8bb)"}
    (D / "gate_conc_fullscale.json").write_text(json.dumps(art))
    Path("figures/conc_fullscale.json").write_text(json.dumps(out, indent=1))
    print("\nwrote data/gate_conc_fullscale.json + "
          "figures/conc_fullscale.json")


if __name__ == "__main__":
    main()
