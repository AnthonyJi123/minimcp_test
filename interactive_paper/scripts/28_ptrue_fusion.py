"""Probe (+) p(True) fusion — the representation-layer lever, internal
half (8bl). Runs entirely on in-repo data: frozen_conc_{calib,test}
feats, ptrue.shard*.parquet, calib/v3 labels.

Question: does the 5b verbalized self-eval signal add anything ON TOP
of the deployed-lineage L22 probe? 5b measured both solo and concluded
"gate on self-eval, probe as auxiliary"; the duplex work then went
probe-only because ptrue collapses under AUDIO input on the deployed
backbone (app: audio collapse table — trap dead at p_yes .556). The
paper also recorded the fix (repeat-then-judge on the model's own
transcript restores introspection), so fusion stays a live direction:
this script measures the ceiling of that direction where the signals
already exist (text-side, internal test-240).

A-priori variant: probe (+) ptrue_pre — pre-answer is the pre-decode
deployable signal (5b's own conclusion); post-draft needs the answer
first and (measured here) mis-calibrates calib->test. Other variants
printed as reference only.

Output: figures/ptrue_fusion.json.
Usage (from interactive_paper/): python scripts/28_ptrue_fusion.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

D = Path("data")
RNG = np.random.default_rng(42)
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}


def load_feats(tag):
    ids, X = [], []
    for p in sorted(D.glob(f"frozen_conc_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += [str(i) for i in z["ids"]]
        X.append(z["X"])
    X = np.concatenate(X)
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    return list(df["id"]), X[df["row"].to_numpy()]


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p) - np.log(1 - p)


def boot_delta(y, s_new, s_old, n=10000):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx))
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], s_new[b]) -
                    roc_auc_score(y[b], s_old[b]))
    return (round(float(np.mean(vals)), 3),
            *[round(v, 3) for v in np.percentile(vals, [2.5, 97.5])])


def remix(lo, eo, esc):
    acc = float(np.where(esc, eo, lo).mean())
    k = int(esc.sum())
    n = len(lo)
    if not 0 < k < n:
        return acc, k / n, acc, None
    rnd = []
    for _ in range(2000):
        r = np.zeros(n, dtype=bool)
        r[RNG.choice(n, k, replace=False)] = True
        rnd.append(np.where(r, eo, lo).mean())
    rnd = np.array(rnd)
    return acc, k / n, float(rnd.mean()), float((rnd >= acc).mean())


def main():
    cids, Xc = load_feats("calib")
    tids, Xt = load_feats("test")
    cf = pd.read_parquet(D / "calib_features.parquet").set_index("id")
    y_c = cf["escalate_label"].reindex(cids).astype(int).to_numpy()

    tr = pd.read_parquet(D / "frozen_v3_traces.parquet")
    loc = tr[tr["mode"] == "local"].groupby("id")["heard_ok"].max()
    y_t = (1 - loc.reindex(tids)).to_numpy().astype(float)
    keep = ~np.isnan(y_t)
    tids = [i for i, k in zip(tids, keep) if k]
    Xt, y_t = Xt[keep], y_t[keep].astype(int)
    esc_arm = tr[tr["mode"] == "escalated"].groupby("id")["heard_ok"].max()
    eo = esc_arm.reindex(tids).astype(float).to_numpy()
    lo = 1 - y_t
    print(f"calib n={len(y_c)} (fail {y_c.mean():.3f}) | "
          f"test n={len(y_t)} (fail {y_t.mean():.3f})")

    # anchor: the shipped 360-row conc probe must reproduce 8bb's .818
    art = json.loads((D / "gate_conc_frozen.json").read_text())
    a_ship = roc_auc_score(y_t, Xt @ np.array(art["w"]) + art["b"])
    print(f"anchor shipped conc360 test AUC = {a_ship:.3f} (8bb .818)")

    pt = pd.concat([pd.read_parquet(p) for p in
                    sorted(glob.glob(str(D / "ptrue.shard*.parquet")))]
                   ).set_index("id")
    P_c, P_t = pt.reindex(cids), pt.reindex(tids)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    best = None
    for C in (3e-5, 1e-4, 3e-4, 1e-3):
        oof = cross_val_predict(
            LogisticRegression(C=C, max_iter=5000), Xc, y_c, cv=cv,
            method="predict_proba")[:, 1]
        a = roc_auc_score(y_c, oof)
        if best is None or a > best[1]:
            best = (C, a, oof)
    C, oof_auc, probe_oof = best
    clf = LogisticRegression(C=C, max_iter=5000).fit(Xc, y_c)
    probe_t = clf.predict_proba(Xt)[:, 1]
    a_probe = roc_auc_score(y_t, probe_t)
    print(f"probe refit C={C}: calib OOF {oof_auc:.3f}, test {a_probe:.3f}")

    out = {"anchor_shipped": round(float(a_ship), 3), "C": C,
           "probe": {"oof_auc": round(float(oof_auc), 3),
                     "test_auc": round(float(a_probe), 3)},
           "solo": {}, "fusion": {}}
    for col in ("p_yes_pre", "p_yes_post"):
        out["solo"][col] = {
            "calib_auc": round(float(roc_auc_score(y_c, -P_c[col])), 3),
            "test_auc": round(float(roc_auc_score(y_t, -P_t[col])), 3)}
        print(f"solo {col:<11} calib {out['solo'][col]['calib_auc']:.3f} "
              f"test {out['solo'][col]['test_auc']:.3f}")

    variants = {"probe+pre": ["p_yes_pre"],          # a-priori variant
                "probe+post": ["p_yes_post"],
                "probe+pre+post": ["p_yes_pre", "p_yes_post"]}
    fused = {}
    for name, cols in variants.items():
        Fc = np.column_stack([lg(probe_oof)]
                             + [-lg(P_c[c].to_numpy()) for c in cols])
        Ft = np.column_stack([lg(probe_t)]
                             + [-lg(P_t[c].to_numpy()) for c in cols])
        stk_oof = cross_val_predict(
            LogisticRegression(C=1.0, max_iter=1000), Fc, y_c, cv=cv,
            method="predict_proba")[:, 1]
        stk = LogisticRegression(C=1.0, max_iter=1000).fit(Fc, y_c)
        s_t = stk.predict_proba(Ft)[:, 1]
        d = boot_delta(y_t, s_t, probe_t)
        out["fusion"][name] = {
            "stacker_oof_auc": round(float(roc_auc_score(y_c, stk_oof)), 3),
            "test_auc": round(float(roc_auc_score(y_t, s_t)), 3),
            "delta_vs_probe": d}
        fused[name] = (stk_oof, s_t)
        print(f"fusion {name:<15} test "
              f"{out['fusion'][name]['test_auc']:.3f}  "
              f"d={d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")

    # margin translation: remix at calib-OOF quantile thresholds,
    # probe vs the a-priori fusion, vs matched random
    print(f"\n{'signal':<11}{'tier':<14}{'esc':>5}{'acc':>7}{'rand':>7}"
          f"{'margin':>8}{'p':>8}")
    out["remix"] = {}
    for sig, (oof_s, s_t) in [("probe", (probe_oof, probe_t)),
                              ("fusion_pre", fused["probe+pre"])]:
        out["remix"][sig] = {}
        for tier, rate in RATES.items():
            thr = float(np.quantile(oof_s, 1 - rate))
            acc, er, rnd, p = remix(lo, eo, s_t >= thr)
            out["remix"][sig][tier] = {
                "esc_rate": round(er, 3), "acc": round(acc, 3),
                "random_matched": round(rnd, 3),
                "perm_p": (round(p, 4) if p is not None else None)}
            print(f"{sig:<11}{tier:<14}{er:>5.2f}{acc:>7.3f}{rnd:>7.3f}"
                  f"{acc - rnd:>+8.3f}"
                  + (f"{p:>8.4f}" if p is not None else "       -"))

    Path("figures/ptrue_fusion.json").write_text(json.dumps(out, indent=1))
    print("\nwrote figures/ptrue_fusion.json")


if __name__ == "__main__":
    main()
