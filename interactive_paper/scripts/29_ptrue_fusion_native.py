"""Probe (+) p(True) fusion, EXTERNAL half (8bl GPU half, item b).

28 measured the fusion ceiling internally (conc feats + text ptrue:
.818 -> .845, margin +.073 -> +.095) but n=240 cannot power +.027 and
the complementarity argument lives on the external pools (probe native
external mean ~.709; 5b: ptrue transfers per-pool with no inversion).
This scores the DEPLOYED-SHAPE fusion in the native regime everywhere:

  probe branch  = the deployed v2 recipe verbatim (scripts/27):
                  calib+exp+exp2 native feats + fresh TRAIN rows,
                  C from gate_native.json; calib-block OOF via 5-fold
                  CV over the full recipe set (unbiased stacker input).
  ptrue branch  = repeat-then-judge on the model's OWN transcript —
                  internal 600 from asr_minicpm-o45-audio.shard*
                  (p_yes_transcript, collected 6a), externals from
                  rtj_{pool}.shard*.parquet (p_yes_rtj, modal_rtj.py).
  stacker       = LR(C=1) on the 360 calib rows: [logit(probe_oof),
                  -logit(p_yes)] -> escalate_label (28's shape).

Eval per pool (internal test-240 + striviaqa/swebq/sllama/sdqa/sreason):
solo AUCs, fusion AUC + bootstrap delta CI vs probe, and the margin
translation — remix vs matched-random at each signal's own per-pool
quantile thresholds (the scripts/26 deployable story), so esc rates
match by construction. Labels mirror scripts/26: native judged parquet
for the local arm, conclive "always" tier for the expert arm.

Output: figures/ptrue_fusion_native.json.
Usage (from interactive_paper/): python scripts/29_ptrue_fusion_native.py
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
EXTERNAL = [("striviaqa", "oab_ok"), ("swebq", "oab_ok"),
            ("sllama", "oab_ok"), ("sdqa", "heard_ok"),
            ("sreason", "heard_ok")]


def load_feats(tag):
    ids, X = [], []
    for p in sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += [str(i) for i in z["ids"]]
        X.append(z["X"])
    if not X:
        raise FileNotFoundError(f"no native feats for tag {tag}")
    X = np.concatenate(X)
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    return list(df["id"]), X[df["row"].to_numpy()]


def lg(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p) - np.log(1 - p)


def boot_delta(y, s_new, s_old, n=4000):
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


def load_rtj_internal():
    fs = sorted(glob.glob(str(D / "asr_minicpm-o45-audio.shard*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df = df.drop_duplicates("id", keep="last").set_index("id")
    return df["p_yes_transcript"]


def load_rtj_pool(pool):
    fs = sorted(glob.glob(str(D / f"rtj_{pool}.shard*.parquet")))
    if not fs:
        raise FileNotFoundError(f"no rtj shards for {pool}")
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df = df.drop_duplicates("id", keep="last").set_index("id")
    return df


def main():
    art = json.loads((D / "gate_native.json").read_text())
    C = art["C"]

    # ---- probe branch: deployed v2 recipe (scripts/27 verbatim) ----
    blocks, block_ids = [], []
    for tag, lab_file in [("calib", "calib_features.parquet"),
                          ("exp", "expansion_labels.parquet"),
                          ("exp2", "expansion2_labels.parquet")]:
        ids, X = load_feats(tag)
        lab = pd.read_parquet(D / lab_file).set_index("id")["escalate_label"]
        y = lab.reindex(ids).to_numpy().astype(float)
        keep = ~np.isnan(y)
        blocks.append((X[keep], y[keep].astype(int)))
        block_ids.append([i for i, k in zip(ids, keep) if k])
    fl = pd.read_parquet(D / "fresh_labels.parquet")
    fl = fl[fl["escalate_label"].notna()]
    lab_f = dict(zip(fl["id"], fl["escalate_label"].astype(int)))
    split_f = dict(zip(fl["id"], fl["split"]))
    ids_fr, X_fr = load_feats("fresh")
    tr_j = [j for j, i in enumerate(ids_fr)
            if i in lab_f and split_f.get(i) == "train"]
    blocks.append((X_fr[tr_j],
                   np.array([lab_f[ids_fr[j]] for j in tr_j])))
    block_ids.append([ids_fr[j] for j in tr_j])
    Xc = np.concatenate([b[0] for b in blocks])
    yc = np.concatenate([b[1] for b in blocks])
    n_calib = len(block_ids[0])
    print(f"recipe n={len(yc)} (calib block {n_calib}, esc "
          f"{yc.mean():.3f})")

    oof = cross_val_predict(
        LogisticRegression(C=C, max_iter=5000), Xc, yc,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba")[:, 1]
    lr = LogisticRegression(C=C, max_iter=5000).fit(Xc, yc)
    print(f"probe LR(C={C}): recipe OOF AUC "
          f"{roc_auc_score(yc, oof):.3f}")

    # ---- stacker on the calib block (the rows with RTJ signal) ----
    rtj_int = load_rtj_internal()
    cal_ids = block_ids[0]
    p_cal = rtj_int.reindex(cal_ids).to_numpy()
    keep = ~np.isnan(p_cal)
    F_cal = np.column_stack([lg(oof[:n_calib][keep]),
                             -lg(p_cal[keep])])
    y_cal = yc[:n_calib][keep]
    stk = LogisticRegression(C=1.0, max_iter=1000).fit(F_cal, y_cal)
    stk_oof = cross_val_predict(
        LogisticRegression(C=1.0, max_iter=1000), F_cal, y_cal,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba")[:, 1]
    print(f"stacker: calib n={len(y_cal)} (rtj missing "
          f"{int((~keep).sum())}), OOF AUC "
          f"{roc_auc_score(y_cal, stk_oof):.3f}, coefs "
          f"{np.round(stk.coef_[0], 3).tolist()}")

    out = {"C": C, "n_recipe": int(len(yc)),
           "stacker_calib_n": int(len(y_cal)),
           "stacker_oof_auc": round(float(roc_auc_score(y_cal, stk_oof)), 3),
           "stacker_coefs": np.round(stk.coef_[0], 4).tolist(),
           "pools": {}}

    def eval_pool(pool, ids_p, Xp, lo_map, eo_map, p_rtj, p_txt=None):
        rows = [i for i in ids_p
                if i in lo_map and i in p_rtj and not
                np.isnan(p_rtj[i])]
        sel = {i: j for j, i in enumerate(ids_p)}
        Xr = Xp[[sel[i] for i in rows]]
        lo = np.array([lo_map[i] for i in rows], dtype=float)
        y = (1 - lo).astype(int)
        pr = np.array([p_rtj[i] for i in rows])
        s_probe = lr.predict_proba(Xr)[:, 1]
        s_fuse = stk.predict_proba(
            np.column_stack([lg(s_probe), -lg(pr)]))[:, 1]
        res = {"n": len(rows),
               "probe_auc": round(float(roc_auc_score(y, s_probe)), 3),
               "rtj_auc": round(float(roc_auc_score(y, -pr)), 3),
               "fusion_auc": round(float(roc_auc_score(y, s_fuse)), 3),
               "delta_vs_probe": boot_delta(y, s_fuse, s_probe)}
        if p_txt is not None:
            pt = np.array([p_txt.get(i, np.nan) for i in rows])
            m = ~np.isnan(pt)
            if m.sum() > 20:
                res["textq_auc"] = round(
                    float(roc_auc_score(y[m], -pt[m])), 3)
        # margin translation where the expert arm exists
        er_rows = [k for k, i in enumerate(rows) if i in eo_map]
        if len(er_rows) > 20:
            lo_r = lo[er_rows]
            eo_r = np.array([eo_map[rows[k]] for k in er_rows],
                            dtype=float)
            res["remix"] = {}
            for sig, sc in [("probe", s_probe[er_rows]),
                            ("fusion", s_fuse[er_rows])]:
                res["remix"][sig] = {}
                for tier, rate in RATES.items():
                    thr = float(np.quantile(sc, 1 - rate))
                    acc, er, rnd, p = remix(lo_r, eo_r, sc >= thr)
                    res["remix"][sig][tier] = {
                        "esc_rate": round(er, 3), "acc": round(acc, 3),
                        "random_matched": round(rnd, 3),
                        "perm_p": (round(p, 4) if p is not None
                                   else None)}
        out["pools"][pool] = res
        d = res["delta_vs_probe"]
        print(f"{pool:<11} n={res['n']:<4} probe {res['probe_auc']:.3f} "
              f"rtj {res['rtj_auc']:.3f} -> fusion "
              f"{res['fusion_auc']:.3f}  d={d[0]:+.3f} "
              f"[{d[1]:+.3f},{d[2]:+.3f}]"
              + (f"  txtq {res['textq_auc']:.3f}"
                 if "textq_auc" in res else ""))
        if "remix" in res:
            for tier in RATES:
                a, b = (res["remix"]["probe"][tier],
                        res["remix"]["fusion"][tier])
                print(f"   {tier:<13} margin probe "
                      f"{a['acc'] - a['random_matched']:+.3f} "
                      f"(p={a['perm_p']}) -> fusion "
                      f"{b['acc'] - b['random_matched']:+.3f} "
                      f"(p={b['perm_p']})")

    # ---- internal test-240 ----
    tids, Xt = load_feats("test")
    tr = pd.read_parquet(D / "frozen_v3_traces.parquet")
    loc = tr[tr["mode"] == "local"].groupby("id")["heard_ok"].max()
    esc = tr[tr["mode"] == "escalated"].groupby("id")["heard_ok"].max()
    lo_map = loc.dropna().astype(int).to_dict()
    eo_map = esc.dropna().astype(int).to_dict()
    eval_pool("frozen_test", tids, Xt, lo_map, eo_map,
              rtj_int.to_dict())

    # ---- externals ----
    for pool, ecol in EXTERNAL:
        try:
            ids_p, Xp = load_feats(pool)
            rtj = load_rtj_pool(pool)
        except FileNotFoundError as e:
            print(f"{pool}: SKIPPED ({e})")
            continue
        j = pd.read_parquet(D / f"frozen_native_{pool}_judged.parquet")
        j = j.dropna(subset=["adequate"]).drop_duplicates("id",
                                                          keep="last")
        lo_map = dict(zip(j["id"], j["adequate"].astype(int)))
        cl = pd.read_parquet(D / f"{pool}_conclive_traces.parquet")
        a = cl[cl.tier == "always"].dropna(subset=[ecol])
        a = a.drop_duplicates("id", keep="last")
        eo_map = dict(zip(a["id"], a[ecol].astype(int)))
        eval_pool(pool, ids_p, Xp, lo_map, eo_map,
                  rtj["p_yes_rtj"].to_dict(),
                  rtj["p_yes_textq"].to_dict()
                  if "p_yes_textq" in rtj else None)

    Path("figures/ptrue_fusion_native.json").write_text(
        json.dumps(out, indent=1))
    print("\nwrote figures/ptrue_fusion_native.json")


if __name__ == "__main__":
    main()
