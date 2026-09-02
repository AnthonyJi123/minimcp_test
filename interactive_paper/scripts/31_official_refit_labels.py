"""8bq official-config refit with a selectable LABEL SOURCE and
per-language operating points. Same features/recipe as scripts/26
(caliboff+expoff+exp2off+exp3off+exp3zhoff + freshoff train, C=3e-4
nested-confirmed), plus:

  --source tb|native   tb = turn-based escalate_label parquets (the
                       pre-8bq deployed recipe); native = the judged
                       label of the SAME official-native trace the
                       features were read from (fresh_fast keeps its
                       prior 1). Review item 1.
  eot_thresholds_lang  {en, zh} tier thresholds quantiled on the
                       core-mix OOF of each language separately
                       (review item 3); demo_duplex picks by ?lang=.
  manifest             sha1 of query ids / features / labels so the
                       receipt (scripts/27) can refuse a mismatch.
  label_source         stored in the artifact for scripts/27.

Guards (pre-registered, same as 26): testoff AUC vs the previous
artifact, reported on BOTH the v3-harness labels and the official-
native judged labels; freshoff-heldout fire separation.

Usage: .venv_boot\\Scripts\\python.exe scripts\\31_official_refit_labels.py --source native [--dry]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

D = Path("data")
PARTS = [("caliboff", "calib_features.parquet"),
         ("expoff", "expansion_labels.parquet"),
         ("exp2off", "expansion2_labels.parquet"),
         ("exp3off", "expansion3_labels.parquet"),
         ("exp3zhoff", "expansion3zh_labels.parquet"),
         # 8bq targeted zh coverage: native labels only (no turn-based
         # answer pass was run for it) — skipped when absent
         ("exp4zhoff", None),
         # 8bq-2 bilingual reasoning coverage (native labels only)
         ("exp5rsoff", None),
         # 8bq-2 in-context rows: same queries as calib / exp3zh, but
         # arriving as the SECOND turn after a carrier Q&A (native
         # labels judged on the in-context answer)
         ("calibctx", None), ("exp3zhctx", None)]
ZH_TAGS = {"exp3zhoff", "exp4zhoff", "exp3zhctx"}
OPTIONAL = {"exp4zhoff", "exp5rsoff", "calibctx", "exp3zhctx"}
# per-row language for mixed-language tags: pool name prefix
POOL_FILES = {"exp4zhoff": "queries_expansion4zh.jsonl",
              "exp5rsoff": "queries_expansion5rs.jsonl"}


def row_lang(tag, ids):
    if tag in POOL_FILES:
        pool = {q["id"]: q["pool"] for q in (json.loads(l) for l in open(
            D / POOL_FILES[tag], encoding="utf-8") if l.strip())}
        return ["zh" if pool.get(i, "").startswith("zh-") else "en"
                for i in ids]
    return ["zh" if tag in ZH_TAGS else "en"] * len(ids)
RATES = (("conservative", .15), ("balanced", .30), ("aggressive", .50))


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


def labels_for(tag, lab_file, source):
    if source == "native":
        return pd.read_parquet(
            D / f"frozen_native_{tag}_judged.parquet").set_index("id")[
            "escalate_label"]
    df = pd.read_parquet(D / lab_file)
    if "escalate_label" not in df:
        df["escalate_label"] = 1 - df["adequate"].astype(float)
    return df.set_index("id")["escalate_label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="native", choices=["tb", "native"])
    ap.add_argument("--C", type=float, default=3e-4)
    ap.add_argument("--dry", action="store_true",
                    help="report guards, do not write gate_native.json")
    a = ap.parse_args()

    ids, Xs, ys, lang = [], [], [], []
    for tag, lab_file in PARTS:
        try:
            i, X = load_feats(tag)
            if lab_file is None and a.source != "native":
                raise FileNotFoundError(f"{tag}: native labels only")
            y = labels_for(tag, lab_file, a.source).reindex(i).to_numpy(float)
        except FileNotFoundError as e:
            if tag in OPTIONAL:
                print(f"{tag}: skipped ({e})")
                continue
            raise
        k = ~np.isnan(y)
        kept = [q for q, kk in zip(i, k) if kk]
        ids += kept
        Xs.append(X[k])
        ys.append(y[k].astype(int))
        lang += row_lang(tag, kept)
        print(f"{tag}: {k.sum()} rows, fail {y[k].mean():.3f}")
    n0 = len(ids)

    fl = pd.read_parquet(D / "fresh_labels.parquet")
    fl = fl[fl["escalate_label"].notna()]
    lab_f = dict(zip(fl["id"], fl["escalate_label"].astype(int)))
    if a.source == "native":
        nf = labels_for("freshoff", None, "native")
        for i, p in zip(fl["id"], fl["pool"]):
            if p != "fresh_fast" and i in nf.index and pd.notna(nf[i]):
                lab_f[i] = int(nf[i])
    split_f = dict(zip(fl["id"], fl["split"]))
    ids_fr, X_fr = load_feats("freshoff")
    tr_j = [j for j, i in enumerate(ids_fr)
            if i in lab_f and split_f.get(i) == "train"]
    he_j = [j for j, i in enumerate(ids_fr)
            if i in lab_f and split_f.get(i) != "train"]
    ids += [ids_fr[j] for j in tr_j]
    X_tr = np.concatenate(Xs + [X_fr[tr_j]])
    y_tr = np.concatenate(ys + [[lab_f[ids_fr[j]] for j in tr_j]])
    lang = np.array(lang + ["en"] * len(tr_j))
    print(f"train {len(y_tr)} (core {n0} + fresh {len(tr_j)}), "
          f"source={a.source}, fail {y_tr.mean():.3f}")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(LogisticRegression(C=a.C, max_iter=5000),
                            X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y_tr, oof)
    print(f"OOF AUC {auc:.3f} (C={a.C})")
    clf = LogisticRegression(C=a.C, max_iter=5000).fit(X_tr, y_tr)

    core = np.arange(len(y_tr)) < n0
    thr = {t: float(np.quantile(oof[core], 1 - b)) for t, b in RATES}
    thr_lang = {}
    for lg in ("en", "zh"):
        m = core & (lang == lg)
        thr_lang[lg] = {t: float(np.quantile(oof[m], 1 - b))
                        for t, b in RATES}
    print("global thresholds: "
          + "  ".join(f"{t}={v:.4f}" for t, v in thr.items()))
    for lg, tl in thr_lang.items():
        print(f"  {lg}: " + "  ".join(f"{t}={v:.4f}" for t, v in tl.items()))

    old = json.loads((D / "gate_native.json").read_text())
    tst_ids, Xt = load_feats("testoff")
    labels = {}
    tr = pd.read_parquet(D / "frozen_v3_traces.parquet")
    loc = tr[tr["mode"] == "local"].groupby("id")["heard_ok"].max()
    labels["v3"] = (1 - loc.reindex(tst_ids)).to_numpy(float)
    labels["native"] = pd.read_parquet(
        D / "frozen_native_testoff_judged.parquet").set_index("id")[
        "escalate_label"].reindex(tst_ids).to_numpy(float)
    s_new = clf.predict_proba(Xt)[:, 1]
    s_old = Xt @ np.array(old["w"]) + old["b"]
    for name, y_t in labels.items():
        k = ~np.isnan(y_t)
        print(f"guard1 testoff[{name}] AUC: previous {roc_auc_score(y_t[k].astype(int), s_old[k]):.3f}"
              f" -> this {roc_auc_score(y_t[k].astype(int), s_new[k]):.3f}")
    for t, v in thr.items():
        print(f"  test fire@{t}: {float((s_new >= v).mean()):.2f}")

    s_he = clf.predict_proba(X_fr[he_j])[:, 1]
    y_he = np.array([lab_f[ids_fr[j]] for j in he_j])
    for t, v in thr.items():
        print(f"guard2 {t:<13} fresh-heldout fire: "
              f"fast {(s_he[y_he == 1] >= v).mean():.2f} "
              f"never {(s_he[y_he == 0] >= v).mean():.2f}")

    # zh operating point on the external zh pool (sreasonoff, conclive
    # never labels): fire rate at the global vs the zh threshold
    try:
        z_ids, Xz = load_feats("sreasonoff")
        sz = clf.predict_proba(Xz)[:, 1]
        for t in ("balanced", "aggressive"):
            print(f"  sreason fire@{t}: global {float((sz >= thr[t]).mean()):.2f}"
                  f"  zh-thr {float((sz >= thr_lang['zh'][t]).mean()):.2f}")
    except FileNotFoundError:
        pass

    man = {"query_ids": hashlib.sha1("\n".join(ids).encode()).hexdigest(),
           "features": hashlib.sha1(np.ascontiguousarray(X_tr)).hexdigest(),
           "labels": hashlib.sha1("\n".join(
               f"{i}\t{v}" for i, v in zip(ids, y_tr)).encode()).hexdigest()}
    art = dict(old)
    art.update(w=clf.coef_[0].tolist(), b=float(clf.intercept_[0]),
               C=a.C, train_n=int(len(y_tr)), eot_thresholds=thr,
               eot_thresholds_lang=thr_lang, label_source=a.source,
               manifest=man, oof_auc=round(float(auc), 4),
               recipe="scripts/31 official-config refit, label source "
                      f"{a.source} (core caliboff+expoff+exp2off+exp3off+"
                      "exp3zhoff + fresh, official serving params, "
                      "per-language thresholds)")
    if a.dry:
        print("--dry: gate_native.json NOT written")
        return
    (D / "gate_native.json").write_text(json.dumps(art))
    print("wrote data/gate_native.json")


if __name__ == "__main__":
    main()
