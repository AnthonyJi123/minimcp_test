"""Reviewer item C (GPT pass 2026-08-26): the TF-IDF router is
intentionally weak --- would a modern frozen-embedding classifier on the
SAME data close the gap to the internal probe? Mirrors
modal_stream.py::router_baseline exactly (same splits, same 5-fold OOF
seed, same LOPO, same expert-inject tradeoff), only the features change:
OpenAI text-embedding-3-large -> LogisticRegression(C=1.0).

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _embed_router.py::t
"""
import json

import modal

from modal_app import (DATA, EVAL_EXPERT, FEATURES, OPENAI, QUERIES,
                       gate_data, util_image, _read_jsonl)

app = modal.App("embed-router")
img = util_image


@app.function(image=img, volumes={DATA: gate_data}, secrets=[OPENAI],
              timeout=60 * 30, cpu=8)
def t():
    import numpy as np
    import pandas as pd
    from openai import OpenAI
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, log_loss,
                                 roc_auc_score)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    f = pd.read_parquet(FEATURES)[["id", "pool", "split",
                                   "escalate_label"]]
    qtext = {q["id"]: q["query"] for q in _read_jsonl(QUERIES)}
    f = f[f["escalate_label"].notna() & f["id"].isin(qtext)]
    f["text"] = f["id"].map(qtext)
    cal = f[f["split"] == "calib"].reset_index(drop=True)
    tst = f[f["split"] == "test"].reset_index(drop=True)
    y_cal = cal["escalate_label"].astype(int).to_numpy()

    cl = OpenAI()

    def embed(texts):
        out = []
        for i in range(0, len(texts), 128):
            r = cl.embeddings.create(model="text-embedding-3-large",
                                     input=list(texts[i:i + 128]))
            out.extend(d.embedding for d in r.data)
        return np.asarray(out, dtype=np.float32)

    print(f"embedding {len(cal)} calib + {len(tst)} test texts...",
          flush=True)
    X_cal = embed(cal["text"].tolist())
    X_tst = embed(tst["text"].tolist())
    print(f"  dims={X_cal.shape[1]}", flush=True)

    def make_lr():
        return LogisticRegression(C=1.0, max_iter=3000)

    oof = cross_val_predict(make_lr(), X_cal, y_cal,
                            cv=StratifiedKFold(5, shuffle=True,
                                               random_state=42),
                            method="predict_proba")[:, 1]
    print("=== embedding router (text-embedding-3-large -> LR) ===",
          flush=True)
    print(f"  calib OOF AUC = {roc_auc_score(y_cal, oof):.3f} "
          f"(TF-IDF .669 / probe h_prompt .828 / pool-oracle .715)",
          flush=True)

    print("--- LOPO (train without a pool, test on it) ---", flush=True)
    lopo = {}
    for p in sorted(cal["pool"].unique()):
        m = (cal["pool"] != p).to_numpy()
        te_y = cal.loc[~m, "escalate_label"].astype(int)
        lr = make_lr().fit(X_cal[m], y_cal[m])
        s = lr.predict_proba(X_cal[~m])[:, 1]
        if te_y.nunique() < 2:
            print(f"  {p:15s} single-class (fail rate {te_y.mean():.2f})"
                  f" — mean score {s.mean():.3f}", flush=True)
            continue
        lopo[p] = float(roc_auc_score(te_y, s))
        print(f"  {p:15s} AUC={lopo[p]:.3f}", flush=True)

    lr_full = make_lr().fit(X_cal, y_cal)
    tst = tst.merge(pd.read_parquet(EVAL_EXPERT)[["id",
                                                  "expert_adequate"]],
                    on="id")
    X_tst = X_tst[:len(tst)] if len(X_tst) != len(tst) else X_tst
    s_test = lr_full.predict_proba(X_tst)[:, 1]
    small_ok = 1 - tst["escalate_label"].astype(int).to_numpy()
    exp_ok = tst["expert_adequate"].astype(float).to_numpy()
    order = np.argsort(-s_test)
    n = len(tst)
    accs, rand = [], []
    a0, a1 = small_ok.mean(), exp_ok.mean()
    for k in range(n + 1):
        esc = np.zeros(n, dtype=bool)
        esc[order[:k]] = True
        accs.append(np.where(esc, exp_ok, small_ok).mean())
        rand.append(a0 + (a1 - a0) * k / n)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = float(trap(np.array(accs) - np.array(rand), dx=1 / n))
    y_tst = tst["escalate_label"].astype(int).to_numpy()
    test_auc = float(roc_auc_score(y_tst, s_test))
    print("--- test tradeoff (expert-inject) ---", flush=True)
    print(f"  test AUC = {test_auc:.3f}; area over random = {area:+.3f} "
          f"(TF-IDF +.040 / midlayer_L22 +.064)", flush=True)
    for k_frac in (0.15, 0.30, 0.50):
        k = int(round(k_frac * n))
        print(f"  acc @ {k_frac:.0%} escalation = {accs[k]:.3f}",
              flush=True)

    out = {"oof_auc": float(roc_auc_score(y_cal, oof)),
           "lopo": lopo, "test_auc": test_auc, "area": area,
           "model": "text-embedding-3-large -> LR(C=1.0), protocol = "
                    "router_baseline (5-fold OOF seed 42)"}
    with open(f"{DATA}/embed_router.json", "w") as fh:
        json.dump(out, fh)
    gate_data.commit()
    print(f">>> wrote {DATA}/embed_router.json", flush=True)
