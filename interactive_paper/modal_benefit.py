"""Benefit-trained probe refit (P0-R q3 follow-up, 8bc).

The deployed gate trains on y = local-wrong (expert-agnostic by design);
app:fixedthr re-SCORES that probe against the expert-benefit oracle
(y = local-wrong AND expert-right) and pays 0.63--0.81 external /
0.840->0.732 internal. Open question: is that the label's price or the
probe's? Here the same 12288-d L22 read (eoth2 stored hiddens, CPU-only)
is re-TRAINED on benefit labels and compared on identical features.

Stage 1 needs expert outcomes on the 2310 train rows (gold-text gpt-5.5
low + standard judge, expert-cache deduped) -> /data/train_ceiling.parquet.
Stage 2 fits + evaluates -> /data/benefit_refit.json.

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run modal_benefit.py::train_ceiling --limit 10   # smoke
  modal run modal_benefit.py::train_ceiling
  modal run modal_benefit.py::benefit_refit
"""
import json
import os
import sys

import modal

from modal_app import (gen_app, util_image, gate_data, DATA, OPENAI,
                       API_REGION, _read_jsonl)

HERE = os.path.dirname(os.path.abspath(__file__))
util_bn = util_image.add_local_file(os.path.join(HERE, "modal_app.py"),
                                    "/root/modal_app.py")

ART_V3 = f"{DATA}/gate_v3_frozen.json"
CEIL = f"{DATA}/train_ceiling.parquet"
EOTH2 = f"{DATA}/eoth2"
K_EOT = 8
LAYERS2 = [14, 18, 22, 26, 30]

TRAIN_Q = [(f"{DATA}/queries.jsonl", "calib"),
           (f"{DATA}/queries_expansion.jsonl", None),
           (f"{DATA}/queries_expansion2.jsonl", None)]
TRAIN_LABELS = [(f"{DATA}/calib_features.parquet", "calib"),
                (f"{DATA}/expansion_labels.parquet", None),
                (f"{DATA}/expansion2_labels.parquet", None)]
# eval pools: (name, eoth2 tag, traces file, ceiling file, local col, ceil col)
EVALS = [
    ("internal_test", "frozen", "frozen_v3_traces.parquet",
     "eval_expert.parquet", "heard_ok", "expert_adequate"),
    ("striviaqa", "striviaqa", "striviaqa_v3_traces.parquet",
     "striviaqa_ceiling.parquet", "oab_ok", "oab_ok"),
    ("swebq", "swebq", "swebq_v3_traces.parquet",
     "swebq_ceiling.parquet", "oab_ok", "oab_ok"),
    ("sllama", "sllama", "sllama_v3_traces.parquet",
     "sllama_ceiling.parquet", "oab_ok", "oab_ok"),
    ("sdqa", "sdqa", "sdqa_v3_traces.parquet",
     "sdqa_ceiling.parquet", "heard_ok", "adequate"),
    ("sreason", "sreason", "sreason_v3_traces.parquet",
     "sreason_ceiling.parquet", "heard_ok", "adequate"),
]


@gen_app.function(image=util_bn, volumes={DATA: gate_data}, secrets=[OPENAI],
                  timeout=60 * 120, region=API_REGION)
def train_ceiling(limit: int = 0):
    """gpt-5.5 (low) answers the GOLD text of every train-pool query;
    standard judge scores it. Mirrors modal_bench.py::ceiling."""
    import asyncio
    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate
    from modal_app import EXPERT_CACHE

    qs = []
    for path, split in TRAIN_Q:
        rows = _read_jsonl(path)
        if split:
            rows = [q for q in rows if q.get("split") == split]
        qs += rows
    if limit:
        qs = qs[:limit]
    print(f">>> train ceiling: {len(qs)} gold queries -> "
          f"{escalate.EXPERT_MODEL} low", flush=True)
    answers = asyncio.run(escalate.ask_expert_many(
        [q["query"] for q in qs], concurrency=3, effort="low",
        cache_dir=EXPERT_CACHE))
    rows = [{"id": q["id"], "pool": q.get("pool"), "query": q["query"],
             "reference_answer": q.get("reference_answer"),
             "answer": a.get("answer"), "latency_s": a.get("latency_s"),
             "error": a.get("error")}
            for q, a in zip(qs, answers)]
    judged = asyncio.run(escalate.judge_many(
        [dict(r) for r in rows if r["answer"]], concurrency=8))
    adq = {r["id"]: r["adequate"] for r in judged}
    out = pd.DataFrame(rows)
    out["adequate"] = [adq.get(i) for i in out["id"]]
    if not limit:
        out.to_parquet(CEIL)
        gate_data.commit()
    ok = out["adequate"].dropna()
    print(f">>> expert-right rate {ok.mean():.3f} (n={len(ok)}, "
          f"errors {out['error'].notna().sum()})", flush=True)


def _load2(tag):
    import glob as _glob
    import numpy as np
    shards = sorted(_glob.glob(f"{EOTH2}_{tag}.shard*.npz"))
    if not shards:
        raise FileNotFoundError(f"no eoth2 shards for {tag}")
    ids, E, M, ELEN = [], [], [], []
    for s in shards:
        z = np.load(s, allow_pickle=True)
        ids += [str(x) for x in z["ids"]]
        E.append(z["H_eot"])
        M.append(z["H_mean"])
        ELEN.append(z["eot_len"])
    import numpy as np
    return (ids, np.concatenate(E), np.concatenate(M),
            np.concatenate(ELEN))


def _feat(E, M, ELEN, layers, modes):
    import numpy as np
    parts = []
    for L in layers:
        j = LAYERS2.index(L)
        for m in modes:
            if m == "eot_last":
                parts.append(E[:, j, -1, :].astype(np.float32))
            elif m == "eot_mean":
                He = E[:, j].astype(np.float32)
                ln = np.clip(ELEN.astype(np.int32), 1, K_EOT)
                mask = (np.arange(K_EOT)[None, :]
                        >= (K_EOT - ln[:, None])).astype(np.float32)
                parts.append((He * mask[:, :, None]).sum(1) / ln[:, None])
            elif m == "user_mean":
                parts.append(M[:, j].astype(np.float32))
    return np.concatenate(parts, axis=1)


@gen_app.function(image=util_bn, volumes={DATA: gate_data},
                  timeout=60 * 60, memory=32768)
def benefit_refit() -> str:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    v3 = json.load(open(ART_V3))
    w3, b3 = np.array(v3["w"]), v3["b"]
    layers, modes = v3["layer_set"], v3["modes"]

    ceil = pd.read_parquet(CEIL).set_index("id")["adequate"]

    # ---- train matrix ---------------------------------------------------
    fail = {}
    for path, split in TRAIN_LABELS:
        df = pd.read_parquet(path)
        if split:
            df = df[df["split"] == split]
        df = df[df["escalate_label"].notna()]
        fail.update(dict(zip(df["id"], df["escalate_label"].astype(int))))

    IDS, Xp = [], []
    for tag in ("frozen", "expansion", "expansion2"):
        ids, E, M, ELEN = _load2(tag)
        keep = [j for j, i in enumerate(ids)
                if i in fail and i in ceil.index
                and ceil.loc[i] is not None and not pd.isna(ceil.loc[i])]
        IDS += [ids[j] for j in keep]
        Xp.append(_feat(E[keep], M[keep], ELEN[keep], layers, modes))
        print(f"train {tag}: {len(keep)} rows", flush=True)
    X = np.concatenate(Xp)
    y_fail = np.array([fail[i] for i in IDS])
    y_bene = np.array([int(fail[i] == 1 and bool(ceil.loc[i]))
                       for i in IDS])
    print(f"train n={len(IDS)}, fail rate {y_fail.mean():.3f}, "
          f"benefit rate {y_bene.mean():.3f}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    best = None
    for C in (3e-5, 1e-4, 3e-4, 1e-3):
        oof = cross_val_predict(
            LogisticRegression(C=C, max_iter=5000), X, y_bene, cv=cv,
            method="predict_proba")[:, 1]
        a = roc_auc_score(y_bene, oof)
        print(f"  C={C}: benefit OOF AUC={a:.3f}", flush=True)
        if best is None or a > best[1]:
            best = (C, a)
    C, oof_auc = best
    clf = LogisticRegression(C=C, max_iter=5000).fit(X, y_bene)

    # ---- eval -----------------------------------------------------------
    rng = np.random.default_rng(42)

    def boot(y, s, n=10000):
        vals = []
        while len(vals) < n:
            b = rng.choice(len(y), len(y))
            if len(np.unique(y[b])) < 2:
                continue
            vals.append(roc_auc_score(y[b], s[b]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return (round(float(roc_auc_score(y, s)), 3),
                round(float(lo), 3), round(float(hi), 3))

    def pdelta(y, sn, so, n=10000):
        vals = []
        while len(vals) < n:
            b = rng.choice(len(y), len(y))
            if len(np.unique(y[b])) < 2:
                continue
            vals.append(roc_auc_score(y[b], sn[b])
                        - roc_auc_score(y[b], so[b]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return (round(float(np.mean(vals)), 3),
                round(float(lo), 3), round(float(hi), 3))

    out = {"train_n": int(len(IDS)), "C": C,
           "benefit_oof_auc": round(float(oof_auc), 3),
           "benefit_rate_train": round(float(y_bene.mean()), 3),
           "pools": {}}
    for name, tag, traces_f, ceil_f, lcol, ccol in EVALS:
        ids, E, M, ELEN = _load2(tag)
        if name == "internal_test":
            qs = _read_jsonl(f"{DATA}/queries.jsonl")
            test_ids = {q["id"] for q in qs if q.get("split") == "test"}
            keep = [j for j, i in enumerate(ids) if i in test_ids]
            ids = [ids[j] for j in keep]
            E, M, ELEN = E[keep], M[keep], ELEN[keep]
        Xe = _feat(E, M, ELEN, layers, modes)
        tr = pd.read_parquet(f"{DATA}/{traces_f}")
        loc = (tr[tr.tier == "never"].drop_duplicates("id", keep="last")
               .set_index("id")[lcol])
        ce = (pd.read_parquet(f"{DATA}/{ceil_f}")
              .drop_duplicates("id", keep="last").set_index("id")[ccol])
        lv = loc.reindex(ids).to_numpy().astype(float)
        ev = ce.reindex(ids).to_numpy().astype(float)
        k = ~(np.isnan(lv) | np.isnan(ev))
        yb = ((lv[k] == 0) & (ev[k] == 1)).astype(int)
        yf = (1 - lv[k]).astype(int)
        s3 = Xe[k] @ w3 + b3
        sb = clf.predict_proba(Xe[k])[:, 1]
        r = {"n": int(k.sum()),
             "benefit_rate": round(float(yb.mean()), 3),
             "bAUC_fail_probe": boot(yb, s3),
             "bAUC_benefit_probe": boot(yb, sb),
             "bAUC_delta": pdelta(yb, sb, s3),
             "fAUC_fail_probe": boot(yf, s3) if 0 < yf.mean() < 1 else None,
             "fAUC_benefit_probe": (boot(yf, sb)
                                    if 0 < yf.mean() < 1 else None)}
        out["pools"][name] = r
        print(f"{name:<14} n={r['n']:>4} bene-rate {r['benefit_rate']:.3f} "
              f"| bAUC fail-probe {r['bAUC_fail_probe'][0]:.3f} -> "
              f"benefit-probe {r['bAUC_benefit_probe'][0]:.3f} "
              f"(d {r['bAUC_delta'][0]:+.3f} "
              f"[{r['bAUC_delta'][1]:+.3f},{r['bAUC_delta'][2]:+.3f}]) "
              f"| fAUC {r['fAUC_fail_probe'][0] if r['fAUC_fail_probe'] else 0:.3f}"
              f"->{r['fAUC_benefit_probe'][0] if r['fAUC_benefit_probe'] else 0:.3f}",
              flush=True)

    with open(f"{DATA}/benefit_refit.json", "w") as fh:
        json.dump(out, fh, indent=1)
    gate_data.commit()
    return json.dumps(out)


@gen_app.local_entrypoint()
def run_refit():
    print(benefit_refit.remote()[:2000])
