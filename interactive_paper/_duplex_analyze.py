"""Analysis for the duplex-validation sweep (todo P2 / reviewer #1).

AUC of the frozen v4 gate score against frozen local-failure labels,
same 240 ids, three read regimes:
  frozen  = the paper's headless EOT-prefill loop (frozen_v3_traces)
  clean   = deployed voice loop, talker head ON, clean turns
  overlap = deployed voice loop, query spoken OVER the talker,
            barge-in seeds the turn

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _duplex_analyze.py::t
"""
import modal

app = modal.App("duplex-analyze")
vol = modal.Volume.from_name("gate-data")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("pandas", "pyarrow", "numpy", "scikit-learn"))


@app.function(image=img, volumes={"/data": vol}, timeout=600)
def t():
    import json

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    tr = pd.read_parquet("/data/frozen_v3_traces.parquet")
    print("tiers:", tr["tier"].value_counts().to_dict())

    # local-failure label per id: heard_ok of non-escalated (mode=local)
    # rows; a query that escalated in every tier has no local label
    loc = tr[tr["mode"] == "local"].groupby("id")["heard_ok"].max()
    label = (1 - loc.astype(int)).rename("fail")
    print(f"labels: {len(label)} ids, fail rate {label.mean():.3f}")

    # frozen surrogate score per id (mean over tiers; near-identical)
    froz = tr.groupby("id")["eot_score"].mean().rename("s_frozen")

    arms = {}
    for arm in ("clean", "overlap"):
        rows = [json.loads(x) for x in
                open(f"/data/duplex_sweep/{arm}.jsonl") if x.strip()]
        arms[arm] = pd.DataFrame(rows).drop_duplicates(
            "id", keep="last").set_index("id")
        print(f"{arm}: {len(arms[arm])} records")

    ov = arms["overlap"]
    print(f"overlap: barged {int(ov['barged'].sum())}/{len(ov)}, "
          f"resumes on {int((ov['n_resumes'] > 0).sum())} pairs")

    df = pd.concat([label, froz,
                    arms["clean"]["eot_score"].rename("s_clean"),
                    ov["eot_score"].rename("s_overlap"),
                    ov["barged"]], axis=1, join="inner").dropna(
        subset=["fail", "s_frozen", "s_clean", "s_overlap"])
    print(f"joined: {len(df)} ids with all three scores + label")

    rng = np.random.default_rng(0)

    def auc_ci(y, s, n_boot=2000):
        a = roc_auc_score(y, s)
        bs = []
        y = np.asarray(y)
        s = np.asarray(s)
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) < 2:
                continue
            bs.append(roc_auc_score(y[i], s[i]))
        return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    out = {}
    for c in ("s_frozen", "s_clean", "s_overlap"):
        a, lo, hi = auc_ci(df["fail"], df[c])
        out[c] = f"{a:.3f} [{lo:.3f}, {hi:.3f}]"
        print(f"AUC {c:10s}: {a:.3f}  [{lo:.3f}, {hi:.3f}]")
    b = df[df["barged"] == True]  # noqa: E712
    a, lo, hi = auc_ci(b["fail"], b["s_overlap"])
    print(f"AUC overlap barged-only (n={len(b)}): "
          f"{a:.3f}  [{lo:.3f}, {hi:.3f}]")

    # paired bootstrap on AUC deltas vs frozen
    for c in ("s_clean", "s_overlap"):
        ds = []
        y = df["fail"].to_numpy()
        s0 = df["s_frozen"].to_numpy()
        s1 = df[c].to_numpy()
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) < 2:
                continue
            ds.append(roc_auc_score(y[i], s1[i])
                      - roc_auc_score(y[i], s0[i]))
        print(f"dAUC {c} - frozen: {np.mean(ds):+.3f} "
              f"[{np.percentile(ds, 2.5):+.3f}, "
              f"{np.percentile(ds, 97.5):+.3f}]")

    # score stability across regimes
    print(f"corr(frozen, clean)   r={df['s_frozen'].corr(df['s_clean']):.3f}")
    print(f"corr(frozen, overlap) r={df['s_frozen'].corr(df['s_overlap']):.3f}")
    print(f"corr(clean, overlap)  r={df['s_clean'].corr(df['s_overlap']):.3f}")

    # would the balanced threshold make the same decision?
    thr = json.load(open("/data/midlayer_gate_audio_v4.json")
                    )["eot_thresholds"]["balanced"]
    for c in ("s_clean", "s_overlap"):
        agree = ((df[c] >= thr) == (df["s_frozen"] >= thr)).mean()
        fire = (df[c] >= thr).mean()
        print(f"{c}: fire@balanced {fire:.3f}, "
              f"decision agreement vs frozen {agree:.3f}")

    # timing
    for arm in ("clean", "overlap"):
        d = arms[arm]
        r = d["eot_read_ms"].dropna()
        f = d["first_audio_client_ms"].dropna()
        print(f"{arm}: eot_read_ms p50={r.median():.0f} "
              f"p95={r.quantile(.95):.0f} max={r.max():.0f}; "
              f"eot->first_audio_ms n={len(f)} p50={f.median():.0f} "
              f"p95={f.quantile(.95):.0f}")
