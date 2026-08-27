"""Concurrent-prefill arm analysis: does the frozen probe read survive
when the EOT read happens while the model is mid-generation, with the
target audio interleaved into the same KV stream?

Mirrors _duplex_analyze.py's construction exactly so the result slots
into app:duplexval's table (frozen .866 / clean .871 / overlap .856):
  label    = 1 - max(heard_ok) over each id's local rows (v3 traces)
  s_frozen = mean eot_score over tiers (headless surrogate)
Probe vintage note: concurrent scores come from the v3 gate artifact
(gate_v3_frozen.json); 8aq's clean/overlap used the deployed v4 gate
(same features/layer; calibration adds FreshQA rows).

Usage (from interactive_paper/): .venv_boot\\Scripts\\python.exe
scripts\\11_concurrent_auc.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = Path("data")
BAL_THR = 0.613  # gate_v3_frozen.json balanced (as run)


def main():
    tr = pd.read_parquet(D / "frozen_v3_traces.parquet")
    loc = tr[tr["mode"] == "local"].groupby("id")["heard_ok"].max()
    label = (1 - loc.astype(int)).rename("fail")
    froz = tr.groupby("id")["eot_score"].mean().rename("s_frozen")

    rows = []
    for p in sorted(D.glob("frozen_concurrent_traces.jsonl.shard*")):
        rows += [json.loads(l) for l in open(p, encoding="utf-8")
                 if l.strip()]
    cc = pd.DataFrame(rows).drop_duplicates("id", keep="last").set_index("id")
    print(f"concurrent rows: {len(cc)}; "
          f"gen_active_at_eot: {int(cc['gen_active_at_eot'].sum())}; "
          f"full coverage: "
          f"{int((cc['n_concurrent'] == cc['n_chunks']).sum())}")

    df = pd.concat([label, froz, cc["eot_score"].rename("s_conc"),
                    cc["gen_active_at_eot"]], axis=1, join="inner").dropna()
    print(f"joined: {len(df)} ids, fail rate {df['fail'].mean():.3f}")
    print(f"score shift: frozen mean {df['s_frozen'].mean():.3f} -> "
          f"concurrent mean {df['s_conc'].mean():.3f}")
    print(f"score corr (pearson): "
          f"{np.corrcoef(df['s_frozen'], df['s_conc'])[0, 1]:.3f}")

    rng = np.random.default_rng(0)

    def auc_ci(y, s, n_boot=2000):
        a = roc_auc_score(y, s)
        y, s = np.asarray(y), np.asarray(s)
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) < 2:
                continue
            bs.append(roc_auc_score(y[i], s[i]))
        return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    for c in ("s_frozen", "s_conc"):
        a, lo, hi = auc_ci(df["fail"], df[c])
        print(f"AUC {c:9s}: {a:.3f}  [{lo:.3f}, {hi:.3f}]")

    sub = df[df["gen_active_at_eot"]]
    if len(sub) < len(df):
        a, lo, hi = auc_ci(sub["fail"], sub["s_conc"])
        print(f"AUC s_conc gen-active-only (n={len(sub)}): "
              f"{a:.3f}  [{lo:.3f}, {hi:.3f}]")

    # paired bootstrap on the AUC delta vs the headless surrogate
    y = df["fail"].to_numpy()
    s0, s1 = df["s_frozen"].to_numpy(), df["s_conc"].to_numpy()
    ds = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        ds.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s0[i]))
    print(f"dAUC concurrent - frozen: {np.mean(ds):+.3f} "
          f"[{np.percentile(ds, 2.5):+.3f}, {np.percentile(ds, 97.5):+.3f}]")

    # decision agreement at the deployed balanced threshold, and with the
    # concurrent scores requantiled to the same realized fire rate
    d0 = df["s_frozen"] >= BAL_THR
    d1 = df["s_conc"] >= BAL_THR
    print(f"fire rate: frozen {d0.mean():.3f}, concurrent@same-thr "
          f"{d1.mean():.3f}; agreement {(d0 == d1).mean():.3f}")
    thr_q = np.quantile(df["s_conc"], 1 - d0.mean())
    d1q = df["s_conc"] >= thr_q
    print(f"requantiled thr {thr_q:.3f}: fire {d1q.mean():.3f}; "
          f"agreement {(d0 == d1q).mean():.3f}")


if __name__ == "__main__":
    main()
