"""W2+D1 (2026-08-25): paired-bootstrap CIs for the v3 live table
(paper Table tab:live), full pool (n=240) and speakable subset (n=218).

Reproduces live_v3_figures.py's outcome construction (heard_ok per row;
gold-inject = expert_adequate on escalated rows) and adds the delta-vs-floor
and channel-cost CIs the paper table needs. Point estimates must match
figures/live_dualview.json and figures/fair_figures.json exactly.

Inputs: data/frozen_v3_traces.parquet, data/eval_expert.parquet,
figures/fair_subset_audit.json. CPU-only. B=10k, seed 42.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["never", "conservative", "balanced", "aggressive"]
B, SEED = 10_000, 42


def main():
    df = pd.read_parquet(ROOT / "data" / "frozen_v3_traces.parquet")
    exp = pd.read_parquet(ROOT / "data" / "eval_expert.parquet").set_index("id")
    flagged = set(json.load(open(ROOT / "figures" / "fair_subset_audit.json")))

    piv_h, piv_g = {}, {}
    for arm in ARMS:
        d = df[df["tier"] == arm].set_index("id")
        h = d["heard_ok"].astype(int)
        esc = d["mode"].eq("escalated")
        gold = np.where(esc, [int(bool(exp.loc[i, "expert_adequate"]))
                              for i in d.index], h)
        piv_h[arm], piv_g[arm] = h, pd.Series(gold, index=d.index)

    ids_all = np.array(piv_h["never"].index)
    rng = np.random.default_rng(SEED)
    for label, ids in [("FULL", ids_all),
                       ("FAIR", np.array([i for i in ids_all
                                          if i not in flagged]))]:
        n = len(ids)
        print(f"== {label} n={n}")
        H = np.stack([piv_h[a].loc[ids].values for a in ARMS])
        G = np.stack([piv_g[a].loc[ids].values for a in ARMS])
        esc_rates = [df[(df["tier"] == a) & (df["id"].isin(ids))]["mode"]
                     .eq("escalated").mean() for a in ARMS]
        idx = rng.integers(0, n, (B, n))
        hm, gm = H[:, idx].mean(2), G[:, idx].mean(2)
        for k, a in enumerate(ARMS):
            heard, gold = H[k].mean(), G[k].mean()
            dh, cc = hm[k] - hm[0], gm[k] - hm[k]
            print(f"{a:13s} esc={esc_rates[k]:.3f} heard={heard:.3f} "
                  f"CI[{np.percentile(hm[k],2.5):.3f},{np.percentile(hm[k],97.5):.3f}] "
                  f"dVsFloor={heard-H[0].mean():+.3f} "
                  f"CI[{np.percentile(dh,2.5):+.3f},{np.percentile(dh,97.5):+.3f}] "
                  f"gold={gold:.3f} "
                  f"CI[{np.percentile(gm[k],2.5):.3f},{np.percentile(gm[k],97.5):.3f}] "
                  f"chan={gold-heard:+.3f} "
                  f"CI[{np.percentile(cc,2.5):+.3f},{np.percentile(cc,97.5):+.3f}]")
        alw = np.array([int(bool(exp.loc[i, "expert_adequate"])) for i in ids])
        ab = alw[idx].mean(1)
        print(f"{'always(synth)':13s} esc=1.000 gold={alw.mean():.3f} "
              f"CI[{np.percentile(ab,2.5):.3f},{np.percentile(ab,97.5):.3f}]")


if __name__ == "__main__":
    main()
