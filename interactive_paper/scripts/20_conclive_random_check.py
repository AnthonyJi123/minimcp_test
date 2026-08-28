# -*- coding: utf-8 -*-
"""P1 review fix: matched-random references + permutation tests for
tab:conclive (the concurrent-regime full-loop table).

Convention follows 18_nvda_random_check.py / fig:dualview: for each pool
and each gated tier, at that tier's REALIZED escalation count k, draw
20k random id-subsets of size k; remix accuracy = always-arm outcome on
the subset, never-arm outcome elsewhere (per id, paired). Report the
random EV, MC 95% CI, and p = P(random >= measured gate arm). Also a
bootstrap 95% CI for each measured arm.

Usage (from interactive_paper/):
  .venv_boot\Scripts\python.exe scripts\20_conclive_random_check.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data")
NPERM = 20000
NBOOT = 10000
rng = np.random.default_rng(8)

POOLS = [("striviaqa", "oab_ok"), ("swebq", "oab_ok"), ("sllama", "oab_ok"),
         ("sdqa", "heard_ok"), ("sreason", "heard_ok"),
         ("frozen", "heard_ok"), ("valpaca", "score")]
GATED = ["conservative", "balanced", "aggressive"]


def load(pool, col):
    if pool == "valpaca":
        df = pd.read_parquet(D / "valpaca_conclive_scored.parquet")
    else:
        df = pd.read_parquet(D / f"{pool}_conclive_traces.parquet")
    df = df[df[col].notna()].copy()
    return df


def analyze(df, col, tag):
    nv = df[df.tier == "never"].set_index("id")[col].astype(float)
    al = df[df.tier == "always"].set_index("id")[col].astype(float)
    ids = nv.index.intersection(al.index)
    loc, exp = nv.loc[ids].values, al.loc[ids].values
    n = len(ids)
    out = {"n": n, "never": round(loc.mean(), 4),
           "always": round(exp.mean(), 4), "tiers": {}}
    for t in GATED:
        g = df[df.tier == t]
        k = int((g["mode"] == "escalated").sum())
        acc = float(g[col].mean())
        # measured-arm bootstrap CI
        vals = g[col].astype(float).values
        bs = rng.choice(vals, (NBOOT, len(vals)), replace=True).mean(axis=1)
        blo, bhi = np.percentile(bs, [2.5, 97.5])
        # matched-rate random remix null
        null = np.empty(NPERM)
        for i in range(NPERM):
            m = np.zeros(n, bool)
            m[rng.choice(n, k, replace=False)] = True
            null[i] = np.where(m, exp, loc).mean()
        p = (np.sum(null >= acc) + 1) / (NPERM + 1)
        lo, hi = np.percentile(null, [2.5, 97.5])
        out["tiers"][t] = {
            "k": k, "rate": round(k / len(g), 3), "acc": round(acc, 4),
            "acc_ci": [round(blo, 4), round(bhi, 4)],
            "rand_ev": round((1 - k / n) * loc.mean() + (k / n) * exp.mean(), 4),
            "rand_mc": round(null.mean(), 4),
            "rand_ci": [round(lo, 4), round(hi, 4)],
            "p_rand_ge_gate": float(f"{p:.3g}")}
        print(f"{tag:<18} {t:<13} k={k:>3} ({k/len(g):>5.1%}) "
              f"acc={acc:.3f} [{blo:.3f},{bhi:.3f}] "
              f"rand={null.mean():.3f} [{lo:.3f},{hi:.3f}] p={p:.2e}")
    return out


def main():
    results = {}
    for pool, col in POOLS:
        df = load(pool, col)
        results[pool] = analyze(df, col, pool)
    # frozen speakable subset
    flagged = set(json.load(open("figures/fair_subset_audit.json")))
    fz = load("frozen", "heard_ok")
    fz = fz[~fz["id"].isin(flagged)]
    results["frozen_speakable"] = analyze(fz, "heard_ok", "frozen_speakable")
    Path("figures/conclive_random.json").write_text(
        json.dumps(results, indent=2))
    print("\nwrote figures/conclive_random.json")


if __name__ == "__main__":
    main()
