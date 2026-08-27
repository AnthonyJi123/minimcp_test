# -*- coding: utf-8 -*-
"""8av check: nvda random 行进表前的验证。

1) 从 per-query 数据复算 local/expert/tiers_A/random_matched,
   与 data/nvda_remix.json 逐值对比 (应完全一致);
2) 置换检验: 每个 pool x budget(15/30/50), 随机等预算子集重混 20000 次,
   p = P(random remix >= gate remix); 另给 random 的 MC 均值/95% CI,
   以及 4-QA-pool 平均口径 (表里 avg 列) 的联合置换检验。
"""
import json

import numpy as np
import pandas as pd

D = "data"
RATES = (0.15, 0.30, 0.50)
POOLS = ("striviaqa", "swebq", "sllama", "sdqa")
NPERM = 20000
rng = np.random.default_rng(8)

ref = json.load(open(f"{D}/nvda_remix.json"))
EXP = pd.read_parquet(f"{D}/nvda_expert_outcomes.parquet")
S = pd.read_parquet(f"{D}/nvda_scores.parquet")

pool_d = {}
for pool in POOLS:
    nv = pd.read_parquet(f"{D}/nvda_{pool}.parquet").set_index("id")
    loc_col = "oab_ok" if "oab_ok" in nv.columns and pool != "sdqa" \
        else "adequate"
    e = EXP[EXP["pool"] == pool].set_index("id")
    d = S[S.pool == pool][["id", "score"]].copy()
    d["loc"] = d["id"].map(nv[loc_col]).astype(float)
    d["exp"] = d["id"].map(e["expert_ok"]).astype(float)
    d = d.dropna(subset=["loc", "exp", "score"])
    d = d.sort_values("score", ascending=False).reset_index(drop=True)
    pool_d[pool] = d

print("== 1) 复算 vs nvda_remix.json ==")
bad = 0
for pool, d in pool_d.items():
    r = ref[pool]
    checks = {
        "n": (len(d), r["n"]),
        "local": (d["loc"].mean(), r["local_acc"]),
        "expert": (d["exp"].mean(), r["expert_acc_transcript"]),
    }
    for rate in RATES:
        k = int(round(rate * len(d)))
        gate = np.concatenate([d["exp"][:k], d["loc"][k:]]).mean()
        rnd = (1 - rate) * d["loc"].mean() + rate * d["exp"].mean()
        checks[f"gate@{rate}"] = (gate, r["tiers_A_relayfree"][str(rate)])
        checks[f"rnd@{rate}"] = (rnd, r["random_matched"][str(rate)])
    for name, (mine, theirs) in checks.items():
        ok = abs(mine - theirs) < 1e-9
        bad += not ok
        if not ok:
            print(f"  MISMATCH {pool} {name}: {mine:.6f} vs {theirs:.6f}")
print("  all match" if bad == 0 else f"  {bad} mismatches")

print("\n== 2) 置换检验 (NPERM=%d) ==" % NPERM)
print(f"{'pool':<10} {'rate':>5} {'gate':>6} {'rnd(EV)':>8} "
      f"{'rnd MC mean [95% CI]':>22} {'p(rnd>=gate)':>13}")
perms = {}   # (pool, rate) -> null accuracy draws
for pool, d in pool_d.items():
    loc, exp = d["loc"].values, d["exp"].values
    n = len(d)
    for rate in RATES:
        k = int(round(rate * n))
        gate = np.concatenate([exp[:k], loc[k:]]).mean()
        null = np.empty(NPERM)
        for i in range(NPERM):
            idx = rng.choice(n, k, replace=False)
            m = np.zeros(n, bool); m[idx] = True
            null[i] = np.where(m, exp, loc).mean()
        perms[pool, rate] = null
        p = (np.sum(null >= gate) + 1) / (NPERM + 1)
        lo, hi = np.percentile(null, [2.5, 97.5])
        print(f"{pool:<10} {rate:>5} {gate*100:6.1f} "
              f"{((1-rate)*loc.mean()+rate*exp.mean())*100:8.1f} "
              f"{null.mean()*100:6.1f} [{lo*100:.1f}, {hi*100:.1f}]"
              f" {p:13.2e}")

print("\n== 3) 4-pool 平均口径 (表 avg 列) ==")
for rate in RATES:
    gate_avg = np.mean([np.concatenate(
        [pool_d[p]["exp"][:int(round(rate*len(pool_d[p])))],
         pool_d[p]["loc"][int(round(rate*len(pool_d[p]))):]]).mean()
        for p in POOLS])
    null_avg = np.mean([perms[p, rate] for p in POOLS], axis=0)
    p = (np.sum(null_avg >= gate_avg) + 1) / (NPERM + 1)
    lo, hi = np.percentile(null_avg, [2.5, 97.5])
    print(f"rate {rate}: gate avg {gate_avg*100:.1f} vs random MC "
          f"{null_avg.mean()*100:.1f} [{lo*100:.1f}, {hi*100:.1f}], "
          f"p={p:.2e}")
