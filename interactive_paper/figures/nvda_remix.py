# -*- coding: utf-8 -*-
"""NVDA escalation-tier re-mix (2026-08-26): tab:transfer 的 NVDA 版。

同 8ad rate_curve 的重混逻辑: 按 NVDA probe 分数 top-r 换成 expert 结果,
其余用 NVDA 本地结果; judge 与 tab:transfer 逐池对齐 (OAB 池=官方 gpt-4o
judge, sdqa=我们的 judge, valpaca=VoiceBench 1-5)。expert 结果两个口径:
  A (主): 全体 expert 答案 (实测+补跑) 统一由官方 judge 直接判 —
     无转述税, 跨模型最干净 (NVDA 的 relay 通道离线不可测);
  B (对照): 有实测 relay 结果的 id 用 live 实测 (含 MiniCPM 转述税),
     其余落回 A — 与 tab:transfer 口径最近。
输入 (先 modal volume get): nvda_scores.parquet, nvda_{pool}.parquet
(oab_ok 列), nvda_expert_outcomes.parquet, nvda_scores_valpaca.parquet,
nvda_valpaca.parquet; 本地 {pool}_v3_traces.parquet / valpaca_v3_scored。
输出: data/nvda_remix.json + 打印的表。
"""
import json

import numpy as np
import pandas as pd

D = "../data"
RATES = (0.0, 0.15, 0.30, 0.50, 1.0)
POOLS = ("striviaqa", "swebq", "sllama", "sdqa")

S = pd.read_parquet(f"{D}/nvda_scores.parquet")
EXP = pd.read_parquet(f"{D}/nvda_expert_outcomes.parquet")
out = {}


def remix(d, rates):
    d = d.sort_values("score", ascending=False).reset_index(drop=True)
    res = {}
    for r in rates:
        k = int(round(r * len(d)))
        res[r] = float(np.mean(np.concatenate(
            [d["exp"][:k].values, d["loc"][k:].values])))
    return res, d


for pool in POOLS:
    nv = pd.read_parquet(f"{D}/nvda_{pool}.parquet").set_index("id")
    loc_col = "oab_ok" if "oab_ok" in nv.columns and pool != "sdqa" \
        else "adequate"
    e = EXP[EXP["pool"] == pool].set_index("id")

    d = S[S.pool == pool][["id", "score"]].copy()
    d["loc"] = d["id"].map(nv[loc_col]).astype(float)
    d["exp"] = d["id"].map(e["expert_ok"]).astype(float)
    d["src"] = d["id"].map(e["src"])
    n0 = len(d)
    d = d.dropna(subset=["loc", "exp", "score"])
    if len(d) < n0:
        print(f"!! {pool}: dropped {n0 - len(d)} ids w/ missing outcome")

    # 口径 B: 实测 relay 结果覆盖 (escalated 行的最终口径列)
    tr = pd.read_parquet(f"{D}/{pool}_v3_traces.parquet")
    okc = "oab_ok" if "oab_ok" in tr.columns else "heard_ok"
    meas = (tr[tr["mode"] == "escalated"].drop_duplicates("id")
            .set_index("id")[okc])
    d["expB"] = d["id"].map(meas).fillna(d["exp"]).astype(float)

    dA = d.rename(columns={"exp": "exp"})
    resA, dd = remix(dA, RATES)
    resB, _ = remix(d.assign(exp=d["expB"]), RATES)
    rnd = {r: float((1 - r) * d["loc"].mean() + r * d["exp"].mean())
           for r in RATES}
    # AUC of the probe against the OFFICIAL local-fail label
    from sklearn.metrics import roc_auc_score
    y = 1 - d["loc"]
    auc = float(roc_auc_score(y, d["score"])) if y.nunique() > 1 else None
    out[pool] = {
        "n": len(d), "judge": "OAB" if pool != "sdqa" else "ours",
        "local_acc": float(d["loc"].mean()),
        "expert_acc_transcript": float(d["exp"].mean()),
        "auc_official_label": auc,
        "tiers_A_relayfree": resA, "tiers_B_measured_relay": resB,
        "random_matched": rnd,
        "n_measured_relay": int(d["id"].map(meas).notna().sum()),
    }

# ---- valpaca (VB 1-5, mean score) ----------------------------------------
try:
    sv = pd.read_parquet(f"{D}/nvda_scores_valpaca.parquet")
    e = EXP[EXP["pool"] == "valpaca"].set_index("id")
    d = sv[["id", "score", "vb_score"]].rename(
        columns={"vb_score": "loc"}).copy()
    d["exp"] = d["id"].map(e["expert_score"]).astype(float)
    tr = pd.read_parquet(f"{D}/valpaca_v3_scored.parquet")
    meas = (tr[tr["mode"] == "escalated"].drop_duplicates("id")
            .set_index("id")["score"])
    d["expB"] = d["id"].map(meas).fillna(d["exp"]).astype(float)
    n0 = len(d)
    d = d.dropna(subset=["loc", "exp", "score"])
    if len(d) < n0:
        print(f"!! valpaca: dropped {n0 - len(d)} ids w/ missing outcome")
    resA, _ = remix(d, RATES)
    resB, _ = remix(d.assign(exp=d["expB"]), RATES)
    rnd = {r: float((1 - r) * d["loc"].mean() + r * d["exp"].mean())
           for r in RATES}
    out["valpaca"] = {
        "n": len(d), "judge": "VB 1-5",
        "local_acc": float(d["loc"].mean()),
        "expert_acc_transcript": float(d["exp"].mean()),
        "tiers_A_relayfree": resA, "tiers_B_measured_relay": resB,
        "random_matched": rnd,
        "n_measured_relay": int(d["id"].map(meas).notna().sum()),
    }
except FileNotFoundError as e:
    print(f"!! valpaca skipped: {e}")

with open(f"{D}/nvda_remix.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

hdr = f"{'pool':<11}{'judge':<7}{'n':>4} {'0%':>6} {'15%':>6} {'30%':>6} " \
      f"{'50%':>6} {'100%':>6} {'AUC':>6}  rnd50%"
print("\n=== A 口径 (relay-free, 主表) ===")
print(hdr)
for p, o in out.items():
    t = o["tiers_A_relayfree"]
    auc = o.get("auc_official_label")
    print(f"{p:<11}{o['judge']:<7}{o['n']:>4} " +
          " ".join(f"{t[r]:>6.3f}" for r in RATES) +
          (f" {auc:>6.3f}" if auc else "    ---") +
          f"  {o['random_matched'][0.5]:.3f}")
print("\n=== B 口径 (measured relay where available) ===")
for p, o in out.items():
    t = o["tiers_B_measured_relay"]
    print(f"{p:<11}" + " ".join(f"{t[r]:>6.3f}" for r in RATES) +
          f"   (measured {o['n_measured_relay']}/{o['n']})")
