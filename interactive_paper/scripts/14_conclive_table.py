"""8at final numbers: every cell of the concurrent-regime transfer table,
judge-matched to the turn-based main table (OAB official for
TriviaQA/WebQ/Llama, ours for SD-QA/Reasoning/frozen; VB 1-5 for
AlpacaEval), plus in-regime probe AUC per pool, the frozen speakable
subset, and random references at realized rates.

Usage (from interactive_paper/): .venv_boot\\Scripts\\python.exe
scripts\\14_conclive_table.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = Path("data")
TIERS = ["never", "conservative", "balanced", "aggressive", "always"]


def tier_table(df, col):
    out = {}
    for t in TIERS:
        g = df[df.tier == t]
        if not len(g):
            continue
        out[t] = {"n": len(g), "esc": round(float((g["mode"] ==
                  "escalated").mean()), 3),
                  "acc": round(float(g[col].mean()), 3)}
    return out


def auc_never(df, col):
    nv = df[df.tier == "never"]
    y = 1 - nv[col].astype(int)
    if y.mean() in (0, 1):
        return None
    return round(float(roc_auc_score(y, nv["eot_score"])), 3)


def main():
    results = {}
    for pool, col in [("striviaqa", "oab_ok"), ("swebq", "oab_ok"),
                      ("sllama", "oab_ok"), ("sdqa", "heard_ok"),
                      ("sreason", "heard_ok"), ("frozen", "heard_ok")]:
        df = pd.read_parquet(D / f"{pool}_conclive_traces.parquet")
        if col not in df.columns:
            print(f"!! {pool}: column {col} missing, has {list(df.columns)}")
            col = "heard_ok"
        df = df[df[col].notna()].copy()
        r = {"judge": col, "tiers": tier_table(df, col),
             "auc_in_regime": auc_never(df, col)}
        results[pool] = r

    # valpaca (VB 1-5)
    vp = pd.read_parquet(D / "valpaca_conclive_scored.parquet")
    vcol = "score"
    vp = vp[vp[vcol].notna()]
    results["valpaca"] = {"judge": "VB 1-5",
                          "tiers": tier_table(vp, vcol),
                          "auc_in_regime": None}

    # frozen speakable subset (paper's pre-registered filter)
    flagged = set(json.load(open("figures/fair_subset_audit.json")))
    fz = pd.read_parquet(D / "frozen_conclive_traces.parquet")
    fz = fz[fz["heard_ok"].notna() & ~fz["id"].isin(flagged)]
    results["frozen_speakable"] = {"judge": "heard_ok",
                                   "tiers": tier_table(fz, "heard_ok"),
                                   "auc_in_regime": auc_never(fz, "heard_ok")}

    # print LaTeX-ready rows (accuracies in %)
    name = {"striviaqa": "TriviaQA", "swebq": "WebQ", "sllama": "Llama Q.",
            "sdqa": "SD-QA", "sreason": "Reason.\\ zh"}
    order = ["striviaqa", "swebq", "sllama", "sdqa", "sreason"]
    print("\n=== concurrent transfer table (judge-matched, %) ===")
    hdr = "tier      " + "".join(f"{name[p]:>12}" for p in order)
    print(hdr)
    for t in TIERS:
        row = f"{t:<10}"
        for p in order:
            d = results[p]["tiers"].get(t)
            row += f"{d['acc']*100:>11.1f} " if d else f"{'---':>12}"
        print(row)
    print("esc@tier  " + "".join(
        f"{results[p]['tiers'].get('aggressive', {}).get('esc', 0)*100:>11.1f} "
        for p in order))
    print("AUC       " + "".join(
        f"{results[p]['auc_in_regime'] or 0:>11.3f} " for p in order))
    print("\nfrozen full:", results["frozen"]["tiers"])
    print("frozen speakable:", results["frozen_speakable"]["tiers"],
          "| AUC", results["frozen_speakable"]["auc_in_regime"])
    print("valpaca:", results["valpaca"]["tiers"])

    Path("figures/conclive_table.json").write_text(
        json.dumps(results, indent=2))
    print("\nwrote figures/conclive_table.json")


if __name__ == "__main__":
    main()
