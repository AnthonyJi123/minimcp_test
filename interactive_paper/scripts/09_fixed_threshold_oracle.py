"""P2b: (1) one absolute threshold frozen on the internal pool, applied verbatim
to every external pool; (2) expert-benefit oracle y = local-wrong AND expert-right,
with the oracle frontier at matched escalation budgets.

Runs in gold-inject space (expert answers the gold text): the ceiling parquets
cover 100% of ids, so any threshold is counterfactually evaluable. The heard view
only observes the ~50% of ids that some arm actually escalated, so a free-moving
threshold is not evaluable there.

Usage (from interactive_paper/): .venv_boot\Scripts\python.exe
scripts\09_fixed_threshold_oracle.py  (.venv_ip lacks sklearn)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "figures" / "fixed_threshold_oracle.json"
SEED = 42
B = 10000

# (paper name, traces file, ceiling file, local label col, ceiling label col).
# The label column follows the judge the paper scores that pool with: OAB's own
# judge where an official protocol exists, ours elsewhere. Mixing them is not
# allowed -- on Web Questions the two disagree by 13 points on the ceiling.
POOLS = [
    ("Speech TriviaQA", "striviaqa_v3_traces.parquet", "striviaqa_ceiling.parquet", "oab_ok", "oab_ok"),
    ("Speech Web Questions", "swebq_v3_traces.parquet", "swebq_ceiling.parquet", "oab_ok", "oab_ok"),
    ("Llama Questions", "sllama_v3_traces.parquet", "sllama_ceiling.parquet", "oab_ok", "oab_ok"),
    ("SD-QA", "sdqa_v3_traces.parquet", "sdqa_ceiling.parquet", "heard_ok", "adequate"),
    ("Reasoning QA (zh)", "sreason_v3_traces.parquet", "sreason_ceiling.parquet", "heard_ok", "adequate"),
]
INTERNAL = ("internal frozen test", "frozen_v3_traces.parquet", "eval_expert.parquet",
            "heard_ok", "expert_adequate")


def load(traces_f, ceiling_f, local_col, ceil_col):
    """Per-id gate score, local correctness, expert-on-gold-text correctness."""
    tr = pd.read_parquet(DATA / traces_f)
    # eot_score is identical across the three gated arms; the never arm was a
    # separate audio pass with its own jitter, so never read the score off it.
    score = tr[tr.tier == "balanced"].set_index("id")["eot_score"]
    local = tr[tr.tier == "never"].set_index("id")[local_col].astype(int)
    ce = pd.read_parquet(DATA / ceiling_f).set_index("id")
    expert = ce[ceil_col].astype(int)
    df = pd.DataFrame({"score": score, "local": local, "expert": expert.reindex(score.index)})
    assert not df.isna().any().any(), f"{traces_f}: incomplete join"
    return df


def acc_at(df, esc_mask):
    """Gold-inject accuracy: escalated ids take the expert's outcome."""
    return float(np.where(esc_mask, df.expert.values, df.local.values).mean())


def random_acc(df, rate):
    """Expected accuracy of escalating a uniformly random `rate` fraction."""
    return float((1 - rate) * df.local.mean() + rate * df.expert.mean())


def oracle_acc(df, rate):
    """Escalate the highest-benefit ids first: y=1 gains +1, harmful ids lose 1."""
    n = len(df)
    k = int(round(rate * n))
    benefit = np.sort(df.expert.values - df.local.values)[::-1]
    return float(df.local.mean() + benefit[:k].sum() / n)


def boot_delta(df, esc_mask, rate, rng):
    """Paired bootstrap over ids: gate accuracy minus random-at-matched-rate."""
    n = len(df)
    gate_row = np.where(esc_mask, df.expert.values, df.local.values)
    loc, exp = df.local.values, df.expert.values
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        rnd = (1 - rate) * loc[idx].mean() + rate * exp[idx].mean()
        out[b] = gate_row[idx].mean() - rnd
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    rng = np.random.default_rng(SEED)

    # --- freeze one absolute threshold on the internal pool -------------------
    thr_all = json.loads((DATA / "gate_v3_thresholds_corrected.json").read_text())
    # the live internal run used threshold_old; threshold_corrected is a post-hoc
    # recomputation that only reproduces 95% of the realized modes.
    TAU = float(thr_all["frozen"]["threshold_old"]["balanced"])

    internal = load(*INTERNAL[1:])
    report = {
        "tau": TAU,
        "tau_source": "internal frozen pool, balanced tier (threshold_old), as deployed",
        "internal_rate_at_tau": float((internal.score >= TAU).mean()),
        "seed": SEED,
        "bootstrap": B,
        "view": "gold-inject (expert answers the gold text)",
        "pools": [],
    }

    for name, tf, cf, lc, cc in POOLS + [INTERNAL]:
        df = load(tf, cf, lc, cc)
        esc = (df.score >= TAU).values
        rate = float(esc.mean())

        y_bene = ((1 - df.local) * df.expert).values  # local wrong AND expert right
        y_fail = (1 - df.local).values

        row = {
            "pool": name,
            "n": int(len(df)),
            "fixed_tau": {
                "rate": rate,
                "acc": acc_at(df, esc),
                "random_acc_matched": random_acc(df, rate),
                "oracle_acc_matched": oracle_acc(df, rate),
            },
            "floor_local": float(df.local.mean()),
            "ceiling_expert": float(df.expert.mean()),
            "benefit_base_rate": float(y_bene.mean()),
            "auc_vs_local_fail": float(roc_auc_score(y_fail, df.score)) if 0 < y_fail.mean() < 1 else None,
            "auc_vs_expert_benefit": float(roc_auc_score(y_bene, df.score)) if 0 < y_bene.mean() < 1 else None,
        }
        lo, hi = boot_delta(df, esc, rate, rng)
        row["fixed_tau"]["delta_vs_random_ci"] = [lo, hi]
        row["fixed_tau"]["beats_random"] = bool(lo > 0)

        # gate vs oracle vs random across the deployed budget grid
        row["frontier"] = [
            {
                "budget": b,
                "gate": acc_at(df, (df.score >= np.quantile(df.score, 1 - b)).values),
                "random": random_acc(df, b),
                "oracle": oracle_acc(df, b),
            }
            for b in (0.15, 0.30, 0.50)
        ]
        report["pools"].append(row)

    OUT.write_text(json.dumps(report, indent=2))

    # --- console table -------------------------------------------------------
    print(f"frozen tau = {TAU:.4f}  (internal realized rate {report['internal_rate_at_tau']:.3f})\n")
    print(f"{'pool':<22} {'n':>4} {'rate':>6} {'acc':>6} {'rand':>6} {'d95CI':>16} "
          f"{'AUCfail':>8} {'AUCbene':>8} {'bene%':>6}")
    for r in report["pools"]:
        f = r["fixed_tau"]
        ci = f"[{f['delta_vs_random_ci'][0]:+.3f},{f['delta_vs_random_ci'][1]:+.3f}]"
        af = "  n/a  " if r["auc_vs_local_fail"] is None else f"{r['auc_vs_local_fail']:.3f}"
        ab = "  n/a  " if r["auc_vs_expert_benefit"] is None else f"{r['auc_vs_expert_benefit']:.3f}"
        print(f"{r['pool']:<22} {r['n']:>4} {f['rate']:>6.3f} {f['acc']:>6.3f} "
              f"{f['random_acc_matched']:>6.3f} {ci:>16} {af:>8} {ab:>8} "
              f"{r['benefit_base_rate']:>6.3f}")

    print("\nfrontier (gate / random / oracle) at matched budgets")
    for r in report["pools"]:
        cells = "  ".join(f"{d['budget']:.0%}: {d['gate']:.3f}/{d['random']:.3f}/{d['oracle']:.3f}"
                          for d in r["frontier"])
        print(f"{r['pool']:<22} {cells}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
