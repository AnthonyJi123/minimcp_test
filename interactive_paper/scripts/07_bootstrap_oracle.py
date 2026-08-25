"""D2 statistics hardening: paired bootstrap for the gate-vs-pool-oracle
tradeoff-area residual (+0.012, RESULTS "Pool-oracle baseline" 2026-07-29).

Reproduces eval_assemble's area computation locally (same trapezoid over the
threshold-swept curve, same random-expectation reference line), then resamples
the 240 test queries with replacement, recomputing BOTH areas per replicate
(pool-oracle scores stay fixed from calib — they are part of the router
definition, not the test data). Reports the point estimates (must match
+0.054 / +0.042 / +0.012), the 95% percentile CI of the paired delta, and
one/two-sided bootstrap p-values.

CPU-only, no Modal. Inputs: data/calib_features.parquet, data/gate_config.json,
data/eval_expert.parquet, data/eval_paraphrase.parquet (pulled from gate-data).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gate import Probe  # noqa: E402

SEED = 42
B = 10_000


def area_vs_random(scores, s, outcome):
    """Trapezoid area between the threshold-swept hybrid curve and the
    random-escalation expectation line, exactly as in eval_assemble."""
    small_acc, overall = s.mean(), outcome.mean()
    ts = np.concatenate([[np.inf], np.unique(scores)[::-1], [-np.inf]])
    pts = []
    for t in ts:
        esc = scores >= t
        pts.append((esc.mean(), np.where(esc, outcome, s).mean()))
    pts = np.array(pts)
    order = np.argsort(pts[:, 0])
    rate, acc = pts[order, 0], pts[order, 1]
    rand = (1 - rate) * small_acc + rate * overall
    diff = acc - rand
    return float(np.sum((diff[1:] + diff[:-1]) / 2 * np.diff(rate)))


def main():
    cfg = json.load(open(ROOT / "data" / "gate_config.json"))
    probe = Probe.from_config(cfg)
    df = pd.read_parquet(ROOT / "data" / "calib_features.parquet")
    test = df[df["split"] == "test"].reset_index(drop=True)
    exp = pd.read_parquet(ROOT / "data" / "eval_expert.parquet").set_index("id")
    para = pd.read_parquet(ROOT / "data" / "eval_paraphrase.parquet").set_index("id")

    def b(x):
        return 1 if x is True or x == 1 else 0

    ids = test["id"].values
    s = np.array([b(x) for x in test["adequate"].values])
    e = np.array([b(exp.loc[i, "expert_adequate"]) for i in ids])
    gate_scores = np.array([probe.score(list(h)) for h in test["h_prompt"]])
    pools = test["pool"].values

    calib = df[df["split"] == "calib"]
    pool_fail = {
        pl: float(np.mean([1 - b(x)
                           for x in calib.loc[calib["pool"] == pl, "adequate"]]))
        for pl in np.unique(pools)
    }
    oracle_scores = np.array([pool_fail[pl] for pl in pools])

    n = len(test)
    print(f"test n={n} | small={s.mean():.3f} | expert={e.mean():.3f}")
    print("calib pool fail rates:",
          {k: round(v, 3) for k, v in sorted(pool_fail.items(),
                                             key=lambda kv: -kv[1])})

    a_gate = area_vs_random(gate_scores, s, e)
    a_oracle = area_vs_random(oracle_scores, s, e)
    print(f"\npoint estimates: gate {a_gate:+.4f} | oracle {a_oracle:+.4f} | "
          f"delta {a_gate - a_oracle:+.4f}")
    print("(reference: RESULTS 2026-07-29 = +0.054 / +0.042 / +0.012)")

    rng = np.random.default_rng(SEED)
    deltas = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        deltas[i] = (area_vs_random(gate_scores[idx], s[idx], e[idx])
                     - area_vs_random(oracle_scores[idx], s[idx], e[idx]))

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_le0 = float(np.mean(deltas <= 0))
    p_two = 2 * min(p_le0, 1 - p_le0)
    print(f"\npaired bootstrap (B={B}, seed {SEED}):")
    print(f"  delta mean {deltas.mean():+.4f} | 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  P(delta<=0) = {p_le0:.4f} | two-sided p = {p_two:.4f}")
    print("  verdict:", "SIGNIFICANT" if p_two < 0.05 else
          "NOT significant — soften system.tex:21 to 'on par with the pool oracle'")


if __name__ == "__main__":
    main()
