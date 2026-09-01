"""Per-pool (label-free) threshold recalibration for the native gate.

Motivation (8be validity): thresholds are the fragile part of
calibration for the third regime in a row — global calib-quantile
thresholds fire 64-73% on external pools at the 50% nominal tier and
fire 0% on Reasoning-zh at every deployed tier (zh scores sit below the
en calib distribution). Both are operating-point failures, not probe
failures, and both are fixable WITHOUT labels: a pool's own score
quantiles hit the nominal fire rate by construction.

Deployability framing: a static per-pool quantile is the converged
limit of ONLINE quantile tracking — keep a sliding window of recent
onset scores and threshold at its (1-rate) quantile. No labels, no
pool identity, no oracle. The window simulation below shows realized
fire rates converge to nominal within ~1 window of warmup.

Outputs:
  data/gate_native_pooled.json      global + per-pool thresholds
  figures/native_validity_pooled.json  the scripts/23 remix table
      recomputed at (a) per-pool static and (b) windowed thresholds —
      same schema as native_validity.json, so the gallery figure can
      be pointed at either.

Usage (from interactive_paper/): python scripts/26_pool_thresholds.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data")
RNG = np.random.default_rng(42)
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
WINDOW = 100  # sliding-window size for the online tracker
POOLS = [
    ("frozen", "test", None),
    ("striviaqa", "striviaqa", "oab_ok"),
    ("swebq", "swebq", "oab_ok"),
    ("sllama", "sllama", "oab_ok"),
    ("sdqa", "sdqa", "heard_ok"),
    ("sreason", "sreason", "heard_ok"),
]


def load_feats(tag):
    ids, X = [], []
    for p in sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += list(z["ids"])
        X.append(z["X"])
    if not X:
        raise FileNotFoundError(f"no native feats for tag {tag}")
    X = np.concatenate(X)
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    return list(df["id"]), X[df["row"].to_numpy()]


def windowed_fire(sc, rate, warmup=WINDOW):
    """Online tracker: threshold each item at the (1-rate) quantile of
    the previous WINDOW scores (arrival order = shuffled, seed 42).
    Returns the fire mask over the post-warmup stream."""
    order = RNG.permutation(len(sc))
    s = sc[order]
    fire = np.zeros(len(s), dtype=bool)
    for i in range(warmup, len(s)):
        thr = np.quantile(s[max(0, i - WINDOW):i], 1 - rate)
        fire[i] = s[i] >= thr
    return fire[warmup:], order[warmup:]


def remix(lo, eo, esc):
    acc = float(np.where(esc, eo, lo).mean())
    k = int(esc.sum())
    n = len(lo)
    if 0 < k < n:
        rnd = []
        for _ in range(2000):
            r = np.zeros(n, dtype=bool)
            r[RNG.choice(n, k, replace=False)] = True
            rnd.append(np.where(r, eo, lo).mean())
        rnd = np.array(rnd)
        return acc, k / n, float(rnd.mean()), float((rnd >= acc).mean())
    return acc, k / n, acc, None


def main():
    art = json.loads((D / "gate_native.json").read_text())
    w, b = np.array(art["w"]), art["b"]
    thr_global = art["eot_thresholds"]

    out = {"rates": RATES, "window": WINDOW,
           "global": thr_global, "pools": {}}
    valid = {}
    print(f"{'pool':<11}{'tier':<14}{'thr_glob':>9}{'fire_g':>8}"
          f"{'thr_pool':>9}{'fire_p':>8}{'fire_win':>9}")
    for pool, tag, ecol in POOLS:
        ids, X = load_feats(tag)
        s = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        score = dict(zip(ids, s))

        j = pd.read_parquet(D / f"frozen_native_{tag}_judged.parquet")
        j = j.dropna(subset=["adequate"]).drop_duplicates("id",
                                                          keep="last")
        local_ok = dict(zip(j["id"], j["adequate"].astype(int)))
        if pool == "frozen":
            f = pd.read_parquet(D / "frozen_v3_traces.parquet")
            e = f[f["mode"] == "escalated"].groupby("id")[
                "heard_ok"].max()
            exp_ok = e.dropna().astype(int).to_dict()
        else:
            cl = pd.read_parquet(D / f"{pool}_conclive_traces.parquet")
            a = cl[cl.tier == "always"].dropna(subset=[ecol])
            a = a.drop_duplicates("id", keep="last")
            exp_ok = dict(zip(a["id"], a[ecol].astype(int)))
        rows = [i for i in ids
                if i in local_ok and i in exp_ok and i in score]
        lo = np.array([local_ok[i] for i in rows])
        eo = np.array([exp_ok[i] for i in rows])
        sc = np.array([score[i] for i in rows])
        n = len(rows)

        pooled = {t: float(np.quantile(sc, 1 - r))
                  for t, r in RATES.items()}
        out["pools"][pool] = pooled
        pv = {"n": n, "local_floor": round(float(lo.mean()), 3),
              "expert_ceiling": round(float(eo.mean()), 3), "tiers": {}}
        pv["tiers"]["never"] = {"esc_rate": 0.0,
                                "acc": pv["local_floor"],
                                "random_matched": pv["local_floor"],
                                "perm_p": None}
        for tier, rate in RATES.items():
            fire_g = float((sc >= thr_global[tier]).mean())
            acc, er, rnd, p = remix(lo, eo, sc >= pooled[tier])
            fw, ow = windowed_fire(sc, rate)
            acc_w, er_w, rnd_w, p_w = remix(lo[ow], eo[ow], fw)
            pv["tiers"][tier] = {
                "esc_rate": round(er, 3), "acc": round(acc, 3),
                "random_matched": round(rnd, 3),
                "perm_p": (round(p, 4) if p is not None else None),
                "windowed": {"esc_rate": round(er_w, 3),
                             "acc": round(acc_w, 3),
                             "random_matched": round(rnd_w, 3),
                             "perm_p": (round(p_w, 4)
                                        if p_w is not None else None)}}
            print(f"{pool:<11}{tier:<14}{thr_global[tier]:>9.4f}"
                  f"{fire_g:>8.2f}{pooled[tier]:>9.4f}{er:>8.2f}"
                  f"{er_w:>9.2f}")
        pv["tiers"]["always"] = {"esc_rate": 1.0,
                                 "acc": pv["expert_ceiling"],
                                 "random_matched": pv["expert_ceiling"],
                                 "perm_p": None}
        valid[pool] = pv

    (D / "gate_native_pooled.json").write_text(json.dumps(out, indent=1))
    Path("figures/native_validity_pooled.json").write_text(
        json.dumps(valid, indent=1))
    print("\nwrote data/gate_native_pooled.json + "
          "figures/native_validity_pooled.json")


if __name__ == "__main__":
    main()
