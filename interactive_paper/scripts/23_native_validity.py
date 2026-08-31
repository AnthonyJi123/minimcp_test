"""Native-regime validity tables (8be): the 8ad remix arithmetic on the
native dumps.

Per pool: native onset score (gate_native probe on the dumped features)
selects the escalation set at each tier threshold; gated accuracy mixes
the NATIVE local outcome (gpt-5.4-mini judged, deployed decoding config)
with the CACHED expert outcome (always-arm conclive traces for external
pools / frozen_v3 escalated arm for frozen test). Matched-random control
+ permutation p, 8ad style. Offline remix — no live escalation cost.

Usage: .venv_boot\\Scripts\\python.exe scripts\\23_native_validity.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data")
RNG = np.random.default_rng(42)
POOLS = [
    ("frozen", "test", None),          # expert col resolved specially
    ("striviaqa", "striviaqa", "oab_ok"),
    ("swebq", "swebq", "oab_ok"),
    ("sllama", "sllama", "oab_ok"),
    ("sdqa", "sdqa", "heard_ok"),
    ("sreason", "sreason", "heard_ok"),
]
TIERS = ["never", "conservative", "balanced", "aggressive", "always"]


def load_feats(tag):
    ids, X = [], []
    for p in sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += list(z["ids"])
        X.append(z["X"])
    X = np.concatenate(X)
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    return list(df["id"]), X[df["row"].to_numpy()]


def main():
    art = json.loads((D / "gate_native.json").read_text())
    w, b = np.array(art["w"]), art["b"]
    thr = art["eot_thresholds"]

    out = {}
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

        pool_out = {"n": n, "local_floor": round(float(lo.mean()), 3),
                    "expert_ceiling": round(float(eo.mean()), 3),
                    "tiers": {}}
        print(f"\n== {pool} (n={n})  local {lo.mean():.3f}  "
              f"expert {eo.mean():.3f}")
        for tier in TIERS:
            if tier == "never":
                esc = np.zeros(n, dtype=bool)
            elif tier == "always":
                esc = np.ones(n, dtype=bool)
            else:
                esc = sc >= thr[tier]
            acc = float(np.where(esc, eo, lo).mean())
            k = int(esc.sum())
            if 0 < k < n:
                rnd = []
                for _ in range(2000):
                    r = np.zeros(n, dtype=bool)
                    r[RNG.choice(n, k, replace=False)] = True
                    rnd.append(np.where(r, eo, lo).mean())
                rnd = np.array(rnd)
                p = float((rnd >= acc).mean())
                rnd_m = float(rnd.mean())
            else:
                p, rnd_m = None, acc
            pool_out["tiers"][tier] = {
                "esc_rate": round(k / n, 3), "acc": round(acc, 3),
                "random_matched": round(rnd_m, 3),
                "perm_p": (round(p, 4) if p is not None else None)}
            print(f"  {tier:<13} esc={k / n:.2f}  acc={acc:.3f}  "
                  f"rand={rnd_m:.3f}"
                  + (f"  p={p:.4f}" if p is not None else ""))
        out[pool] = pool_out

    Path("figures/native_validity.json").write_text(
        json.dumps(out, indent=1))
    print("\nwrote figures/native_validity.json")


if __name__ == "__main__":
    main()
