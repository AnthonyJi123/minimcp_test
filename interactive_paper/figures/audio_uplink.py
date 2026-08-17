"""Rescore the dual-view with the audio-direct-to-expert uplink ($0).

Third counterfactual next to heard (blue) and gold-inject (green):
escalated rows re-scored with the audio-expert verdict (the original
pool wav sent straight to gpt-audio — no self-transcription). RESULTS
8q follow-up.

Needs data/audio_expert.parquet:
  modal volume get --force gate-data audio_expert.parquet ../data/
Run from figures/: ..\\..\\.venv_ip\\Scripts\\python audio_uplink.py
"""
import json

import numpy as np
import pandas as pd

DATA = "../data"


def b(x):
    return 1 if x is True or x == 1 else 0


def main():
    tiers = ["never", "conservative", "balanced", "aggressive"]
    df = pd.read_parquet(f"{DATA}/gated_traces_v2.parquet")
    df = df[df["tier"].isin(tiers) & df["heard_ok"].notna()]
    heard = df.pivot(index="id", columns="tier", values="heard_ok").dropna()
    esc = (df.pivot(index="id", columns="tier", values="mode")
           .loc[heard.index] == "escalated")
    exp = pd.read_parquet(f"{DATA}/eval_expert.parquet").set_index("id")
    au = pd.read_parquet(f"{DATA}/audio_expert.parquet").set_index("id")
    unfair = set(json.load(open("fair_subset_audit.json")))

    gold = np.array([b(exp.loc[i, "expert_adequate"]) for i in heard.index])
    audio = np.array([b(au.loc[i, "adequate"]) if i in au.index else np.nan
                      for i in heard.index])
    A = heard[tiers].to_numpy(float)
    E = esc[tiers].to_numpy(bool)
    B = np.where(E, gold[:, None], A)                       # gold-inject
    C = np.where(E, audio[:, None], A)                      # audio-uplink
    fair = np.array([i not in unfair for i in heard.index])

    # escalated-subset head-to-head (the 8d five-arm frame, now n=132)
    eids = [i for j, i in enumerate(heard.index) if E[j].any()]
    sub = pd.DataFrame({
        "pool": [exp.loc[i, "pool"] if i in exp.index else au.loc[i, "pool"]
                 for i in eids],
        "gold": [gold[list(heard.index).index(i)] for i in eids],
        "audio": [audio[list(heard.index).index(i)] for i in eids]})
    sub = sub.dropna()
    print(f"=== escalated subset, n={len(sub)}: audio-direct vs gold ===")
    for p, g in sub.groupby("pool"):
        print(f"  {p:15s} n={len(g):3d}  gold {g['gold'].mean():.2f}  "
              f"audio {g['audio'].mean():.2f}")
    print(f"  {'ALL':15s} n={len(sub):3d}  gold {sub['gold'].mean():.2f}  "
          f"audio {sub['audio'].mean():.2f}")

    rng = np.random.default_rng(42)

    def curve(mask, label):
        a, bb, cc, e = A[mask], B[mask], C[mask], E[mask]
        n = mask.sum()
        idx = rng.integers(0, n, size=(10000, n))
        bC = cc[idx].mean(axis=1)
        print(f"\n=== {label} (n={n}) ===")
        print(f"{'tier':13s} {'esc':>5s} {'heard':>6s} {'audio-up':>9s} "
              f"{'gold-inj':>8s}   audio-up 95% CI")
        for j, t in enumerate(tiers):
            lo, hi = np.percentile(bC[:, j], [2.5, 97.5])
            print(f"{t:13s} {e[:, j].mean():5.2f} {a[:, j].mean():6.3f} "
                  f"{cc[:, j].mean():9.3f} {bb[:, j].mean():8.3f}   "
                  f"[{lo:.3f},{hi:.3f}]")

    curve(np.ones(len(heard), bool), "FULL pool")
    curve(fair, "FAIR / speakable subset")


if __name__ == "__main__":
    main()
