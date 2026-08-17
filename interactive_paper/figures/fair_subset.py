"""Input-side fairness audit + fair-subset dual-view recompute ($0).

RESULTS §8q. "Unfair" is decided from the INPUT ONLY (query text + wav
bytes), never from model outcomes — otherwise excluding is outcome
cherry-picking:
  latex      : formula-symbolic content — backslash commands, ^{ _{ ^x
               exponents, $..$ only when the inside contains math
               operators (plain dollar amounts like "$815.50" are
               speakable and NOT flagged)
  broken_wav : render is digital silence (figures/wav_audit.json)
Rare entities / hard names are NOT flagged: mishearing them is the
phenomenon under study, not a defect.

Run from figures/: ..\\..\\.venv_ip\\Scripts\\python fair_subset.py
"""
import json
import re

import numpy as np
import pandas as pd

DATA = "../data"

BSLASH = re.compile(r"\\[A-Za-z]{2,}")
CARET = re.compile(r"\^\{|_\{|\^\(|\^-?\w")
DOLLAR = re.compile(r"\$([^$]+)\$")


def is_latex(q):
    if BSLASH.search(q) or CARET.search(q):
        return True
    m = DOLLAR.search(q)
    return bool(m and re.search(r"[\\^_=]", m.group(1)))


def b(x):
    return 1 if x is True or x == 1 else 0


def main():
    bad_wavs = {r["id"] for r in json.load(open("wav_audit.json"))
                if r["peak"] < 500}

    tiers = ["never", "conservative", "balanced", "aggressive"]
    df = pd.read_parquet(f"{DATA}/gated_traces_v2.parquet")
    df = df[df["tier"].isin(tiers) & df["heard_ok"].notna()]
    heard = df.pivot(index="id", columns="tier", values="heard_ok").dropna()
    esc = (df.pivot(index="id", columns="tier", values="mode")
           .loc[heard.index] == "escalated")
    exp = pd.read_parquet(f"{DATA}/eval_expert.parquet").set_index("id")
    qtext = df.drop_duplicates("id").set_index("id")["query"]
    pool = df.drop_duplicates("id").set_index("id")["pool"]

    audit = {}
    for i in heard.index:
        f = []
        if is_latex(qtext.loc[i]):
            f.append("latex")
        if i in bad_wavs:
            f.append("broken_wav")
        audit[i] = f
    unfair = sorted(i for i in heard.index if audit[i])
    print(f"test ids: {len(heard)} | unfair: {len(unfair)} ",
          pd.Series([pool.loc[i] for i in unfair]).value_counts().to_dict())
    print("  ids:", unfair)

    gold = np.array([b(exp.loc[i, "expert_adequate"]) for i in heard.index])
    A = heard[tiers].to_numpy(float)
    E = esc[tiers].to_numpy(bool)
    B = np.where(E, gold[:, None], A)
    fair = np.array([i not in set(unfair) for i in heard.index])

    rng = np.random.default_rng(42)

    def curve(mask, label):
        a, bb, e, g = A[mask], B[mask], E[mask], gold[mask]
        n = mask.sum()
        idx = rng.integers(0, n, size=(10000, n))
        bA, bB = a[idx].mean(axis=1), bb[idx].mean(axis=1)
        print(f"\n=== {label} (n={n}) ===")
        for j, t in enumerate(tiers):
            lo, hi = np.percentile(bA[:, j], [2.5, 97.5])
            gap = bB[:, j] - bA[:, j]
            lg, hg = np.percentile(gap, [2.5, 97.5])
            print(f"{t:13s} esc={e[:, j].mean():.2f} "
                  f"heard {a[:, j].mean():.3f} [{lo:.3f},{hi:.3f}]  "
                  f"gold-inj {bb[:, j].mean():.3f}  "
                  f"gap {bb[:, j].mean() - a[:, j].mean():+.3f} [{lg:+.3f},{hg:+.3f}]")
        print(f"{'always(synth)':13s} gold {g.mean():.3f}")

    curve(np.ones(len(heard), bool), "FULL pool (paper today)")
    curve(fair, "FAIR / speakable subset")

    print("\n=== flag share of escalated heard-fails ===")
    for j, t in enumerate(tiers[1:], 1):
        fail = E[:, j] & (A[:, j] == 0)
        flagged = [i for i in heard.index[fail] if audit[i]]
        print(f"  {t:13s} esc-fails={fail.sum():3d}  flagged={len(flagged):3d}")

    json.dump({i: audit[i] for i in unfair},
              open("fair_subset_audit.json", "w"), indent=1)
    print("\nwritten: figures/fair_subset_audit.json")


if __name__ == "__main__":
    main()
