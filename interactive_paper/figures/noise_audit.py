"""Replication-noise audit of the live arms (2026-08-20, $0).

Motivated by the threshold-overshoot "fix" that recovered nothing: if
correcting a 11-point rate error moves accuracy by +.004, how big is
the noise we have been narrating against?

Measured here, from the existing traces only:
  * local-decode replication flips — SAME query, SAME audio, kept local
    in two different arms, judge verdict disagrees: 2.3-18.8% per pool;
  * repeat-escalation flips (expert sampling + judge): 0.7-10.6%;
  * the resulting paired SE on an arm-vs-arm accuracy delta: .009-.028.

Left  : the frozen pool's reconstructed rate curve (every point is a
        measured outcome, re-mixed) with a paired bootstrap band; the
        v2/v3 "regression" and the overshoot both sit inside it.
Right : forest plot of all 18 v2->v3 live deltas with McNemar CIs, vs
        the two claims that DO survive the audit.
"""
import json
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, GREEN, RED = "#2a78d6", "#1baf7a", "#b00"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e4de"
ARMS = ("never", "conservative", "balanced", "aggressive")
TIERS = ("conservative", "balanced", "aggressive")
POOLS = {"frozen": "heard_ok", "striviaqa": "oab_ok", "swebq": "oab_ok",
         "sllama": "oab_ok", "sreason": "heard_ok", "sdqa": "heard_ok"}
DATA = "../data"


def arm(bench, ver, tier, col):
    d = pd.read_parquet(f"{DATA}/{bench}{ver}_traces.parquet")
    c = col if col in d.columns else "heard_ok"
    return d[d["tier"] == tier].set_index("id")[c].dropna().astype(int)


def mcnemar(x, y):
    """returns delta, se, n_discordant (y - x, paired)."""
    n01 = int(((x == 0) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n = len(x)
    d = (n01 - n10) / n
    se = np.sqrt(max(n01 + n10, 1)) / n
    return d, se, n01 + n10


# ---------------- left panel data: frozen rate curve ----------------------
curves = json.load(open(f"{DATA}/rate_curves.json"))
f = curves["frozen"]
rs, ys = np.array(f["rate"]), np.array(f["acc"])

df = pd.read_parquet(f"{DATA}/frozen_v3_traces.parquet")
df = df[df["tier"].isin(ARMS)]
A = df.pivot(index="id", columns="tier", values="heard_ok")
M = df.pivot(index="id", columns="tier", values="mode")
S = df.pivot(index="id", columns="tier", values="eot_score")
keep = A[list(ARMS)].notna().all(axis=1)
A, M, S = A[keep], M[keep], S[keep]
score = S["aggressive"].to_numpy(float)
local = A["never"].to_numpy(float)
X = np.full(len(A), np.nan)
for i in range(len(A)):
    for t in TIERS:
        if M[t].iloc[i] == "escalated" and pd.notna(A[t].iloc[i]):
            X[i] = float(A[t].iloc[i])
            break

rng = np.random.default_rng(42)
n = len(score)
order = np.argsort(-score)
BOOT = 2000
band = np.zeros((BOOT, len(rs)))
for bi in range(BOOT):
    idx = rng.integers(0, n, n)
    sc, lo, xx = score[idx], local[idx], X[idx]
    od = np.argsort(-sc)
    for j, r in enumerate(rs):
        k = int(round(r * n))
        y = lo.copy()
        sel = od[:k]
        ok = ~np.isnan(xx[sel])
        y[sel[ok]] = xx[sel[ok]]
        band[bi, j] = np.nanmean(y)
lo_b, hi_b = np.percentile(band, [2.5, 97.5], axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
for ax in (ax1, ax2):
    ax.grid(color=GRID, lw=.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)

# NO random line in this panel on purpose: this curve is the
# DEPLOYED channel (expert answers the talker's transcript) while
# the only all-query random reference we can build is the gold-text
# one — mixing channels here invites a "gate < random" misreading.
# The channel-matched comparison lives in the dualview figures.
ax1.fill_between(rs, lo_b, hi_b, color=BLUE, alpha=.15, zorder=2,
                 label="95% paired bootstrap band")
ax1.plot(rs, ys, "-", lw=1.9, color=BLUE, zorder=4,
         label="gate curve (measured outcomes, re-mixed)")
for t in TIERS:
    r, acc = f["arm_rates"][t], f["arm_acc"][t]
    ax1.plot(r, acc, "o", ms=7, color=BLUE, zorder=5)
    ax1.annotate(f"{t[:4]} @{r:.0%}", (r, acc), xytext=(5, -12),
                 textcoords="offset points", fontsize=7.5, color=BLUE)
ax1.plot(.5625, .621, "s", ms=7, color=RED, zorder=5)
ax1.annotate("v2 aggressive .621 @56%\n(the “regression” —\ninside "
             "the band)", (.5625, .621), xytext=(12, 2),
             textcoords="offset points", fontsize=7.5, color=RED,
             ha="left")
a_fix = f["acc_at_budget"]["aggressive"]
ax1.plot(.50, a_fix, "*", ms=14, color=GREEN, zorder=6)
ax1.annotate(f"at budget 50%: {a_fix:.3f}\n(vs .596 fired at 61% —\n"
             "11% fewer expert calls,\nsame accuracy)", (.50, a_fix),
             xytext=(18, -108), textcoords="offset points", fontsize=7.5,
             color=GREEN, ha="left",
             arrowprops=dict(arrowstyle="-", color=GREEN, lw=.9,
                             alpha=.6))
ax1.set_xlabel("escalation rate", fontsize=9)
ax1.set_ylabel("deployed-channel accuracy (our judge)", fontsize=9)
ax1.set_title("Correcting an 11-point rate error moves accuracy by "
              "+.004\nfrozen pool (n=240), deployed channel — how big\n"
              "is our own replication noise?",
              fontsize=9.5, loc="left")
ax1.set_xlim(-.02, .72)
ax1.legend(loc="upper left", fontsize=7.5, frameon=False)

# ---------------- right panel: forest plot of v2->v3 deltas ---------------
rows = []
for bench, col in POOLS.items():
    for t in TIERS:
        x = arm(bench, "_v2", t, col)
        y = arm(bench, "_v3", t, col)
        ids = x.index.intersection(y.index)
        d, se, nd = mcnemar(x.loc[ids], y.loc[ids])
        rows.append((f"{bench[:9]} {t[:4]}", d, se))
ys_ = np.arange(len(rows))[::-1]
ax2.axvline(0, color=INK, lw=1, alpha=.5, zorder=3)
for (lab, d, se), yy in zip(rows, ys_):
    sig = abs(d) > 1.96 * se
    c = RED if sig else MUT
    ax2.plot([d - 1.96 * se, d + 1.96 * se], [yy, yy], color=c, lw=1.4,
             alpha=.75, zorder=4)
    ax2.plot(d, yy, "o", ms=4.5, color=BLUE if not sig else RED, zorder=5)
ax2.set_yticks(ys_)
ax2.set_yticklabels([r[0] for r in rows], fontsize=6.8)
ax2.set_xlabel("v2 → v3 accuracy delta, paired (95% McNemar CI)",
               fontsize=9)
ax2.set_title("All 18 live v2→v3 deltas cross zero\n"
              "(what survives: the probe's split, z=6.5 — see caption)",
              fontsize=9.5, loc="left")
ax2.set_xlim(-.12, .12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"noise_audit.{ext}", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote noise_audit.{png,pdf}")
print("frozen budget-corrected aggressive:", round(a_fix, 3))
