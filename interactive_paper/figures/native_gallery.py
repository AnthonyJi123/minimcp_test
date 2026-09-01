"""Native-regime gallery figures (8be/8bf): regime chain + scaling,
validity small-multiples, floor-control behavior.
Run: .venv_boot\\Scripts\\python.exe figures\\native_gallery.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BLUE, ORANGE, TEAL = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, PINK, GREY = "#eda100", "#e87ba4", "#8a97a5"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})

refit = json.load(open("figures/native_refit.json"))
valid = json.load(open("figures/native_validity.json"))

# ---- fig 1: regime chain + scaling curve --------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
ax = axes[0]
regimes = ["turn-based\n(ideal)", "native duplex\n(deployed)",
           "concurrent\n(harness)"]
internal = [0.877, 0.830, 0.760]
external = [0.771, 0.709, 0.689]
x = np.arange(3)
ax.bar(x - 0.18, internal, 0.34, color=BLUE, label="internal test")
ax.bar(x + 0.18, external, 0.34, color=ORANGE, label="external mean")
for i, (a, b) in enumerate(zip(internal, external)):
    ax.text(i - 0.18, a + .008, f"{a:.3f}", ha="center", fontsize=10)
    ax.text(i + 0.18, b + .008, f"{b:.3f}", ha="center", fontsize=10)
ax.set_xticks(x, regimes)
ax.set_ylim(.6, .95)
ax.set_ylabel("probe AUC (in-regime)")
ax.legend(frameon=False, loc="upper right")
ax.set_title("Read-point regimes: the deployed native read is nearly free", fontsize=12)

ax = axes[1]
c = refit["scaling_curve"]
ns = [r["n"] for r in c]
ax.plot(ns, [r["auc_internal"] for r in c], "o-", color=BLUE,
        label="internal")
ax.plot(ns, [r["auc_external_mean"] for r in c], "s-", color=ORANGE,
        label="external mean")
ax.axhline(0.877, color=BLUE, ls=":", lw=1)
ax.axhline(0.771, color=ORANGE, ls=":", lw=1)
ax.text(2310, 0.879, "turn-based .877", fontsize=9, color=BLUE,
        ha="right")
ax.text(2310, 0.773, "turn-based .771", fontsize=9, color=ORANGE,
        ha="right")
ax.set_xlabel("in-regime calibration rows")
ax.set_ylabel("AUC")
ax.legend(frameon=False, loc="lower right")
ax.set_title("Native calibration scaling: not saturated at 2310", fontsize=12)
fig.tight_layout()
fig.savefig("figures/native_regimechain.png", dpi=170)
plt.close(fig)

# ---- fig 2: validity small-multiples ------------------------------------
ORDER = ["frozen", "striviaqa", "swebq", "sllama", "sdqa", "sreason"]
NICE = {"frozen": "our pool", "striviaqa": "TriviaQA",
        "swebq": "WebQ", "sllama": "LlamaQ", "sdqa": "SD-QA",
        "sreason": "Reasoning-zh"}
TIERS = ["never", "conservative", "balanced", "aggressive", "always"]
fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharey=False)
for ax, pool in zip(axes.flat, ORDER):
    d = valid[pool]
    xs = [d["tiers"][t]["esc_rate"] for t in TIERS]
    ys = [d["tiers"][t]["acc"] for t in TIERS]
    rs = [d["tiers"][t]["random_matched"] for t in TIERS]
    # oracle selector: escalate the (local-wrong, expert-right) items
    # first. The benefit-pool size is joint-free bounded in
    # [ceil-floor, min(1-floor, ceil)]; the band spans the two.
    fl, ce = d["local_floor"], d["expert_ceiling"]
    pb_lo, pb_hi = max(0.0, ce - fl), min(1 - fl, ce)
    rg = np.linspace(0, 1, 201)

    def oracle(pb):
        acc = fl + np.minimum(rg, pb)
        tail = rg >= pb
        acc[tail] = (fl + pb) + (ce - fl - pb) * (rg[tail] - pb) / (1 - pb)
        return acc

    ax.fill_between(rg, oracle(pb_lo), oracle(pb_hi), color=TEAL,
                    alpha=.13, lw=0)
    ax.plot(rg, oracle(pb_lo), color=TEAL, lw=1.3, alpha=.8,
            label="oracle selector")
    ax.plot(xs, rs, "--", color=GREY, lw=1.6, label="matched random")
    ax.plot(xs, ys, "o-", color=BLUE, lw=2, label="gated (native)")
    ax.set_title(f"{NICE[pool]}  (n={d['n']})", fontsize=11)
    ax.set_xlim(-.04, 1.04)
    for t in ("balanced", "aggressive"):
        p = d["tiers"][t]["perm_p"]
        i = TIERS.index(t)
        if p is not None and p < .05:
            ax.annotate("*", (xs[i], ys[i]),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", color=ORANGE, fontsize=15)
        # fraction of the oracle-over-random margin the gate captures
        # (exact when the tier rate <= ceil-floor, else optimistic end)
        om = min(xs[i], pb_lo) - xs[i] * (ce - fl)
        if om > 0 and xs[i] > 0:
            ax.annotate(f"{(ys[i] - rs[i]) / om:.0%}", (xs[i], ys[i]),
                        textcoords="offset points", xytext=(3, -13),
                        fontsize=9, color=ORANGE)
axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
for ax in axes[1]:
    ax.set_xlabel("escalation rate")
for r in range(2):
    axes[r][0].set_ylabel("delivered accuracy")
fig.suptitle("Native full duplex: gated accuracy vs matched-random vs the"
             " oracle bound\n(* = permutation p<.05; orange % = share of the"
             " oracle-over-random margin captured; Reasoning-zh never fires)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("figures/native_validity.png", dpi=170)
plt.close(fig)

# ---- fig 3: floor behavior ----------------------------------------------
rows = [json.loads(l) for p in glob.glob("data/floor.jsonl.shard*")
        for l in open(p, encoding="utf-8") if l.strip()]
if not rows:
    print("wrote figures/native_{regimechain,validity}.png "
          "(no data/floor.jsonl.shard*; floor fig left untouched)")
    raise SystemExit


def cell(arm, phase, kind):
    return [r for r in rows if r["arm"] == arm and r["phase"] == phase
            and r["kind"] == kind and r["stim_at"] is not None]


fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
ax = axes[0]
kinds = ["bcs", "bcl", "stop", "bq"]
names = ["backchannel\n(short)", "backchannel\n(long)",
         "\"Stop!\"", "question\n(sustained)"]
frac = []
for k in kinds:
    rs = cell("ans", "ans", k)
    y = sum(1 for r in rs
            if r["yield_at"] is not None
            and r["yield_at"] - r["stim_at"] <= 6)
    frac.append(y / max(1, len(rs)))
cols = [TEAL, TEAL, ORANGE, ORANGE]
ax.bar(range(4), frac, color=cols)
for i, f in enumerate(frac):
    ax.text(i, f + .02, f"{f:.0%}", ha="center", fontsize=11)
ax.set_xticks(range(4), names, fontsize=10)
ax.set_ylim(0, 1)
ax.set_ylabel("yield ≤6 chunks")
ax.set_title("Native yielding: sustained speech, not commands\n"
             "(teal = should hold, orange = should yield; no VAD/ASR)",
             fontsize=11)

ax = axes[1]
ph_kinds = [("wait", "bcs"), ("wait", "stop"), ("wait", "bq"),
            ("relay", "bcs"), ("relay", "stop"), ("relay", "bq")]
names2 = ["wait+bc", "wait+stop", "wait+Q",
          "relay+bc", "relay+stop", "relay+Q"]
rd = []
for ph, k in ph_kinds:
    rs = cell("esc", ph, k)
    rd.append(np.mean([r["relay_done"] for r in rs]) if rs else 0)
cols2 = [TEAL, ORANGE, ORANGE, TEAL, ORANGE, ORANGE]
ax.bar(range(6), rd, color=cols2)
for i, f in enumerate(rd):
    ax.text(i, f + .02, f"{f:.0%}", ha="center", fontsize=11)
ax.set_xticks(range(6), names2, fontsize=10)
ax.set_ylim(0, 1.09)
ax.set_ylabel("relay completes")
ax.set_title("Stims during escalation: backchannels harmless;\n"
             "a NEW QUESTION during the wait derails 70% of relays",
             fontsize=11)
fig.tight_layout()
fig.savefig("figures/native_floor.png", dpi=170)
plt.close(fig)
print("wrote figures/native_{regimechain,validity,floor}.png")
