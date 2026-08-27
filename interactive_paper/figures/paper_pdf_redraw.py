# -*- coding: utf-8 -*-
"""Paper-quality PDF redraws of the PNG-only figures (2026-08-26 feedback):
layer_sweep, tradeoff, tradeoff_ptrue, receipt_compare, overlap + a lossless
PDF wrap of the teaser_v2 illustration. Big fonts/legends, vector output.

Data: figures/_voldata/ (fetched from gate-data volume) + data/gate_config.json.
Run:  .venv_ip/Scripts/python.exe figures/paper_pdf_redraw.py   (from interactive_paper/)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from gate import Probe

VOL = "figures/_voldata"
OUT = ["figures", "paper/figures"]

# validated categorical palette (dataviz reference, light mode, fixed order)
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14.5,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUT, "ytick.color": MUT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42,
})


def save(fig, name):
    for d in OUT:
        fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(d, name + ".png"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, "->", ", ".join(OUT))


# ---------------------------------------------------------------- layer sweep
def fig_layer_sweep():
    MODELS = [  # (tag, label, color, linestyle, lw)
        ("minicpm-o45", "MiniCPM-o 4.5 (duplex)", C1, "-", 3.0),
        ("qwen3-8b", "Qwen3-8B (its raw backbone)", C2, "--", 2.2),
        ("minicpm-o26", "MiniCPM-o 2.6 (duplex)", C3, "-", 2.2),
        ("qwen2.5-7b", "Qwen2.5-7B (raw)", C4, "--", 2.2),
        ("qwen2.5-omni-7b", "Qwen2.5-Omni (streaming)", C5, ":", 2.2),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, pool, ptitle in zip(axes, ("last", "mean"),
                                ("last-token read (deployed position)",
                                 "mean-pooled read")):
        for tag, label, col, ls, lw in MODELS:
            with open(f"{VOL}/layer_sweep_{tag}.json") as f:
                data = json.load(f)
            L = data["n_layers"]
            x = [(c["layer"] + 1) / L for c in data["curves"]]
            y = [c.get(f"{pool}_lopo_hard-math") for c in data["curves"]]
            ax.plot(x, y, ls, color=col, lw=lw, label=label,
                    solid_capstyle="round")
        ax.axhline(0.5, color=MUT, lw=1.2, ls=(0, (3, 2)))
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("relative depth")
        ax.set_title(ptitle)
    x22 = 23 / 36  # L22 of 36, matching the sweep's (layer+1)/L axis
    axes[0].axvline(x22, color=INK, lw=1.2, ls=(0, (1, 2)))
    axes[0].text(x22 - 0.02, 0.972, "L22 (deployed)", color=INK,
                 fontsize=12, ha="right", va="top")
    axes[1].text(0.015, 0.512, "chance", color=MUT, fontsize=11.5)
    axes[0].set_ylabel("LOPO hard-math AUC")
    axes[0].set_ylim(0.3, 1.0)
    axes[0].legend(loc="lower left", framealpha=0.95, borderpad=0.7,
                   handlelength=2.6)
    fig.tight_layout()
    save(fig, "layer_sweep")


# ------------------------------------------------------------------- tradeoff
def _b(x):
    return 1 if x is True or x == 1 else 0


def _load_test():
    cfg = json.load(open("data/gate_config.json"))
    probe = Probe.from_config(cfg)
    df = pd.read_parquet(f"{VOL}/calib_features.parquet")
    test = df[df["split"] == "test"].reset_index(drop=True)
    exp = pd.read_parquet(f"{VOL}/eval_expert.parquet").set_index("id")
    ids = test["id"].values
    s = np.array([_b(x) for x in test["adequate"].values])
    e = np.array([_b(exp.loc[i, "expert_adequate"]) for i in ids])
    return cfg, probe, df, test, s, e


def fig_tradeoff():
    cfg, probe, df, test, s, e = _load_test()
    para = pd.read_parquet(f"{VOL}/eval_paraphrase.parquet").set_index("id")
    p = np.array([_b(para.loc[i, "paraphrase_adequate"])
                  for i in test["id"].values])
    scores = np.array([probe.score(list(h)) for h in test["h_prompt"]])
    pools = test["pool"].values
    small_acc, big_acc = s.mean(), e.mean()

    def hybrid(esc, outcome):
        return np.where(esc, outcome, s).mean()

    def sweep(sc):
        ts = np.concatenate([[np.inf], np.unique(sc)[::-1], [-np.inf]])
        c = np.array([[(sc >= t).mean(), hybrid(sc >= t, e),
                       hybrid(sc >= t, p)] for t in ts])
        return c[np.argsort(c[:, 0])]

    curve = sweep(scores)
    calib = df[df["split"] == "calib"]
    pool_fail = {pl: float(np.mean([1 - _b(x) for x in
                                    calib.loc[calib["pool"] == pl,
                                              "adequate"]]))
                 for pl in np.unique(pools)}
    ocurve = sweep(np.array([pool_fail[pl] for pl in pools]))

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.plot(curve[:, 0] * 100, curve[:, 1] * 100, "-", color=C1, lw=2.6,
            label="gated escalation (expert-inject)")
    ax.plot(curve[:, 0] * 100, curve[:, 2] * 100, "-", color=C2, lw=2.2,
            label="gated escalation (paraphrase relay)")
    ax.plot(ocurve[:, 0] * 100, ocurve[:, 1] * 100, "-s", color=C3, lw=1.8,
            ms=6, label="pool-oracle (type-only) baseline")
    r = np.linspace(0, 1, 101)
    ax.plot(r * 100, ((1 - r) * small_acc + r * big_acc) * 100,
            color=MUT, lw=1.6, ls=(0, (4, 3)), label="random escalation")
    for tier in ("conservative", "balanced", "aggressive"):
        t = cfg["thresholds"][tier]
        esc = scores >= t
        ax.scatter([esc.mean() * 100], [hybrid(esc, e) * 100], s=55,
                   color=INK, zorder=5)
        ax.annotate(tier, (esc.mean() * 100, hybrid(esc, e) * 100),
                    textcoords="offset points", xytext=(7, -13),
                    fontsize=12, color=INK)
    ax.axhline(small_acc * 100, color=MUT, lw=1, ls=":")
    ax.axhline(big_acc * 100, color=MUT, lw=1, ls=":")
    ax.text(100.5, small_acc * 100, "small-only", color=MUT, fontsize=11.5,
            va="center")
    ax.text(100.5, big_acc * 100, "big-only", color=MUT, fontsize=11.5,
            va="center")
    ax.set_xlim(0, 100)
    ax.set_xlabel("escalation rate (%)")
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="lower right", framealpha=0.95, borderpad=0.7)
    fig.tight_layout()
    save(fig, "tradeoff")


def fig_tradeoff_ptrue():
    cfg, probe, df, _, _, _ = _load_test()
    df = df[df["escalate_label"].notna()].reset_index(drop=True)
    pt = pd.concat([pd.read_parquet(f"{VOL}/ptrue.shard{i}.parquet")
                    for i in range(4)], ignore_index=True)
    df = df.merge(pt, on="id", validate="one_to_one")
    df["probe_score"] = [probe.score(list(h)) for h in df["h_prompt"]]
    df["ptrue_pre_score"] = 1.0 - df["p_yes_pre"]
    df["ptrue_post_score"] = 1.0 - df["p_yes_post"]
    test = df[df["split"] == "test"].reset_index(drop=True)
    exp = pd.read_parquet(f"{VOL}/eval_expert.parquet").set_index("id")
    ids = test["id"].values
    s = np.array([_b(x) for x in test["adequate"].values])
    e = np.array([_b(exp.loc[i, "expert_adequate"]) for i in ids])
    small_acc, big_acc = s.mean(), e.mean()

    def curve(sc):
        ts = np.concatenate([[np.inf], np.unique(sc)[::-1], [-np.inf]])
        pts = np.array([[(sc >= t).mean(),
                         np.where(sc >= t, e, s).mean()] for t in ts])
        return pts[np.argsort(pts[:, 0])]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, col, colr in [("mid-layer probe (pre-answer)", "probe_score", C1),
                            ("$p(\\mathrm{True})$-pre", "ptrue_pre_score", C2),
                            ("$p(\\mathrm{True})$-post (full draft)",
                             "ptrue_post_score", C3)]:
        c = curve(test[col].to_numpy())
        ax.plot(c[:, 0] * 100, c[:, 1] * 100, "-", color=colr, lw=2.4,
                label=name)
    r = np.linspace(0, 1, 101)
    ax.plot(r * 100, ((1 - r) * small_acc + r * big_acc) * 100, color=MUT,
            lw=1.6, ls=(0, (4, 3)), label="random escalation")
    ax.set_xlim(0, 100)
    ax.set_xlabel("escalation rate (%)")
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="lower right", framealpha=0.95, borderpad=0.7)
    fig.tight_layout()
    save(fig, "tradeoff_ptrue")


# ------------------------------------------------------------ receipt compare
def fig_receipt():
    rr = json.load(open(f"{VOL}/router_baseline.json"))["receipt"]
    pr = json.load(open(f"{VOL}/probe_receipt.json"))
    text, audio = pr[0], pr[1]
    SIGS = [
        ("Router (query text)", "#898781", rr["oof_acc"],
         rr["majority_acc_calib"], rr["test_acc"], rr["majority_acc_test"],
         rr["train_logloss"], rr["oof_logloss"], rr["test_logloss"]),
        ("Text probe (h_prompt)", C1, text["oof_acc"],
         text["majority_calib"], text["test_acc"], text["majority_test"],
         text["train_logloss"], text["oof_logloss"], text["test_logloss"]),
        ("Audio probe (L22, live gate)", C2, audio["oof_acc"],
         audio["majority_calib"], audio["test_acc"], audio["majority_test"],
         audio["train_logloss"], audio["oof_logloss"],
         audio["test_logloss"]),
    ]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5))
    w = 0.34
    for i, (name, col, oa, om, ta, tm, *_r) in enumerate(SIGS):
        ax.bar(i - w / 2, oa, w * .92, color=col, zorder=3)
        ax.bar(i + w / 2, ta, w * .92, color=col, alpha=.55, zorder=3)
        for x, v in ((i - w / 2, oa), (i + w / 2, ta)):
            ax.text(x, v + .015, f"{v:.3f}", ha="center", fontsize=11.5,
                    color=INK)
        for x, m in ((i - w / 2, om), (i + w / 2, tm)):
            ax.plot([x - w * .46, x + w * .46], [m, m], color=INK, lw=1.8,
                    ls=(0, (3, 2)), zorder=4)
    ax.set_xticks(range(len(SIGS)))
    ax.set_xticklabels([s[0].replace(" (", "\n(") for s in SIGS],
                       fontsize=12.5, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy @ 0.5 threshold")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) Accuracy on our eval splits\nsolid = calib OOF, "
                 "faded = test, dashed = majority baseline", fontsize=13.5)
    ys = np.arange(len(SIGS))[::-1]
    for yv, (name, col, *_a, tr_l, oof_l, te_l) in zip(ys, SIGS):
        bx.plot([tr_l, oof_l], [yv, yv], color=col, lw=2.4, zorder=2)
        bx.scatter([tr_l], [yv], s=80, facecolor="white", edgecolor=col,
                   lw=2.4, zorder=3)
        bx.scatter([oof_l], [yv], s=80, color=col, zorder=3)
        bx.scatter([te_l], [yv], s=80, color=col, marker="s", zorder=3)
        for v, dy in ((tr_l, .20), (oof_l, .20), (te_l, -.33)):
            bx.text(v, yv + dy, f"{v:.3f}", ha="center", fontsize=11,
                    color=MUT)
        bx.text(-.05, yv, name.split(" (")[0], ha="right", va="center",
                fontsize=12.5, color=INK)
    bx.set_yticks([])
    bx.set_xlim(-.05, .9)
    bx.set_ylim(-.6, len(SIGS) - .4)
    bx.set_xlabel("log-loss")
    bx.grid(axis="x")
    bx.grid(axis="y", visible=False)
    bx.spines["left"].set_visible(False)
    bx.set_title("(b) Training vs eval loss\nopen = train, filled = calib "
                 "OOF, square = test", fontsize=13.5)
    fig.tight_layout()
    save(fig, "receipt_compare")


# -------------------------------------------------------------------- overlap
def fig_overlap():
    b = pd.read_parquet(f"{VOL}/latency_bench.parquet")
    b = b[~b["warmup"]]
    gate = {a: float(b[f"{a}_l22_s"].median()) for a in ("text", "audio")}
    local = {a: {p: g[f"{a}_answer_s"].to_numpy()
                 for p, g in b.groupby("pool")} for a in ("text", "audio")}
    big = pd.read_parquet(f"{VOL}/eval_expert.parquet")
    big = big[big["expert_error"].isna() & (big["expert_latency"] > 0)]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for arm, ls in (("text", "-"), ("audio", "--")):
        ready = np.sort(gate[arm] + big["expert_latency"].to_numpy())
        ax.plot(ready, np.linspace(0, 1, len(ready)), ls, color=C1, lw=2.4,
                label=f"cloud result ready ({arm} gate)")
        dur = np.sort(np.concatenate(list(local[arm].values())))
        ax.plot(dur, np.linspace(0, 1, len(dur)), ls, color=C2, lw=2.4,
                label=f"local answer duration ({arm})")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)
    ax.set_xlabel("seconds after query end")
    ax.set_ylabel("CDF")
    ax.legend(loc="lower right", framealpha=0.95, borderpad=0.7)
    fig.tight_layout()
    save(fig, "overlap")


# ---------------------------------------------------- teaser_v2 lossless wrap
def fig_teaser_v2():
    img = plt.imread("paper/figures/teaser_v2.png")
    h, w = img.shape[:2]
    dpi = 100.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    ax.grid(False)
    for d in OUT:
        fig.savefig(os.path.join(d, "teaser_v2.pdf"), dpi=dpi, pad_inches=0)
    plt.close(fig)
    print("wrote teaser_v2.pdf (lossless raster wrap)")


if __name__ == "__main__":
    fig_layer_sweep()
    fig_tradeoff()
    fig_tradeoff_ptrue()
    fig_receipt()
    fig_overlap()
    fig_teaser_v2()
