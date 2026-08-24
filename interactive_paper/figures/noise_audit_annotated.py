# -*- coding: utf-8 -*-
"""讲解版之三:noise_audit(测量方法审计)。左:frozen 池重建曲线+
噪声带;右:18 个 v2→v3 live 差值的森林图;下:大白话解读。
计算逻辑照搬 noise_audit.py,只换版式。"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

BLUE, GREEN, RED = "#2a78d6", "#1e9e50", "#b00"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e4de"
ARMS = ("never", "conservative", "balanced", "aggressive")
TIERS = ("conservative", "balanced", "aggressive")
POOLS = {"frozen": "heard_ok", "striviaqa": "oab_ok", "swebq": "oab_ok",
         "sllama": "oab_ok", "sreason": "heard_ok", "sdqa": "heard_ok"}
DATA = "../data"

# ---- 左图数据:重建曲线 + bootstrap 带(照搬 noise_audit.py) ----------
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
BOOT = 2000
band = np.zeros((BOOT, len(rs)))
for bi in range(BOOT):
    idx = rng.integers(0, n, n)
    sc, lo_, xx = score[idx], local[idx], X[idx]
    od = np.argsort(-sc)
    for j, r in enumerate(rs):
        k = int(round(r * n))
        y = lo_.copy()
        sel = od[:k]
        ok = ~np.isnan(xx[sel])
        y[sel[ok]] = xx[sel[ok]]
        band[bi, j] = np.nanmean(y)
lo_b, hi_b = np.percentile(band, [2.5, 97.5], axis=0)

# ---- 右图数据:v2→v3 配对差值 ----------------------------------------
rows = []
for bench, col in POOLS.items():
    for t in TIERS:
        o = []
        for ver in ("_v2", "_v3"):
            d = pd.read_parquet(f"{DATA}/{bench}{ver}_traces.parquet")
            c = col if col in d.columns else "heard_ok"
            o.append(d[d["tier"] == t].set_index("id")[c]
                     .dropna().astype(int))
        ids = o[0].index.intersection(o[1].index)
        x, y = o[0].loc[ids], o[1].loc[ids]
        n01 = int(((x == 0) & (y == 1)).sum())
        n10 = int(((x == 1) & (y == 0)).sum())
        nn = len(ids)
        rows.append((f"{bench[:9]} {t[:4]}", (n01 - n10) / nn,
                     np.sqrt(max(n01 + n10, 1)) / nn))

# ---- 版式 ------------------------------------------------------------
fig = plt.figure(figsize=(11.5, 10.6))
gs = fig.add_gridspec(2, 2, height_ratios=[5.0, 4.4], hspace=.12,
                      wspace=.24)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
axc = fig.add_subplot(gs[1, :])
axc.axis("off")
for ax in (ax1, ax2):
    ax.grid(color=GRID, lw=.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.5)

ax1.fill_between(rs, lo_b, hi_b, color=BLUE, alpha=.15, zorder=2)
ax1.plot(rs, ys, "-", lw=2.0, color=BLUE, zorder=4)
for t in TIERS:
    r, acc = f["arm_rates"][t], f["arm_acc"][t]
    ax1.plot(r, acc, "o", ms=6.5, color=BLUE, zorder=5)
ax1.plot(.5625, .621, "s", ms=8, color=RED, zorder=5)
ax1.annotate("旧探针(v2)的 .621\n当时叫\"退步\"", (.5625, .621),
             xytext=(-8, 16), textcoords="offset points", fontsize=8.5,
             color=RED, ha="right")
a_fix = f["acc_at_budget"]["aggressive"]
ax1.plot(.50, a_fix, "*", ms=15, color=GREEN, zorder=6)
ax1.annotate("阈值修正到 50% 后:.600\n(实测超发 61% 时:.596)",
             (.50, a_fix), xytext=(6, -66), textcoords="offset points",
             fontsize=8.5, color=GREEN,
             arrowprops=dict(arrowstyle="-", color=GREEN, lw=.8,
                             alpha=.6))

def tag(ax, x, y, nlab, color):
    ax.annotate(nlab, (x, y), fontsize=13, color="#fff", ha="center",
                va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=.18", fc=color, ec="none"))

tag(ax1, .18, .513, "①", BLUE)
tag(ax1, .32, .615, "②", BLUE)
tag(ax1, .655, .638, "③", RED)
tag(ax1, .445, .603, "④", GREEN)
ax1.set_xlim(-.02, .72)
ax1.set_xlabel("升级率", fontsize=9.5)
ax1.set_ylabel("准确率(我们的判分器)", fontsize=9.5)
ax1.set_title("frozen 池:重建曲线 + 我们自己的噪声带",
              fontsize=10.5, loc="left")

ys_ = np.arange(len(rows))[::-1]
ax2.axvline(0, color=INK, lw=1, alpha=.5, zorder=3)
for (lab, dd, se), yy in zip(rows, ys_):
    ax2.plot([dd - 1.96 * se, dd + 1.96 * se], [yy, yy], color=MUT,
             lw=1.4, alpha=.75, zorder=4)
    ax2.plot(dd, yy, "o", ms=4.5, color=BLUE, zorder=5)
ax2.set_yticks(ys_)
ax2.set_yticklabels([r[0] for r in rows], fontsize=6.6)
ax2.set_xlim(-.12, .12)
ax2.set_xlabel("探针 v2 → v3 的准确率变化(配对 95% 区间)", fontsize=9.5)
ax2.set_title("18 个 live 差值全部跨过 0 线", fontsize=10.5, loc="left")
tag(ax2, .095, ys_[2], "⑤", MUT)

def _wrap(t, w=60):
    out, line = [], ""
    for ch in t:
        line += ch
        if len(line) >= w and ch in "。;;,,)】\" ":
            out.append(line)
            line = ""
    if line:
        out.append(line)
    return chr(10).join(out)


COMMENTS = [
    ("①", BLUE, "左图蓝线不是新实验:是把已经测过的结果【重新混合】画出的连续曲线。"
     "为什么可以这么做:三个档位的升级集合完全嵌套、探针分数逐位相同,所以"
     "\"升级率 = r 时会发生什么\"对每道题都有现成的实测答案,$0 成本。"),
    ("②", BLUE, "蓝色阴影带 = 我们自己的复现噪声(95% 区间)。它的来源实测过:"
     "同一道题、同一段音频、两次都留在本地,判分结果有 2.3%~18.8% 的概率翻转"
     "(专家采样 + 判分器波动)。换算下来,一个臂的准确率天生有 ±.02~.03 的抖动。"),
    ("③", RED, "红方块:旧探针(v2)在这个档的 .621。新探针(v3)测得 .596,当时我们写成"
     "\"v3 退步了\"——但它完全落在噪声带里,配对检验 p=.44。那个\"退步\"是我们"
     "对着噪声讲了故事,已在 §8ad 正式撤回。"),
    ("④", GREEN, "绿星:阈值本该升 50% 却实际升了 61%(标定漂移)。把它修回 50%,"
     "准确率只动 +.004——所以这个 bug 伤的是【成本】(白白多花 11% 的专家调用),"
     "不是准确率。修正后的阈值已入库。"),
    ("⑤", MUT, "右图:新旧探针在 6 个数据集 × 3 个档位的全部 18 个 live 对比,"
     "置信区间全部跨 0——【单次 live 实验分辨不了小于 3 分的差别】。"
     "v3 更好的证据在离线 AUC(统计上紧得多,.860→.879),不在这些曲线上。"),
    ("⑥", INK, "给读者的规矩(本图存在的意义):比较两个版本/两个策略,要用配对检验、"
     "AUC、同率 precision 这类低方差读数;臂准确率的小数点后第二位是噪声,"
     "谁拿它讲故事都不要信——包括我们自己(③就是案例)。"),
]
y = .99
for mark, c, txt in COMMENTS:
    wrapped = _wrap(txt)
    nlines = wrapped.count(chr(10)) + 1
    axc.annotate(mark, (.008, y), xycoords="axes fraction", fontsize=12,
                 color="#fff", ha="center", va="top",
                 bbox=dict(boxstyle="circle,pad=.15", fc=c, ec="none"))
    axc.text(.038, y, wrapped, transform=axc.transAxes, fontsize=9.3,
             color=INK, va="top",
             bbox=dict(boxstyle="round,pad=.35", fc="#fbfbfa",
                       ec=GRID, lw=.8))
    y -= .052 * nlines + .048

fig.suptitle("noise_audit 讲解版 — 我们的测量精度有多高,哪些差别不该解读",
             fontsize=12, x=.09, ha="left")
fig.savefig("noise_audit_annotated.png", dpi=200, bbox_inches="tight")
fig.savefig("noise_audit_annotated.pdf", bbox_inches="tight")
print("wrote noise_audit_annotated.{png,pdf}")
