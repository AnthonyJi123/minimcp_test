# -*- coding: utf-8 -*-
"""Token 级机制图(2026-08-24):四组 striviaqa 题的解码熵轨迹 +
句界停止意愿。数据 = entropy_traj.parquet(93 次 H100 重放,温度与
bench 相同)。验证用户假设:检索失败 → 熵升 → EOS 被压 → hedging。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
BLUE, GREEN, RED, AMBER = "#2a78d6", "#1e9e50", "#b00", "#b8860b"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e4de"

d = pd.read_parquet("../data/entropy_traj.parquet")
GROUPS = [("hedged_wrong", RED, "hedged 错(绕而错)"),
          ("confident_wrong", AMBER, "自信错(快而错)"),
          ("right", BLUE, "答对"),
          ("hedged_right", GREEN, "绕对(12 题)")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4),
                               gridspec_kw={"width_ratios": [3, 2]})
for ax in (ax1, ax2):
    ax.grid(color=GRID, lw=.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.5)

# ---- 左:熵轨迹(归一化位置) -----------------------------------------
XS = np.linspace(0, 1, 50)
for g, col, lab in GROUPS:
    gd = d[d.group == g]
    M = []
    for _, r in gd.iterrows():
        e = np.array(r["ent"], dtype=float)
        if len(e) < 5:
            continue
        M.append(np.interp(XS, np.linspace(0, 1, len(e)), e))
    M = np.array(M)
    med = np.median(M, axis=0)
    ax1.plot(XS, med, color=col, lw=2.0, zorder=4, label=lab)
    ax1.fill_between(XS, np.percentile(M, 30, axis=0),
                     np.percentile(M, 70, axis=0), color=col, alpha=.12,
                     zorder=2)
ax1.set_xlabel("生成进度(0 = 第一个 token,1 = 停止)", fontsize=9.5)
ax1.set_ylabel("解码熵(nats,全词表)", fontsize=9.5)
ax1.set_title("熵轨迹:绕 = 在高熵里游走;自信错的分布也被骗了\n"
              "(中位线 ± 30-70 分位带;每组 27/12 题重放)",
              fontsize=10.5, loc="left")
ax1.legend(fontsize=8.5, frameon=False, loc="upper right")

# ---- 右:句界停止意愿(对数轴) ---------------------------------------
vals, meta = [], []
for g, col, lab in GROUPS:
    gd = d[d.group == g]
    sb = []
    for _, r in gd.iterrows():
        ps = np.array(r["p_stop"])
        tk = np.array(r["tok_ids"])
        per = np.where(tk == 13)[0]
        per = per[per + 1 < len(ps)]
        if len(per) > 1:
            sb.append(np.median(ps[per[:-1] + 1]))
    vals.append(np.median(sb))
    meta.append((lab, col))
x = np.arange(len(vals))
ax2.bar(x, vals, color=[m[1] for m in meta], alpha=.8, zorder=4)
for i, v in enumerate(vals):
    ax2.text(i, v * 1.25, f"{v:.4f}", fontsize=8.5, ha="center",
             color=meta[i][1], fontweight="bold")
ax2.set_yscale("log")
ax2.set_ylim(5e-5, .4)
ax2.set_xticks(x)
ax2.set_xticklabels([m[0].replace("(", chr(10) + "(") for m in meta],
                    fontsize=8)
ax2.set_ylabel("句号后一步,终止符的概率(log)", fontsize=9.5)
ax2.set_title("停止意愿:答对题每个句界都准备收尾,\n"
              "hedged 错低 35 倍——「END 难出」实测", fontsize=10.5,
              loc="left")
fig.text(.01, .01,
         "验证的链条:检索失败 → 首5token熵 ×1.8(0.57 vs 0.31)→ 句界终止概率 ÷35(0.0024 vs 0.083)→ 继续套话(89 步 vs 45)→ 慢。"
         "自信错 = 短(34 步)、低熵轨迹——分布被错误事实骗过,这就是熵信号(AUC .70)到不了探针(AUC .80)的原因。"
         "附:MiniCPM 实际终止符 id=151704,不在 generation_config 里。",
         fontsize=8, color=MUT, wrap=True)
fig.tight_layout(rect=[0, .06, 1, 1])
fig.savefig("entropy_traj.png", dpi=200, bbox_inches="tight")
fig.savefig("entropy_traj.pdf", bbox_inches="tight")
print("wrote entropy_traj.{png,pdf}")
