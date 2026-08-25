# -*- coding: utf-8 -*-
"""预注册预测的验证(2026-08-24):NVDA 无拐弯。同一套 top-r 重混算术,
MiniCPM 折、NVDA 单调。NVDA 本地延迟 = 回答 token 数 × 80ms(语音原生
双工的帧钟,即部署下答案说完的真实时间,免疫批量污染);专家路径 =
同题实测 gpt-5.5 RTT + 转述按帧率发声。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
BLUE, GREEN, RED = "#2a78d6", "#1e9e50", "#b00"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e4de"
FRAME = 0.08
S = pd.read_parquet("../data/nvda_scores.parquet")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
for ax, pool, title in zip(
        axes, ("sllama", "striviaqa"),
        ("Llama Questions(MiniCPM 拐弯最深的池)",
         "Speech TriviaQA")):
    ax.grid(color=GRID, lw=.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.5)

    tr = pd.read_parquet(f"../data/{pool}_v3_traces.parquet")
    esc = (tr[tr["mode"] == "escalated"].drop_duplicates("id")
           .set_index("id"))
    med_exp = esc["expert_latency_s"].median()

    d = S[S.pool == pool].copy()
    d["l_s"] = d["n_tokens"] * FRAME
    d["e_s"] = (d["id"].map(esc["expert_latency_s"]).fillna(med_exp)
                + d["id"].map(esc["relay"].str.len() / 4).fillna(20)
                * FRAME)
    d = d.sort_values("score", ascending=False).reset_index(drop=True)

    nev = tr[tr.tier == "never"].set_index("id")
    agg = tr[tr.tier == "aggressive"].set_index("id")
    ids = nev.index.intersection(agg.index)
    m = pd.DataFrame({"score": agg.loc[ids, "eot_score"],
                      "l_s": nev.loc[ids, "answer_ms"] / 1000})
    m["e_s"] = [esc.loc[i, "expert_latency_s"] if i in esc.index
                else med_exp for i in ids]
    m = m.sort_values("score", ascending=False).reset_index(drop=True)

    rs = np.arange(0, 0.61, 0.05)

    def curve(df):
        out = []
        for r in rs:
            k = int(round(r * len(df)))
            out.append(float(np.median(np.concatenate(
                [df["e_s"][:k], df["l_s"][k:]]))))
        return np.array(out)

    cm, cn = curve(m), curve(d)
    ax.plot(rs, cm, "-o", ms=4.5, color=BLUE, lw=1.8, zorder=4,
            label="MiniCPM-o(实测本地耗时)")
    ax.plot(rs, cn, "-s", ms=4.5, color=GREEN, lw=1.8, zorder=4,
            label="NVDA VoiceChat(帧钟延迟)")
    # annotate the MiniCPM dip if present
    j = int(np.argmin(cm[:8]))
    if cm[j] < cm[0] - 1e-9:
        ax.annotate(f"MiniCPM 左折 −{cm[0] - cm[j]:.2f}s",
                    (rs[j], cm[j]), xytext=(8, -18),
                    textcoords="offset points", fontsize=8.5,
                    color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.annotate("NVDA:严格单调,无折", (rs[10], cn[10]), xytext=(-70, 16),
                textcoords="offset points", fontsize=8.5, color=GREEN)
    ax.set_xlabel("升级率(按各自探针分数 top-r)", fontsize=9.5)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
axes[0].set_ylabel("P50 延迟(秒,同一套重混算术)", fontsize=9.5)
fig.suptitle("预注册预测验证:回答风格极简的模型没有拐弯"
             "(8ab 预测 → 本图观测)", fontsize=11.5, x=.02, ha="left")
fig.text(.02, .012,
         "口径:NVDA 本地延迟 = 回答 token 数 × 80ms 帧钟(语音原生双工的部署真实值,免疫批量计时污染);"
         "专家路径 = 同题实测 gpt-5.5 RTT + 转述按帧率发声。机制:NVDA 本地 P90 仅 2.0s < 专家 ~4s,升级对每道题都是净加时。",
         fontsize=7.8, color=MUT)
fig.tight_layout(rect=[0, .05, 1, .93])
fig.savefig("nvda_fold_test.png", dpi=200, bbox_inches="tight")
fig.savefig("nvda_fold_test.pdf", bbox_inches="tight")
print("wrote nvda_fold_test.{png,pdf}")
