# -*- coding: utf-8 -*-
"""自带解读的图5(swebq)讲解版(2026-08-24,用户要求"画一个图、
图底下写 comment 讲每条线怎么解释")。上半:三种运行形态的所有线,
每条挂圈号;下半:圈号对应的解读文字,直接印在图里。数据全部来自
bench_figures.json(v3 实测)。"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

BLUE, GREEN, RED = "#2a78d6", "#1e9e50", "#b00"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e4de"

d = json.load(open("bench_figures.json"))["swebq"]
esc = np.array(d["esc"])
heard = np.array(d["heard"])
lo = heard - np.array(d["heard_ci"][0])
hi = np.array(d["heard_ci"][1]) - heard
OFFICIAL, CHAT, QWEN = .702, .716, .749

fig = plt.figure(figsize=(8.6, 10.2))
gs = fig.add_gridspec(2, 1, height_ratios=[5.2, 4.8], hspace=.06)
ax = fig.add_subplot(gs[0])
axc = fig.add_subplot(gs[1])
axc.axis("off")

ax.grid(color=GRID, lw=.7, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=9)

# ① 官方锚线(离线 chat 模式)
ax.axhline(OFFICIAL, color=INK, ls=(0, (1, 3)), lw=1.6, alpha=.7, zorder=2)
ax.text(.985, OFFICIAL - .006, "官方 70.2(离线 chat 模式)",
        fontsize=9, color=MUT, ha="right", va="top",
        transform=ax.get_yaxis_transform())
# ② 我们的离线对照(桥)
ax.axhline(CHAT, color=BLUE, ls=(0, (5, 3)), lw=1.4, alpha=.55, zorder=2)
ax.text(.985, CHAT + .005, "我们的离线对照 .716(同批音频、同判分器)",
        fontsize=9, color=BLUE, alpha=.8, ha="right",
        transform=ax.get_yaxis_transform())
# ④ 对比模型(也是离线)
ax.axhline(QWEN, color=MUT, ls=(0, (5, 2, 1, 2)), lw=1.2, alpha=.55,
           zorder=2)
ax.text(.985, QWEN + .005, "Qwen3-Omni-30B 官方 .749(离线)",
        fontsize=9, color=MUT, ha="right",
        transform=ax.get_yaxis_transform())
# ③ 我们的实时曲线
ax.errorbar(esc, heard, yerr=[lo, hi], fmt="-o", ms=7, color=BLUE,
            capsize=3.5, lw=2.0, zorder=5)
for j, a in enumerate(("never", "conservative", "balanced", "aggressive")):
    off = (11, 2) if a == "aggressive" else (7, -26)
    ax.annotate(f"{a}\n{esc[j]:.0%} · {heard[j]:.3f}",
                (esc[j], heard[j]), xytext=off,
                textcoords="offset points", fontsize=8.5, color=BLUE)

# 两个"差距"箭头:loop 代价 和 路由赢回
ax.annotate("", xy=(0.028, heard[0] + .004), xytext=(0.028, CHAT - .004),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.3))
ax.text(0.042, (heard[0] + CHAT) / 2, "loop 代价 −.148",
        fontsize=9, color=RED, va="center")
ax.annotate("", xy=(0.5, heard[3] - .004), xytext=(0.5, heard[0] + .004),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
ax.text(0.514, (heard[0] + heard[3]) / 2, "gate 赢回 +.164",
        fontsize=9, color=GREEN, va="center")

# 圈号标记
def tag(x, y, n, color):
    ax.annotate(n, (x, y), fontsize=13, color="#fff", ha="center",
                va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=.18", fc=color, ec="none"))

tag(.062, OFFICIAL, "①", INK)
tag(.115, CHAT, "②", BLUE)
tag(.24, .636, "③", BLUE)
tag(.062, QWEN, "④", MUT)
tag(.435, .703, "⑤", MUT)

ax.set_xlim(-.02, .78)
ax.set_ylim(.46, .80)
ax.set_xlabel("实际升级率(送给 gpt-5.5 的比例)", fontsize=10)
ax.set_ylabel("准确率(OpenAudioBench 官方判分器)", fontsize=10)
ax.set_title("图5 讲解版 — Speech Web Questions:三种运行形态,"
             "什么能比、什么不能比\n(n=250,probe v3;所有线同一把判分尺)",
             fontsize=11, loc="left")

def _wrap(t, w=46):
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
    ("①", INK, "官方 70.2 是离线 chat 模式(turn-based:整段音频一次给模型、"
     "无对话人格、可写长答案)。它是【锚线】,用来对齐绝对水平——"
     "不能拿我们的曲线和它比大小,因为运行形态不同。"),
    ("②", BLUE, "桥:同 250 条音频、切回离线模式、同一判分器,我们得 .716 ≈ 官方 "
     ".702(+.014 = 250 题子采样噪声)。这条线证明【官方数字我们复现得出来】,"
     "下面所有差距都不是\"能力丢了\"或\"测错了\"。"),
    ("③", BLUE, "唯一的\"系统\"数字:实时双工 loop 的四个臂(同题、同 loop、同尺,"
     "内部配对)。floor .568 比②低 .148 = 【loop 代价】:实时人格答案短"
     "(中位 820 vs 2186 字符)+ 512 token 截断 + 流式分块。曲线内部的"
     "上升(.568→.732)才是 gate 的因果贡献。"),
    ("④", MUT, "对比模型也全是离线数字。当我们的【实时】@50% (.732) 接近它的"
     "【离线】.749 时,这个比较方向对我们是保守的——离线普遍高于实时,"
     "若 Qwen 被迫跑实时只会更低。反向比较(拿我们离线打别人离线)不做:"
     "对方无双工接口,完全对齐的实时对局不存在。"),
    ("⑤", MUT, "@50% 的 .732 回到官方线①之上:实时系统靠路由把 turn-based "
     "让掉的分挣回来了。注意误差棒 ±.05:复现噪声地板(§8ad)是每臂 "
     "±.02-.03,小于 3 分的差别不要解读。"),
    ("⑥", GREEN, "严格的对比阶梯是四层:无 router(.568)< 随机升级@同预算 < "
     "我们的 router(.732)< 全升级(.864)。本图只画了首尾两层;"
     "router 对随机的净超额(+.045/+.035/+.066)在正式图5(dualview)"
     "的灰虚线上——那才是\"挑得准\"的证据,这里的 +.164 是\"机制+挑得准\"的总和。"),
]
y = .99
for mark, c, txt in COMMENTS:
    wrapped = _wrap(txt)
    nlines = wrapped.count(chr(10)) + 1
    axc.annotate(mark, (.012, y), xycoords="axes fraction", fontsize=12,
                 color="#fff", ha="center", va="top",
                 bbox=dict(boxstyle="circle,pad=.15", fc=c, ec="none"))
    axc.text(.05, y, wrapped, transform=axc.transAxes, fontsize=9.3,
             color=INK, va="top",
             bbox=dict(boxstyle="round,pad=.35", fc="#fbfbfa",
                       ec=GRID, lw=.8))
    y -= .045 * nlines + .045

fig.savefig("swebq_annotated.png", dpi=200, bbox_inches="tight")
fig.savefig("swebq_annotated.pdf", bbox_inches="tight")
print("wrote swebq_annotated.{png,pdf}")
