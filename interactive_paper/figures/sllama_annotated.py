# -*- coding: utf-8 -*-
"""讲解版之二:sllama(Llama Questions)——"选择性升级 ≥ 全部升级"
这个全项目最强结果,含 8ad 统计修正后的诚实表述。格式同
swebq_annotated.py:上图挂圈号,下方印对应解读。数据:
bench_figures.json(v3)+ 8ad 配对检验数字。"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

BLUE, GREEN, RED = "#2a78d6", "#1e9e50", "#b00"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e4de"

d = json.load(open("bench_figures.json"))["sllama"]
esc = np.array(d["esc"])
heard = np.array(d["heard"])
lo = heard - np.array(d["heard_ci"][0])
hi = np.array(d["heard_ci"][1]) - heard
ALWAYS = d["ceiling"]                      # .928,每题都给 gpt-5.5

fig = plt.figure(figsize=(8.6, 10.2))
gs = fig.add_gridspec(2, 1, height_ratios=[5.2, 4.8], hspace=.06)
ax = fig.add_subplot(gs[0])
axc = fig.add_subplot(gs[1])
axc.axis("off")

ax.grid(color=GRID, lw=.7, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=9)

# ② 全部升级线
ax.axhline(ALWAYS, color=GREEN, ls=":", lw=1.8, alpha=.8, zorder=2)
ax.text(.985, ALWAYS - .005, "全部升级:每题都给 gpt-5.5 = .928",
        fontsize=9, color=GREEN, ha="right", va="top",
        transform=ax.get_yaxis_transform())
# ① 实时曲线
ax.errorbar(esc, heard, yerr=[lo, hi], fmt="-o", ms=7, color=BLUE,
            capsize=3.5, lw=2.0, zorder=5)
for j, a in enumerate(("never", "conservative", "balanced", "aggressive")):
    off = (10, 4) if a == "aggressive" else (7, -26)
    ax.annotate(f"{a}\n{esc[j]:.0%} · {heard[j]:.3f}",
                (esc[j], heard[j]), xytext=off,
                textcoords="offset points", fontsize=8.5, color=BLUE)

# ③ @50% 与全升级的对比箭头
ax.annotate("", xy=(.55, heard[3]), xytext=(.55, ALWAYS),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
ax.text(.563, (heard[3] + ALWAYS) / 2, "+.020(p=.125,见③)",
        fontsize=8.5, color=INK, va="center")

# ④ 分解框:把 aggressive 档的 250 题拆开
ax.text(.585, .868,
        "把 @50% 的 250 题拆开看(④):\n"
        "  留本地的 125 题:小模型 .976,专家 .968 → 打平\n"
        "  送云端的 125 题:小模型 .696,专家 .888 → 专家救回",
        fontsize=8.6, color=INK, va="top",
        bbox=dict(boxstyle="round,pad=.4", fc="#fbfbfa", ec=GRID, lw=.9))

def tag(x, y, n, color):
    ax.annotate(n, (x, y), fontsize=13, color="#fff", ha="center",
                va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=.18", fc=color, ec="none"))

tag(.24, .916, "①", BLUE)
tag(.06, ALWAYS, "②", GREEN)
tag(.505, .939, "③", INK)
tag(.565, .878, "④", MUT)

ax.set_xlim(-.02, 1.0)
ax.set_ylim(.78, 1.0)
ax.set_xlabel("实际升级率(送给 gpt-5.5 的比例)", fontsize=10)
ax.set_ylabel("准确率(OpenAudioBench 官方判分器)", fontsize=10)
ax.set_title("sllama 讲解版 — Llama Questions:会挑的路由,花一半的钱"
             "拿到全升级的分\n(n=250,probe v3;全项目最强正面结果,"
             "含统计修正后的表述)", fontsize=11, loc="left")

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
    ("①", BLUE, "蓝色曲线 = 实时系统,同样 250 道题跑四遍,唯一区别是送给 "
     "gpt-5.5 的比例。这个池是英文短事实题,小模型底子好(不升级也有 .836),"
     "所以整条曲线都贴着天花板。"),
    ("②", GREEN, "绿色虚线 = \"无脑策略\":每道题都送 gpt-5.5,得 .928。"
     "注意它花的专家调用是 @50% 档的【两倍】。直觉上它该是准确率上界——"
     "这张图的看点就是它不是。"),
    ("③", INK, "@50% 的 .948 比全升级的 .928 高 .020——但诚实地说(§8ad "
     "配对检验):250 题里两种策略只在 7 题上结果不同(6 比 1),p=.125,"
     "\"更高\"不能单独下结论。能下的结论是:【至少打平,而专家调用省了一半】。"),
    ("④", MUT, "为什么会这样(这部分是铁的):探针把 250 题干净地分成两半——"
     "它留在本地的 125 题,小模型自己 .976(专家也只有 .968,配对 p=1.0,"
     "打平);它送云的 125 题,小模型只有 .696,专家救到 .888(p<.0001)。"
     "这个分割的强度 z=6.5——探针真的知道哪些题小模型会挂。"),
    ("⑤", GREEN, "所以正确的头条不是\"小模型打败了 gpt-5.5\"(那句在打平的 "
     "125 题上不成立),而是:【\"全部送云\"不是上界,会挑的路由用一半的成本"
     "拿到同样好(方向上更好)的结果】。对系统设计者:专家预算减半,零代价。"),
    ("⑥", MUT, "此池官方没有公布 MiniCPM 数字,所以没有官方锚线;判分器是 "
     "OpenAudioBench 官方的(gpt-4o + 官方 prompt),对比公平。此池复现噪声"
     "全项目最低(同题重测翻转率 2.3%),④的分解在所有池里最可信。"),
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

fig.savefig("sllama_annotated.png", dpi=200, bbox_inches="tight")
fig.savefig("sllama_annotated.pdf", bbox_inches="tight")
print("wrote sllama_annotated.{png,pdf}")
