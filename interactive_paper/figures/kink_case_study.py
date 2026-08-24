# -*- coding: utf-8 -*-
"""拐弯案例走查(2026-08-24,用户:"从错题簿找一个典型,用 trace 说明
拐弯怎么出现")。单题 sllama0164 的两个世界 + 池效应,全部实测毫秒,
无一处模拟。讲解版版式。"""
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

df = pd.read_parquet("../data/sllama_v3_traces.parquet")
n = df[(df.tier == "never") & (df.id == "sllama0164")].iloc[0]
c = df[(df.tier == "conservative") & (df.id == "sllama0164")].iloc[0]
THR = json.load(open("../data/gate_v3_thresholds_corrected.json"))[
    "sllama"]["threshold_corrected"]["conservative"]
nev = df[df.tier == "never"].set_index("id")
cons = df[df.tier == "conservative"].set_index("id")
esc_ids = cons[cons["mode"] == "escalated"].index
P_ALL = nev["answer_ms"].median() / 1000                       # 1.50
P_ESC = nev.loc[nev.index.intersection(esc_ids),
                "answer_ms"].median() / 1000                    # 2.38
P_KEPT = cons[cons["mode"] == "local"]["answer_ms"].median() / 1000  # 0.94

A_dec = n["answer_ms"] / 1000                                  # 3.103
B_eot = c["eot_read_ms"] / 1000                                # .021
B_exp = c["expert_latency_s"]                                  # 1.68
B_rel = c["relay_ms"] / 1000                                   # .619
B_tot = B_eot + B_exp + B_rel
AUD = n["audio_s"]                                             # 2.52

fig = plt.figure(figsize=(10.5, 11.2))
gs = fig.add_gridspec(3, 1, height_ratios=[3.4, 1.6, 5.2], hspace=.30)
ax = fig.add_subplot(gs[0])          # timelines
axp = fig.add_subplot(gs[1])         # pool effect
axc = fig.add_subplot(gs[2])
axc.axis("off")

# ---------------- Row 1: 两个世界的时间线(实测) ----------------------
ax.set_xlim(-AUD - .2, 3.6)
ax.set_ylim(-.4, 2.6)
ax.set_yticks([])
ax.grid(color=GRID, lw=.7, axis="x", zorder=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.axvline(0, color=INK, lw=1.2, alpha=.7, zorder=3)
ax.text(0, 2.52, "用户话音落下(t=0)\n探针在此读数", fontsize=8.5,
        color=INK, ha="center")

def bar(y, x0, w, color, label=None, alpha=1.0, h=.34):
    ax.barh(y, w, left=x0, height=h, color=color, alpha=alpha, zorder=4)
    if label:
        ax.text(x0 + w / 2, y, label, fontsize=8, color="#fff",
                ha="center", va="center", zorder=5, fontweight="bold")

# 音频段(两个世界相同)
for y in (2.0, 0.9):
    bar(y, -AUD, AUD, MUT, alpha=.30)
ax.text(-AUD / 2, 2.30, "音频流入 2.5 s(1 s/块)", fontsize=8, color=MUT,
        ha="center")
# 世界A:探针关
bar(2.0, 0, A_dec, BLUE, f"本地解码 {A_dec:.2f} s(487 字符的绕)")
ax.text(A_dec + .06, 2.0, "× 答错", fontsize=11, color=RED, va="center",
        fontweight="bold")
ax.text(-AUD - .15, 2.0, "世界 A\n探针关", fontsize=9, ha="right",
        va="center", color=INK)
# 世界B:探针开(cons 臂实测)
bar(0.9, 0, B_eot, INK)
bar(0.9, B_eot, B_exp, GREEN, f"gpt-5.5 {B_exp:.2f} s")
bar(0.9, B_eot + B_exp, B_rel, GREEN, f"转述 {B_rel:.2f} s", alpha=.55)
ax.annotate(f"句尾读数 {c['eot_read_ms']:.0f} ms\n(+stall 掩护 21 ms)",
            (B_eot / 2, 0.9), xytext=(-8, -40),
            textcoords="offset points", fontsize=8, color=INK,
            ha="right",
            arrowprops=dict(arrowstyle="-", color=INK, lw=.7, alpha=.6))
ax.text(B_tot + .06, 0.9, "对 √", fontsize=11, color=GREEN,
        va="center", fontweight="bold")
ax.text(-AUD - .15, 0.9, "世界 B\n探针开", fontsize=9, ha="right",
        va="center", color=INK)
# 差值箭头
ax.annotate("", xy=(B_tot, 1.55), xytext=(A_dec, 1.55),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
ax.text((A_dec + B_tot) / 2, 1.66, f"快 {A_dec - B_tot:.2f} s,且从错变对",
        fontsize=9.5, color=GREEN, ha="center", fontweight="bold")
# 探针分数(B 世界音频段上)
sc = list(c["scores"])
for i, s in enumerate(sc):
    x = -AUD + (i + 1) * AUD / len(sc)
    ax.annotate(f"{s:.2f}", (x - AUD / len(sc) / 2, 0.9), xytext=(0, 26),
                textcoords="offset points", fontsize=7.5, color=BLUE,
                ha="center")
ax.annotate(f"eot={c['eot_score']:.3f} ≥ 阈值 {THR:.3f} → 升级",
            (0.02, 0.9), xytext=(30, 30), textcoords="offset points",
            fontsize=8.5, color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=.7, alpha=.6))
ax.set_xlabel("秒(t=0 为话音落下;左侧为流式听音阶段)", fontsize=9)
ax.set_title("一道题的两个世界 — sllama0164「锡克教有几位祖师?」"
             "(参考答案:十)\n同一段音频、同一台 H100,全部为实测毫秒",
             fontsize=11, loc="left")

# ---------------- Row 2: 池效应 ---------------------------------------
axp.set_xlim(0, 2.8)
axp.set_ylim(-.6, 2.6)
axp.set_yticks([])
axp.grid(color=GRID, lw=.7, axis="x", zorder=0)
for sp in ("top", "right", "left"):
    axp.spines[sp].set_visible(False)
rows = [("never 臂:全部 250 题都本地答", P_ALL, MUT),
        ("其中探针要送走的 38 题(都是上面这种)", P_ESC, RED),
        ("cons 臂:送走后留下的 212 题", P_KEPT, BLUE)]
for i, (lab, v, col) in enumerate(rows):
    y = 2 - i
    axp.barh(y, v, height=.42, color=col, alpha=.75, zorder=4)
    axp.text(v + .04, y, f"{v:.2f} s", fontsize=9, va="center", color=col,
             fontweight="bold")
    axp.text(0.02, y + .33, lab, fontsize=8.5, color=INK)
axp.set_xlabel("本地解码 P50(秒)—— 整臂中位随之 1.52 → 1.17 s,"
               "即图8 的左折", fontsize=9)

# ---------------- Row 3: 讲解 -----------------------------------------
def _wrap(t, w=52):
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
    ("①", RED, "世界 A(探针关,never 臂实测):小模型不会这道题,但它没有说"
     "\"不知道\",而是绕——487 字符,先断言\"只有一位 Guru\",再展开一段"
     "辨析(\"However… to avoid confusion…\"),解码整整 3.10 秒,判分:错。"),
    ("②", BLUE, "世界 B(探针开,conservative 臂实测):同一段音频流入时,"
     "探针逐块的运行分数已经在高位(0.71/0.67/0.66);话音一落,21 毫秒读出 "
     "eot=0.631 ≥ 阈值 0.513 → 升级。注意:此刻本地回答一个字都还没生成"
     "——探针读的是内部状态,不是回答文本。"),
    ("③", GREEN, "升级路径:stall 话术(21 ms)掩护下,gpt-5.5 用 1.68 秒给出"
     "\"十位人间祖师\",talker 用 0.62 秒转述——总计 2.32 秒。"
     "比世界 A 快 0.78 秒,并且从错变对。这不是特例:cons 档送走的 38 题里,"
     "这样\"又快又对\"的结构反复出现(它们的本地 P50 高达 2.38 秒)。"),
    ("④", MUT, "拐弯就是把中间那根红条从本地队列里抽走:全池本地 P50 本来 "
     "1.50 秒;38 道慢题(P50 2.38 秒)离开后,留下 212 题的 P50 掉到 0.94 秒。"
     "38 题走专家路径(约 3 秒)但只占 15% 的质量,压不过 85% 变快的本地质量"
     "——于是整臂中位 1.52 → 1.17 秒,左折出现。"),
    ("⑤", INK, "边界(8ab Addendum 4 的修正):探针挑的是\"会错\",不是"
     "\"会绕\"——它对不绕的自信错抓得一样准(81% vs 81%)。拐弯只在"
     "\"会错的题恰好也慢\"的池子出现(sllama/striviaqa 的 hedging、"
     "sreason 的长推理链);sdqa 错得又快又干脆,就没有拐弯。"),
]
y = .995
for mark, col, txt in COMMENTS:
    wrapped = _wrap(txt)
    nl = wrapped.count(chr(10)) + 1
    axc.annotate(mark, (.010, y), xycoords="axes fraction", fontsize=12,
                 color="#fff", ha="center", va="top",
                 bbox=dict(boxstyle="circle,pad=.15", fc=col, ec="none"))
    axc.text(.045, y, wrapped, transform=axc.transAxes, fontsize=9.2,
             color=INK, va="top",
             bbox=dict(boxstyle="round,pad=.35", fc="#fbfbfa", ec=GRID,
                       lw=.8))
    y -= .052 * nl + .042

fig.savefig("kink_case_study.png", dpi=200, bbox_inches="tight")
fig.savefig("kink_case_study.pdf", bbox_inches="tight")
print("wrote kink_case_study.{png,pdf}")
