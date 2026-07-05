# MiniCPM-o 4.5 — 全双工能力总表

模型 `openbmb/MiniCPM-o-4_5` (9B) · 单卡 H100 · 1Hz duplex loop · 注入受控事件于 chunk T。
1 chunk ≈ 1 秒。✅=强 / ⚠️=隐患 / ❌=bad case。p5/p7/p8/p9 为 5-run 率,其余为单跑。

| # | 能力 | 注入的事件 | 关键结果 | 判定 |
|---|------|-----------|---------|------|
| p1 | 边说边听(说话时执行语音指令) | 说话中说"switch to English" | 立即照做,延迟 **0** chunk,因果成立 | ✅ |
| p2 | 用户打断("停") | 说话中说"停" | **3** chunk 后停并致歉("好,我停下来了") | ✅ |
| p3 | 视觉打断(改口) | 画面 A→B 切换 | **3** chunk 后改描述新画面,因果成立 | ✅ |
| p4 | 跨模态冲突(信谁) | 图红 vs 语音说"绿" | **两个方向都信视觉**(忽略语音) | ✅ 但存偏置 |
| p6 | 主动提醒(该静则静) | 空闲→突发警示画面 | 空闲不误报,警示后 **0** chunk 提醒 | ✅ |
| **p5** | **Backchannel(附和≠轮换)** | 用户 monologue 中"嗯/对" | **5/5 抢话**,插整段实义回复(短backchannel率 0) | ❌ |
| **p7** | **在线纠错(承认说错)** | 2个点→5个点 | 会念新值,但 **0/5 显式纠正**;因果率仅 0.6 | ❌ |
| **p8** | **长程稳定性** | 连续 120 chunk(≈2min) | 不崩,但 **VRAM +1.4GB、KV 线性 +80/chunk、说话延迟峰 808ms** | ⚠️ |
| **p9** | **记忆(隔距离检索)** | 早期给暗号,隔 15/32 chunk 追问 | **复述率 1.0 但答对率仅 0.2**(当场跟读得出、追问答不出) | ❌ |

## 四个 bad case(对应文章 Further Questions)

| Bad case | 一句话 | 数字 | 方向 |
|---|---|---|---|
| Backchannel 抢话 | 把用户附和当成轮到自己 | 抢话 5/5 | Proactive / turn-taking |
| 不显式纠错 | 念新结论但不否定旧结论 → 前后矛盾 | 显式纠正 0/5 | Talker 幻觉 |
| 记得住≠取得出 | KV 里有暗号但检索不回来 | 复述 1.0 / recall 0.2 | Long-term 记忆 compaction |
| 资源无界增长 | context 随时间线性堆积 → <1min 触顶 | VRAM +1.4GB / 120chunk | Context overflow |

数据来源: `results_modal/*.metric.json` · 逐 chunk trace: `results_modal/*.jsonl`
