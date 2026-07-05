# MiniCPM-o 4.5 — Full-Duplex Bad Cases (vibe check → 量化)

模型: `openbmb/MiniCPM-o-4_5` (9B), 单卡 H100 80GB, bf16/sdpa。
方法: harness 以 1 Hz 驱动 duplex loop, 在已知 chunk **T** 注入受控事件,
从模型每秒的 `is_listen` / `text` / `end_of_turn` / `kv_cache_length` 流里读指标。
下表每个数字都是 **5-run 失败率**(不是单跑 vibe),来源 `results_modal/`。

## Headline: 四个稳定复现的 bad case

| Bad case | 5-run 结果 | 严重度 | 对应 Further Question |
|---|---|---|---|
| **Backchannel 抢话** | 用户附和音期间 **5/5 抢话** (`interrupted_user_rate=1.0`, 短backchannel率=0.0) | 致命 | Proactive / turn-taking |
| **不显式纠正** | 看到新证据会念新值, 但 **0/5 显式承认之前说错** (`explicit_correction=0.0`) | 明显 | Talker 幻觉 |
| **记忆当场记、隔会儿忘** | 复述暗号 **5/5**, 但被问时答对仅 **1/5** (`recall=0.2`) | 致命 | Long-term 记忆 / compaction |
| **长程无界增长** | 120 chunk 不崩, 但 VRAM **+1.4GB**、KV **80/chunk 线性涨**、speak 延迟峰值 808ms | 明显 | Long-term / context overflow |

---

## 逐条 bad case

### BC-1 · Backchannel 抢话 (turn-taking) — 致命
- **场景**: 用户连续讲一段 monologue, 中间只有"嗯""对"这类附和,模型本应继续 listen。
- **实际**: 5/5 在用户还没说完(onset chunk 19.2 < 用户结束 chunk 20)就插话,且插的是一整段实义回复而非短backchannel(`is_short_backchannel_rate=0.0`)。
- **实录**: `嗯听上去你今天早上遇到的事情确实是挺多的，一连串的事情发生确实会让人`
- **为什么是 bad case**: 全双工最核心的 turn-taking 判断失败,把"附和"误当"轮换"。直接支撑文章里 proactive/interrupt 需要专门训练的论点。
- 证据: `results_modal/p5_backchannel.{metric.json,jsonl}`

### BC-2 · 看到新证据不显式纠正 (Talker 幻觉) — 明显
- **场景**: 画面 2 个点 → 变 5 个点,模型已就旧值(2)做过陈述(`stated_2_before_change=0.8`)。
- **实际**: 会在 ~3 chunk 后念出新数字,但 **0/5** 显式说"我刚才说错了/更正为5"(`explicit_correction_rate=0.0`);因果有效率仅 0.6。
- **为什么是 bad case**: talker 无自我否定/纠错话术,旧结论和新结论并列输出 → 用户侧表现为幻觉/前后矛盾。对应"talker 该学会 invoke thinker 并纠偏"。
- 证据: `results_modal/p7_online_correction.{metric.json,jsonl}`

### BC-3 · 记忆当场记、隔会儿忘 (Long-term 记忆) — 致命
- **场景**: 早期给出暗号"蓝色河马7392",隔 q1(15 chunk 后)/ q2(32 chunk 后)追问。
- **实际**: 复述率 **1.0**(当场会跟读),但真正追问时 recall 仅 **q1=0.2 / q2=0.2**;有效记忆距离 q2 只有 ~19 chunk。
- **实录(答非所问)**: `会。今天的暗号是蓝色河马7392。`(把追问答成了复述,没有真正检索)
- **为什么是 bad case**: 说明 KV 里"有"但检索不出 → 直接印证 Long-Term 方向里"thinker/talker 需要对历史 token 做 compaction/传递"。
- 证据: `results_modal/p9_memory_budget.{metric.json,jsonl}`

### BC-4 · 长程资源无界增长 (context overflow) — 明显
- **场景**: 连续 120 chunk(≈2min)持续输入。
- **实际**: 未崩溃(crashed=0),但 **VRAM +1404MB**、**KV 线性增长 80.3/chunk**、speak 延迟峰值 **808ms**、listen 漂移 33.6ms。趋势外推即文章说的"< 1min 触顶"。
- **为什么是 bad case**: 量化了 context 随时间无界积累的斜率 → 直接支撑 Full-Duplex Long-Term 里 omni-gating / compaction 的必要性。
- 证据: `results_modal/p8_long_horizon.{metric.json,jsonl}`(逐 chunk 的 kv/vram 曲线在 jsonl 里)

---

## 复现方式(Modal H100)
```bash
PYTHONUTF8=1 python -m modal run modal_app.py::run_eval --probes "p5 p7 p8 p9" --repeat 5
PYTHONUTF8=1 python -m modal volume get --force minicpm-o45-results report.md results_modal/report.md
```
完整逐 chunk trace 在各 `*.jsonl`;`report.md` 是自动汇总。
