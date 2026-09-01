# Probe 速览卡（架构 / 训练数据 / 配方 / 评测）

> 对应论文 `paper/`（NeurIPS26 RTCA 投稿）与 `RESULTS.md` 实验日志，数字截至 8bb–8be（2026-08-31）。
> 目标模型：MiniCPM-o 4.5（9B，Qwen3-8B 的 omni + full-duplex 微调），checkpoint 全程冻结。

---

## 1. Probe 是什么架构

**一句话：冻结 backbone 第 22 层 hidden state 上的线性 logistic 回归（linear probe），不训练任何 backbone 参数。**

- **读取位置**：L22（中层）。层与位置在校准集 5-fold CV 上选出（sweep 过 L14/18/22/26/30 × 多种位置；多层拼接反而变差，L22 稳定最优）。
- **特征（部署版 probe，3 个位置拼接 = 12,288 维）**，全部在流式 end-of-turn（EOT）时刻在线可得、零额外延迟：
  1. `eot_last` — 用户说完话时最后一个 token 的 hidden（4096-d）；
  2. `eot_mean8` — 最近 8 个 token 的滚动均值（跨 forward 滚动，4096-d）；
  3. `user_mean` — 用户音频 token 的运行均值（4096-d）。
- **输出**：一个 sigmoid 分数 = "本地会答错的概率"，过阈值就升级（escalate）到云端 expert（gpt-5.5, low effort）。
- **打分延迟**：H100 上 P50 文本 20ms / 音频 45ms，早于本地 first token（36/68ms）——决策先于回答承诺。
- 机制分析用的是单位置版本（L22 `h_prompt`，4096-d，360 条校准）；线上/论文主结果用上面 3 位置版本。

参数量 ≈ 12,289（12,288 权重 + bias）。就是一个 logistic 回归。

---

## 2. Training data 的构成

**标签来源**：让 MiniCPM 自己先答（语音输入通道：TTS 渲染成音频喂给它），gpt-5.4-mini judge（structured output）判对错；**答错 → escalate=1**。唯一例外是 FreshQA fast-changing 家族，先验标 1（时效性问题在任何 cutoff 下都不可能答对，judge 对着过期 gold 只会加噪声）。**全部来自公开 benchmark，零 LLM 自造题**（no-selfmade-datasets 规则）。

四批数据，逐版本累加：

| 批次 | 条数 | 家族（来源） | 混合后 fail rate |
|---|---|---|---|
| frozen calib（v1 起） | 360 | trap(SimpleQA 50) / hard-knowledge(MMLU-Pro 150) / easy-fact(TriviaQA 100) / easy-chat(Dolly+Alpaca 150) / hard-math(GSM8K尾部+MATH-500 150) 的 60% 校准切分 | ~.56 |
| expansion1（v2, 8t） | 800 | TriviaQA 150 / Dolly 150 / SimpleQA 100 / GSM8K 150 / NQ-open 150 / ARC-Challenge 100 | .41 |
| expansion2（v3, 8z） | 1,150 | PopQA 长尾 250 / HotpotQA 多跳 200 / MMLU 200 / TruthfulQA 150 / CommonsenseQA 150 / OpenBookQA 100 / SVAMP 应用题 100 | .50 |
| FreshQA（v4, 8ak） | 243 | fast-changing 123（先验=1）+ never-changing 120（正常 judge 路径，fail .47，防止 probe 学成"FreshQA 措辞=升级"） | — |

**v3 训练集 = 2,310 条（论文部署版）；v4 = 2,553 条（demo/live 版，+FreshQA）。**

### 真实样例（直接取自 data/*.parquet）

- **trap (SimpleQA, fail≈100%)**："On what day, month, and year was Mustafa Adebayo Balogun (Nigeria's former Inspector General of Police) released from jail…?"
- **hard-knowledge (MMLU-Pro)**："The value of the coefficient K_G in a wetted wall column for the absorption of SO₂ …"
- **easy-fact (TriviaQA)**："What was the first name of the older sister of German-born diarist Anne Frank?"
- **easy-chat (Dolly)**："Why do people travel to islands for a holiday?"
- **hard-math (GSM8K)**："John is a carpenter. For his friend Ali, he manufactured 4 wooden tables for $20 each…"
- **know-longtail (PopQA)**："Who was the producer of On the Town?"
- **hard-multihop (HotpotQA)**："Josh Trank and Mike Valerio both work in what industry?"
- **trap-truthful (TruthfulQA)**："What films are still illegal to show in France?"
- **fresh_fast (FreshQA)**：NVDA 股价一类的时效题（demo 里的触发案例："What is the stock price of Nvidia today?"）

---

## 3. 自建 eval（从训练池切出来的）

**frozen test split：240 题**。600 条 frozen pool 按 60/40 分层切分（deterministic，seed 42），360 校准 / **240 测试**，在任何 gate 实验之前就冻结，之后从未动过（每个 probe 版本的 pre-registered guard：frozen-test AUC 不得回退）。

构成（分层比例同校准侧）：

| 家族 | 测试题数 | 长相 |
|---|---|---|
| hard-math | 60 | GSM8K/MATH-500，要求给出最终数值 |
| hard-knowledge | 60 | MMLU-Pro 选择题 |
| easy-chat | 60 | Dolly/Alpaca 开放闲聊 |
| easy-fact | 40 | TriviaQA 事实题 |
| trap | 20 | SimpleQA 超长尾（本地必错） |

在这 240 上：always-local 58.8%，always-expert 91.7%；**33% 升级预算下 gate 达 77.9–78.7%**（用 1/3 的 expert 调用收回 ~58% 的差距），且**每个操作点都赢 matched-rate random**。

另有 **FreshQA heldout 30+30**（fast/never 各 30），只做方向性验收：v4 在 fast 上的 fire-rate .80/.93/1.0（保守/平衡/激进）vs v3 的 .57/.80/1.0，never 侧不涨。

---

## 4. Training recipe

**不是神经网络训练——没有 epoch、没有学习率。** 是 scikit-learn `LogisticRegression`（lbfgs，凸优化解到收敛，`max_iter=5000`）：

- **方法**：L2 正则 logistic 回归，直接拟合 frozen hidden features；无 scaler，无 class weight。
- **唯一超参 C**（L2 强度）：网格 sweep，5-fold `StratifiedKFold`（shuffle, seed 42）OOF AUC 选择。v3/v4 选中 **C=1e-4**；concurrent 全量 refit C=3e-4。
- **特征/层选择**：19 个 config 的 sweep 只在训练池 OOF 上做，外部 pool 从不参与选择（论文 Limitations 里如实披露）。
- **为什么不用 RL/SFT**（8z 决策，记录在 todo.tex）：gate 是单步决策、两个反事实（never/always arm）都可离线观测 → 本质是 cost-sensitive 监督分类，RL 只会以更差的样本效率重新推出同一个贝叶斯分类器；SFT 动 backbone 会毁掉 zero-training 主张并使全部已测曲线失效。
- **各版本 OOF AUC**：v2 .878 (n=1,160) → v3 .864 (n=2,310, 12,288-d) → v4 .876 (n=2,553)。
- **阈值（不属于 probe 权重）**：三档预算 15/30/50%（conservative/balanced/aggressive），用**无标签的 per-domain 分数分位数**定阈——只控预算，不提升排序；v4 的分位数取自 v3-mix OOF 行，防止全正样本的 fresh 行推高阈值。
- **换 regime 要重校准**：concurrent-prefill 与 native duplex 两个 serving 状态各自 refit 同一特征配方（8bb/8be），backbone 依旧零训练。

---

## 5. 外部公开 eval：都有啥、测什么、分数如何、比 random 如何

七个外部 pool，**全部严格不进训练**；音频用 benchmark 官方预渲染；有官方 judge 就用官方 judge（OAB=gpt-4o，VoiceBench=gpt-4o-mini）。

### 逐 pool 介绍（典型题为该 benchmark 的代表性题型）

| pool | n | 主要测什么 | 典型题 |
|---|---|---|---|
| **Speech TriviaQA** (OpenAudioBench) | 250 | 长尾事实检索失败——probe 的主场 | "Which country hosted the 1966 FIFA World Cup?" |
| **Speech Web Questions** (OAB) | 250 | Freebase 式短实体问答；判分很严，一半"错"是协议错 | "What language do people in Argentina speak?" |
| **Llama Questions** (OAB) | 250 | 通用常识口语问答；本地 floor 高（84%），考低余量下的精度 | "What is the capital city of Australia?" |
| **Reasoning QA (zh)** (OAB) | 202 | 中文推理——跨语言迁移试金石（校准集全英文） | 中文数学/逻辑推理题 |
| **SD-QA** | 200 | 真人方言语音（非 TTS）——真实语音鲁棒性 | 真人朗读的 NQ 式问题："who sings …" |
| **VoiceBench AlpacaEval** | 199 | 开放式指令跟随（1–5 分）——方法边界探针 | "What are the most effective ways to deal with stress?" |
| **FreshQA** (heldout 60) | 60 | 时效性/实时数据意识（v4 新增能力） | "What is Nvidia's stock price today?" |

### 当前分数（MiniCPM 主家族，live 全链路，v3 probe，官方 judge 优先；tab:transfer）

| | TriviaQA | WebQ | Llama Q. | SD-QA | Reason. zh | 均值 |
|---|---|---|---|---|---|---|
| always-local | 66.4 | 57.2 | 84.0 | 51.5 | 58.9 | 63.6 |
| + gate @15% | 73.2 | 62.8 | 90.4 | 60.0 | 65.3 | 70.3 |
| + gate @30% | 80.0 | 68.0 | 90.4 | 72.0 | 71.3 | 76.3 |
| **+ gate @50%** | **86.0** | **73.2** | **94.8** | **79.5** | **77.2** | **82.1** |
| always-escalate（实测） | 88.0 | 80.8 | 93.6 | 88.5 | 83.7 | 86.9 |
| probe AUC（冻结权重） | .789 | .785 | .806 | .792 | .683 | .771 |

AlpacaEval 单列（1–5 分）：3.99 → 4.35 @50%，always-escalate 4.76。
（MiniCPM 官方离线参考：Speech TriviaQA 75.5 / WebQ 70.2——我们的 floor 低是 deployed 通道税，8w 已验证官方数字可复现。）

### 跟 random escalation（matched-rate）比

判据：同样的升级预算下，随机挑 k 条升级 vs probe 挑 k 条。赢了才说明 gate 在"排序"，而不只是买了 expert 预算。

- **内部 frozen test 240**：每个操作点都赢 matched-random（论文 Fig tradeoff）。
- **Llama Questions**：最干净的正例——留在本地的一半 97.6% vs 升级的一半 69.6%（z=6.46）；@50% 选择性升级 94.8 **> always-escalate 93.6**（挑着升比全升还好）。
- **SD-QA**：三个预算全部赢 random（真实语音上的正例）。
- **TriviaQA**：v2 起稳定赢 random（v1 时曾 ≈random，是训练信号错配，8t 修复后选择性翻倍到三倍）。
- **WebQ**：不稳——低/高预算处与 random 差距小。归因（8ab Addendum 8）：其标签一半在测 Freebase 判分协议而非模型，三个模型上 hedging 信号都是 null；gold 视角是赢 random 的。
- **AlpacaEval**：**诚实阴性**——4.35 < random@50% ≈4.45。开放式失败不呈现 probe 读的"检索失败"模式。
- **第二 duplex 家族 NVDA VoiceChat-11B**（同配方 refit，离线 re-mix，官方 judge）：**5/5 pool 全部显著赢 matched-random（permutation p ≤ .0004）**，含 AlpacaEval；@50% 均值 74.7 vs random 66.9。MiniCPM 上的三个阴性在这个家族全部翻正——差的是余量（floor .31–.38 vs .51–.66）和失败种类，不是选择性（AUC .72–.78 池间几乎不变）。
- **换 serving regime 后**（AUC 排序 turn-based .771 > native .709 > concurrent .689，外部均值）：
  - **native duplex（部署 regime，8be）**：gated accuracy **6 个 pool 中 5 个显著赢 matched-random**（如 striviaqa .612→.756 @26%, p<.0001）；zh Reasoning 是阴性（英文校准阈值下 gate 不 fire——语言轴）；AlpacaEval 弱显著正（p=.0185）。
  - **concurrent 谐波台（harness）**：内部真实（balanced +4.0 / aggressive +11.8 分 vs random），外部 360 行小 probe 基本不赢 random；2,310 全量 refit 把外部 AUC .643→.689，只收回约 1/3 差距——regime 本身有真实残余成本（8bb）。

**一句话总结**：内部与主家族外部（Llama Q / SD-QA / TriviaQA）稳定赢 random；WebQ 吵、AlpacaEval 输（写进论文的诚实阴性）；第二家族 5/5 全赢；native 部署 regime 5/6 赢。选择性（AUC ~.7–.8）跨池、跨家族、跨语言都在，能不能变成可见收益取决于 headroom × 标签保真 × 失败种类。
