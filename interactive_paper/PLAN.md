# Zero-Training Escalation Gate for MiniCPM-o 4.5

> 本文档是完整的执行计划,面向没有任何前置对话上下文的执行者。请按 Phase 顺序执行,
> 每个 Phase 末尾有明确的 go/no-go 检查点。**全程不训练、不微调任何模型参数**
> (Phase 2 的线性 probe 是 sklearn 拟合,不算模型训练)。

---

## 1. 项目背景与目标

### 想法来源
- **MiniCPM-o 4.5**(arXiv:2604.27393)提出 Omni-Flow 全双工流式框架:模型在约 1 秒的
  时间窗里同时感知(视觉 V / 音频 A token)和输出(silent `sl` / speak `sp` / 文本 token),
  实现边看边听边说。但它是 9B 小模型,知识和推理密集型任务明显弱于大模型
  (MMLU-Pro/GPQA 类任务、Speech Web Questions 等)。
- **"Enabling Real-Time Conversations with Minimal Training Costs"**(arXiv:2409.11727)
  证明:双工交互能力本质是"控制策略"而非"能力",可以用极少训练(甚至推理侧改造)获得。

### 本项目做什么
给 MiniCPM-o 4.5 外挂一个**零训练的升级门(escalation gate)**,构成 System 1 / System 2 架构:

1. **旁路观测**:decode 循环中每步"偷听"LLM 的 hidden state `h_t` 和 logits(免费副产品)。
2. **零训练信号**:token 熵、logit margin(top1−top2)、线性 probe(`w·h_t`,sklearn 拟合)。
3. **阈值门**:EMA 平滑 + 连续 k 步超标才触发(防抖)。
4. **触发后**:由外部 wrapper(不是模型自己)拦截,让小模型生成一段**蒸馏 query**
   (不传原始 context,避免大模型 prefill 爆炸),异步发给云端大模型(Claude API)。
5. **结果回注**:大模型答案作为新输入注入小模型 context,小模型口语化转述。
6. 等待期间小模型继续正常运转(拖延话术靠 prompt 注入;MiniCPM-o 原生的 `sl` token
   天然支持"沉默等待")。

### 核心研究问题(按优先级)
- **RQ1**:零训练信号(熵/margin/probe)对"小模型答不好这个 query"的判别力有多强?(AUC)
- **RQ2**:hybrid 系统(小模型 + 选择性升级)相对 small-only 和 big-only,
  在 accuracy–escalation rate–latency 三维上的 tradeoff 曲线长什么样?
- **RQ3**(stretch):这套机制能否嵌入流式/全双工推理而不破坏实时性?

### 明确不做的事(防止跑偏)
- ❌ 不做任何 SFT / LoRA / RL。
- ❌ 不先做 UI / 语音 demo。v1 全部用文本模态验证(MiniCPM-o 支持纯文本 chat)。
- ❌ 不在本地跑大模型(只有一张 H100,升级目标用 Claude API)。
- ❌ 不实现完整全双工音视频管线,除非 Phase 0–5 全部完成(那是 Phase 6 stretch)。

---

## 2. 资源与环境

| 资源 | 说明 |
|------|------|
| GPU | 1× H100 (80GB),Modal 平台 |
| 小模型 | MiniCPM-o 4.5(9B:Qwen3-8B backbone + SigLIP + Whisper + 语音解码器)。bf16 权重约 18–20GB,H100 富余 |
| 大模型 | Claude API。裁判用 `claude-opus-4-8`(或更强),升级目标用 `claude-sonnet-5`(思考模式按需) |
| 密钥 | Modal secrets:`huggingface`(HF_TOKEN)、`anthropic`(ANTHROPIC_API_KEY)。若不存在,提示用户创建:`modal secret create anthropic ANTHROPIC_API_KEY=...` |

**执行前必须现场核实的事**(不要凭记忆写死):
1. HF 上 MiniCPM-o 4.5 的确切 repo 名(参考:2.6 版是 `openbmb/MiniCPM-o-2_6`,4.5 大概率是
   `openbmb/MiniCPM-o-4_5`,用 HF 搜索确认)。**读模型卡**,确认:依赖版本(transformers 版本
   常被 pin)、`trust_remote_code=True`、纯文本 chat 的调用方式、是否要 `init_tts=False` 之类
   的开关来跳过语音模块。
2. Claude API 当前可用的模型 id(用 claude-api skill 或文档确认)。
3. Modal 当前 API 用法(`modal.App`、`@app.cls(gpu="H100")`、`modal.Volume`)。

**降级预案**(按顺序,任何一步卡住超过半天就降级并在 RESULTS.md 记录原因):
- MiniCPM-o 4.5 的自定义代码难以 hook → 换 `openbmb/MiniCPM-o-2_6`(更成熟,架构同族)。
- 还不行 → 用 `Qwen/Qwen3-8B` 纯文本先把 gate 机制全链路验证完,再回头移植。
  (gate 只依赖 h_t 和 logits,与具体模型解耦,这个降级不损伤核心结论。)

---

## 3. 仓库结构(在当前目录初始化 git)

```
interactive_paper/
├── PLAN.md              # 本文件
├── RESULTS.md           # 执行日志:每个 Phase 的结果、数字、决策、踩坑。边做边写
├── modal_app.py         # Modal app:模型加载 + 带信号的生成接口
├── src/
│   ├── signals.py       # 熵 / margin / hidden state 提取
│   ├── gate.py          # EMA + 滞回阈值门(纯 python,可单测)
│   ├── probe.py         # 线性 probe 拟合与推理(sklearn)
│   ├── distill.py       # 蒸馏 query 生成 prompt
│   ├── escalate.py      # Claude API 调用(升级 + 裁判两种角色)
│   └── inject.py        # 结果回注 + 转述 prompt
├── data/
│   ├── queries.jsonl    # 校准/评测 query 池(带 source 与 split 字段)
│   └── calib_features.parquet  # 每条 query 的信号特征 + 标签
├── scripts/
│   ├── 01_smoke_test.py
│   ├── 02_build_queries.py
│   ├── 03_collect_signals.py
│   ├── 04_calibrate.py
│   ├── 05_run_e2e.py
│   └── 06_eval.py
└── figures/             # ROC、tradeoff 曲线等
```

---

## Phase 0 — Modal 环境 + 模型冒烟测试(预计 0.5 天)

**目标**:MiniCPM-o 4.5 在 H100 上以纯文本模式跑通生成,并测得基线 decode 速度。

步骤:
1. `pip install modal` 本地装好,`modal token` 已配置(如未配置,停下来让用户跑 `modal setup`)。
2. 写 `modal_app.py` 骨架(版本以模型卡为准!先读模型卡再 pin)。
3. `scripts/01_smoke_test.py`:发 3 条 query(一条闲聊、一条 GSM8K 题、一条 GPQA 风格题),
   打印回答和 tokens/sec。
4. 把 decode 速度、显存占用、加载耗时记入 RESULTS.md。

**Go/No-Go**:模型能对话、输出连贯中英文、decode ≥ 20 tok/s。不达标 → 走降级预案。

---

## Phase 1 — 信号提取:自定义 decode 循环(预计 1 天)

**目标**:每个 decode step 拿到 `h_t`(最后一层 hidden state 的最后位置)和 logits,
计算三个标量信号,且不显著拖慢推理。

关键点:
- **不要用** `model.generate(output_hidden_states=True)` 全量返回(内存爆炸且笨重)。
  写自定义循环:先 prefill 拿 `past_key_values`,然后逐 token `forward(use_cache=True)`。
  每步只取 `out.hidden_states[-1][:, -1, :]` 和 `out.logits[:, -1, :]`。
  - 若逐步开 `output_hidden_states` 太慢,备选:在 LLM 最后一层 decoder layer 上注册
    forward hook,只抓最后位置,开销更小。
  - MiniCPM-o 的 LLM 主干在 remote code 里(属性名可能是 `model.llm`),先
    `print(model)` 摸清结构再下 hook。
- `src/signals.py` 实现(对 float32 计算,避免 bf16 数值问题):
  - `entropy(logits)`:softmax 后 `-(p * log p).sum()`
  - `margin(logits)`:top1 与 top2 的 **logprob** 差
  - `hidden(h_t)`:直接返回向量(留给 probe)
- 沿用 chat template:query 结束后开始 decode,记录**前 K=16 步**的信号序列 +
  prompt 最后一个 token 的 `h`。
- 顺手做个 sanity check:对同一条难题和一条闲聊各画一条熵随 step 的曲线,肉眼确认
  难题的熵普遍更高(不严格,只是确认接线没错)。

**产出**:`SmallModel.chat_with_signals(messages, k=16)` →
`{"text", "signals": {"entropy": [...], "margin": [...], "h_prompt": [...], "h_steps": [[...]...]}}`

**Go/No-Go**:信号提取使 decode 变慢 < 30%;两条 sanity 曲线方向符合直觉。

---

## Phase 2 — 校准数据集 + 离线判别力分析(预计 1–1.5 天)⭐ 全项目最关键检查点

**目标**:回答 RQ1 — 零训练信号到底能不能区分"小模型会答砸的 query"。

### 2.1 构建 query 池(`scripts/02_build_queries.py`,目标 ~600 条)
| 池 | 数量 | 来源 | 预期 |
|----|------|------|------|
| easy-chat | 150 | 用 Claude API 批量生成日常闲聊/简单指令(中英各半) | 小模型几乎全对 |
| easy-fact | 100 | TriviaQA / CMMLU 里的简单常识题 | 大多能对 |
| hard-math | 150 | GSM8K test 后段 + MATH-500 抽样 | 部分答错 |
| hard-knowledge | 150 | MMLU-Pro / GPQA-main 抽样 | 大量答错 |
| trap | 50 | 用 Claude 生成"看似简单实则长尾"的问题(冷门实体、多跳事实) | 高错误率且**模型往往自信** |

每条记录:`{"id", "pool", "query", "reference_answer"(有则填), "split"}`。
**按 60/40 切 calib/test,切分固定随机种子,test 只在 Phase 5 碰一次。**

### 2.2 收集信号与标签(`scripts/03_collect_signals.py`)
1. 小模型 greedy 生成每条 query 的完整回答 + 前 16 步信号。**用 Modal 的
   `.map()`/batch 并行**,600 条在单 H100 上应在 1–2 小时内完成。
2. 标签:`claude-opus-4-8` 当裁判,输入 query + 参考答案(如有)+ 小模型回答,
   输出 `adequate ∈ {0,1}`(rubric:事实正确、推理正确、完整回答了问题)。
   `escalate_label = 1 - adequate`。
   - 裁判 prompt 固定存进 repo;抽 30 条人工核对裁判质量,一致率 < 85% 就修 rubric。
3. 存 `data/calib_features.parquet`。

### 2.3 判别力分析(`scripts/04_calibrate.py`)
- 标量特征:`mean/max entropy@K`、`min/mean margin@K`(K ∈ {4, 8, 16} 都算)。
- probe:`LogisticRegression` 于 (a) `h_prompt`,(b) 前 8 步 `h` 的均值。calib 集内 5-fold CV。
- 组合:标量特征 + probe 分数再过一个 LR。
- 输出:每个信号的 **ROC-AUC 表** + ROC 曲线图(`figures/roc.png`)。
- 分池分析:特别看 trap 池——预期熵/margin 在这里失效(自信地胡说),probe 应显著更好。
  这是论文叙事的关键证据,单独记录。

**Go/No-Go(硬性)**:
- 最佳信号 AUC ≥ 0.75 → 继续。
- 0.65–0.75 → 继续,但在 RESULTS.md 标注"信号偏弱,tradeoff 曲线预期平缓"。
- < 0.65 → **停下**。写清失败分析(分池 AUC、失败案例),等用户决策。不要自行开始训练。

---

## Phase 3 — 在线阈值门(预计 0.5 天)

**目标**:把离线信号变成 decode 循环里的实时触发器。

- `src/gate.py`,纯 Python 类,不依赖 torch,配套单元测试:

```python
class EscalationGate:
    def __init__(self, threshold, k_consecutive=4, ema_alpha=0.3, cooldown_steps=64):
        ...
    def update(self, score: float) -> bool:
        # EMA 平滑;连续 k 步超阈值 → 触发;触发后进入 cooldown 防重复
```

- score 用 Phase 2 选出的最佳信号(大概率是组合 LR 分数;LR 权重就是一组常数,
  推理时是一次点积,零成本)。
- 阈值选择:在 calib 集上画 precision-recall 曲线,默认选 **precision ≥ 0.8** 对应的点
  (误触发 = 多余的大模型调用 + 多余的"我想想",体验代价高于漏触发)。
  同时保留 3 个档位(conservative / balanced / aggressive)进 config。
- 集成进 `chat_with_signals` → `chat_gated()`:decode 中途触发则**立即停止生成**,
  返回 `{"partial_text", "triggered": True, "trigger_step"}`。
- 在 calib 集上跑通:报告 easy 池误触发率、hard 池触发召回率,和离线 AUC 对得上。

---

## Phase 4 — 升级链路 E2E(预计 1 天)

**目标**:触发 → 蒸馏 query → Claude → 回注转述,全链路打通并测延迟。

1. **蒸馏 query**(`src/distill.py`):触发后,向小模型追加一条注入 prompt:
   "把用户当前的问题和必要背景压缩成一段给专家的独立提问,不超过 200 字",
   让小模型自己生成蒸馏 query。**不传原始对话全文给大模型**(这是设计决策:
   避免大模型 prefill 长 context,也是论文卖点之一)。
   - 对照条件(消融用):直接把原始对话文本发给大模型。两种都实现,eval 时对比。
2. **升级调用**(`src/escalate.py`):`claude-sonnet-5`,普通模式与 extended thinking
   两档。异步(`anthropic` SDK 的 async client),记录首 token 延迟与总延迟。
3. **回注**(`src/inject.py`):把大模型答案包成
   `<result>...</result>` 注入小模型 context,追加指令"用口语把上面专家结论转述给用户,
   若与你之前说的部分矛盾以专家结论为准"。小模型生成最终回答。
4. `scripts/05_run_e2e.py`:单条 query 的完整 trace(每一步的文本、耗时)打印成
   可读日志。跑 10 条 hard 池 query,人工过一遍 trace,确认蒸馏 query 质量和转述忠实度。

**Go/No-Go**:E2E 跑通;蒸馏 query 在 10 条人工检查中 ≥ 8 条忠实抓住原问题;
升级链路端到端额外延迟 P50 有记录(数字本身不设门槛,记录即可)。

---

## Phase 5 — 系统评测(预计 1 天)

**目标**:回答 RQ2,产出论文级数字。**只用 test split(~240 条)。**

四个系统条件:
| 条件 | 说明 |
|------|------|
| small-only | MiniCPM-o 4.5 直接回答(下限) |
| big-only | 全部发给 Claude(上限 + 成本参照) |
| hybrid-gate | 本项目:gate 触发才升级(3 个阈值档位都跑) |
| hybrid-random | 消融:随机升级,升级率与 hybrid-gate balanced 档对齐 |

指标(`scripts/06_eval.py`,裁判同 Phase 2,裁判不知道答案来自哪个条件):
- **accuracy**(裁判判 adequate 的比例)
- **escalation rate**
- **latency**:P50/P95(小模型独答 vs 升级链路分开报)
- **cost**:每 100 条 query 的 API 花费估算
- 核心图:**accuracy vs escalation-rate 曲线**(扫阈值),hybrid-gate 曲线应显著在
  hybrid-random 上方——这条曲线是整个项目的中心结果,存 `figures/tradeoff.png`。
- 消融:蒸馏 query vs 原始 context(accuracy 与延迟两个维度)。

**产出**:RESULTS.md 完整章节 + 两张核心图 + 一段 200 字结论
(信号判别力如何、hybrid 提升多少、代价多少、trap 池上 probe 是否救了熵)。

---

## Phase 6(Stretch,默认不做)— 流式/双工集成

仅当 Phase 0–5 全部完成且用户明确要求时再做:
- 把 gate 嵌入 MiniCPM-o 的流式推理路径(时间窗内逐步喂音频/视频 chunk)。
- 触发后注入拖延话术 prompt("我想想"),等待期间模型输出 `sl` 保持沉默,
  `<result>` 作为新输入注入后续时间窗。
- 测:触发到拖延话术出口的延迟、回注后转述是否自然、用户中途插话是否被正常处理。

---

## 执行纪律

1. **边做边写 RESULTS.md**:每个 Phase 一节,记录数字、图、决策、踩坑和放弃的路线。
   这是最终写论文的原始素材。
2. **省钱**:Modal 按秒计费。开发调试用 `modal run` 短任务,不要挂常驻容器;
   批量任务(Phase 2.2)一次排满再跑。H100 约 $4–5/小时,全项目 GPU 预算目标 < $150。
   Claude API 预算目标 < $50(裁判调用是大头,batch 处理)。
3. **随机种子固定**(42),所有采样可复现。
4. **卡住策略**:任何步骤卡住超过半天,执行降级预案或在 RESULTS.md 写明阻塞点后
   暂停等用户,不要发明计划外的新方案(尤其不要开始训练模型)。
5. Phase 2 的 go/no-go 是全项目最重要的决策点,到达时无论结果如何都停下来向用户汇报。
