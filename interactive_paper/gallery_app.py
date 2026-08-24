"""Phone-friendly gallery for the figure deliverable, with a written
interpretation under every figure (what it shows / why we win or lose /
whether the comparison is fair). Deploy: modal deploy gallery_app.py
PNGs are bundled at deploy time — redeploy to refresh."""
import os

import modal

TOKEN = "62dc5cd9"

# (file, title, verdict-class, interpretation-html)
FIGS = [
    ("fair_dualview", "图1 · 我们的数据集 — escalation vs acc（speakable 子集 n=218）", "win", """
<b>看什么</b>：蓝线是部署实测（专家读小模型的转写），绿线是信道对照（专家读原文），灰虚线是随机升级。
<b>数字</b>：never .436 → aggressive .670，绿线到 .771，全升级上限 .922。
<b>为什么我们好</b>：三档全部在随机线之上（超额 +.027/+.048/+.083），说明探针挑的题确实更该挑。
<b>为什么不够好</b>：蓝绿之间还有 .083 的差距，那是语音信道成本——小模型自己的转写把题目送坏了（§8r 已证明不是 TTS 的锅）。
<b>公平吗</b>：内部完全公平（同一 loop、同一批题、逐题配对）。但这是我们自己造的题库，不能单独用来说服别人，所以才有后面五个公开集。
"""),
    ("fair_pareto_latency", "图2 · 我们的数据集 — latency vs acc", "win", """
<b>看什么</b>：横轴是端到端 P50 延迟，纵轴同上。
<b>数字</b>：1.83s/.436 → 4.01s/.670，即 +23 分要多等 2.2 秒。
<b>怎么读</b>：balanced（3.5 秒）是性价比拐点；aggressive 再快也快不过专家的往返时间。
<b>注意</b>：这是中位数视角，长尾另算（P99 会被专家的推理尾巴拉到 30 秒以上，见 latency 表）。
<b>公平吗</b>：延迟全部来自真实会话时间戳，不是模型估算；唯一未计入的是语音合成（我们的 loop 输出文本）。
"""),
    ("striviaqa_dualview", "图3 · Speech TriviaQA — escalation vs acc（含对比模型）", "win", """
<b>⭐ 这是最有力的一张。</b> 我们 aggressive 档 <b>.860</b>（v3），而 3.3 倍参数的 Qwen3-Omni-30B 官方是 .629，MiniCPM 自己的官方离线数是 .755，Kimi-Audio 是 .419。
<b>为什么我们好</b>：9B 小模型 + 路由，在实时流式条件下打赢了大三倍的单体模型 23 分——<b>路由的收益大于把语音模型放大</b>。相对随机的超额 +.033/+.062/+.080，三档全胜。
<b>为什么 floor 低于官方</b>：我们的 never 臂 .664 vs 官方 .755，差距已完全拆解——其中 4.8 分是实时流式 loop 的代价（我们自己的离线 chat-mode 对照 = .712），另 4.3 分是 250 题子采样 + 协议细节。没有无法解释的残差。
<b>公平吗</b>：<b>是</b>，而且是这批图里最严格的一张——全部分数（含官方线、Qwen3-Omni、Kimi）都在 OpenAudioBench 自己的判分器（gpt-4o + 官方 prompt）下。唯一要声明的是：对比模型的数字是<b>离线</b>的，我们的曲线是<b>实时</b>的，所以正确读法是拿它们和我们的 chat-mode 线比。
"""),
    ("striviaqa_pareto", "图4 · Speech TriviaQA — latency vs acc", "win", """
<b>数字</b>：1.23s/.664 → 2.02s/.860，即 +20 分只多等 0.8 秒。
<b>为什么这么便宜</b>：这个池的题很短，本地解码本身就快（P50 0.93 秒），而升级行的专家往返 P50 3.0 秒——但因为只有一半的题升级，中位数被拉动得很少。
<b>公平吗</b>：延迟是实测；对比模型没有公开的延迟数据，所以这张图上没有它们的线（不能拿别人未测量的东西画上去）。
"""),
    ("swebq_dualview", "图5 · Speech Web Questions — escalation vs acc（含对比模型）", "mixed", """
<b>数字</b>：never .568 → aggressive <b>.732</b>（v3），超过 MiniCPM 官方 .702，略低于 Qwen3-Omni-30B 的 .749。
<b>为什么我们好</b>：同样是 9B 打到 30B 的水平线附近；超额 +.045/+.035/+.066。
<b>为什么不如上一张漂亮</b>：这个池的天花板本身低（gpt-5.5 也只有 .856），因为 WebQ 的参考答案是 Freebase 实体列表，判分严格；而且 v1→v2 在 aggressive 档是<b>退步</b>的（超额 +.092 → +.066），是唯一一个旧探针挑得更好的臂，我没有换个档位讲。
<b>公平吗</b>：是，同一把官方判分尺子。<b>但这里有个教训必须记住</b>：用我们自己的判分器时这个池只有 .464，比官方低 25 分——绝对分对判分协议极度敏感，跨来源比较必须先对齐判分器。
"""),
    ("swebq_pareto", "图6 · Speech Web Questions — latency vs acc", "mixed", """
<b>数字</b>：1.74s → 2.99s 换 +16 分。
<b>怎么读</b>：这个池的本地答案较长（中位 406 字符），所以 never 臂的基准延迟就比 TriviaQA 高。
<b>公平吗</b>：同上，延迟实测、无对比模型线。
"""),
    ("sllama_dualview", "图7 · Llama Questions — escalation vs acc ⭐⭐ 选择性升级 > 全部升级", "win", """
<b>⭐⭐ 这是全项目最强的正面结果。</b> aggressive 档 <b>.948</b> 高于"全部升级"的 .928。
<b>为什么会这样</b>：拆开 aggressive 档的 250 题——探针判定为简单、留在本地的 125 题，小模型自己得 <b>.976</b>，gpt-5.5 .968；判定为难、送上云的 125 题，小模型 .696，gpt-5.5 <b>.888</b>。
<b>这意味着什么（2026-08-20 统计修正）</b>：探针把 .976 和 .696 两个子集干净分开，配对检验 <b>z=6.46</b>，这是真正的逐题判别力（单一题型池，不可能靠题型捷径）；专家的优势<b>全部集中在难的那一半</b>（.696→.888，McNemar p&lt;.0001）。但"小模型在简单题上<b>打败</b>专家"这句话不成立——.976 vs .968 配对检验 p=1.00，是<b>打平</b>。所以正确的说法是：<b>全部送云端要花两倍的专家调用，换不到可测量的收益</b>；.948&gt;.928 这个点估计本身 p=.125（两边都贴近天花板，功效不足）。
<b>为什么超额数字反而不大</b>（+.027/+.047/+.054）：因为 floor 已经 .840、天花板 .924，总空间只有 8 分，任何策略的绝对增量都被压缩。
<b>公平吗</b>：是，官方判分器。此池是全部池子里复现噪声最低的（同题同音频在两臂都留本地时判分翻转率仅 2.3%），所以这里的结论最可信；此池官方没有公布 MiniCPM 数字，所以没有官方线。
"""),
    ("sllama_pareto", "图8 · Llama Questions — latency vs acc", "win", """
<b>看点</b>：曲线在 conservative 档<b>向左拐</b>（1.52s → 1.19s）——升级反而更快。
<b>为什么</b>：本地长答案的解码时间（长的能到几秒）有时比专家往返还慢；升级掉一部分长题反而降低了中位延迟。这说明<b>路由不总是"用延迟换准确率"，有时两者兼得</b>。
<b>公平吗</b>：是，实测时间戳。
"""),
    ("sreason_dualview", "图9 · Reasoning QA（中文，执行型失败）— escalation vs acc", "win", """
<b>看什么</b>：这是我们唯一的<b>执行型失败</b>外部验证（推理算错，而不是不知道某个事实），而且是中文的。
<b>数字</b>：never .584 → aggressive <b>.762</b>，天花板 .871，超额 +.000/+.027/+.059。
<b>为什么重要</b>：探针几乎完全在英文上校准，却能在中文推理题上把准确率拉高 18 分——<b>跨语言迁移成立</b>。三种失败类型的外部验证到此齐了。<b>（2026-08-20 修正）</b>v2→v3 在这个池上的 live 增益（+.010~.030）落在我们自己的复现噪声里（该池同题判分翻转率 16.9%，配对 SE .028，McNemar p&gt;.4）——<b>跨语言的证据在离线 AUC（+.062）上成立，live 曲线只是方向一致，不构成独立确认</b>。
<b>为什么 conservative 档没超额</b>：最保守档只升级 15%，在推理型任务上探针的排序能力还不足以在如此小的预算内选中真正会错的题。
<b>公平吗</b>：内部公平。<b>但这张图不能和官方比</b>——官方没有公布这个子集的 MiniCPM 数字，而且它的官方评分用的是逐题 rubric（打分prompt 列），我们没有复制，所以分数是我们自己的判分器口径，图上不画官方线。
"""),
    ("sreason_pareto", "图10 · Reasoning QA（中文）— latency vs acc", "mixed", """
<b>数字</b>：3.17s → 3.84s 换 +18 分。
<b>为什么基准延迟最高</b>：推理题的本地答案最长（要写推理过程），所以 never 臂就已经 3.2 秒。
<b>公平吗</b>：延迟实测；语音合成未计入。
"""),
    ("sdqa_dualview", "图11 · SD-QA 真人语音 — escalation vs acc", "win", """
<b>看什么</b>：唯一一个用<b>真人录音</b>（非合成语音）的池，直接堵掉"你们的结论只在 TTS 上成立"这个质疑。
<b>数字</b>：never .510 → aggressive <b>.785</b>，天花板 .930。超额 +.052/<b>+.125</b>/+.100 是所有池里最高的。
<b>为什么这里表现最好</b>：空间大（.42）、失败是典型的检索型、题目短且口语化——四个公平条件全部满足，探针最能发挥。
<b>公平吗</b>：内部公平。官方没有公布 SD-QA 的 MiniCPM 数字，所以没有官方线；判分是我们自己的口径（全池一致）。
"""),
    ("sdqa_pareto", "图12 · SD-QA 真人语音 — latency vs acc", "win", """
<b>数字</b>：1.39s → 2.96s 换 +27.5 分——这是所有池里性价比最高的一条曲线。
<b>公平吗</b>：是，实测。
"""),
    ("valpaca_dualview", "图13 · VoiceBench AlpacaEval（官方浓缩表那一行）— escalation vs judge score", "loss", """
<b>⚠️ 这是负面结果，我们如实报。</b> aggressive 4.26，低于同升级率的随机线（≈4.45），超额 +.026/<b>−.088</b>/<b>−.083</b>。
<b>为什么我们差</b>：实测原因——探针挑中的题在 never 臂得分 3.90，没挑中的 3.98，<b>零区分度</b>。根因是开放式指令没有"我不知道这个事实"这种离散失败事件可读，每个回答都有部分分（分数区间被压到 1.0 分宽）。这正是我们三失败种类里的第三类（元认知型）盲区——训练再多也救不了，属于方法的边界。
<b>那为什么曲线还是上升的</b>：因为升级行确实从 3.90 涨到 4.71——收益全部来自"gpt-5.5 长文写得好"，不来自"挑得准"，随机挑同样多的题收益一样。
<b>公平吗</b>：判分是公平的（VoiceBench 自己的 gpt-4o-mini + 原文 prompt，我们逐字复制）。但<b>绝对分和官方 4.8 的比较不公平</b>：我们实时 loop 的 floor 是 3.94，而同样音频走离线 chat 模式是 <b>4.86</b>（超过官方 4.8）——0.9 分的差距全部是 loop 造成的，机制是回答长度（chat 模式中位 2186 字符 vs 实时 820），因为双工系统提示让模型给简短口语回答，而 AlpacaEval 奖励完整长文。
"""),
    ("valpaca_pareto", "图14 · VoiceBench AlpacaEval — latency vs judge score", "loss", """
<b>数字</b>：4.59s → 9.69s，是所有池里延迟最高的。
<b>为什么这么慢</b>：AlpacaEval 的回答是长篇论述，<b>本地解码</b>而非专家往返才是延迟主因——这也解释了为什么这张图的延迟随升级率上升得这么陡。
<b>怎么用这张图</b>：它和图13 一起构成一个完整的负面案例——在开放式生成任务上，我们的方法既贵又不比随机好。诚实地画出来比藏起来强。
"""),
    ("noise_audit", "图17 · 我们自己的复现噪声有多大（方法审计）", "mixed", """
<b>为什么有这张图</b>：我们本来是去修 frozen 池的阈值超发（应该升 50% 实际升了 61%）。把同一批实测结果按 50% 重新混合，准确率只从 .596 变成 .600——<b>+.004，等于没有</b>。这说明超发是<b>成本 bug 不是准确率 bug</b>（修正后省 11.3% 的专家调用，准确率不变），也逼出了下一个问题：那我们一直在解读的那些零点零几，到底有多少是真的？
<b>左图</b>：frozen 池的连续曲线，每一个点都是<b>实测结果重新混合</b>出来的（三个档位的升级集合完全嵌套 + 探针分数逐位相同，所以可以这么做，$0）。灰带是配对 bootstrap 95% 区间。v2 的 .621 和 v3 的 .596 都落在带子里。
<b>右图</b>：18 个 v2→v3 的 live 差值全部跨过 0 线（McNemar p=.16~1.00）。
<b>噪声地板</b>：同一道题、同一段音频、在两个臂里都留在本地，判分结果的翻转率是 <b>2.3%~18.8%</b>（frozen 15.5%）；换算成臂间差值的配对标准误是 .009~.028。<b>所以任何小于 3 个点的差异，单跑一次 live sweep 是测不出来的。</b>
<b>什么活下来了</b>：离线 AUC 的提升（统计量紧得多）、探针的逐题判别力（sllama z=6.46）、以及升级对难题那一半的提升（p&lt;.0001）。
"""),
    ("kink_case_study", "图21 · 案例走查:一道题的两个世界(拐弯怎么出现)", "win", """
<b>用途</b>:用错题簿里的一道典型题(sllama0164「锡克教有几位祖师」)的<b>完整实测 trace</b> 讲拐弯:世界A(探针关)本地绕 487 字符、3.10 秒、答错;世界B(探针开)话音落下 21ms 读出 eot=0.631≥0.513 → 升级,gpt-5.5 1.68s + 转述 0.62s = 2.32 秒,答对——<b>同一道题快 0.78 秒且从错变对</b>。下半用三根条讲池效应:38 道这种慢题(P50 2.38s)离开本地队列后,留下 212 题 P50 掉到 0.94s,整臂中位 1.52→1.17s = 图8 的左折。含 Addendum 4 的边界修正(探针挑"会错"不挑"会绕")。
"""),
    ("swebq_annotated", "图18 · 讲解版:图5 每条线是什么、什么能比什么不能比", "win", """
<b>用途</b>:自带解读的教学版——上图每条线挂圈号,图内直接印 ①-⑥ 的大白话注释:官方线是别的测法(离线 chat)、浅蓝虚线证明官方分我们测得出来、蓝曲线才是真系统、+.164 = 机制+挑得准的总和、随机对照在正式图5 的灰虚线上。开会投这张,不用口头解释坐标系。
"""),
    ("sllama_annotated", "图19 · 讲解版:会挑的路由,花一半的钱拿到全升级的分", "win", """
<b>用途</b>:最强正面结果的教学版,含 §8ad 统计修正后的诚实表述:.948 vs .928 本身 p=.125 不单独下结论;铁的是探针的分割(.976/.696,z=6.5)和难半边的提升(.696→.888,p&lt;.0001);正确头条 = "全部送云不是上界,选择性路由省一半专家调用零代价"。
"""),
    ("noise_audit_annotated", "图20 · 讲解版:我们的测量精度有多高,哪些差别不该解读", "mixed", """
<b>用途</b>:方法审计的教学版。左:重建曲线+复现噪声带(同题重测判分翻转 2.3%~18.8%);红方块 = 曾被我们叫"退步"的 v2 点,其实在带内(p=.44,已撤回);绿星 = 阈值超发修正,只救成本不救准确率。右:18 个 v2→v3 live 差值全部跨零 → 单次 live 分辨不了 &lt;3 分。规矩:比较用配对检验/AUC,别读臂准确率的小数点。
"""),
    ("nvda_layer_sweep", "图15 · NVDA VoiceChat-11B — 探针层扫（第二个全双工家族）", "win", """
<b>看什么</b>：把我们的探针配方原样搬到 NVIDIA NemotronLabs-VoiceChat-11B（Nemotron Nano v2 9B 主干，56 层 Mamba2/attention 混合架构——和 MiniCPM 完全不同的架构家族）。
<b>数字</b>：中段 L30-34 最强（OOF AUC .714），两端弱（L2 .693 / L54 .682）。
<b>为什么重要</b>：这是论文 §9 预注册的"第二家族"测试。"中层语义带最可读"的结构在一个 Mamba 混合主干上复现了——探针读的不是 MiniCPM 的私有特征，是全双工语音模型的共性结构。
<b>公平吗</b>：校准只用了我们冻结的 600 题池（vs MiniCPM v3 的 2310），判分器同一把（gpt-5.4-mini ref-anchored）；离线重放口径，非实时 loop。
"""),
    ("nvda_transfer", "图16 · NVDA VoiceChat-11B — 冻结方法论迁移（AUC 对比 MiniCPM）", "win", """
<b>看什么</b>：同样三个读数（eot_last / +窗口均值 / +user-audio 均值）、同样的 C=1e-4 逻辑回归，在 NVDA 模型上从零校准，然后直接测 4 个外部公开池。
<b>数字</b>：OOF .790；striviaqa .781 / swebq .793 / sdqa .754 / sllama .701——绿线是 MiniCPM v3（.79-.81），600 条校准就摸到了同一水平带；特征叠加的增益模式也和 MiniCPM 完全一致（.714→.761→.790）。
<b>边界（如实报）</b>：sreason（中文）在此模型上 fail rate = 1.000——NVDA VoiceChat 是英文单语模型，听中文音频直接幻觉英文答案，跨语言迁移在它身上没有对应物，该池无 AUC 可算。另外它的知识 floor 明显低于 MiniCPM（striviaqa 本地正确率 .32 vs .62，同判分器）——底座弱不妨碍探针读失败信号，反而 base-fail 高信号更足。
<b>公平吗</b>：判分器与标签定义完全同口径；尚未跑实时 4 臂曲线（需要把 streaming loop 移植到 NeMo，是下一步的花钱决定）。
"""),
]

VERDICT = {"win": ("✓ 有利", "#1e9e50"), "mixed": ("~ 有保留", "#b8860b"),
           "loss": ("✗ 不利（如实报告）", "#b00")}

app = modal.App("figures-gallery")
HERE = os.path.dirname(os.path.abspath(__file__))
image = modal.Image.debian_slim().pip_install("fastapi[standard]")
for name, _, _, _ in FIGS:
    image = image.add_local_file(os.path.join(HERE, "figures", f"{name}.png"),
                                 f"/root/figs/{name}.png")


@app.function(image=image, timeout=60 * 5, min_containers=0)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response

    api = FastAPI()
    blocks = []
    for name, title, verdict, text in FIGS:
        label, color = VERDICT[verdict]
        blocks.append(
            f'<div class=fig><h3>{title}</h3>'
            f'<span class=badge style="background:{color}">{label}</span>'
            f'<img src="/{TOKEN}/f/{name}.png" loading=lazy>'
            f'<div class=interp>{text}</div></div>')
    page = f"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Figures + interpretation</title><style>
body{{font-family:-apple-system,system-ui,sans-serif;max-width:900px;
margin:0 auto;padding:1rem;background:#fafafa;color:#111;line-height:1.55}}
.fig{{background:#fff;border:1px solid #ddd;border-radius:10px;
padding:.8rem;margin:1.2rem 0}}
h1{{font-size:1.2rem}}h3{{font-size:.98rem;margin:.2rem 0 .4rem}}
img{{width:100%;height:auto;border-radius:4px;margin:.5rem 0}}
.badge{{color:#fff;font-size:.72rem;padding:.12rem .5rem;border-radius:99px}}
.interp{{font-size:.86rem;background:#fbfbfa;border-left:3px solid #ccc;
padding:.6rem .7rem;border-radius:0 6px 6px 0}}
.interp b{{color:#000}}
.note{{background:#fff;border-left:4px solid #333;padding:.7rem;
font-size:.86rem}}</style></head><body>
<h1>14 张图 + 逐图解读</h1>
<div class=note><b>怎么读这套图</b>：每张图下面写了三件事——我们在这张图上<b>为什么占优</b>、
<b>为什么吃亏</b>、以及<b>这个比较公不公平</b>。<br><br>
统一口径：五个外部集用<b>官方判分器</b>（OpenAudioBench 的 gpt-4o / VoiceBench 的
gpt-4o-mini，均逐字复制官方 prompt）；对比模型（Qwen3-Omni-30B、Kimi-Audio、官方
MiniCPM）的数字来自同一张官方表，全部是<b>离线 chat 模式</b>，而我们的曲线是<b>实时流式</b>的——
所以图上还画了"我们自己的离线 chat-mode 线"作为同协议参照。<br><br>
<b>一句话总结</b>：失败可观测的任务（事实型、检索型、推理型）上路由有效，
最好的一张图上 9B+路由打赢 30B 单体模型、甚至打赢"全部升级"；
开放式生成任务（AlpacaEval）上路由无效，这是方法的边界，如实报告。</div>
{''.join(blocks)}</body></html>"""

    @api.get(f"/{TOKEN}", response_class=HTMLResponse)
    def index():
        return page

    @api.get(f"/{TOKEN}/f/{{name}}.png")
    def fig(name: str):
        path = f"/root/figs/{name}.png"
        if not os.path.exists(path) or "/" in name:
            raise HTTPException(404)
        return Response(open(path, "rb").read(), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=600"})

    return api
