# -*- coding: utf-8 -*-
"""Add the /cases side-by-side demo page (escalation before/after with
real answers + latency) to demo_app.py."""
NL = chr(10)
p = 'demo_app.py'
s = open(p, encoding='utf-8').read()

# ---- 1. server: curated cases builder + route ---------------------------
BUILD = '''

# hand-picked exemplars for the /cases page: id -> (category chinese)
CASES_PICK = [
    ("sllama", "sllama0164", "winfast",
     "拐弯的来源:升级反而更快"),
    ("striviaqa", "striviaqa0192", "winfast",
     "拐弯的来源:升级反而更快"),
    ("striviaqa", "striviaqa0074", "bothright",
     "本地也对,但绕得慢——升级更快"),
    ("sllama", "sllama0024", "tradeoff",
     "典型形态:多花几秒,换回正确"),
    ("sllama", "sllama0000", "kept",
     "探针分很低 → 正确地留在本地"),
    ("striviaqa", "striviaqa0137", "missed",
     "自信错:答得又快又斩钉截铁,15% 预算漏网;50% 档被抓回"),
]
_HEDGE_RE = ("there is no|no widely|not a real|i'?m not sure|not sure|"
             "might be|likely|however|unfortunately|i don'?t know|"
             "as an ai|apolog|unclear|difficult to|depends on")


def _cases():
    import re
    import pandas as pd
    thr = {}
    for pool in ("striviaqa", "sllama"):
        a = json.load(open(f"{DATA}/gate_v3_{pool}.json"))
        t = a.get("eot_thresholds") or a
        thr[pool] = {k: float(v) for k, v in t.items()
                     if isinstance(v, (int, float))}
    hedge = re.compile(_HEDGE_RE, re.I)

    def mark(txt):
        return hedge.sub(lambda m: f"<mark>{m.group(0)}</mark>",
                         str(txt or ""))

    out = []
    for pool, qid, cat, cat_label in CASES_PICK:
        df = pd.read_parquet(f"{DATA}/{pool}_v3_traces.parquet")
        nev = df[(df["tier"] == "never") & (df["id"] == qid)].iloc[0]
        esc_row, esc_tier = None, None
        for t in ("conservative", "balanced", "aggressive"):
            r = df[(df["tier"] == t) & (df["id"] == qid)]
            if len(r) and r.iloc[0]["mode"] == "escalated":
                esc_row, esc_tier = r.iloc[0], t
                break
        cons = df[(df["tier"] == "conservative") & (df["id"] == qid)]
        cons = cons.iloc[0] if len(cons) else nev
        case = {"id": qid, "pool": pool, "cat": cat,
                "cat_label": cat_label,
                "q": str(nev["query"]),
                "ref": str(nev["reference_answer"])[:160],
                "audio_s": float(nev["audio_s"]),
                "off": {"ans": mark(nev["answer"]),
                        "ms": float(nev["answer_ms"]),
                        "ok": int(nev["oab_ok"]),
                        "chars": len(str(nev["answer"]))}}
        if esc_row is not None:
            e = esc_row
            tot = (e["eot_read_ms"]
                   + max(e["stall_ms"] or 0,
                         (e["expert_latency_s"] or 0) * 1000)
                   + (e["relay_ms"] or 0))
            case["on"] = {"mode": "escalated", "tier": esc_tier,
                          "score": float(e["eot_score"]),
                          "thr": thr[pool].get(esc_tier),
                          "eot_ms": float(e["eot_read_ms"]),
                          "stall_ms": float(e["stall_ms"] or 0),
                          "expert_s": float(e["expert_latency_s"] or 0),
                          "relay_ms": float(e["relay_ms"] or 0),
                          "total_ms": float(tot),
                          "expert": str(e["expert_answer"]),
                          "relay": str(e["relay"]),
                          "ok": int(e["oab_ok"])}
        else:
            case["on"] = {"mode": "local",
                          "score": float(cons["eot_score"]),
                          "thr": thr[pool].get("conservative"),
                          "ans": mark(cons["answer"]),
                          "ms": float(cons["answer_ms"]),
                          "ok": int(cons["oab_ok"])}
        out.append(case)
    return out
'''

anchor = '@app.function(image=web_image, volumes={DATA: gate_data}, timeout=60 * 10,'
assert s.count(anchor) == 1
s = s.replace(anchor, BUILD + NL + NL + anchor)

# route inside web(): after the index route
a = '''    @api.get(f"/{TOKEN}", response_class=HTMLResponse)
    def index():
        return PAGE.replace("__TOKEN__", TOKEN).replace(
            "__AGG__", json.dumps(AGG)).replace(
            "__THR__", json.dumps(THR))'''
b = '''    @api.get(f"/{TOKEN}", response_class=HTMLResponse)
    def index():
        return PAGE.replace("__TOKEN__", TOKEN).replace(
            "__AGG__", json.dumps(AGG)).replace(
            "__THR__", json.dumps(THR))

    CASES = _cases()

    @api.get(f"/{TOKEN}/cases", response_class=HTMLResponse)
    def cases_page():
        return CASES_PAGE.replace("__TOKEN__", TOKEN).replace(
            "__CASES__", json.dumps(CASES, ensure_ascii=False))'''
assert s.count(a) == 1
s = s.replace(a, b)

# header link on the main page
a2 = '  <span id=state class=sub></span>'
b2 = ('  <a href="/__TOKEN__/cases" style="font-size:.8rem">案例对比 →</a>\n'
      '  <span id=state class=sub></span>')
assert s.count(a2) == 1
s = s.replace(a2, b2)

# ---- 2. the cases page --------------------------------------------------
CASES_PAGE = r'''

CASES_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>升级前后对比 — 真实案例</title><style>
:root{--blue:#2a78d6;--green:#1e9e50;--red:#b00;--amber:#b8860b;
--ink:#0b0b0b;--mut:#52514e;--grid:#e6e4de;--bg:#faf9f7;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
header{background:var(--card);border-bottom:1px solid var(--grid);
padding:.8rem 1.2rem;position:sticky;top:0;z-index:9}
h1{font-size:1.05rem;margin:0;font-weight:650}
.sub{color:var(--mut);font-size:.8rem;margin-top:.15rem}
.wrap{max-width:1080px;margin:0 auto;padding:1rem}
.case{background:var(--card);border:1px solid var(--grid);
border-radius:12px;margin-bottom:1.4rem;overflow:hidden}
.chead{padding:.8rem 1rem;border-bottom:1px solid var(--grid)}
.cat{display:inline-block;font-size:.72rem;padding:.14rem .6rem;
border-radius:99px;color:#fff;margin-right:.5rem;vertical-align:middle}
.cat.winfast{background:var(--green)}.cat.bothright{background:var(--blue)}
.cat.tradeoff{background:var(--amber)}.cat.kept{background:var(--blue)}
.cat.missed{background:var(--red)}
.q{font-size:1rem;font-weight:600;margin:.4rem 0 .1rem}
.ref{color:var(--mut);font-size:.8rem}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:0}
@media(max-width:800px){.cols{grid-template-columns:1fr}}
.col{padding:.85rem 1rem}
.col.left{border-right:1px solid var(--grid);background:#fcfcfb}
.col h3{margin:0 0 .5rem;font-size:.78rem;text-transform:uppercase;
letter-spacing:.04em;color:var(--mut)}
.lat{font-size:1.7rem;font-weight:700;font-variant-numeric:tabular-nums}
.lat small{font-size:.85rem;font-weight:400;color:var(--mut)}
.pill{display:inline-block;font-size:.74rem;padding:.12rem .55rem;
border-radius:99px;color:#fff;vertical-align:middle;margin-left:.5rem}
.pill.ok{background:var(--green)}.pill.no{background:var(--red)}
.ans{margin-top:.55rem;font-size:.86rem;background:#fff;
border:1px solid var(--grid);border-radius:8px;padding:.6rem .7rem;
max-height:170px;overflow:auto;white-space:pre-wrap}
.col.left .ans{background:#fff}
mark{background:#ffe9a8;padding:0 .15rem;border-radius:3px}
.gate{font:12px/1.5 ui-monospace,Menlo,monospace;background:#f4f7fb;
border:1px solid #dbe6f5;border-radius:8px;padding:.45rem .6rem;
margin-bottom:.55rem;color:var(--ink)}
.gate b{color:var(--blue)}
.bar{display:flex;height:14px;border-radius:6px;overflow:hidden;
margin:.45rem 0 .2rem;border:1px solid var(--grid)}
.bar div{height:100%}
.seg{font-size:.72rem;color:var(--mut);margin-bottom:.3rem}
.delta{padding:.55rem 1rem;font-weight:650;font-size:.9rem;
border-top:1px solid var(--grid)}
.delta.win{color:var(--green);background:#f0f9f2}
.delta.trade{color:var(--amber);background:#fdf8ec}
.delta.miss{color:var(--red);background:#fdf0f0}
.delta.keep{color:var(--blue);background:#f0f5fd}
.note{color:var(--mut);font-size:.8rem;max-width:1080px;
margin:0 auto;padding:0 1rem 2rem}
a{color:var(--blue)}
</style></head><body>
<header><h1>升级前后对比 — 六个真实案例</h1>
<div class=sub>左 = 探针关(never 臂实测) · 右 = 探针开(该题实际被升级/保留的那一档实测) ·
所有回答、毫秒、判分均来自 live sweep 的 trace,无任何改写 ·
<a href="/__TOKEN__">← 回主 demo</a></div></header>
<div class=wrap id=root></div>
<div class=note>黄色高亮 = 回答文本里的 hedging 措辞。判分 = OpenAudioBench
官方判分器。延迟口径:用户话音落下 → 回答文本完成。</div>
<script>
const CASES=__CASES__;
const $=(s)=>document.querySelector(s);
const fmt=(ms)=>(ms/1000).toFixed(2);
const verdict=(ok)=>`<span class="pill ${ok?'ok':'no'}">${ok?'判分:对':'判分:错'}</span>`;
function offCol(c){
  return `<div class="col left"><h3>探针关 · 只靠小模型</h3>
   <span class=lat>${fmt(c.off.ms)}<small> s</small></span>${verdict(c.off.ok)}
   <div class=seg>本地解码 ${c.off.chars} 字符</div>
   <div class=ans>${c.off.ans}</div></div>`;
}
function onCol(c){
  const o=c.on;
  if(o.mode==="escalated"){
    const tot=o.total_ms;
    const segs=[[o.eot_ms,"#0b0b0b","句尾读数"],
      [Math.max(o.stall_ms,o.expert_s*1000),"#1e9e50","gpt-5.5"],
      [o.relay_ms,"#8fcda4","转述"]];
    const bar=segs.map(s=>`<div style="width:${100*s[0]/tot}%;background:${s[1]}"
      title="${s[2]}"></div>`).join("");
    return `<div class=col><h3>探针开 · ${o.tier} 档,实际被升级</h3>
     <span class=lat>${fmt(tot)}<small> s</small></span>${verdict(o.ok)}
     <div class=gate>探针读数 <b>${o.score.toFixed(3)}</b> ≥ 阈值
       ${o.thr.toFixed(3)} → <b>升级</b>(读数仅 ${o.eot_ms|0} ms,
       回答尚未生成)</div>
     <div class=bar>${bar}</div>
     <div class=seg>读数 ${o.eot_ms|0}ms → gpt-5.5 ${o.expert_s.toFixed(2)}s
       (stall 掩护 ${o.stall_ms|0}ms)→ 转述 ${(o.relay_ms/1000).toFixed(2)}s</div>
     <div class=ans><b>talker 转述:</b>${o.relay}
       \\n\\n<b>专家原文:</b>${o.expert}</div></div>`;
  }
  return `<div class=col><h3>探针开 · 判定无需升级</h3>
   <span class=lat>${fmt(o.ms)}<small> s</small></span>${verdict(o.ok)}
   <div class=gate>探针读数 <b>${o.score.toFixed(3)}</b> &lt; 阈值
     ${o.thr.toFixed(3)} → <b>留在本地</b>(省一次专家调用)</div>
   <div class=ans>${o.ans}</div></div>`;
}
function delta(c){
  const o=c.on;
  if(c.cat==="kept")
    return `<div class="delta keep">✔ 正确保留:本地又快又对,专家预算留给真正难的题</div>`;
  if(c.cat==="missed")
    return `<div class="delta miss">✘ 15% 预算下漏网的"自信错"——回答毫无 hedging、探针分 ${o.score.toFixed(3)} 不够高;在 50% 档(aggressive)被抓回并答对。这类错是升级率往上提时曲线继续上涨的原因</div>`;
  const d=(c.off.ms-o.total_ms)/1000;
  const acc=c.off.ok? "两边都对" : (o.ok? "错 → 对":"仍错");
  if(d>0) return `<div class="delta win">▲ 升级反而快 ${d.toFixed(2)} s,且${acc}——这类题就是延迟拐弯的来源</div>`;
  return `<div class="delta trade">▼ 多花 ${(-d).toFixed(2)} s,换来${acc}——升级的典型形态(用延迟买准确率)</div>`;
}
$("#root").innerHTML=CASES.map(c=>`
 <div class=case>
  <div class=chead><span class="cat ${c.cat}">${c.cat_label}</span>
   <span class=ref>${c.pool} · ${c.id} · 音频 ${c.audio_s.toFixed(1)}s</span>
   <div class=q>${c.q}</div>
   <div class=ref>参考答案:${c.ref}</div></div>
  <div class=cols>${offCol(c)}${onCol(c)}</div>
  ${delta(c)}
 </div>`).join("");
</script></body></html>"""
'''
s = s.rstrip() + NL + CASES_PAGE

open(p, 'w', encoding='utf-8', newline=NL).write(s)
import ast
ast.parse(s)
print('cases page added, syntax ok')
