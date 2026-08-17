"""Phone-friendly listening page for the frozen TTS pool.

Serves the wavs straight off the gate-data volume (with the truncated
streaming-write RIFF header patched on the fly) plus the gold-vs-heard
text for each case. Deploy: modal deploy listen_app.py

Read-only, scales to zero, ~$0.
"""
import json
import os
import struct

import modal

TOKEN = "62dc5cd9"

app = modal.App("tts-listen")
vol = modal.Volume.from_name("gate-data")
image = (modal.Image.debian_slim()
         .pip_install("fastapi[standard]")
         .add_local_file("data/listen_pack/cases.json", "/root/cases.json"))

GOOD = [
    ("q0578", "生僻人名 + 缩写全对：Gordon Plotkin / ACM SIGPLAN 一字不差"),
    ("q0588", "长句 + 版本号全对：patch 5.2.0 / Elemental Mastery 一字不差"),
]
BAD = [
    ("q0552", "生僻人名被换掉：Mustafa Adebayo Balogun → 听成 Mustapha Arabo "
              "Balogun。听：alloy 念这个名字清不清楚？"),
    ("q0250", "生僻人名：Taurek → 听成 Turek"),
    ("q0271", "⭐ 运算符丢失：Estimate 999 − 103 → 听成 nine hundred ninety "
              "nine hundred and three。听：减号到底念没念出来？"),
    ("q0163", "金额数字：$815.50 → 听成 eight hundred and fifty dollars and "
              "fifty cents（1 丢了）"),
    ("q0212", "LaTeX 公式：Z-transform，四个选项全毁。听 TTS 怎么念 z^-n"),
    ("q0233", "普通听错：选项内容被转岔（题面里的弯引号是正常字符，"
              "之前疑似乱码是显示问题）"),
]
OTHER = [
    ("q0208", "⚠️ 整段 49 秒纯静音（601 个里唯一的坏渲染）——模型说"
              "「音频是静音的」是对的。不用听完，确认没声音即可"),
    ("q0213", "不是听错：模型直接答 D) Mongolia 而不转述题目（指令遵循失败）"),
    ("q0237", "不是听错：96 秒长题，模型输出了答案而非转录"),
    ("q0169", "对照-良性：相似度低只因数字被拼成英文，内容完好"),
    ("q0164", "LaTeX：sqrt 项丢失，选项里的 10 位小数被打乱"),
    ("q0256", "源文本就是裸 LaTeX（2-k\\Omega、1-\\muF）——听 TTS 怎么念"),
]


def _page(cases):
    import html as h

    def block(cid, note):
        c = cases[cid]
        return f"""<div class=case>
<h3>{cid} <span class=pool>{h.escape(c['pool'])}</span></h3>
<p class=note>{h.escape(note)}</p>
<audio controls preload=none src="/{TOKEN}/a/{cid}.wav"></audio>
<table>
<tr><th>GOLD<br><small>TTS 被要求念的原文</small></th><td>{h.escape(c['query'])}</td></tr>
<tr><th>HEARD<br><small>MiniCPM 转录出来的</small></th><td>{h.escape(c['transcript'] or '')}</td></tr>
</table></div>"""

    parts = ["""<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>TTS 听音包</title><style>
body{font-family:-apple-system,system-ui,sans-serif;max-width:820px;margin:0 auto;
padding:1rem;line-height:1.5;background:#fafafa;color:#111}
.case{border:1px solid #ddd;border-radius:10px;padding:.8rem;margin:.9rem 0;background:#fff}
h1{font-size:1.3rem}h2{font-size:1.05rem;border-bottom:2px solid #333;padding-bottom:.3rem;margin-top:2rem}
h3{margin:.2rem 0;font-size:1rem}.pool{font-size:.75em;color:#777;font-weight:400}
.note{color:#b00;margin:.35rem 0;font-size:.9rem}
table{width:100%;border-collapse:collapse;margin-top:.5rem;font-size:.82rem}
th{width:6.5em;text-align:left;vertical-align:top;color:#555;padding:.35rem;background:#f4f4f4;font-weight:600}
th small{font-weight:400;color:#888}
td{padding:.35rem;border-bottom:1px solid #eee;white-space:pre-wrap;word-break:break-word}
audio{width:100%;margin-top:.4rem}
.intro{background:#fff;border-left:4px solid #333;padding:.7rem;font-size:.9rem}
</style></head><body>
<h1>听音包：现在的 TTS 念出来是什么样</h1>
<div class=intro>音频池 = 冻结的 600 题，OpenAI <b>tts-1</b> · <b>alloy</b> 音色。<br><br>
判断标准：bad case 里 <b>你自己都听不清</b>（人名含糊、减号没念）→ 是 TTS 的锅，
重渲染有救；你 <b>听得清清楚楚但模型转错了</b> → 是小模型耳朵的锅，换 TTS 没用。
（Whisper 换耳朵只回收 4pp，偏向后者，但以你亲耳听到的为准。）</div>"""]
    parts.append("<h2>✅ Good cases — 先听这两个建立基准</h2>")
    parts += [block(c, n) for c, n in GOOD]
    parts.append("<h2>❌ Bad cases — 真·听错</h2>")
    parts += [block(c, n) for c, n in BAD]
    parts.append("<h2>其他：坏文件 / 不是听错 / 对照</h2>")
    parts += [block(c, n) for c, n in OTHER]
    parts.append("</body></html>")
    return "\n".join(parts)


@app.function(image=image, volumes={"/data": vol}, timeout=60 * 10,
              min_containers=0)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import HTMLResponse, Response

    api = FastAPI()
    cases = json.load(open("/root/cases.json", encoding="utf-8"))
    html_page = _page(cases)

    @api.get(f"/{TOKEN}", response_class=HTMLResponse)
    def index():
        return html_page

    @api.get(f"/{TOKEN}/a/{{qid}}.wav")
    def audio(qid: str, request: Request):
        if qid not in cases:
            raise HTTPException(404)
        path = f"/data/audio_pool/{qid}.wav"
        if not os.path.exists(path):
            raise HTTPException(404)
        buf = bytearray(open(path, "rb").read())
        # streaming writes left RIFF/data sizes at 0xFFFFFFFF -> players
        # report a 24-day duration and refuse to seek; patch to real size.
        n = len(buf)
        struct.pack_into("<I", buf, 4, n - 8)
        struct.pack_into("<I", buf, 40, n - 44)
        data = bytes(buf)

        rng = request.headers.get("range")
        headers = {"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"}
        if rng and rng.startswith("bytes="):
            spec = rng[6:].split(",")[0]
            lo_s, _, hi_s = spec.partition("-")
            lo = int(lo_s) if lo_s else 0
            hi = int(hi_s) if hi_s else n - 1
            hi = min(hi, n - 1)
            chunk = data[lo:hi + 1]
            headers["Content-Range"] = f"bytes {lo}-{hi}/{n}"
            return Response(chunk, status_code=206, media_type="audio/wav",
                            headers=headers)
        return Response(data, media_type="audio/wav", headers=headers)

    return api
