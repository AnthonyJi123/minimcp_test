"""Floor-control sweep (todo P1 2026-08-27): does the escalation gate
break the talker's barge-in vs backchannel discrimination?

Grid (cell = phase:stim:arm):
  phase ans   -- overlap injected while the talker speaks a LOCAL answer
                 (non-firing queries; BOTH arms g0/g1 -> the paired
                 orthogonality claim: gate reads once at EOT and never
                 touches the floor state machine)
  phase stall/wait/relay -- the three phases that exist only under
                 escalation (firing queries, g1 only): stall sentence,
                 expert-wait silence, relay speech. The actual risk
                 surface of "escalation broke full-duplex".
  stim  bcs   -- short backchannel (<1.2 s; "Okay." ...)   -> expect resume
        bcl   -- long lexicon-only continuer (tests the >=1.2 s
                 sustained-commit rule vs the leaky accumulator)
        stop  -- short out-of-lexicon command ("Stop!")    -> expect interrupt
        bq    -- frozen-pool query spoken over the talker  -> expect
                 sustained interrupt + next-turn seeding

Injection is streamed at REAL-TIME pacing (0.128 s/frame) so client
latencies are honest; query audio streams fast (0.03) like _ws_duplex.
One ws session per pair (connect => _turn_reset, no cross-pair state).
Outcomes: resume | interrupt | no_duck | no_phase | wrong_fire |
no_fire | no_eot. Restartable: pair_ids in the JSONL are skipped.

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _ws_floor.py::sweep --limit 1              # smoke: 1/cell
  modal run _ws_floor.py::sweep                        # full
  modal run _ws_floor.py::report                       # per-cell table
"""
import json

import modal

from modal_app import OPENAI

app = modal.App("ws-floor-sweep")
vol = modal.Volume.from_name("gate-data")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy",
                    "pandas", "pyarrow", "openai")
       .add_local_file("modal_app.py", "/root/modal_app.py"))

BASE = "https://rhe9527--gate-demo-voice.modal.run/62dc5cd9"
WS = "wss://rhe9527--gate-demo-voice.modal.run/62dc5cd9/ws"
OUT = "/data/floor_sweep/floor.jsonl"
STIM_DIR = "/data/floor_sweep/stim"

# lexicon-only variants (bcs/bcl MUST pass demo_app._classify_burst if
# the ASR path runs: <=4 tokens, all in BACKCHANNEL, en+zh)
STIM_TEXT = {
    "bcs": ["Okay.", "Yeah.", "Mm-hm.", "嗯。",
            "好的。"],
    "bcl": ["Uh-huh, yeah, okay.", "Oh, okay, go on.",
            "Oh wow, right, right.",
            "嗯，好的，继续。"],
    "stop": ["Stop!", "Wait, stop!", "别说了。"],
}
N_ANS = 40    # pairs per ans-phase cell (paired across g0/g1)
N_ESC = 16    # pairs per escalated-phase cell (g1 only)

CELLS = ([("ans", s, a) for s in ("bcs", "bcl", "stop", "bq")
          for a in ("g0", "g1")]
         + [(p, s, "g1") for p in ("stall", "wait", "relay")
            for s in ("bcs", "bq")])


@app.function(image=img, volumes={"/data": vol}, secrets=[OPENAI],
              timeout=60 * 60 * 5)
async def sweep(limit: int = 0, only: str = ""):
    import asyncio
    import os
    import time as _time
    import urllib.request

    import librosa
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import websockets
    from openai import OpenAI

    os.makedirs(STIM_DIR, exist_ok=True)
    t0 = _time.time()
    r = None
    while _time.time() - t0 < 480:
        try:
            r = json.load(urllib.request.urlopen(f"{BASE}/ready",
                                                 timeout=25))
            break
        except Exception as e:
            print(f"  warm poll: {type(e).__name__}")
            await asyncio.sleep(4)
    assert r and r.get("ready"), "GPU never became ready"
    print(f"ready: {r}")

    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = {json.loads(x)["pair_id"] for x in fh if x.strip()}

    # ---- stimuli (TTS once, cached on the volume) ----
    cl = OpenAI()
    stim = {}
    for kind, texts in STIM_TEXT.items():
        stim[kind] = []
        for j, tx in enumerate(texts):
            p = f"{STIM_DIR}/{kind}{j}.wav"
            if not os.path.exists(p):
                r = cl.audio.speech.create(model="tts-1", voice="onyx",
                                           input=tx,
                                           response_format="wav")
                au, _ = librosa.load(__import__("io").BytesIO(r.content),
                                     sr=16000, mono=True)
                sf.write(p, au, 16000)
            au, _ = librosa.load(p, sr=16000, mono=True)
            if kind == "bcl":
                # TTS says these in <1.2 s; stretch so the LONG stratum
                # actually probes the sustained-commit rule
                au = librosa.effects.time_stretch(au, rate=0.62)
            stim[kind].append((tx, au))
            print(f"stim {kind}{j} '{tx}' {len(au)/16000:.2f}s")
    vol.commit()

    # ---- pools ----
    tr = pd.read_parquet("/data/frozen_v3_traces.parquet")
    loc = tr[tr["mode"] == "local"]
    agg = (tr.groupby("id").agg(score=("eot_score", "mean")).reset_index())
    lagg = (loc.groupby("id")
            .agg(score=("eot_score", "mean"), ans_ms=("answer_ms", "mean"),
                 ok=("heard_ok", "mean")).reset_index())
    have = lambda i: os.path.exists(f"/data/audio_pool/{i}.wav")

    def max_pause(qid):
        # longest internal VAD-silence: pool TTS pauses >1.25 s trip
        # the server VAD mid-query and the query's own tail then reads
        # as a barge-in (39% of pairs in the first launch). Measure it
        # the way the server hears it: 0.128 s frame RMS of
        # audio+stream-noise against the adaptive-floor speech
        # threshold (~floor*3.5 with floor ~0.008)
        au, _ = librosa.load(f"/data/audio_pool/{qid}.wav", sr=16000,
                             mono=True)
        n = 2048
        run = best = 0
        for i in range(0, len(au) - n, n):
            f = au[i:i + n]
            rms = float(np.sqrt((f * f).mean() + 0.008 ** 2))
            if rms < 0.028:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best * n / 16000.0

    def pause_ok(cands, need, cap=1.0):
        out = []
        for i in cands:
            if have(i) and max_pause(i) < cap:
                out.append(i)
            if len(out) >= need:
                break
        return out

    ans_q = lagg[(lagg["score"] < 0.25) & (lagg["ans_ms"] > 5000)
                 & (lagg["ok"] > 0.5)].sort_values("score")
    ans_q = pause_ok(list(ans_q["id"]), 30)
    if len(ans_q) < 12:   # relax selection, then the pause cap
        more = lagg[(lagg["score"] < 0.35) & (lagg["ans_ms"] > 3000)
                    & (lagg["ok"] > 0.4)].sort_values("score")
        ans_q = pause_ok(list(more["id"]), 24)
    if len(ans_q) < 12:
        ans_q = pause_ok(list(more["id"]), 24, cap=1.15)
    # thr(balanced) from a throwaway hello (retry while a stale
    # session still holds the server lock)
    thr = None
    for _ in range(40):
        async with websockets.connect(WS + "?tier=balanced&probe_on=0",
                                      max_size=None,
                                      open_timeout=300) as w:
            hello = json.loads(await w.recv())
        if "thr" in hello:
            thr = hello["thr"]
            break
        print(f"  busy, retrying: {hello}")
        await asyncio.sleep(8)
    assert thr is not None, "server stayed busy"
    esc_q = agg[agg["score"] > thr + 0.05].sort_values(
        "score", ascending=False)
    esc_q = pause_ok(list(esc_q["id"]), 30)
    barge_q = [i for i in agg.sample(frac=1, random_state=13)["id"]
               if have(i)]
    assert ans_q and esc_q, (len(ans_q), len(esc_q))
    print(f"thr={thr} ans_q={len(ans_q)} esc_q={len(esc_q)}")

    rng = np.random.default_rng(23)
    FR = 2048

    def frames(au, gain=1.0):
        au = au * gain + rng.normal(0, 0.008, len(au))
        i16 = (au * 32767).clip(-32767, 32767).astype(np.int16)
        return [i16[i:i + FR].tobytes() for i in range(0, len(i16), FR)]

    def noise():
        return ((rng.normal(0, 0.008, FR) * 32767)
                .clip(-32767, 32767).astype(np.int16).tobytes())

    def load(qid):
        au, _ = librosa.load(f"/data/audio_pool/{qid}.wav", sr=16000,
                             mono=True)
        return au

    TAIL = np.zeros(int(1.6 * 16000))

    # ---- pair list (deterministic; interleave cells) ----
    pairs = []
    for (ph, st, arm) in CELLS:
        if only and f"{ph}:{st}:{arm}" not in only:
            continue
        n = N_ANS if ph == "ans" else N_ESC
        if limit:
            n = min(n, limit)
        for i in range(n):
            qid = (ans_q[i % len(ans_q)] if ph == "ans"
                   else esc_q[i % len(esc_q)])
            b = barge_q[i % len(barge_q)]
            if b == qid:
                b = barge_q[(i + 1) % len(barge_q)]
            pairs.append(dict(pair_id=f"{ph}:{st}:{arm}:{i:03d}",
                              phase=ph, stim=st, arm=arm, qid=qid,
                              var=i, bq=b))
    order = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in order if pairs[i]["pair_id"] not in done]
    print(f">>> {len(pairs)} pairs pending")

    def save(rec):
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if sum(1 for _ in open(OUT)) % 8 == 0:
            vol.commit()

    async def run_pair(pr):
        probe_on = 1 if pr["arm"] == "g1" else 0
        url = f"{WS}?tier=balanced&probe_on={probe_on}"
        if pr["stim"] == "bq":
            s_text, s_au = pr["bq"], load(pr["bq"])
        else:
            tx, au = stim[pr["stim"]][pr["var"] % len(stim[pr["stim"]])]
            s_text, s_au = tx, au
        rec = dict(pr, stim_text=str(s_text)[:60],
                   stim_s=round(len(s_au) / 16000, 2), thr=thr)
        async with websockets.connect(url, max_size=None,
                                      open_timeout=120) as ws:
            hello = json.loads(await ws.recv())
            if hello.get("type") != "hello":
                raise RuntimeError(f"no hello: {hello}")
            st = dict(eots=[], turns=[], ducks=[], resumes=[],
                      interrupts=[], audio=[], phases=[], speech=[])

            async def rx():
                while True:
                    try:
                        m = json.loads(await ws.recv())
                    except Exception:
                        return
                    t, now = m.get("type"), _time.time()
                    if t == "audio":
                        st["audio"].append(now)
                    elif t == "phase":
                        st["phases"].append((m["v"], now))
                    elif t == "eot":
                        st["eots"].append((m, now))
                    elif t == "turn":
                        st["turns"].append((m, now))
                    elif t == "duck":
                        st["ducks"].append(now)
                    elif t == "resume":
                        st["resumes"].append((m, now))
                    elif t == "interrupt":
                        st["interrupts"].append((m, now))
                    elif t == "speech":
                        st["speech"].append((m, now))

            rxt = asyncio.create_task(rx())

            async def stream(au, dt, gain=1.0):
                for f in frames(au, gain):
                    await ws.send(f)
                    await asyncio.sleep(dt)

            async def idle(cond, tmax):
                t1 = _time.time()
                while not cond() and _time.time() - t1 < tmax:
                    await ws.send(noise())
                    await asyncio.sleep(0.128)
                return bool(cond())

            def cur_phase():
                return st["phases"][-1][0] if st["phases"] else None

            try:
                await stream(load(pr["qid"]), 0.03)
                await stream(TAIL, 0.03)
                if not await idle(lambda: st["eots"], 90):
                    rec.update(outcome="no_eot")
                    return rec
                eot, t_eot = st["eots"][-1]
                rec.update(eot_score=eot["score"], fired=eot["fired"])
                if pr["phase"] == "ans" and eot["fired"]:
                    rec.update(outcome="wrong_fire")
                    return rec
                if pr["phase"] != "ans" and not eot["fired"]:
                    rec.update(outcome="no_fire")
                    return rec

                # ---- wait for the injection window ----
                if pr["phase"] == "ans":
                    okp = await idle(
                        lambda: cur_phase() == "answering"
                        and len(st["audio"]) >= 2, 30)
                    extra = 0.9
                elif pr["phase"] == "stall":
                    okp = await idle(
                        lambda: cur_phase() == "escalating"
                        and len(st["audio"]) >= 1, 30)
                    extra = 0.4
                elif pr["phase"] == "wait":
                    okp = await idle(
                        lambda: cur_phase() == "escalating"
                        and len(st["audio"]) >= 1, 30)
                    extra = 2.4      # past the ~2 s canned stall
                else:                # relay
                    okp = await idle(
                        lambda: cur_phase() == "relaying"
                        and st["audio"]
                        and st["audio"][-1] > st["phases"][-1][1], 90)
                    extra = 0.7
                if not okp:
                    rec.update(outcome="no_phase",
                               phase_seen=cur_phase())
                    return rec
                t1 = _time.time()
                while _time.time() - t1 < extra:
                    await ws.send(noise())
                    await asyncio.sleep(0.128)
                if st["ducks"] or st["resumes"] or st["interrupts"]:
                    # floor events BEFORE our injection: the VAD ended
                    # the turn early and the query's own tail barged in
                    pre = min([st["ducks"][0]] if st["ducks"] else []
                              + [w for _, w in st["interrupts"]]
                              + [w for _, w in st["resumes"]])
                    rec.update(outcome="early_eot",
                               pre_ducks=len(st["ducks"]),
                               pre_ints=len(st["interrupts"]),
                               pre_res=len(st["resumes"]),
                               pre_ms=int((pre - t_eot) * 1000))
                    return rec

                # ---- inject at real-time pacing ----
                t_inj = _time.time()
                rec.update(phase_at_inject=cur_phase())
                await stream(s_au, 0.128, gain=1.7)
                await idle(lambda: st["resumes"] or st["interrupts"],
                           max(6.0, len(s_au) / 16000) + 8)

                ms = lambda w: int((w - t_inj) * 1000)
                rec.update(
                    duck_ms=ms(st["ducks"][0]) if st["ducks"] else None,
                    phase_at_resolve=cur_phase())
                if st["interrupts"]:
                    m, w = st["interrupts"][0]
                    rec.update(outcome="interrupt", interrupt_ms=ms(w),
                               why=m.get("why"),
                               heard=str(m.get("heard", ""))[:80])
                elif st["resumes"]:
                    m, w = st["resumes"][0]
                    rec.update(outcome="resume", resume_ms=ms(w),
                               heard=str(m.get("heard", ""))[:80])
                else:
                    # duck engaged but no resolution: did the talker
                    # simply finish during the overlap (short relay)?
                    got_turn = await idle(
                        lambda: st["turns"]
                        and st["turns"][-1][1] > t_inj, 25)
                    if (st["ducks"] and got_turn
                            and not st["turns"][-1][0].get(
                                "interrupted")):
                        rec.update(outcome="turn_done_first",
                                   turn_interrupted=False,
                                   mode=st["turns"][-1][0].get("mode"))
                    else:
                        rec.update(outcome="no_duck")
                    return rec

                # ---- resolution details ----
                if rec["outcome"] == "resume":
                    # answer/relay must run to natural completion
                    if await idle(lambda: st["turns"], 150):
                        t_rec, w = st["turns"][-1]
                        rec.update(
                            turn_interrupted=t_rec.get("interrupted"),
                            mode=t_rec.get("mode"),
                            answer_len=len(str(t_rec.get("answer") or "")),
                            audio_after_resume=bool(
                                st["audio"]
                                and st["audio"][-1]
                                > st["resumes"][0][1]))
                    else:
                        rec.update(turn_interrupted=None)
                else:
                    if await idle(lambda: st["turns"], 90):
                        t_rec, w = st["turns"][-1]
                        rec.update(
                            turn_interrupted=t_rec.get("interrupted"),
                            mode=t_rec.get("mode"))
                    # seeding: the barge speech must become the next
                    # turn (bq only) -- wait for the seeded EOT
                    if pr["stim"] == "bq":
                        n_e = len(st["eots"])
                        await stream(TAIL, 0.03)
                        seeded = await idle(
                            lambda: len(st["eots"]) > n_e, 45)
                        rec.update(seed_next_eot=bool(seeded))
                        if seeded:
                            rec.update(
                                seed_eot_score=st["eots"][-1][0]["score"])
                rec.update(wall_s=round(_time.time() - t_eot, 1))
                return rec
            finally:
                rxt.cancel()

    n_done = 0
    for pr in pairs:
        for attempt in (1, 2):
            try:
                rec = await run_pair(pr)
                break
            except Exception as e:
                print(f"  !! {pr['pair_id']} {type(e).__name__}: "
                      f"{str(e)[:100]}" + (" retrying" if attempt == 1
                                           else ""), flush=True)
                rec = dict(pr, outcome="error", err=str(e)[:150])
                await asyncio.sleep(6)
        save(rec)
        n_done += 1
        print(f"[{n_done}/{len(pairs)}] {pr['pair_id']} "
              f"-> {rec.get('outcome')} "
              f"duck={rec.get('duck_ms')} res={rec.get('resume_ms')} "
              f"int={rec.get('interrupt_ms')} why={rec.get('why')} "
              f"heard={rec.get('heard', '')!r:.40} "
              f"ti={rec.get('turn_interrupted')} "
              f"seed={rec.get('seed_next_eot')}", flush=True)
    vol.commit()
    print(">>> SWEEP DONE")


@app.function(image=img, volumes={"/data": vol}, timeout=600)
def report():
    import collections
    import os
    if not os.path.exists(OUT):
        print("no results yet")
        return
    rows = [json.loads(x) for x in open(OUT) if x.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["phase"], r["stim"], r["arm"])].append(r)
    print(f"{'cell':<16}{'n':>4}{'resume':>8}{'intr':>6}{'noduck':>8}"
          f"{'other':>7}  med duck/res/int ms   seed")
    for k in sorted(by):
        rs = by[k]
        n = len(rs)
        cnt = collections.Counter(r["outcome"] for r in rs)
        med = lambda f: (sorted(f)[len(f) // 2] if f else "-")
        d = med([r["duck_ms"] for r in rs if r.get("duck_ms")])
        re_ = med([r["resume_ms"] for r in rs if r.get("resume_ms")])
        it = med([r["interrupt_ms"] for r in rs if r.get("interrupt_ms")])
        sd = [r for r in rs if "seed_next_eot" in r]
        seed = (f"{sum(r['seed_next_eot'] for r in sd)}/{len(sd)}"
                if sd else "-")
        print(f"{':'.join(k):<16}{n:>4}{cnt['resume']:>8}"
              f"{cnt['interrupt']:>6}{cnt['no_duck']:>8}"
              f"{n - cnt['resume'] - cnt['interrupt'] - cnt['no_duck']:>7}"
              f"  {d}/{re_}/{it}   {seed}")
