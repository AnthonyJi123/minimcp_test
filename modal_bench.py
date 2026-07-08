"""Modal app: offline audio/omni benchmarks for MiniCPM-o 4.5.

Fills the three columns the duplex work (modal_fdb.py) doesn't cover:
LibriSpeech WER, Big Bench Audio, Daily-Omni. All use the raw `model.chat`
path (bench/model.py), not the duplex loop.

Reuses modal_app.py's validated model image + weights Volume; adds a `bench-data`
Volume for datasets + predictions. Datasets download on cheap CPU boxes; only
inference touches the H100.

Usage (pilot-first — small limits validate the harness before a full run):
    modal run modal_bench.py::download_librispeech
    modal run modal_bench.py::download_bigbench
    modal run modal_bench.py::download_dailyomni
    modal run modal_bench.py::pilot                 # ~30/benchmark on one H100
    modal run modal_bench.py::run_librispeech --limit 0 --workers 4
    modal run modal_bench.py::run_bigbench --limit 0 --workers 4
    modal run modal_bench.py::run_dailyomni --limit 0 --workers 2
    modal run modal_bench.py::score                 # WER + accuracy from preds
"""
import json
import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))

# Reuse modal_app.py's weights Volume, but define the handle locally so this app
# never imports modal_app — that keeps bench fully self-contained (no cross-module
# container import) and its image free of base_image's trailing add_local_dir.
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"

DATA = "/data"
PRED = f"{DATA}/preds"
LS_ROOT = f"{DATA}/librispeech"
BBH_ROOT = f"{DATA}/bigbench"
DO_ROOT = f"{DATA}/dailyomni"

app = modal.App("bench-minicpmo45")
data = modal.Volume.from_name("bench-data", create_if_missing=True)
weights = modal.Volume.from_name("minicpm-o45-weights")

# Mirrors modal_app.py's validated stack (torch 2.8 + transformers 4.51, SDPA,
# no flash-attn) but built standalone so bench's extra pips precede the local-dir
# add (Modal requires add_local_* to be the final build steps). Bench uses the
# raw AutoModel + minicpmo.utils, so it needs neither the MiniCPM-o-Demo clone
# nor the duplex harness dir that base_image carries.
bench_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsndfile1")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install(
        "minicpmo-utils[all]",       # librosa 0.9.0 + audio/video stack
        "transformers==4.51.0",      # 4.52+ breaks MiniCPM's Resampler init
        "accelerate==1.12.0",
        "setuptools<81",             # librosa 0.9 imports pkg_resources
        "pydantic>=2.11",
        "PyYAML",
        "soundfile",
        "opencv-python-headless",
        "huggingface_hub[hf_transfer]",
        "jiwer",                     # WER
        "datasets>=2.18",
        "pandas",
        "decord",                    # video frame decode for omni QA
    )
    .add_local_dir(os.path.join(HERE, "bench"), "/workspace/bench")
)

util_image = (modal.Image.debian_slim(python_version="3.11")
              .pip_install("huggingface_hub[hf_transfer]", "datasets>=2.18",
                           "soundfile", "pandas", "pyarrow")
              .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
              .add_local_dir(os.path.join(HERE, "bench"), "/workspace/bench"))

GPU_VOL = {"/workspace/models": weights, DATA: data}


# ------------------------------------------------------------------ downloads
@app.function(image=util_image, volumes={DATA: data}, timeout=60 * 60)
def download_librispeech(splits: str = "test-clean"):
    """Fetch + unpack the openslr LibriSpeech test tarball(s)."""
    import tarfile
    import urllib.request

    sys.path.insert(0, "/workspace/bench")
    from librispeech import OPENSLR

    for split in splits.split():
        marker = os.path.join(LS_ROOT, "LibriSpeech", split)
        if os.path.isdir(marker) and os.listdir(marker):
            print(f">>> {split}: already present")
            continue
        os.makedirs(LS_ROOT, exist_ok=True)
        tgz = os.path.join(LS_ROOT, f"{split}.tar.gz")
        if not os.path.exists(tgz):
            print(f">>> downloading {split} ...", flush=True)
            urllib.request.urlretrieve(OPENSLR[split], tgz)
        print(f">>> extracting {split} ...", flush=True)
        with tarfile.open(tgz) as tf:
            tf.extractall(LS_ROOT)
        os.remove(tgz)
    data.commit()
    print(">>> done")


@app.function(image=util_image, volumes={DATA: data}, timeout=60 * 60)
def download_bigbench():
    """Snapshot the mp3s + build manifest.jsonl of the text metadata.

    The repo's main branch holds only data/question_{id}.mp3 + README; the
    text fields (category, official_answer) live in HF's auto-converted parquet,
    which `load_dataset` resolves. mp3 filenames encode the id, so we keep just
    the text metadata and compute audio paths from the id at inference time.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    import glob

    REPO = "ArtificialAnalysis/big_bench_audio"
    if len(glob.glob(f"{BBH_ROOT}/data/*.mp3")) < 1000:  # resume any partial pull
        snapshot_download(REPO, repo_type="dataset", local_dir=BBH_ROOT)
        data.commit()
    manifest = f"{BBH_ROOT}/manifest.jsonl"
    if not os.path.exists(manifest):
        # text fields live only in HF's auto-converted parquet; read the text
        # columns directly (reading the audio column is what hung load_dataset).
        api = HfApi()
        pqs = [f for f in api.list_repo_files(REPO, repo_type="dataset",
                                              revision="refs/convert/parquet")
               if f.endswith(".parquet")]
        n = 0
        with open(manifest, "w", encoding="utf-8") as fh:
            for f in pqs:
                p = hf_hub_download(REPO, f, repo_type="dataset",
                                    revision="refs/convert/parquet")
                names = set(pq.read_schema(p).names)
                cols = [c for c in ("id", "category", "official_answer") if c in names]
                tbl = pq.read_table(p, columns=cols).to_pydict()
                for i in range(len(tbl[cols[0]])):
                    fh.write(json.dumps({
                        "id": int(tbl["id"][i]),
                        "category": str(tbl["category"][i]),
                        "official_answer": str(tbl["official_answer"][i]),
                    }, ensure_ascii=False) + "\n")
                    n += 1
        print(f">>> manifest: {n} rows from {len(pqs)} parquet file(s)")
        data.commit()
    rows = _bbh_manifest()
    miss = [r["id"] for r in rows if not os.path.exists(r["path"])]
    print(f">>> {len(rows)} rows; {len(miss)} missing mp3; sample: {rows[0]}")


def _bbh_manifest() -> list:
    """Read manifest.jsonl; resolve each mp3 as data/question_{id}.mp3."""
    rows = []
    with open(f"{BBH_ROOT}/manifest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            d["path"] = f"{BBH_ROOT}/data/question_{d['id']}.mp3"
            rows.append(d)
    return rows


@app.function(image=util_image, volumes={DATA: data}, timeout=60 * 60 * 2)
def download_dailyomni():
    """Snapshot liarliar/Daily-Omni and extract Videos.tar."""
    import glob
    import tarfile

    from huggingface_hub import snapshot_download

    if not (os.path.isdir(DO_ROOT) and os.path.exists(f"{DO_ROOT}/qa.json")):
        snapshot_download("liarliar/Daily-Omni", repo_type="dataset",
                          local_dir=DO_ROOT)
        data.commit()
    vids = os.path.join(DO_ROOT, "Videos")
    if not (os.path.isdir(vids) and os.listdir(vids)):
        for tar in glob.glob(f"{DO_ROOT}/Videos*.tar"):
            print(f">>> extracting {os.path.basename(tar)} ...", flush=True)
            with tarfile.open(tar) as tf:
                tf.extractall(DO_ROOT)
        data.commit()
    n_v = len(os.listdir(vids)) if os.path.isdir(vids) else 0
    sys.path.insert(0, "/workspace/bench")
    from dailyomni import load_qa
    qa = load_qa(f"{DO_ROOT}/qa.json")
    print(f">>> {len(qa)} QA rows, {n_v} video files; sample: {qa[0]}")


# ------------------------------------------------------------------ inference
@app.function(image=bench_image, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 4)
def infer_librispeech(items: list, split: str) -> int:
    """items: [[uid, flac_path, ref], ...]. Writes preds/librispeech/{split}.jsonl."""
    sys.path.insert(0, "/workspace/bench")
    import librosa
    from model import BenchModel

    m = BenchModel(MODEL_DIR)
    print(f">>> model loaded {m.load_s:.1f}s | {len(items)} utts", flush=True)
    os.makedirs(f"{PRED}/librispeech", exist_ok=True)
    out = f"{PRED}/librispeech/{split}.shard{items[0][0]}.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for k, (uid, path, ref) in enumerate(items):
            try:
                wav, _ = librosa.load(path, sr=16000, mono=True)
                hyp = m.asr(wav)
            except Exception as e:
                print(f"[{k}] {uid} FAILED: {e}", flush=True)
                continue
            fh.write(json.dumps({"uid": uid, "ref": ref, "hyp": hyp},
                                ensure_ascii=False) + "\n")
            n += 1
            if k < 3 or k % 100 == 0:
                print(f"[{k}] {uid} :: {hyp[:70]!r}", flush=True)
    data.commit()
    return n


@app.function(image=bench_image, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 4)
def infer_bigbench(rows: list) -> int:
    """rows: manifest slices. Writes preds/bigbench.shard*.jsonl."""
    sys.path.insert(0, "/workspace/bench")
    import librosa
    from model import BenchModel
    import bigbench

    m = BenchModel(MODEL_DIR)
    print(f">>> model loaded {m.load_s:.1f}s | {len(rows)} qs", flush=True)
    os.makedirs(PRED, exist_ok=True)
    out = f"{PRED}/bigbench.shard{rows[0]['id']}.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for k, r in enumerate(rows):
            try:
                wav, _ = librosa.load(r["path"], sr=16000, mono=True)
                output = m.answer_audio(wav, bigbench.PROMPT)
            except Exception as e:
                print(f"[{k}] id={r['id']} FAILED: {e}", flush=True)
                continue
            ok = bigbench.is_correct(r["category"], r["official_answer"], output)
            fh.write(json.dumps({**{k2: r[k2] for k2 in
                                    ("id", "category", "official_answer")},
                                 "output": output, "ok": ok},
                                ensure_ascii=False) + "\n")
            n += 1
            if k < 3:
                print(f"[{k}] {r['category']} gold={r['official_answer']!r} "
                      f"ok={ok} :: {output[:60]!r}", flush=True)
    data.commit()
    return n


@app.function(image=bench_image, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 4)
def infer_dailyomni(rows: list) -> int:
    """rows: qa slices. Writes preds/dailyomni.shard*.jsonl."""
    sys.path.insert(0, "/workspace/bench")
    from model import BenchModel
    import dailyomni

    m = BenchModel(MODEL_DIR)
    print(f">>> model loaded {m.load_s:.1f}s | {len(rows)} qs", flush=True)
    os.makedirs(PRED, exist_ok=True)
    vids = f"{DO_ROOT}/Videos"
    out = f"{PRED}/dailyomni.shard{rows[0]['id']}.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for k, r in enumerate(rows):
            vp, is_temp = dailyomni.prepare_media(vids, r["video_id"])
            if not vp:
                print(f"[{k}] {r['id']} missing video {r['video_id']}", flush=True)
                continue
            try:
                output = m.answer_video(vp, dailyomni.build_prompt(r))
            except Exception as e:
                print(f"[{k}] {r['id']} FAILED: {e}", flush=True)
                continue
            finally:
                if is_temp and os.path.exists(vp):
                    os.remove(vp)
            pred = dailyomni.resolve_pred(output, r["choices"])
            fh.write(json.dumps({"id": r["id"], "answer": r["answer"],
                                 "pred": pred, "duration": r["duration"],
                                 "type": r["type"], "video_id": r["video_id"],
                                 "choices": r["choices"], "output": output},
                                ensure_ascii=False) + "\n")
            n += 1
            if k < 3:
                print(f"[{k}] {r['id']} gold={r['answer']} pred={pred} "
                      f":: {output[:40]!r}", flush=True)
    data.commit()
    return n


# ------------------------------------------------------------------ sharding
def _shard(seq, workers):
    w = max(1, min(workers, len(seq)))
    return [seq[i::w] for i in range(w)]


@app.function(image=util_image, volumes={DATA: data})
def librispeech_items(split: str, limit: int) -> list:
    sys.path.insert(0, "/workspace/bench")
    from librispeech import iter_samples
    items = [[u, p, t] for u, p, t in iter_samples(LS_ROOT, split)]
    items.sort()
    return items[:limit] if limit else items


@app.function(image=util_image, volumes={DATA: data})
def bigbench_rows(limit: int) -> list:
    rows = sorted(_bbh_manifest(), key=lambda r: r["id"])
    return rows[:limit] if limit else rows


@app.function(image=util_image, volumes={DATA: data})
def dailyomni_rows(limit: int) -> list:
    sys.path.insert(0, "/workspace/bench")
    from dailyomni import load_qa
    rows = load_qa(f"{DO_ROOT}/qa.json")
    return rows[:limit] if limit else rows


# ------------------------------------------------------------------ entrypoints
@app.local_entrypoint()
def run_librispeech(split: str = "test-clean", limit: int = 0, workers: int = 4):
    items = librispeech_items.remote(split, limit)
    shards = _shard(items, workers)
    print(f">>> {len(items)} utts / {len(shards)} H100 workers")
    print(">>> total:", sum(infer_librispeech.map(
        shards, kwargs={"split": split})))


@app.local_entrypoint()
def run_bigbench(limit: int = 0, workers: int = 4):
    rows = bigbench_rows.remote(limit)
    shards = _shard(rows, workers)
    print(f">>> {len(rows)} qs / {len(shards)} H100 workers")
    print(">>> total:", sum(infer_bigbench.map(shards)))


@app.local_entrypoint()
def run_dailyomni(limit: int = 0, workers: int = 2):
    rows = dailyomni_rows.remote(limit)
    shards = _shard(rows, workers)
    print(f">>> {len(rows)} qs / {len(shards)} H100 workers")
    print(">>> total:", sum(infer_dailyomni.map(shards)))


@app.local_entrypoint()
def pilot(n: int = 30):
    """One H100: n samples from each benchmark, to validate the harness cheaply."""
    ls = librispeech_items.remote("test-clean", n)
    print(">>> librispeech:", infer_librispeech.remote(ls, split="test-clean"))
    bb = bigbench_rows.remote(n)
    print(">>> bigbench:", infer_bigbench.remote(bb))
    do = dailyomni_rows.remote(n)
    print(">>> dailyomni:", infer_dailyomni.remote(do))
    print(json.dumps(_score.remote(), indent=2, ensure_ascii=False))


@app.function(image=bench_image, volumes={DATA: data})
def _score() -> dict:
    import glob

    sys.path.insert(0, "/workspace/bench")
    import librispeech
    import bigbench
    import dailyomni

    res = {}
    # LibriSpeech
    for split_dir in sorted({os.path.basename(p).split(".shard")[0]
                             for p in glob.glob(f"{PRED}/librispeech/*.jsonl")}):
        refs, hyps = {}, {}
        for f in glob.glob(f"{PRED}/librispeech/{split_dir}.shard*.jsonl"):
            for line in open(f, encoding="utf-8"):
                d = json.loads(line)
                refs[d["uid"]] = d["ref"]
                hyps[d["uid"]] = d["hyp"]
        if refs:
            res[f"librispeech/{split_dir}"] = librispeech.wer(refs, hyps)
    # Big Bench Audio
    bb_rows = []
    for f in glob.glob(f"{PRED}/bigbench.shard*.jsonl"):
        bb_rows += [json.loads(l) for l in open(f, encoding="utf-8")]
    if bb_rows:
        for r in bb_rows:  # re-apply current extractor to stored outputs
            r["ok"] = bigbench.is_correct(r["category"], r["official_answer"],
                                          r.get("output", ""))
        res["bigbench"] = bigbench.score(bb_rows)
    # Daily-Omni
    do_rows = []
    for f in glob.glob(f"{PRED}/dailyomni.shard*.jsonl"):
        do_rows += [json.loads(l) for l in open(f, encoding="utf-8")]
    if do_rows:
        for r in do_rows:  # re-apply current parse heuristic to stored outputs
            if r.get("choices"):
                r["pred"] = dailyomni.resolve_pred(r.get("output", ""), r["choices"])
        res["dailyomni"] = dailyomni.score(do_rows)
    with open(f"{PRED}/scores.json", "w") as fh:
        json.dump(res, fh, indent=2)
    data.commit()
    return res


@app.local_entrypoint()
def score():
    res = _score.remote()
    print("\n" + "=" * 60)
    print(json.dumps(res, indent=2, ensure_ascii=False))
