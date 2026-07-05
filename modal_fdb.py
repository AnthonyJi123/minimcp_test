"""Modal app: run Full-Duplex-Bench v1.0 (arXiv 2503.04721) on MiniCPM-o 4.5.

Reuses the eval harness's proven image + weights Volume (modal_app.py) and adds
a `fdb-data` Volume holding the benchmark dataset and the model's outputs.

Usage:
    modal run modal_fdb.py::download_dataset          # one-time, Drive -> Volume
    modal run modal_fdb.py::smoke                     # 2 samples/subset on H100
    modal run modal_fdb.py::run_infer                 # full run, sharded H100s
    modal run modal_fdb.py::run_infer --subsets "icc_backchannel" --workers 2

ASR + evaluation live in modal_fdb_eval.py (separate app so its heavy NeMo
image never blocks inference runs).
"""
import os
import sys

import modal

from modal_app import image as base_image, weights, HARNESS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = "/data"
DATA_ROOT = f"{DATA_DIR}/v1.0"
EXP_ROOT = f"{DATA_DIR}/exp/minicpm-o-4_5"

# v1.0 subsets -> Google Drive file ids (from the benchmark's release folder)
ZIPS = {
    "candor_pause_handling": "1uls3atEz7bZ1IkVq3Rw0yQyLAjjVkklc",
    "candor_turn_taking": "1sb9mwOqDCK9BEpMb6fDVYOJU1vd5RFlb",
    "icc_backchannel": "1HttNZbi0bYHe7a-CO_9vg_z7hMM8a5h0",
    "synthetic_pause_handling": "1iV0X6z3Z9SrmJvxJ2Hkij3nb8iuWbJyv",
    "synthetic_user_interruption": "1I36wGbPtZObjqI_1h11Rb2s1ulerb65a",
}

app = modal.App("fdb-minicpmo45")

data = modal.Volume.from_name("fdb-data", create_if_missing=True)

# NB: containers re-import this file, which imports modal_app — include it
# explicitly (Modal only auto-mounts the entrypoint file itself).
fdb_image = (base_image
             .add_local_dir(os.path.join(HERE, "fdb"), "/workspace/fdb")
             .add_local_python_source("modal_app"))

util_image = (modal.Image.debian_slim(python_version="3.11")
              .pip_install("gdown")
              .add_local_python_source("modal_app"))


@app.function(image=util_image, volumes={DATA_DIR: data}, timeout=60 * 60)
def download_dataset(force: bool = False):
    """Fetch the five v1.0 subset zips from Google Drive and unpack them so the
    layout is {DATA_ROOT}/{subset}/{id}/input.wav + annotations."""
    import glob
    import shutil
    import zipfile

    import gdown

    os.makedirs(f"{DATA_DIR}/zips", exist_ok=True)
    for name, fid in ZIPS.items():
        dest = os.path.join(DATA_ROOT, name)
        if os.path.isdir(dest) and os.listdir(dest) and not force:
            print(f">>> {name}: already present ({len(os.listdir(dest))} entries)")
            continue
        z = f"{DATA_DIR}/zips/{name}.zip"
        if not os.path.exists(z) or force:
            print(f">>> downloading {name}.zip ...", flush=True)
            gdown.download(id=fid, output=z, quiet=False)
        tmp = f"{DATA_DIR}/tmp_extract/{name}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp)
        # normalize: every dir containing input.wav becomes {dest}/{basename}
        os.makedirs(dest, exist_ok=True)
        n = 0
        for wav in glob.glob(f"{tmp}/**/input.wav", recursive=True):
            src = os.path.dirname(wav)
            sid = os.path.basename(src)
            tgt = os.path.join(dest, sid)
            if not os.path.exists(tgt):
                shutil.move(src, tgt)
            n += 1
        shutil.rmtree(tmp, ignore_errors=True)
        print(f">>> {name}: {n} samples", flush=True)
    shutil.rmtree(f"{DATA_DIR}/tmp_extract", ignore_errors=True)
    data.commit()
    for name in ZIPS:
        d = os.path.join(DATA_ROOT, name)
        print(f"{name}: {len(os.listdir(d)) if os.path.isdir(d) else 'MISSING'}")


@app.function(image=util_image, volumes={DATA_DIR: data})
def list_samples(subset: str) -> list:
    d = os.path.join(DATA_ROOT, subset)
    return sorted(s for s in os.listdir(d)
                  if os.path.exists(os.path.join(d, s, "input.wav")))


@app.function(image=fdb_image, gpu="H100",
              volumes={"/workspace/models": weights, DATA_DIR: data},
              timeout=60 * 60 * 4)
def infer_shard(pairs: list, overwrite: bool = False) -> dict:
    """pairs: list of [subset, sample_id]; one model load per container."""
    sys.path.insert(0, HARNESS_DIR)
    sys.path.insert(0, "/workspace/fdb")
    import torch

    print(">>> GPU:", torch.cuda.get_device_name(0), flush=True)
    from infer import run_shard

    res = run_shard([tuple(p) for p in pairs], DATA_ROOT, EXP_ROOT,
                    overwrite=overwrite)
    data.commit()
    return res


def _shards(subsets, workers, limit):
    pairs = []
    for s in subsets:
        ids = list_samples.remote(s)
        if limit:
            ids = ids[:limit]
        pairs += [(s, i) for i in ids]
    w = max(1, min(workers, len(pairs)))
    return [pairs[i::w] for i in range(w)]


@app.local_entrypoint()
def smoke(limit: int = 2):
    """2 samples per subset on one H100 — validates speech output + timing."""
    shards = _shards(list(ZIPS), workers=1, limit=limit)
    print(infer_shard.remote(shards[0], overwrite=True))


@app.local_entrypoint()
def run_infer(subsets: str = "", workers: int = 6, limit: int = 0,
              overwrite: bool = False):
    subs = subsets.split() if subsets.strip() else list(ZIPS)
    shards = _shards(subs, workers, limit)
    total = sum(len(s) for s in shards)
    print(f">>> {total} samples across {len(shards)} H100 workers")
    done = skipped = 0
    for r in infer_shard.map(shards, kwargs={"overwrite": overwrite}):
        done += r["done"]
        skipped += r["skipped"]
    print(f">>> ALL DONE: {done} ran, {skipped} skipped (already present)")
