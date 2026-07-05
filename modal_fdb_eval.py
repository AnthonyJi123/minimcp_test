"""Modal app: ASR alignment + metric evaluation for Full-Duplex-Bench v1.0.

Separate from modal_fdb.py so the heavy NeMo image never blocks inference runs.
Runs the benchmark's own scripts (sparse clone of DanielLin94144/Full-Duplex-Bench)
for exact metric comparability.

Usage:
    modal run modal_fdb_eval.py::run_asr             # parakeet -> output.json
    modal run modal_fdb_eval.py::run_eval            # local metrics, all tasks
    modal run modal_fdb_eval.py::run_eval --openai-key sk-...   # + GPT rating
"""
import json
import os
import subprocess

import modal

DATA_DIR = "/data"
EXP_ROOT = f"{DATA_DIR}/exp/minicpm-o-4_5"
REPO = "/fdb-repo"
LOG_DIR = f"{DATA_DIR}/eval_logs"

# subset -> (evaluate.py task name, asr crop mode)
SUBSETS = {
    "candor_pause_handling": ("pause_handling", "default"),
    "synthetic_pause_handling": ("pause_handling", "default"),
    "candor_turn_taking": ("smooth_turn_taking", "default"),
    "icc_backchannel": ("backchannel", "default"),
    "synthetic_user_interruption": ("user_interruption", "user_interruption"),
}

CLONE = (
    "git clone --depth 1 --filter=blob:none --sparse "
    f"https://github.com/DanielLin94144/Full-Duplex-Bench {REPO} && "
    f"cd {REPO} && git sparse-checkout set v1_v1.5/get_transcript v1_v1.5/evaluation"
)

app = modal.App("fdb-minicpmo45-eval")

data = modal.Volume.from_name("fdb-data", create_if_missing=True)

asr_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("nemo_toolkit[asr]==2.3.1", "cuda-python>=12.3")
    .run_commands(CLONE)
)

eval_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libsndfile1", "ffmpeg")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1",
                 index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("silero-vad", "scipy", "numpy", "tqdm", "openai",
                 "python-dotenv", "soundfile")
    .run_commands(CLONE)
)


@app.function(image=asr_image, gpu="L4", volumes={DATA_DIR: data}, timeout=60 * 60 * 2)
def asr_all(subsets: list = None):
    """Run the benchmark's parakeet ASR over each subset's output.wav files."""
    for subset in (subsets or list(SUBSETS)):
        root = os.path.join(EXP_ROOT, subset)
        if not os.path.isdir(root):
            print(f">>> {subset}: no outputs yet, skipping")
            continue
        _, mode = SUBSETS[subset]
        cmd = ["python", "asr.py", "--root_dir", root, "--task", mode]
        print(f">>> $ {' '.join(cmd)}", flush=True)
        p = subprocess.run(cmd, cwd=f"{REPO}/v1_v1.5/get_transcript")
        if p.returncode != 0:
            raise RuntimeError(f"ASR failed for {subset}")
        n = sum(1 for d in os.listdir(root)
                if os.path.exists(os.path.join(root, d, "output.json")))
        print(f">>> {subset}: {n} output.json written", flush=True)
    data.commit()


def _ui_tor_latency(root: str) -> dict:
    """user_interruption TOR + latency without the GPT judge (same thresholds
    as the benchmark's eval_user_interruption.py)."""
    turn_dur_thr, turn_words_thr = 1, 3
    tors, lats = [], []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        with open(os.path.join(p, "output.json")) as f:
            chunks = json.load(f)["chunks"]
        with open(os.path.join(p, "interrupt.json")) as f:
            input_end = json.load(f)[0]["timestamp"][1]
        if not chunks:
            tors.append(0)
            continue
        start = chunks[0]["timestamp"][0]
        dur = chunks[-1]["timestamp"][-1] - start
        if dur < turn_dur_thr and len(chunks) <= turn_words_thr:
            tors.append(0)
            continue
        tors.append(1)
        lats.append(max(0.0, start - input_end))
    return {
        "n": len(tors),
        "TOR": sum(tors) / len(tors) if tors else None,
        "latency": sum(lats) / len(lats) if lats else None,
    }


@app.function(image=eval_image, volumes={DATA_DIR: data}, timeout=60 * 60)
def eval_all(openai_key: str = "", subsets: list = None) -> dict:
    os.makedirs(LOG_DIR, exist_ok=True)
    results = {}
    env = dict(os.environ)
    if openai_key:
        env["OPENAI_API_KEY"] = openai_key
    for subset in (subsets or list(SUBSETS)):
        root = os.path.join(EXP_ROOT, subset)
        if not os.path.isdir(root):
            print(f">>> {subset}: no outputs, skipping")
            continue
        task, _ = SUBSETS[subset]
        if task == "user_interruption" and not openai_key:
            r = _ui_tor_latency(root)
            results[subset] = r
            print(f">>> {subset} (no GPT judge): {r}", flush=True)
            continue
        cmd = ["python", "evaluate.py", "--task", task, "--root_dir", root]
        print(f">>> $ {' '.join(cmd)}", flush=True)
        p = subprocess.run(cmd, cwd=f"{REPO}/v1_v1.5/evaluation", env=env,
                           capture_output=True, text=True)
        log = p.stdout + ("\n--- STDERR ---\n" + p.stderr if p.stderr else "")
        with open(os.path.join(LOG_DIR, f"{subset}.log"), "w") as f:
            f.write(log)
        tail = "\n".join(log.strip().splitlines()[-12:])
        results[subset] = tail
        print(f">>> {subset} result tail:\n{tail}", flush=True)
        if p.returncode != 0:
            print(f"!!! {subset} eval exited {p.returncode}", flush=True)
    data.commit()
    return results


@app.local_entrypoint()
def run_asr(subsets: str = ""):
    asr_all.remote(subsets.split() if subsets.strip() else None)


@app.local_entrypoint()
def run_eval(openai_key: str = "", subsets: str = ""):
    res = eval_all.remote(openai_key, subsets.split() if subsets.strip() else None)
    print("\n" + "=" * 70)
    for k, v in res.items():
        print(f"\n### {k}\n{v}")
