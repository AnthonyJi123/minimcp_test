"""Issue #8: export the official-native gate training/eval bundle with a
checksummed manifest. Copies the exact inputs scripts/31 reads (plus the
raw traces and query files) into a staging dir, writes MANIFEST.json +
SHA256SUMS, and zips it.

Usage: .venv_boot\Scripts\python.exe scripts\32_export_official_bundle.py OUT_DIR
"""
import glob
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data")
TRAIN = ["caliboff", "expoff", "exp2off", "exp3off", "exp3zhoff", "freshoff"]
EVAL = ["testoff", "striviaqaoff", "swebqoff", "sllamaoff", "sdqaoff",
        "sreasonoff"]
LABELS = ["calib_features.parquet", "expansion_labels.parquet",
          "expansion2_labels.parquet", "expansion3_labels.parquet",
          "expansion3zh_labels.parquet", "fresh_labels.parquet",
          "frozen_v3_traces.parquet"]
QUERIES = ["queries.jsonl", "queries_expansion.jsonl",
           "queries_expansion2.jsonl", "queries_expansion3.jsonl",
           "queries_expansion3zh.jsonl", "queries_fresh.jsonl",
           "queries_striviaqa.jsonl", "queries_swebq.jsonl",
           "queries_sllama.jsonl", "queries_sdqa.jsonl",
           "queries_sreason.jsonl"]
ARTIFACT = ["gate_native.json"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(out):
    out = Path(out)
    stage = out / "official_native_bundle" / "data"
    if stage.parent.exists():
        shutil.rmtree(stage.parent)
    stage.mkdir(parents=True)
    files, tags = [], {}
    for tag in TRAIN + EVAL:
        feats = sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz"))
        traces = sorted(D.glob(f"frozen_native_{tag}_traces.jsonl.shard*"))
        judged = D / f"frozen_native_{tag}_judged.parquet"
        assert feats and traces and judged.exists(), tag
        ids, dim = [], None
        for p in feats:
            z = np.load(p, allow_pickle=True)
            ids += list(z["ids"])
            dim = int(z["X"].shape[1])
        n_tr = sum(sum(1 for _ in open(p, encoding="utf-8")) for p in traces)
        jd = pd.read_parquet(judged)
        tags[tag] = {
            "split": "train" if tag in TRAIN else "eval",
            "feats_shards": [p.name for p in feats],
            "traces_shards": [p.name for p in traces],
            "judged": judged.name,
            "n_feature_rows": len(ids),
            "n_unique_ids": len(set(ids)),
            "dim": dim,
            "n_trace_rows": n_tr,
            "n_judged_rows": int(len(jd)),
            "n_judged_labelled": int(jd["escalate_label"].notna().sum()),
        }
        files += feats + traces + [judged]
    files += [D / f for f in LABELS + QUERIES + ARTIFACT]
    entries = []
    for p in files:
        assert p.exists(), p
        shutil.copy2(p, stage / p.name)
        entries.append({"path": f"data/{p.name}", "bytes": p.stat().st_size,
                        "sha256": sha256(p)})
    gate = json.load(open(D / "gate_native.json"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "issue": "https://github.com/dyyfk/minimcp_test/issues/8",
        "source_commit": commit,
        "serving_config": {
            "id": "official_cfg (modal_native_dump.py official_cfg=True)",
            "top_k": 20, "duplex.force_listen_count": 3,
            "system_prompt": "You are a friendly assistant.",
            "regime": "native duplex, 1s chunk streaming prefill",
            "feature": "layer 22, [eot_last, eot_mean8, user_mean] -> 12288-d "
                       "float32, read at the model's own onset chunk",
        },
        "deployed_artifact": {
            "file": "data/gate_native.json",
            "train_n": gate["train_n"], "C": gate["C"],
            "label_source": gate["label_source"],
            "recipe": gate["recipe"], "manifest": gate["manifest"],
            "oof_auc": gate["oof_auc"],
        },
        "reconstruct": "scripts/31_official_refit_labels.py --source native "
                       "--dry  (expects train 5228 = core 4986 + fresh 242)",
        "tags": tags,
        "files": entries,
    }
    (stage.parent / "MANIFEST.json").write_bytes(
        json.dumps(manifest, indent=1).encode("utf-8"))
    (stage.parent / "SHA256SUMS").write_bytes(
        "".join(f"{e['sha256']}  {e['path']}\n" for e in entries)
        .encode("utf-8"))
    zp = shutil.make_archive(str(out / "official_native_bundle"), "zip",
                             root_dir=out, base_dir="official_native_bundle")
    print("zip", zp, Path(zp).stat().st_size, "bytes; sha256", sha256(zp))
    for t, v in tags.items():
        print(f"{t:14s} {v['split']:5s} feats={v['n_feature_rows']} "
              f"uniq={v['n_unique_ids']} dim={v['dim']} traces={v['n_trace_rows']} "
              f"judged={v['n_judged_rows']}")
    print("files", len(entries), "total bytes", sum(e["bytes"] for e in entries))


if __name__ == "__main__":
    main(sys.argv[1])
