"""Per-pool label-free thresholds for the concurrent sweep: quantiles of
the in-regime probe's own never-arm score distribution (the never arm IS
the scoring pass), at the deployed nominal rates 15/30/50%.

Usage (from interactive_paper/): .venv_boot\\Scripts\\python.exe
scripts\\13_conc_thresholds.py [pool ...]
Prints the three --thr-override values per pool for the gated launches.
"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path("data")
POOLS = sys.argv[1:] or ["striviaqa", "swebq", "sllama", "sdqa", "sreason"]

out = {}
for pool in POOLS:
    rows = []
    for p in sorted(D.glob(f"{pool}_conclive_traces.jsonl.never.shard*")):
        rows += [json.loads(l) for l in open(p, encoding="utf-8")
                 if l.strip()]
    seen = {}
    for r in rows:
        seen[r["id"]] = r["eot_score"]
    s = np.array(list(seen.values()))
    if not len(s):
        print(f"{pool}: NO never traces yet")
        continue
    thr = {t: float(np.quantile(s, 1 - r))
           for t, r in [("conservative", .15), ("balanced", .30),
                        ("aggressive", .50)]}
    out[pool] = thr
    print(f"{pool}: n={len(s)} mean={s.mean():.3f}  "
          + "  ".join(f"{t}={v:.4f}" for t, v in thr.items()))

(D / "conc_thresholds.json").write_text(json.dumps(out, indent=2))
print("wrote data/conc_thresholds.json")
