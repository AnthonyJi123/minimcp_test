# -*- coding: utf-8 -*-
"""8ax analysis: the two direct turn-control tests (GPT exps 3 and 6).

Inputs (modal_interp.py, gate-data volume -> data/layers/):
  logitlens_{tag}.npz      per-layer log control-token mass + argmax flags,
                           control-token unembedding rows, norm scale.
  logitlens_{tag}_top.json most frequent argmax token per layer.
  cues_{tag}.npz           all-layer last-token states for the same 60
                           queries under neutral / listen / speak suffixes.

Tests (turn-control-repurposing predictions in brackets):
  T1  control-token mass by depth, duplex vs raw backbone [duplex spikes
      late where the cliff is; raw does not].
  T2  final-layer probe direction vs control-token unembeddings, in the
      norm-scaled space the head actually reads [w35 aligns with control
      rows far above the random-row null; w22 does not].
  T3  per-query: corr(final-layer control mass, probe score) [positive].
  T4  cue direction t = mean(h_speak - h_listen) by layer: relative
      displacement size and cos(t, w) [duplex final layer moves along
      w35; raw moves little/unaligned].
Writes data/interp_turncontrol.json.
"""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

D = "data"
OUT = {}


def load_layers(tag):
    ids, hl = [], []
    for s in sorted(glob.glob(f"{D}/layers/layers_{tag}.shard*.npz")):
        z = np.load(s)
        ids += [str(x) for x in z["ids"]]
        hl.append(z["h_last"])
    return np.array(ids), np.concatenate(hl)


def probe_dirs():
    """The LOPO-math probes at L22 / L35, as in scripts 14-17."""
    ids, hl = load_layers("minicpm-o45")
    lab = pd.read_parquet(f"{D}/calib_features.parquet")[
        ["id", "pool", "split", "escalate_label"]]
    lab = lab[(lab["split"] == "calib") & lab["escalate_label"].notna()]
    m = pd.DataFrame({"id": ids, "row": range(len(ids))}).merge(lab, on="id")
    y = m["escalate_label"].astype(int).to_numpy()
    pools = m["pool"].to_numpy()
    r = m["row"].to_numpy()
    tr = pools != "hard-math"
    ws = {}
    for name, li in (("w22", 22), ("w35", 35)):
        X = hl[r, li, :].astype(np.float32)
        lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        ws[name] = lr.coef_[0]
        ws[name + "_scores"] = X @ lr.coef_[0]
    return ws, m


def cosmat(W, v):
    """|cos| of each row of W with v."""
    W = W.astype(np.float64)
    v = v.astype(np.float64)
    return np.abs(W @ v) / (np.linalg.norm(W, axis=1) * np.linalg.norm(v)
                            + 1e-12)


ws, meta = probe_dirs()

# ---- T1 + T3: control-token mass ------------------------------------------
for tag in ("minicpm-o45", "qwen3-8b"):
    p = f"{D}/layers/logitlens_{tag}.npz"
    if not glob.glob(p):
        print(f"T1 {tag}: missing {p}, skipped")
        continue
    z = np.load(p, allow_pickle=True)
    mass, am = z["mass"], z["argmax_ctrl"]
    rec = {
        "mean_log_mass_by_layer": mass.mean(0).tolist(),
        "argmax_ctrl_frac_by_layer": am.mean(0).tolist(),
        "n_ctrl_tokens": int(len(z["ctrl_ids"])),
    }
    if tag == "minicpm-o45":
        order = {q: i for i, q in enumerate(z["ids"].astype(str))}
        rows = [order[q] for q in meta["id"]]
        for wname in ("w22", "w35"):
            rec[f"corr_mass35_{wname}score"] = float(np.corrcoef(
                mass[rows, -1], ws[wname + "_scores"])[0, 1])
        # T2: probe direction vs control unembeddings, norm-scaled space
        g = z["norm_g"].astype(np.float64)
        for wname in ("w22", "w35"):
            v = ws[wname] * g            # direction as the head sees it
            c_ctrl = cosmat(z["W_ctrl"].astype(np.float64), v)
            c_rand = cosmat(z["W_rand"].astype(np.float64), v)
            rec[f"{wname}_ctrl_cos_max"] = float(c_ctrl.max())
            rec[f"{wname}_ctrl_cos_mean"] = float(c_ctrl.mean())
            rec[f"{wname}_rand_cos_max"] = float(c_rand.max())
            rec[f"{wname}_rand_cos_mean"] = float(c_rand.mean())
            top = np.argsort(-c_ctrl)[:8]
            toks = z["ctrl_tokens"].tolist()
            rec[f"{wname}_top_aligned_ctrl"] = [
                (toks[i] if i < len(toks) else int(i), float(c_ctrl[i]))
                for i in top]
    OUT[f"lens_{tag}"] = rec
    print(f"T1 {tag}: log-mass L0 {mass.mean(0)[0]:.2f} -> mid "
          f"{mass.mean(0)[len(mass[0]) // 2]:.2f} -> final "
          f"{mass.mean(0)[-1]:.2f} | argmax-in-ctrl final {am.mean(0)[-1]:.2f}")
    if tag == "minicpm-o45":
        print(f"   corr(mass35, w35 score) = "
              f"{rec['corr_mass35_w35score']:+.3f} | w35 ctrl-cos max "
              f"{rec['w35_ctrl_cos_max']:.3f} vs rand max "
              f"{rec['w35_rand_cos_max']:.3f}")

# ---- T4: cue intervention -------------------------------------------------
for tag in ("minicpm-o45", "qwen3-8b"):
    p = f"{D}/layers/cues_{tag}.npz"
    if not glob.glob(p):
        print(f"T4 {tag}: missing {p}, skipped")
        continue
    z = np.load(p, allow_pickle=True)
    hn = z["h_neutral"].astype(np.float32)
    hl_ = z["h_listen"].astype(np.float32)
    hs = z["h_speak"].astype(np.float32)
    nL = hn.shape[1]
    t_rel, cos35, cos22, dscore = [], [], [], []
    for li in range(nL):
        t = (hs[:, li] - hl_[:, li]).mean(0)
        base = np.linalg.norm(hn[:, li], axis=1).mean()
        t_rel.append(float(np.linalg.norm(t) / (base + 1e-9)))
        if tag == "minicpm-o45":
            cos35.append(float(np.abs(t @ ws["w35"]) /
                               (np.linalg.norm(t) *
                                np.linalg.norm(ws["w35"]) + 1e-12)))
            cos22.append(float(np.abs(t @ ws["w22"]) /
                               (np.linalg.norm(t) *
                                np.linalg.norm(ws["w22"]) + 1e-12)))
    rec = {"t_rel_by_layer": t_rel}
    if tag == "minicpm-o45":
        rec["cos_w35_by_layer"] = cos35
        rec["cos_w22_by_layer"] = cos22
        # per-query probe-score shift at the final layer, in units of the
        # neutral scores' std: does "answer now" move the model toward
        # keep-local the way the inverting probe would predict?
        s_n = hn[:, 35] @ ws["w35"]
        s_l = hl_[:, 35] @ ws["w35"]
        s_s = hs[:, 35] @ ws["w35"]
        sd = s_n.std() + 1e-9
        rec["w35_shift_speak_minus_listen_sd"] = float((s_s - s_l).mean() / sd)
        m_n = hn[:, 22] @ ws["w22"]
        m_l = hl_[:, 22] @ ws["w22"]
        m_s = hs[:, 22] @ ws["w22"]
        sd2 = m_n.std() + 1e-9
        rec["w22_shift_speak_minus_listen_sd"] = float(
            (m_s - m_l).mean() / sd2)
    OUT[f"cues_{tag}"] = rec
    print(f"T4 {tag}: |t|/|h| mid {t_rel[nL // 2]:.3f} -> final "
          f"{t_rel[-1]:.3f}"
          + (f" | cos(t,w35) final {cos35[-1]:.3f}, cos(t,w22) at L22 "
             f"{cos22[22]:.3f} | w35 score shift "
             f"{rec['w35_shift_speak_minus_listen_sd']:+.2f} sd, w22 "
             f"{rec['w22_shift_speak_minus_listen_sd']:+.2f} sd"
             if tag == "minicpm-o45" else ""))

with open(f"{D}/interp_turncontrol.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(">>> wrote data/interp_turncontrol.json")
