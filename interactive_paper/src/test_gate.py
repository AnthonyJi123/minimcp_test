"""Local unit tests for the online gate — pure Python, no torch/modal/sklearn.

Run:  python interactive_paper/src/test_gate.py
Prints "OK (N checks)" on success; raises AssertionError on the first failure.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import EscalationGate, Probe, sigmoid  # noqa: E402

N = 0


def check(cond, msg):
    global N
    N += 1
    assert cond, msg


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --- sigmoid -----------------------------------------------------------------
check(approx(sigmoid(0.0), 0.5), "sigmoid(0)=0.5")
check(sigmoid(50) > 0.999 and sigmoid(-50) < 0.001, "sigmoid saturates")
check(sigmoid(1e4) <= 1.0 and sigmoid(-1e4) >= 0.0, "sigmoid stays in [0,1] (no overflow)")
check(sigmoid(-1.0) < sigmoid(0.0) < sigmoid(1.0), "sigmoid monotone")

# --- Probe -------------------------------------------------------------------
p = Probe(weight=[1.0, 0.0, -2.0], bias=0.5)
# z = 0.5 + 1*1 + 0*9 + (-2)*0.25 = 0.5 + 1 - 0.5 = 1.0
check(approx(p.score([1.0, 9.0, 0.25]), sigmoid(1.0)), "probe dot product")
try:
    p.score([1.0, 2.0])
    check(False, "probe dim mismatch should raise")
except ValueError:
    check(True, "probe dim mismatch raises")

cfg = {
    "probe": {"weight": [0.5, 0.5], "bias": 0.0},
    "thresholds": {"conservative": 0.8, "balanced": 0.5, "aggressive": 0.3},
    "gate": {"k_consecutive": 1, "ema_alpha": 1.0, "cooldown_steps": 4},
}
check(approx(Probe.from_config(cfg).score([2.0, 2.0]), sigmoid(2.0)),
      "Probe.from_config round-trip")

# --- EscalationGate: pre-decode single-shot (k=1, alpha=1) -------------------
g = EscalationGate(threshold=0.5, k_consecutive=1, ema_alpha=1.0, cooldown_steps=0)
check(g.update(0.6) is True, "single-shot fires above threshold")
check(g.update(0.4) is False, "single-shot silent below threshold")
check(g.update(0.5) is True, "single-shot fires exactly at threshold (>=)")

# from_config default tier (balanced=0.5) behaves as single-shot
gc = EscalationGate.from_config(cfg)
check(gc.k == 1 and gc.alpha == 1.0 and gc.threshold == 0.5, "from_config balanced")

# --- hysteresis: k consecutive required --------------------------------------
g = EscalationGate(threshold=0.5, k_consecutive=3, ema_alpha=1.0, cooldown_steps=0)
check(g.update(0.9) is False, "k=3: 1st over -> no fire")
check(g.update(0.9) is False, "k=3: 2nd over -> no fire")
check(g.update(0.9) is True, "k=3: 3rd consecutive over -> fire")
# a dip resets the run
g.reset()
check(g.update(0.9) is False and g.update(0.2) is False and g.update(0.9) is False,
      "k=3: a below-threshold dip resets the consecutive run")

# --- EMA damps a lone spike --------------------------------------------------
g = EscalationGate(threshold=0.6, k_consecutive=1, ema_alpha=0.5, cooldown_steps=0)
check(g.update(0.1) is False, "ema: low baseline no fire")
# ema = 0.5*0.9 + 0.5*0.1 = 0.5 < 0.6 -> a single spike does not fire
check(g.update(0.9) is False, "ema: lone spike damped below threshold")
check(approx(g.ema, 0.5), "ema value tracked")
# sustained high pulls ema over: 0.5*0.9+0.5*0.5 = 0.7 >= 0.6
check(g.update(0.9) is True, "ema: sustained high crosses threshold")

# --- cooldown: one hard span fires at most once ------------------------------
g = EscalationGate(threshold=0.5, k_consecutive=1, ema_alpha=1.0, cooldown_steps=2)
check(g.update(0.9) is True, "cooldown: first fire")
check(g.update(0.9) is False, "cooldown: swallowed (1)")
check(g.update(0.9) is False, "cooldown: swallowed (2)")
check(g.update(0.9) is True, "cooldown: re-arms after cooldown_steps")

# --- validation guards -------------------------------------------------------
for bad in (dict(threshold=0.5, ema_alpha=0.0),
            dict(threshold=0.5, ema_alpha=1.5),
            dict(threshold=0.5, k_consecutive=0),
            dict(threshold=0.5, cooldown_steps=-1)):
    try:
        EscalationGate(**bad)
        check(False, f"bad param {bad} should raise")
    except ValueError:
        check(True, f"bad param {bad} raises")

# --- reset clears state ------------------------------------------------------
g = EscalationGate(threshold=0.5, k_consecutive=2, ema_alpha=0.5, cooldown_steps=3)
g.update(0.9); g.update(0.9)  # fires, enters cooldown, ema warm
g.reset()
check(g.ema is None and g._run == 0 and g._cooldown == 0, "reset clears all state")

print(f"OK ({N} checks)")
