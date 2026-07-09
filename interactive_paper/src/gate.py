"""Online escalation gate: turn the Phase-2 probe into a real-time trigger.

Two pure-Python, torch-free pieces (so they unit-test on CPU and stay decoupled
from the backbone):

  Probe          — the fitted linear probe from Phase 2. score(h) = sigmoid(w·h + b),
                   a single dot product over the 4096-d hidden state. This is the
                   "zero-cost online" scorer the plan promised.
  EscalationGate — EMA smoothing + k-consecutive hysteresis + cooldown over a
                   stream of per-step scores. update(score) -> bool (fire now?).

Phase-2 finding (RESULTS.md): the winning signal is the probe on `h_prompt`, the
prompt's last hidden state, available at PREFILL — so the headline gate is a
single PRE-DECODE decision (feed one score; k_consecutive=1, ema_alpha=1.0, which
degenerates EscalationGate to a plain `score >= threshold`). The streaming
EMA/hysteresis/cooldown path exists for the per-step scalar / duplex use (Phase 6).
"""
from __future__ import annotations

import math


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class Probe:
    """Fitted linear probe: score(h) = sigmoid(w·h + b) = P(small model fails).

    Weights/bias come from the calib-only LogisticRegression fit exported by
    modal_app.py::fit_gate into gate_config.json. `score` matches sklearn's
    predict_proba(...)[:, 1] for the same hidden vector.
    """

    def __init__(self, weight, bias: float):
        self.weight = [float(w) for w in weight]
        self.bias = float(bias)

    def score(self, h) -> float:
        if len(h) != len(self.weight):
            raise ValueError(
                f"hidden dim {len(h)} != probe dim {len(self.weight)}")
        z = self.bias
        w = self.weight
        for i in range(len(w)):
            z += w[i] * h[i]
        return sigmoid(z)

    @classmethod
    def from_config(cls, cfg: dict) -> "Probe":
        p = cfg["probe"]
        return cls(p["weight"], p["bias"])


class EscalationGate:
    """EMA-smoothed, hysteretic, cooldown-guarded threshold trigger.

    update(score) returns True on the step the gate fires. After firing it holds
    off for `cooldown_steps` updates (returns False) so a single hard span
    triggers at most one escalation.

    With k_consecutive=1 and ema_alpha=1.0 this is exactly `score >= threshold`
    — the pre-decode single-shot mode used for the h_prompt probe.
    """

    def __init__(self, threshold: float, k_consecutive: int = 4,
                 ema_alpha: float = 0.3, cooldown_steps: int = 64):
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if k_consecutive < 1:
            raise ValueError("k_consecutive must be >= 1")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0")
        self.threshold = float(threshold)
        self.k = int(k_consecutive)
        self.alpha = float(ema_alpha)
        self.cooldown_steps = int(cooldown_steps)
        self.reset()

    def reset(self) -> None:
        self._ema = None
        self._run = 0          # consecutive over-threshold count
        self._cooldown = 0

    @property
    def ema(self):
        return self._ema

    def update(self, score: float) -> bool:
        # EMA smoothing (first score seeds the EMA).
        if self._ema is None:
            self._ema = score
        else:
            self._ema = self.alpha * score + (1.0 - self.alpha) * self._ema

        # Cooldown after a fire: swallow updates, keep the run reset.
        if self._cooldown > 0:
            self._cooldown -= 1
            self._run = 0
            return False

        if self._ema >= self.threshold:
            self._run += 1
        else:
            self._run = 0

        if self._run >= self.k:
            self._run = 0
            self._cooldown = self.cooldown_steps
            return True
        return False

    @classmethod
    def from_config(cls, cfg: dict, tier: str = "balanced") -> "EscalationGate":
        g = cfg.get("gate", {})
        return cls(threshold=cfg["thresholds"][tier],
                   k_consecutive=g.get("k_consecutive", 1),
                   ema_alpha=g.get("ema_alpha", 1.0),
                   cooldown_steps=g.get("cooldown_steps", 64))
