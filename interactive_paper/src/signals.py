"""Zero-training escalation signals from a decode step's logits / hidden state.

All computation is done in float32 (bf16 softmax/logsumexp is numerically noisy).
These are pure functions over a single step's LLM outputs — no model coupling —
so they unit-test on CPU and stay decoupled from the specific backbone.
"""
from __future__ import annotations


def entropy(logits) -> float:
    """Shannon entropy (nats) of the next-token distribution.

    logits: 1-D tensor over the vocab for one decode step.
    High entropy = the model is unsure which token comes next.
    """
    import torch

    lp = torch.log_softmax(logits.float(), dim=-1)
    return float(-(lp.exp() * lp).sum())


def margin(logits) -> float:
    """Top1 - top2 gap in *logprob* space for one decode step.

    Small margin = the model nearly tied its top two candidates (low confidence).
    """
    import torch

    lp = torch.log_softmax(logits.float(), dim=-1)
    top2 = torch.topk(lp, 2).values
    return float(top2[0] - top2[1])
