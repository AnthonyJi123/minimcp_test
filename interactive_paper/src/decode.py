"""Faithful signal capture during MiniCPM-o 4.5 text generation.

Rather than re-implement MiniCPM's chat template + decode loop (fragile — the
prompt construction lives in remote code), we let `model.chat()` generate
greedily as it normally would and *passively observe* via forward hooks on the
last decoder layer and the LM head. Signals therefore come from the model's real
output, so faithfulness is automatic (the returned text IS the model's answer).

Backbone confirmed in Phase 0: model.llm = Qwen3ForCausalLM, 36 layers, hidden
4096, vocab 151748. Last layer = model.llm.model.layers[35]; head = model.llm.lm_head.

Per forward pass (prefill once, then one per decoded token) we grab the
last-position hidden state and the last-position logits:
    forward 0 (prefill): h = representation of the whole prompt   -> h_prompt
                         logits -> distribution over the 1st generated token
    forward j (decode) : h at the j-th generated token's position
                         logits -> distribution over token j+1
We keep the first K forwards' scalar signals (entropy, margin) and the first K+1
hidden states (index 0 = prompt, 1..K = per-step).
"""
from __future__ import annotations

import signals  # flat import: src/ is mounted on sys.path in the container


class _Collector:
    """Accumulates last-position logits/hidden per forward, capped at k+1."""

    def __init__(self, k: int):
        self.k = k
        self.entropy: list[float] = []
        self.margin: list[float] = []
        self.hiddens: list = []          # each a cpu float32 [4096] tensor

    def on_hidden(self, h_last):
        if len(self.hiddens) <= self.k:
            self.hiddens.append(h_last.float().cpu())

    def on_logits(self, logits_last):
        if len(self.entropy) < self.k:
            self.entropy.append(signals.entropy(logits_last))
            self.margin.append(signals.margin(logits_last))


def _register(model, coll: _Collector):
    last_layer = model.llm.model.layers[-1]
    head = model.llm.lm_head

    def layer_hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out   # decoder layer -> (hidden, ...)
        coll.on_hidden(hs[0, -1, :].detach())

    def head_hook(_m, _inp, out):
        coll.on_logits(out[0, -1, :].detach())

    return [last_layer.register_forward_hook(layer_hook),
            head.register_forward_hook(head_hook)]


def _chat_kwargs(model, tok):
    import inspect
    params = set(inspect.signature(model.chat).parameters)
    kw = {k: v for k, v in dict(
        do_sample=False, generate_audio=False, use_tts_template=False,
        enable_thinking=False).items() if k in params}
    if "tokenizer" in params:
        kw["tokenizer"] = tok
    return kw


def generate(model, tok, user_text: str, max_new_tokens: int = 512) -> str:
    """Plain greedy generation (no signal capture) — for the distill/inject steps."""
    kw = _chat_kwargs(model, tok)
    out = model.chat(msgs=[{"role": "user", "content": [user_text]}],
                     max_new_tokens=max_new_tokens, **kw)
    return out.strip() if isinstance(out, str) else out


def chat_with_signals(model, tok, query: str, k: int = 16,
                      max_new_tokens: int = 512) -> dict:
    """Greedy-generate an answer to `query` and return text + first-K signals.

    Returns:
        text      : the model's answer
        n_forward : number of forward passes observed (≈ prompt+generated)
        entropy   : list[<=k]  Shannon entropy (nats) of each step's next-token dist
        margin    : list[<=k]  top1-top2 logprob gap per step
        h_prompt  : list[4096] last-layer hidden at the final prompt position
        h_mean8   : list[4096] mean of the first up-to-8 per-step hidden states
    """
    coll = _Collector(k)
    handles = _register(model, coll)
    try:
        msgs = [{"role": "user", "content": [query]}]
        text = model.chat(msgs=msgs, max_new_tokens=max_new_tokens,
                          **_chat_kwargs(model, tok))
    finally:
        for h in handles:
            h.remove()

    h_prompt = coll.hiddens[0] if coll.hiddens else None
    step_hiddens = coll.hiddens[1:9]  # per-step hiddens, first up to 8
    if step_hiddens:
        import torch
        h_mean8 = torch.stack(step_hiddens).mean(0)
    else:
        h_mean8 = h_prompt
    return {
        "text": text.strip() if isinstance(text, str) else text,
        "n_forward": len(coll.hiddens),
        "entropy": coll.entropy,
        "margin": coll.margin,
        "h_prompt": h_prompt.tolist() if h_prompt is not None else None,
        "h_mean8": h_mean8.tolist() if h_mean8 is not None else None,
    }
