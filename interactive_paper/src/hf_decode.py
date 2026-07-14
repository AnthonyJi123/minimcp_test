"""Signal capture for plain HuggingFace CausalLM backbones (cross-model check).

Mirrors decode.chat_with_signals but for a vanilla AutoModelForCausalLM (no
MiniCPM remote code): apply_chat_template + greedy generate, with forward hooks
on the last decoder layer and lm_head. Used to test whether the Phase-2/audit
findings (probe AUC, type-shortcut, LOPO failure) replicate on other backbones.
"""
from __future__ import annotations

import signals  # flat import: src/ is mounted on sys.path in the container


class _Collector:
    """Same as decode._Collector: last-position logits/hidden per forward."""

    def __init__(self, k: int):
        self.k = k
        self.entropy: list[float] = []
        self.margin: list[float] = []
        self.hiddens: list = []

    def on_hidden(self, h_last):
        if len(self.hiddens) <= self.k:
            self.hiddens.append(h_last.float().cpu())

    def on_logits(self, logits_last):
        if len(self.entropy) < self.k:
            self.entropy.append(signals.entropy(logits_last))
            self.margin.append(signals.margin(logits_last))


def _register(model, coll: _Collector):
    # vanilla CausalLM layout: model.model.layers[-1], model.lm_head
    last_layer = model.model.layers[-1]
    head = model.lm_head

    def layer_hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        coll.on_hidden(hs[0, -1, :].detach())

    def head_hook(_m, _inp, out):
        coll.on_logits(out[0, -1, :].detach())

    return [last_layer.register_forward_hook(layer_hook),
            head.register_forward_hook(head_hook)]


def _build_inputs(model, tok, query: str):
    kw = {"tokenize": False, "add_generation_prompt": True}
    try:  # Qwen3: disable thinking so signals mirror the MiniCPM setup
        text = tok.apply_chat_template(
            [{"role": "user", "content": query}], enable_thinking=False, **kw)
    except TypeError:  # templates without the enable_thinking kwarg (Mistral etc.)
        text = tok.apply_chat_template(
            [{"role": "user", "content": query}], **kw)
    return tok(text, return_tensors="pt").to(model.device)


def first_token_logits(model, tok, prompt: str):
    """Logits over the first generated token (prefill only) — p(True) baseline."""
    import torch
    inputs = _build_inputs(model, tok, prompt)
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].detach().float().cpu()


def chat_with_signals(model, tok, query: str, k: int = 16,
                      max_new_tokens: int = 512) -> dict:
    """Greedy-generate an answer and return text + first-K signals (same schema
    as decode.chat_with_signals)."""
    import torch

    coll = _Collector(k)
    handles = _register(model, coll)
    inputs = _build_inputs(model, tok, query)
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    finally:
        for h in handles:
            h.remove()
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)

    h_prompt = coll.hiddens[0] if coll.hiddens else None
    step_hiddens = coll.hiddens[1:9]
    if step_hiddens:
        import torch as _t
        h_mean8 = _t.stack(step_hiddens).mean(0)
    else:
        h_mean8 = h_prompt
    return {
        "text": text.strip(),
        "n_forward": len(coll.hiddens),
        "entropy": coll.entropy,
        "margin": coll.margin,
        "h_prompt": h_prompt.tolist() if h_prompt is not None else None,
        "h_mean8": h_mean8.tolist() if h_mean8 is not None else None,
    }
