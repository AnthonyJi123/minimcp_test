"""All-layer prefill hidden-state capture (Phase 5d: destroyed vs relocated).

The Phase-2/5c probe reads ONE point in the network: the LAST decoder layer at
the LAST prompt position. Phase 5c showed that point's cross-pool transfer
degrades with duplex-ness of the fine-tune. Two rival explanations:
  (a) destroyed  — duplex training washes difficulty info out of the model;
  (b) relocated  — duplex training repurposes the late-layer / last-token
       readout (streaming turn control) and the info survives elsewhere.
This helper captures, in a single prefill forward, EVERY decoder layer's
hidden state at (i) the last prompt position and (ii) mean-pooled over all
prompt positions — the depth x position grid that separates (a) from (b).

Hook-based (first forward per layer = prefill) so it works identically for
vanilla CausalLMs, omni Thinkers, and MiniCPM remote-code models, whose prompt
construction we must not re-implement (same faithfulness argument as decode.py).
"""
from __future__ import annotations


def capture_prefill_layers(layer_modules, run_fn):
    """Run `run_fn` once with hooks on every module in `layer_modules`; return
    (h_last, h_mean), each a cpu float32 tensor [n_layers, hidden].

    Only the FIRST forward through each layer is kept — with a generate/chat
    call that is the prefill pass over the full prompt.
    """
    import torch

    n = len(layer_modules)
    h_last: list = [None] * n
    h_mean: list = [None] * n

    def mk(i):
        def hook(_m, _inp, out):
            if h_last[i] is None:
                hs = out[0] if isinstance(out, tuple) else out   # [1, T, d]
                h_last[i] = hs[0, -1, :].detach().float().cpu()
                h_mean[i] = hs[0].detach().float().mean(dim=0).cpu()
        return hook

    handles = [m.register_forward_hook(mk(i))
               for i, m in enumerate(layer_modules)]
    try:
        with torch.no_grad():
            run_fn()
    finally:
        for h in handles:
            h.remove()
    if any(v is None for v in h_last):
        raise RuntimeError("some layers never fired — wrong module list?")
    return torch.stack(h_last), torch.stack(h_mean)
