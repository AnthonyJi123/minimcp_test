"""Core duplex streaming session: wraps MiniCPM-o 4.5 UnifiedProcessor/DuplexView
with a per-chunk recorder.

The model runs a ~1 Hz loop. Each step feeds one 1-second chunk of
(audio_waveform, video frame) via prefill, calls generate, then finalize, and
records a structured row. `is_listen` is the model's per-second speak/stay-silent
decision — the key observable the probes measure against.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

# MiniCPM-o-Demo provides the duplex API (UnifiedProcessor -> DuplexView).
DEMO_PATH = "/workspace/MiniCPM-o-Demo"
if DEMO_PATH not in sys.path:
    sys.path.insert(0, DEMO_PATH)

from core.processors.unified import UnifiedProcessor  # noqa: E402
from core.schemas.duplex import DuplexConfig  # noqa: E402

SR = 16000
CHUNK_SAMPLES = SR  # 1 second / chunk (matches model 1 Hz decision frequency)

DEFAULT_MODEL = "/workspace/models/MiniCPM-o-4_5"
DEFAULT_REF = "/workspace/MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav"


@dataclass
class ChunkRecord:
    chunk_idx: int
    stream_time: float          # seconds of stimulus consumed (chunk_idx + 1)
    wall_time: float            # wall-clock seconds since loop start
    is_listen: bool
    speaking: bool              # convenience: not is_listen
    text: str
    has_audio: bool
    end_of_turn: bool
    gen_latency_ms: float       # wall time of prefill+generate+finalize
    cost_all_ms: Optional[float]
    kv_cache_length: Optional[int]
    vram_mb: Optional[float] = None   # torch.cuda.memory_allocated at this chunk
    label: str = ""             # stimulus label for this chunk (probe annotation)
    event: str = ""             # scheduled-event marker fired at this chunk


class DuplexSession:
    """Load the model once; drive a duplex scenario; record every chunk."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        ref_audio_path: str = DEFAULT_REF,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
        compile: bool = False,
        duplex_config: Optional[DuplexConfig] = None,
    ):
        self.model_path = model_path
        self.ref_audio_path = ref_audio_path
        self.duplex_config = duplex_config or DuplexConfig()
        t0 = time.time()
        self.proc = UnifiedProcessor(
            model_path=model_path,
            ref_audio_path=ref_audio_path,
            device=device,
            attn_implementation=attn_implementation,
            compile=compile,
            duplex_config=self.duplex_config,
        )
        self.load_s = time.time() - t0
        self.duplex = None

    # -- session lifecycle -------------------------------------------------
    def prepare(self, system_prompt: str) -> None:
        self.duplex = self.proc.set_duplex_mode()
        self.duplex.prepare(system_prompt_text=system_prompt,
                            ref_audio_path=self.ref_audio_path)

    def set_break(self) -> None:
        self.duplex.set_break()

    def stop(self) -> None:
        if self.duplex is not None:
            try:
                self.duplex.stop()
            except Exception:
                pass

    def cleanup(self) -> None:
        try:
            self.duplex.cleanup()
        except Exception:
            pass

    # -- the per-chunk step ------------------------------------------------
    def step(self, idx: int, audio: np.ndarray, frame, *,
             loop_t0: float, label: str = "", event: str = "") -> ChunkRecord:
        t = time.time()
        frame_list = [frame] if frame is not None else None
        if torch is not None:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        try:
            self.duplex.prefill(audio_waveform=audio, frame_list=frame_list)
            r = self.duplex.generate()
        finally:
            # finalize MUST run before the next prefill even if generate hiccups
            self.duplex.finalize()
        dt = (time.time() - t) * 1000.0
        kv = getattr(self.proc, "kv_cache_length", None)
        vram = None
        if torch is not None:
            try:
                vram = torch.cuda.max_memory_allocated() / 1e6
            except Exception:
                vram = None
        return ChunkRecord(
            chunk_idx=idx,
            stream_time=float(idx + 1),
            wall_time=time.time() - loop_t0,
            is_listen=bool(r.is_listen),
            speaking=not bool(r.is_listen),
            text=r.text or "",
            has_audio=r.audio_data is not None,
            end_of_turn=bool(r.end_of_turn),
            gen_latency_ms=dt,
            cost_all_ms=getattr(r, "cost_all_ms", None),
            kv_cache_length=int(kv) if kv is not None else None,
            vram_mb=round(vram, 1) if vram is not None else None,
            label=label,
            event=event,
        )

    def run(self, scenario, trace_path: Optional[str] = None,
            verbose: bool = True) -> List[ChunkRecord]:
        """Run a Scenario (see timeline.Scenario) and return all chunk records."""
        records: List[ChunkRecord] = []
        loop_t0 = time.time()
        fh = open(trace_path, "w", encoding="utf-8") if trace_path else None
        try:
            for idx, (audio, frame, label, event) in enumerate(scenario.chunks()):
                rec = self.step(idx, audio, frame, loop_t0=loop_t0,
                                label=label, event=event)
                records.append(rec)
                if fh:
                    fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    fh.flush()
                if verbose:
                    ev = f" <{event}>" if event else ""
                    print(f"[{idx:03d}] {'SPK' if rec.speaking else 'lst'}"
                          f" eot={int(rec.end_of_turn)} kv={rec.kv_cache_length}"
                          f" {rec.gen_latency_ms:.0f}ms{ev} {rec.label} :: {rec.text!r}",
                          flush=True)
                if scenario.stop_on_eot and rec.end_of_turn and scenario.allow_stop(idx):
                    break
        finally:
            if fh:
                fh.close()
        return records
