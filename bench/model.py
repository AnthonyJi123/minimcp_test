"""Raw MiniCPM-o 4.5 chat wrapper for the offline benchmarks.

Unlike the duplex harness (harness/session.py, which drives the 1 Hz
listen/speak loop via UnifiedProcessor), these benchmarks are ordinary offline
QA/ASR: one audio (or audio+video) in, one text answer out. So we load the raw
`AutoModel` and call `model.chat(..., generate_audio=False)` directly — no TTS,
no duplex, no ref voice.

The chat signature has drifted across MiniCPM-o releases (some builds want an
explicit `tokenizer=`, the 4.5 model-card example omits it), so we introspect
`model.chat`'s parameters once and only pass kwargs it actually accepts.
"""
from __future__ import annotations

import inspect
import time

MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"


class BenchModel:
    def __init__(self, model_path: str = MODEL_DIR,
                 attn_implementation: str = "sdpa"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        t0 = time.time()
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            init_vision=True,    # video frames (Daily-Omni)
            init_audio=True,     # audio encoder (all three benchmarks)
            init_tts=False,      # text-out only: skip the TTS head's VRAM/init
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self._chat_params = set(inspect.signature(self.model.chat).parameters)
        self.load_s = time.time() - t0

    def _chat(self, msgs, **want):
        """Call model.chat with only the kwargs this build accepts."""
        kwargs = {k: v for k, v in want.items() if k in self._chat_params}
        if "tokenizer" in self._chat_params:
            kwargs["tokenizer"] = self.tokenizer
        return self.model.chat(msgs=msgs, **kwargs).strip()

    # -- audio-only ---------------------------------------------------------
    def asr(self, audio, max_new_tokens: int = 448) -> str:
        """Transcribe a 16 kHz mono float array to text (greedy)."""
        prompt = ("Please listen to the audio and transcribe the speech into "
                  "text, verbatim. Output only the transcription.")
        msgs = [{"role": "user", "content": [prompt, audio]}]
        return self._chat(msgs, do_sample=False, max_new_tokens=max_new_tokens,
                          use_tts_template=False, generate_audio=False)

    def answer_audio(self, audio, prompt: str, max_new_tokens: int = 512) -> str:
        """Answer a text prompt about / spoken in an audio clip (greedy)."""
        msgs = [{"role": "user", "content": [prompt, audio]}]
        return self._chat(msgs, do_sample=False, max_new_tokens=max_new_tokens,
                          use_tts_template=False, generate_audio=False,
                          enable_thinking=False)

    # -- audio + video ------------------------------------------------------
    def answer_video(self, video_path: str, prompt: str,
                     max_new_tokens: int = 512, stack_frames: int = 1) -> str:
        """Omni QA over a video's interleaved frames + audio (greedy)."""
        from minicpmo.utils import get_video_frame_audio_segments

        frames, audio_segs, _ = get_video_frame_audio_segments(
            video_path, stack_frames=stack_frames)
        content = []
        for f, a in zip(frames, audio_segs):
            content.append(f)
            content.append(a)
        content.append(prompt)
        msgs = [{"role": "user", "content": content}]
        return self._chat(msgs, do_sample=False, max_new_tokens=max_new_tokens,
                          omni_mode=True, max_slice_nums=1, use_image_id=False,
                          use_tts_template=False, generate_audio=False,
                          enable_thinking=False)
