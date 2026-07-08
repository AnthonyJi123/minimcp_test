"""LibriSpeech WER: parse the openslr test set, normalize, score with jiwer.

We use the openslr tarball (test-clean.tar.gz ~346 MB, test-other ~328 MB)
rather than the HF `librispeech_asr` dataset, whose default config also pulls
the multi-GB train splits. Layout inside the tar:

    LibriSpeech/test-clean/{spk}/{chap}/{spk}-{chap}-{utt}.flac
    LibriSpeech/test-clean/{spk}/{chap}/{spk}-{chap}.trans.txt   # "UTTID  TEXT"

Reference transcripts are already upper-cased and punctuation-free; we still run
a light normalizer over both sides so model casing/punctuation doesn't inflate
WER. This is a basic normalizer, not Whisper's EnglishTextNormalizer — good
enough for a pilot; note the difference when comparing to published numbers.
"""
from __future__ import annotations

import glob
import os
import re

OPENSLR = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}

_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def iter_samples(root: str, split: str = "test-clean"):
    """Yield (utt_id, flac_path, ref_text) for every utterance in the split."""
    base = os.path.join(root, "LibriSpeech", split)
    for trans in glob.glob(f"{base}/**/*.trans.txt", recursive=True):
        d = os.path.dirname(trans)
        with open(trans, encoding="utf-8") as fh:
            for line in fh:
                uid, _, text = line.strip().partition(" ")
                if uid:
                    yield uid, os.path.join(d, uid + ".flac"), text


def wer(refs: dict, hyps: dict) -> dict:
    """refs/hyps: {utt_id -> text}. Returns micro-averaged WER over shared ids."""
    import jiwer

    ids = sorted(set(refs) & set(hyps))
    r = [normalize(refs[i]) for i in ids]
    h = [normalize(hyps[i]) for i in ids]
    m = jiwer.process_words(r, h)
    return {
        "n": len(ids),
        "wer": m.wer,
        "substitutions": m.substitutions,
        "deletions": m.deletions,
        "insertions": m.insertions,
        "hits": m.hits,
    }
