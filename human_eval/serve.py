#!/usr/bin/env python3
"""Run the participant UI, API, persistence, and model gateway together."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the MiniCPM human evaluation.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args()
    uvicorn.run("human_eval.backend.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
