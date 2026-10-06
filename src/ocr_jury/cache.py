"""One reading cache per engine, keyed by line_id - independent of splits,
so re-splitting never throws away OCR work."""
from __future__ import annotations

import json
import os

from .data import ROOT

CACHE_DIR = os.path.join(ROOT, "cache")


def cache_path(engine: str) -> str:
    return os.path.join(CACHE_DIR, f"readings_{engine}.jsonl")


def load_readings(engine: str) -> dict[str, dict]:
    """line_id -> record. Tolerates a half-written last line from a killed run."""
    out = {}
    path = cache_path(engine)
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for x in f:
            try:
                rec = json.loads(x)
            except json.JSONDecodeError:
                continue
            out[rec["line_id"]] = rec
    return out
