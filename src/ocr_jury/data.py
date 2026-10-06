"""Loads the English test set of the OCR 18th Century dataset.

Images are read straight out of OCR_Testing.zip, so the ~12k line images
never need to be unpacked onto disk. Schema (inspected, not assumed):
full_english_dataset_relative.csv is tab-separated with columns
file_name (Windows-style path inside the zip) and text (manual transcription).
"""
from __future__ import annotations

import csv
import io
import os
import random
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ZIP_PATH = os.path.join(ROOT, "data", "raw", "OCR_Testing.zip")
SPLIT_DIR = os.path.join(ROOT, "data", "splits")
GT_CSV = "full_english_dataset_relative.csv"


@dataclass(frozen=True)
class Line:
    line_id: str   # zip-internal path, forward slashes
    text: str      # ground-truth transcription, exactly as provided
    source: str    # "Hume" or "Generic"


@lru_cache(maxsize=1)
def _zip() -> zipfile.ZipFile:
    return zipfile.ZipFile(ZIP_PATH)


def load_lines() -> list[Line]:
    raw = _zip().read(GT_CSV).decode("utf-8-sig")
    lines = []
    for row in csv.DictReader(io.StringIO(raw), delimiter="\t"):
        path = row["file_name"].replace("\\", "/")
        lines.append(Line(line_id=path, text=row["text"], source=path.split("/")[0]))
    return lines


def open_image(line: Line) -> Image.Image:
    with _zip().open(line.line_id) as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


def page_key(line_id: str) -> str:
    """File name minus the trailing _<region>_<n>_<m>.png  ->  the page."""
    return re.sub(r"_[^_]+_\d+_\d+\.png$", "", os.path.basename(line_id))


def page_groups(lines: list[Line], min_chars: int = 25) -> dict[str, str]:
    """Links pages that share an identical long line into one group.

    The dataset contains the same Hume pages from two editions (1754 and
    1762), identical line for line. A line-level random split put a dev
    line's twin from the other edition into the training pool - leakage.
    Union-find over pages connected by shared text keeps twins together."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pages_for_text: dict[str, set[str]] = {}
    for ln in lines:
        p = page_key(ln.line_id)
        find(p)
        key = re.sub(r"\s+", " ", ln.text.replace("ſ", "s")).strip().lower()
        if len(key) >= min_chars:
            pages_for_text.setdefault(key, set()).add(p)
    # Edition twins share text on exactly two pages. Running headers
    # ("THE HISTORY OF GREAT BRITAIN.") recur on many pages; linking through
    # them chained 35 unrelated pages into one 1,455-line group, so text
    # found on more than two pages is not used as a link.
    for pages in pages_for_text.values():
        if len(pages) == 2:
            a, b = sorted(pages)
            parent[find(a)] = find(b)
    return {page_key(ln.line_id): find(page_key(ln.line_id)) for ln in lines}


def make_splits(dev_lines: int = 600, test_lines: int = 1000, seed: int = 42) -> None:
    """Group-level split: every page group lands wholly in dev, test, or the
    training pool, so no edition twin straddles the boundary. Groups are
    shuffled with a fixed seed and assigned until each split reaches its
    target line count."""
    lines = load_lines()
    group_of = page_groups(lines)
    members: dict[str, list[Line]] = {}
    for ln in lines:
        members.setdefault(group_of[page_key(ln.line_id)], []).append(ln)

    # Stratify by source: Hume's edition-twin groups are large, so plain
    # random group assignment can skew a split toward one book. Each source
    # contributes to dev/test in proportion to its share of all lines.
    rng = random.Random(seed)
    by_source: dict[str, list[str]] = {}
    for g, ms in members.items():
        by_source.setdefault(ms[0].source, []).append(g)
    dev, test = [], []
    for src, gs in sorted(by_source.items()):
        gs = sorted(gs)
        rng.shuffle(gs)
        share = sum(len(members[g]) for g in gs) / len(lines)
        d_target, t_target = dev_lines * share, test_lines * share
        d, t = [], []
        for g in gs:
            if len(d) < d_target:
                d += members[g]
            elif len(t) < t_target:
                t += members[g]
        dev += d
        test += t

    os.makedirs(SPLIT_DIR, exist_ok=True)
    for name, split in (("dev", dev), ("test", test)):
        with open(os.path.join(SPLIT_DIR, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(l.line_id for l in split)) + "\n")
    print(f"page groups: {len(members)} | dev={len(dev)} test={len(test)} pool={len(lines) - len(dev) - len(test)} "
          f"(seed={seed}) -> {SPLIT_DIR}")


def load_split(name: str) -> list[Line]:
    """dev / test from their ID files; "pool" = every English line in neither.
    The pool is the only data allowed for training (lexicon, TrOCR
    fine-tuning, confidence model, vector-DB memory) - dev tunes, test scores."""
    if name == "pool":
        held_out = set()
        for s in ("dev", "test"):
            with open(os.path.join(SPLIT_DIR, f"{s}.txt"), encoding="utf-8") as f:
                held_out |= {x.strip() for x in f if x.strip()}
        return [l for l in load_lines() if l.line_id not in held_out]
    with open(os.path.join(SPLIT_DIR, f"{name}.txt"), encoding="utf-8") as f:
        ids = {x.strip() for x in f if x.strip()}
    return [l for l in load_lines() if l.line_id in ids]
