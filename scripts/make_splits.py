"""Creates the group-level splits and verifies there is no leakage:
no page group shared with the pool, no long exact-duplicate lines."""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury.cache import load_readings  # noqa: E402
from ocr_jury.data import load_lines, load_split, make_splits, page_groups, page_key  # noqa: E402
from ocr_jury.metrics import normalize  # noqa: E402

make_splits()
group_of = page_groups(load_lines())
pool = load_split("pool")
pool_groups = {group_of[page_key(l.line_id)] for l in pool}
pool_text = {normalize(l.text).lower() for l in pool}
readings = {name: load_readings(name) for name in ("tesseract", "easyocr", "doctr")}

for split_name in ("dev", "test"):
    split = load_split(split_name)
    shared = {group_of[page_key(l.line_id)] for l in split} & pool_groups
    long_dups = sum(1 for l in split if len(l.text) >= 25 and normalize(l.text).lower() in pool_text)
    cached = {name: sum(1 for l in split if l.line_id in r) for name, r in readings.items()}
    print(f"{split_name}: {len(split)} lines {dict(collections.Counter(l.source for l in split))}")
    print(f"   shared page groups with pool: {len(shared)} | long exact duplicates: {long_dups}")
    print(f"   already OCR'd: {cached}")
