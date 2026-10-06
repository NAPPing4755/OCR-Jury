"""Where do dev/pool near-duplicate lines come from? Compares file-name
structure of matched pairs to tell repeated text (headers) from repeated
scans/pages of the same book."""
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury.data import load_split  # noqa: E402
from ocr_jury.metrics import normalize  # noqa: E402


def page_key(line_id):
    """File name minus the trailing _<region>_<n>_<m>.png -> identifies the page."""
    name = os.path.basename(line_id)
    return re.sub(r"_[^_]+_\d+_\d+\.png$", "", name)


pool = load_split("pool")
by_text = {}
for l in pool:
    by_text.setdefault(normalize(l.text).lower(), []).append(l)

dev = load_split("dev")
pairs = [(d, by_text[normalize(d.text).lower()]) for d in dev if normalize(d.text).lower() in by_text]
long_pairs = [(d, m) for d, m in pairs if len(d.text) >= 30]
print(f"dev lines with an EXACT text match in pool: {len(pairs)}  (of which >=30 chars: {len(long_pairs)})")

same_page = sum(1 for d, ms in long_pairs if any(page_key(m.line_id) == page_key(d.line_id) for m in ms))
print(f"  long exact matches where a match is on the SAME page file: {same_page}")

print("\nexamples (dev id | pool id | text):")
for d, ms in long_pairs[:8]:
    print(f"  {os.path.basename(d.line_id)}\n  {os.path.basename(ms[0].line_id)}\n    '{d.text[:70]}'\n")

pages = Counter(page_key(l.line_id) for l in load_split("dev") + pool)
print("distinct page files:", len(pages))
