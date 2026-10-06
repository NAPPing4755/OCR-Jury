"""Sanity check one engine's cached readings: CER on lines from the ORIGINAL
run vs lines added later, plus the worst examples - to catch a broken run."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury.cache import load_readings  # noqa: E402
from ocr_jury.data import ROOT, load_lines  # noqa: E402
from ocr_jury.metrics import cer, line_cer  # noqa: E402

engine = sys.argv[1]
gt = {l.line_id: l.text for l in load_lines()}
recs = list(load_readings(engine).values())
with open(os.path.join(ROOT, "data", "splits", "v0_line_level", "dev.txt"), encoding="utf-8") as f:
    first_run = {x.strip() for x in f if x.strip()}

for name, subset in (("original run", [r for r in recs if r["line_id"] in first_run]),
                     ("added later", [r for r in recs if r["line_id"] not in first_run])):
    if subset:
        print(f"{name}: {len(subset)} lines, CER={cer([gt[r['line_id']] for r in subset], [r['text'] for r in subset]):.4f}, "
              f"empty readings={sum(1 for r in subset if not r['text'].strip())}, errors={sum(1 for r in subset if r['error'])}")

later = [r for r in recs if r["line_id"] not in first_run]
later.sort(key=lambda r: -line_cer(gt[r["line_id"]], r["text"]))
print("\nworst later readings:")
for r in later[:6]:
    print(f"  GT : {gt[r['line_id']]}\n  OUT: {r['text']!r}  err={r['error']}\n")
