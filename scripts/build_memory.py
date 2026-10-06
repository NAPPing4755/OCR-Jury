"""Builds the correction memory from the training pool and reports:
  - retrieval sanity check on a noisy query
  - how much of dev has a (near-)duplicate line in the pool - disclosed so
    any gain from the memory is interpreted honestly
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury.cache import load_readings  # noqa: E402
from ocr_jury.data import load_split  # noqa: E402
from ocr_jury.memory import CorrectionMemory  # noqa: E402
from ocr_jury.metrics import line_cer  # noqa: E402

pool = load_split("pool")
t = time.time()
mem = CorrectionMemory(fresh=True)
mem.fit_encoder([l.text for l in pool])
mem.add([l.text for l in pool], source="pool")
print(f"memory built: {mem.count()} pool lines in {time.time() - t:.1f}s")

query = "both civil and ecclefiaftical, by which the nation had ever been accuf"
print(f"\nquery (noisy OCR): {query}")
for text, score in mem.search(query, k=3):
    print(f"  {score:.3f}  {text}")

tess = {k: v["text"] for k, v in load_readings("tesseract").items()}
dev = [l for l in load_split("dev") if l.line_id in tess]
dup_gt, dup_ocr, helpful = 0, 0, 0
for l in dev:
    (hit_text, _), = mem.search(l.text, k=1)
    if line_cer(l.text, hit_text) <= 0.05:
        dup_gt += 1
    (hit_text2, _), = mem.search(tess[l.line_id], k=1)
    if line_cer(l.text, hit_text2) <= 0.05:
        dup_ocr += 1
    if line_cer(l.text, hit_text2) < line_cer(l.text, tess[l.line_id]):
        helpful += 1
print(f"\ndev lines with a near-duplicate (<=5% CER) in the pool: {dup_gt}/{len(dev)}")
print(f"  ...found when querying with the noisy Tesseract reading: {dup_ocr}/{len(dev)}")
print(f"dev lines where the retrieved correction beats Tesseract's reading: {helpful}/{len(dev)}")
mem.close()
