"""TrOCR smoke test: load, zero-shot read 3 pool lines, one training step,
read again. Catches setup bugs before the long learning loop."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury import trocr  # noqa: E402
from ocr_jury.data import load_split, open_image  # noqa: E402
from ocr_jury.metrics import line_cer  # noqa: E402

t = time.time()
model, proc = trocr.load()
print(f"loaded in {time.time() - t:.0f}s, params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

lines = [l for l in load_split("pool") if len(l.text) > 30][:3]
imgs = [open_image(l) for l in lines]
t = time.time()
reads = trocr.read_batch(model, proc, imgs)
print(f"zero-shot read {len(lines)} lines in {time.time() - t:.1f}s")
for l, (txt, c) in zip(lines, reads):
    print(f"  CER={line_cer(l.text, txt):.3f} conf={c:.2f} | {txt[:70]!r}\n                      GT: {l.text[:70]!r}")

t = time.time()
losses = trocr.train(model, proc, [(imgs[0], lines[0].text), (imgs[1], lines[1].text)], epochs=1, batch_size=2, log_every=1)
print(f"1 training step (batch 2) in {time.time() - t:.1f}s, loss={losses[0]:.3f}")
