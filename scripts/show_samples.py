"""Renders a few dev-set line images with their ground truth, for visual
inspection of what the OCR engines will face."""
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury.data import load_split, open_image, ROOT  # noqa: E402

lines = load_split("dev")
rng = random.Random(7)
picks = rng.sample([l for l in lines if l.source == "Hume"], 5) + rng.sample([l for l in lines if l.source == "Generic"], 3)

fig, axes = plt.subplots(len(picks), 1, figsize=(12, 1.35 * len(picks)))
for ax, ln in zip(axes, picks):
    ax.imshow(open_image(ln))
    ax.set_title(f"[{ln.source}]  GT: {ln.text}", fontsize=9, loc="left")
    ax.axis("off")
fig.tight_layout()
out = os.path.join(ROOT, "outputs", "sample_lines.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=110)
print("saved", out)
for ln in picks:
    img = open_image(ln)
    print(f"{img.size[0]}x{img.size[1]}  {ln.text}")
