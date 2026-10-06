"""Charts from saved results (no experiment is re-run):
  learning_loop.png   - TrOCR CER, review workload, memory hits per round
  systems_<split>.png - CER of every baseline and the jury
  coverage_<split>.png- CER of auto-accepted lines vs share sent to review

    python scripts/make_plots.py --jury tesseract+easyocr+doctr+trocr
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury.data import ROOT  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
ap = argparse.ArgumentParser()
ap.add_argument("--jury", default="tesseract+easyocr+doctr+trocr")
args = ap.parse_args()

hist_path = os.path.join(OUT, "learning_loop", "history.json")
if os.path.exists(hist_path):
    with open(hist_path) as f:
        h = json.load(f)
    rounds = [x["round"] for x in h]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(rounds, [100 * x["dev_cer"] for x in h], "o-", color="#1f77b4")
    axes[0].set_title("TrOCR character error rate (dev subset)")
    axes[0].set_ylabel("CER (%)")
    axes[1].plot(rounds, [100 * x.get("review_rate_at_fixed_threshold", 0) for x in h], "o-", color="#d62728")
    axes[1].set_title("Lines needing human review\n(fixed round-0 confidence bar)")
    axes[1].set_ylabel("% of dev lines")
    mem = [(x["round"], 100 * x["memory_hit_rate"]) for x in h if "memory_hit_rate" in x]
    if mem:
        axes[2].plot(*zip(*mem), "o-", color="#2ca02c")
    axes[2].set_title("Vector-DB memory hits (cos >= 0.90)")
    axes[2].set_ylabel("% of dev lines")
    for ax in axes:
        ax.set_xlabel("learning round (0 = zero-shot)")
        ax.set_xticks(rounds)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_loop.png"), dpi=130)
    print("saved learning_loop.png")

rep_path = os.path.join(OUT, "jury", args.jury, "report.json")
if os.path.exists(rep_path):
    with open(rep_path) as f:
        rep = json.load(f)
    for split in ("dev", "test"):
        if f"{split}_baselines" not in rep:
            continue
        systems = dict(rep[f"{split}_baselines"])
        systems.update({f"jury:{k}": v for k, v in rep[f"{split}_jury"].items()})
        items = sorted(systems.items(), key=lambda kv: kv[1])
        colors = ["#2ca02c" if k.startswith("jury:full") else "#9467bd" if k.startswith("jury") else
                  "#7f7f7f" if k.startswith("oracle") else "#1f77b4" for k, _ in items]
        fig, ax = plt.subplots(figsize=(9, 0.38 * len(items) + 1))
        ax.barh([k for k, _ in items], [100 * v for _, v in items], color=colors)
        for i, (_, v) in enumerate(items):
            ax.text(100 * v + 0.05, i, f"{100 * v:.2f}%", va="center", fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("character error rate (%) - lower is better")
        ax.set_title(f"{split}: baselines vs. OCR Jury  (grey = oracle upper bounds)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"systems_{split}.png"), dpi=130)
        print(f"saved systems_{split}.png")

        cc = rep[f"{split}_confidence"]["coverage_curve"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([100 * c["human_review"] for c in cc], [100 * c["accepted_cer"] for c in cc], "o-")
        ax.set_xlabel("% of lines sent to human review (least confident first)")
        ax.set_ylabel("CER of auto-accepted lines (%)")
        ax.set_title(f"{split}: routing uncertain lines to review")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"coverage_{split}.png"), dpi=130)
        print(f"saved coverage_{split}.png")
