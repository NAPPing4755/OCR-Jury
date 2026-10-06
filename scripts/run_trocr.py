"""Runs a (fine-tuned) TrOCR checkpoint over splits and writes its readings
into the shared engine cache as engine "trocr", so the jury can use it as a
4th reader exactly like the off-the-shelf engines. Batched for speed.

    python scripts/run_trocr.py --checkpoint cache/trocr_checkpoints/round_4 --split dev,test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury import trocr  # noqa: E402
from ocr_jury.cache import cache_path, load_readings  # noqa: E402
from ocr_jury.data import load_split, open_image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--split", default="dev,test")
ap.add_argument("--chunk", type=int, default=32)
args = ap.parse_args()

done = load_readings("trocr")
lines = [l for s in args.split.split(",") for l in load_split(s) if l.line_id not in done]
print(f"trocr ({args.checkpoint}): {len(done)} cached, {len(lines)} to go", flush=True)
model, proc = trocr.load(args.checkpoint)

t0 = time.time()
with open(cache_path("trocr"), "a", encoding="utf-8") as out:
    for i in range(0, len(lines), args.chunk):
        chunk = lines[i:i + args.chunk]
        t = time.time()
        reads = trocr.read_batch(model, proc, [open_image(l) for l in chunk])
        per_line = (time.time() - t) / len(chunk)
        for l, (text, conf) in zip(chunk, reads):
            out.write(json.dumps({"line_id": l.line_id, "text": text, "confidence": conf, "error": None,
                                  "latency_s": round(per_line, 3), "checkpoint": args.checkpoint},
                                 ensure_ascii=False) + "\n")
        out.flush()
        n = i + len(chunk)
        rate = (time.time() - t0) / n
        print(f"trocr: {n}/{len(lines)} ({rate:.2f}s/line, ~{rate * (len(lines) - n) / 60:.0f} min left)", flush=True)
