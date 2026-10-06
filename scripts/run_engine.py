"""Runs ONE OCR engine over a split and caches every reading to JSONL.

Resumable: lines already in the cache are skipped, so an interrupted run
continues where it stopped. Engine failures on a line are recorded in the
cache (with the error text) rather than swallowed, so they show up in analysis.

    python scripts/run_engine.py --engine tesseract --split dev
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury.cache import cache_path  # noqa: E402
from ocr_jury.data import ROOT, load_split, open_image  # noqa: E402
from ocr_jury.engines import ENGINES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--split", default="dev", help="comma-separated, e.g. dev,test")
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
    path = cache_path(args.engine)  # one cache per engine, shared by all splits

    # A killed run can leave a half-written last line. Keep only complete
    # records and rewrite the file, so the cut-off line is simply redone.
    done, good = set(), []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for x in f:
                try:
                    rec = json.loads(x)
                except json.JSONDecodeError:
                    continue
                done.add(rec["line_id"])
                good.append(x if x.endswith("\n") else x + "\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(good)

    wanted = [l for s in args.split.split(",") for l in load_split(s)]
    lines = [l for l in wanted if l.line_id not in done]
    print(f"{args.engine}: {len(done)} cached, {len(lines)} to go", flush=True)
    if not lines:
        return

    engine = ENGINES[args.engine]()
    t_start = time.time()
    recent_errors: list[bool] = []
    with open(path, "a", encoding="utf-8") as out:
        for i, ln in enumerate(lines, 1):
            t = time.time()
            try:
                r = engine.read(open_image(ln))
                rec = {"line_id": ln.line_id, "text": r.text, "confidence": r.confidence, "error": None}
            except Exception as e:  # recorded, not swallowed
                rec = {"line_id": ln.line_id, "text": "", "confidence": None, "error": f"{type(e).__name__}: {e}"}
            rec["latency_s"] = round(time.time() - t, 3)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            # Fail loudly on a systematically broken run instead of quietly
            # filling the cache with empty readings (this happened once:
            # EasyOCR's quantization crashed on ~98% of lines).
            recent_errors = (recent_errors + [rec["error"] is not None])[-50:]
            if len(recent_errors) == 50 and sum(recent_errors) > 10:
                sys.exit(f"{args.engine}: {sum(recent_errors)}/50 recent lines failed - aborting. "
                         f"Last error: {rec['error']}")
            if i % 50 == 0 or i == len(lines):
                rate = (time.time() - t_start) / i
                print(f"{args.engine}: {i}/{len(lines)}  ({rate:.2f}s/line, ~{rate * (len(lines) - i) / 60:.0f} min left)",
                      flush=True)


if __name__ == "__main__":
    main()
