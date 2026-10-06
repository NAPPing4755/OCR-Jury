"""Buckets every remaining jury error (dev predictions) into a taxonomy and
writes outputs/error_analysis.md with counts, share of error characters,
and examples. First matching bucket wins, in the order listed.

    python scripts/error_analysis.py --jury tesseract+easyocr+doctr+trocr
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury.data import ROOT  # noqa: E402
from ocr_jury.jury import agreement_key  # noqa: E402
from ocr_jury.lexicon import is_known  # noqa: E402
from ocr_jury.metrics import normalize  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402


def diff_window(a, b, width=28):
    """Both strings cut to a window around their FIRST difference - a fixed
    60-char prefix hid differences that fell past the cut-off."""
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    start = max(0, i - width // 2)
    clip = lambda s: ("..." if start else "") + s[start:start + width].replace("|", "/") + ("..." if len(s) > start + width else "")  # noqa: E731
    return clip(a), clip(b)


def strip_punct(s):
    return re.sub(r"[^\w]", "", normalize(s))


def bucket(gt, out, engines):
    g, o = normalize(gt), normalize(out)
    if not o:
        return "empty output"
    if len(g) <= 3:
        return "very short line (<=3 chars)"
    if g.replace("f", "s") == o.replace("f", "s"):
        return "long-s / f confusion left unrepaired"
    if g.lower() == o.lower():
        return "letter case only"
    if strip_punct(g) == strip_punct(o):
        return "punctuation / spacing only"
    # Numbers: no plausibility signal (dictionary or language model) can tell
    # "233" from "272", so they're tracked separately.
    if re.findall(r"\d+", g) != re.findall(r"\d+", o):
        return "digits differ (misread numbers, or letters read as digits)"
    if g.rstrip("-").rstrip() == o.rstrip("-").rstrip():
        return "line-end hyphen / fragment"
    raw = {agreement_key(t) for t in engines.values() if t.strip()}
    words = re.findall(r"[A-Za-z']+", o)
    if len(raw) == 1 and agreement_key(out) in raw and words and all(is_known(w) for w in words):
        return "all engines agree on real words (possible ground-truth error)"
    if Levenshtein.distance(g, o) <= 2:
        return "1-2 character misread"
    return "multi-character misread"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jury", default="tesseract+easyocr+doctr+trocr")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    path = os.path.join(ROOT, "outputs", "jury", args.jury, f"{args.split}_predictions.jsonl")
    with open(path, encoding="utf-8") as f:
        preds = [json.loads(x) for x in f]

    total_chars = sum(len(normalize(p["gt"])) for p in preds)
    groups = defaultdict(list)
    for p in preds:
        if p["line_cer"] > 0:
            b = bucket(p["gt"], p["jury"], p["engines"])
            edits = Levenshtein.distance(normalize(p["gt"]), normalize(p["jury"]))
            groups[b].append((edits, p))
    err_lines = sum(len(v) for v in groups.values())
    err_chars = sum(e for v in groups.values() for e, _ in v)

    out = [f"# Error analysis - {args.split} split, jury `{args.jury}`\n",
           f"{len(preds)} lines; {err_lines} ({100 * err_lines / len(preds):.1f}%) have at least one error; "
           f"{err_chars} character edits over {total_chars} reference characters "
           f"(CER {100 * err_chars / total_chars:.2f}%).\n",
           "| Error type | Lines | Share of error characters | Example (ground truth -> jury) |",
           "|---|---|---|---|"]
    for b, items in sorted(groups.items(), key=lambda kv: -sum(e for e, _ in kv[1])):
        chars = sum(e for e, _ in items)
        ex = sorted(items, key=lambda x: x[0])[len(items) // 2][1]  # a typical (median) example
        gt_ex, out_ex = diff_window(normalize(ex["gt"]), normalize(ex["jury"]))
        out.append(f"| {b} | {len(items)} | {100 * chars / err_chars:.1f}% | `{gt_ex}` -> `{out_ex}` |")
    text = "\n".join(out) + "\n"
    dest = os.path.join(ROOT, "outputs", "error_analysis.md")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
