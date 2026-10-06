"""Non-Jev baselines, computed from cached engine readings.

  single engine  - each engine alone
  majority vote  - reading shared by >=2 engines (after normalization);
                   if no two agree, fall back to the best engine by dev CER
  oracle         - per line, whichever engine is closest to ground truth.
                   The upper bound for ANY method that only picks among
                   engine readings, Jev included.

Diagnostics:
  unanimous-but-wrong - all engines agree yet are wrong; an "agree = accept"
                        shortcut would lock these errors in
  no engine correct   - lines no pure selection method can fix
  long-s share        - how much error vanishes if 'f' were also folded to
                        's' (diagnostic ONLY, never used as a metric: it
                        would forgive genuine f/s misreads)

    python scripts/baselines.py --split dev
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury.cache import load_readings  # noqa: E402
from ocr_jury.data import load_split  # noqa: E402
from ocr_jury.jury import agreement_key  # noqa: E402
from ocr_jury.metrics import cer, line_cer, normalize  # noqa: E402

ENGINES = ["tesseract", "easyocr", "doctr"]


def load_all_readings(split):
    return {e: load_readings(e) for e in ENGINES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    readings = load_all_readings(args.split)
    lines = [l for l in load_split(args.split) if all(l.line_id in readings[e] for e in ENGINES)]
    refs = [l.text for l in lines]
    print(f"lines with readings from all {len(ENGINES)} engines: {len(lines)}")
    errors = {e: sum(1 for l in lines if readings[e][l.line_id]["error"]) for e in ENGINES}
    print("engine failures recorded:", errors)

    hyp = {e: [readings[e][l.line_id]["text"] for l in lines] for e in ENGINES}

    print("\n=== Single engines ===")
    single = {}
    for e in ENGINES:
        single[e] = cer(refs, hyp[e])
        print(f"  {e:10} CER normalized={single[e]:.4f}  strict={cer(refs, hyp[e], normalized=False):.4f}")
    best = min(single, key=single.get)
    print(f"  best single engine: {best}")

    # Same agreement notion as the jury (agreement_key), so the comparison
    # is fair; the winning group's reading is taken from its first engine.
    maj, n_majority = [], 0
    for i in range(len(lines)):
        counts = Counter(agreement_key(hyp[e][i]) for e in ENGINES)
        key, votes = counts.most_common(1)[0]
        if votes >= 2:
            maj.append(next(hyp[e][i] for e in ENGINES if agreement_key(hyp[e][i]) == key))
            n_majority += 1
        else:
            maj.append(hyp[best][i])
    print("\n=== Majority vote ===")
    print(f"  CER normalized={cer(refs, maj):.4f}  (>=2 engines agreed on {n_majority}/{len(lines)} lines)")

    oracle, oracle_pick = [], Counter()
    for i, l in enumerate(lines):
        e_best = min(ENGINES, key=lambda e: line_cer(l.text, hyp[e][i]))
        oracle.append(hyp[e_best][i])
        oracle_pick[e_best] += 1
    print("\n=== Oracle (upper bound for pure selection) ===")
    print(f"  CER normalized={cer(refs, oracle):.4f}   picks: {dict(oracle_pick)}")
    gap = single[best] - cer(refs, oracle)
    print(f"  headroom over best single engine: {gap:.4f} absolute ({100 * gap / single[best]:.1f}% relative)")

    unanimous = [i for i in range(len(lines)) if len({agreement_key(hyp[e][i]) for e in ENGINES}) == 1]
    unan_wrong = [i for i in unanimous if line_cer(refs[i], hyp[ENGINES[0]][i]) > 0]
    no_correct = [i for i, l in enumerate(lines) if all(line_cer(l.text, hyp[e][i]) > 0 for e in ENGINES)]
    print("\n=== Diagnostics ===")
    print(f"  all engines identical: {len(unanimous)} lines, of which WRONG: {len(unan_wrong)}")
    print(f"  no engine exactly correct: {len(no_correct)}/{len(lines)} lines "
          f"({100 * len(no_correct) / len(lines):.1f}%)")

    fold_f = lambda s: normalize(s).replace("f", "s")  # noqa: E731
    print("  long-s share of error (diagnostic only - folds f->s on both sides):")
    for e in ENGINES:
        folded = cer([fold_f(r) for r in refs], [fold_f(h) for h in hyp[e]], normalized=False)
        share = (single[e] - folded) / single[e] if single[e] else 0
        print(f"    {e:10} CER {single[e]:.4f} -> {folded:.4f} with f/s folded  "
              f"(~{100 * share:.0f}% of its error is f/s confusion)")

    print("\n=== Sample of unanimous-but-wrong lines ===")
    for i in unan_wrong[:6]:
        print(f"  GT : {refs[i]}\n  ALL: {hyp[ENGINES[0]][i]}\n")


if __name__ == "__main__":
    main()
