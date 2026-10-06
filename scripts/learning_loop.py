"""Continuous human-in-the-loop learning, simulated.

Round 0 : zero-shot TrOCR scored on dev; its confidence threshold is fixed here.
Round r : 1. TrOCR reads a fresh batch of pool lines
          2. the least-confident REVIEW_FRAC go to "human review" - simulated
             by revealing their ground truth (standard in active learning)
          3. reviewed lines join the fine-tuning set AND the vector-DB memory
          4. TrOCR fine-tunes on everything reviewed so far
          5. scored on dev: CER, share of dev lines still below the round-0
             threshold (review workload), and memory hit rate
Only pool data is ever used for training or memory; dev is only scored.

    python scripts/learning_loop.py --rounds 4 --batch 400
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury import trocr  # noqa: E402
from ocr_jury.data import ROOT, load_split, open_image  # noqa: E402
from ocr_jury.memory import CorrectionMemory  # noqa: E402
from ocr_jury.metrics import cer, line_cer  # noqa: E402

REVIEW_FRAC = 0.5
MEMORY_HIT = 0.90   # cosine similarity counted as "memory already knows this line"


def evaluate(model, proc, dev, dev_imgs, threshold, memory):
    t = time.time()
    reads = trocr.read_batch(model, proc, dev_imgs)
    texts = [r[0] for r in reads]
    confs = np.array([r[1] for r in reads])
    refs = [l.text for l in dev]
    res = {
        "dev_cer": cer(refs, texts),
        "dev_exact_lines": int(sum(line_cer(r, h) == 0 for r, h in zip(refs, texts))),
        "mean_confidence": float(confs.mean()),
        "eval_seconds": round(time.time() - t, 1),
    }
    if threshold is not None:
        below = confs < threshold
        res["review_rate_at_fixed_threshold"] = float(below.mean())
        accepted = [i for i in range(len(dev)) if not below[i]]
        res["accepted_cer"] = cer([refs[i] for i in accepted], [texts[i] for i in accepted]) if accepted else None
    if memory is not None and memory.count() > 0:
        hits = sum(1 for h in texts if memory.search(h, k=1)[0][1] >= MEMORY_HIT)
        res["memory_hit_rate"] = hits / len(dev)
        res["memory_size"] = memory.count()
    return res, confs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-eval", type=int, default=200,
                    help="fixed random dev subset scored every round (the final model is scored on all dev/test later)")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "outputs", "learning_loop")
    ckpt_root = os.path.join(ROOT, "cache", "trocr_checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    dev = load_split("dev")
    dev = random.Random(args.seed).sample(dev, min(args.dev_eval, len(dev)))
    dev_imgs = [open_image(l) for l in dev]
    pool = load_split("pool")
    random.Random(args.seed).shuffle(pool)

    memory = CorrectionMemory(path=os.path.join(ROOT, "cache", "qdrant_loop"), fresh=True)
    memory.fit_encoder([l.text for l in pool])   # unsupervised char n-gram encoder on pool text

    model, proc = trocr.load()
    print("round 0: zero-shot TrOCR on dev ...", flush=True)
    res0, confs0 = evaluate(model, proc, dev, dev_imgs, None, None)
    # Fix the review threshold once, at round 0: the median confidence, i.e.
    # zero-shot TrOCR would send half of dev to review. Later rounds are
    # measured against this same bar.
    threshold = float(np.median(confs0))
    res0.update({"round": 0, "labeled": 0, "threshold": threshold, "review_rate_at_fixed_threshold": 0.5})
    history = [res0]
    print(json.dumps(res0), flush=True)

    labeled = []
    cursor = 0
    for r in range(1, args.rounds + 1):
        t0 = time.time()
        batch = pool[cursor:cursor + args.batch]
        cursor += args.batch
        reads = trocr.read_batch(model, proc, [open_image(l) for l in batch])
        order = np.argsort([c for _, c in reads])              # least confident first
        reviewed = [batch[i] for i in order[:int(len(batch) * REVIEW_FRAC)]]
        batch_conf = float(np.mean([c for _, c in reads]))

        # "human" supplies ground truth; images load lazily per batch
        labeled += [((lambda l=l: open_image(l)), l.text) for l in reviewed]
        memory.add([l.text for l in reviewed], source=f"review_round_{r}")

        print(f"round {r}: fine-tuning on {len(labeled)} reviewed lines ...", flush=True)
        t1 = time.time()
        losses = trocr.train(model, proc, labeled, epochs=args.epochs)
        train_s = time.time() - t1
        trocr.save(model, proc, os.path.join(ckpt_root, f"round_{r}"))

        res, _ = evaluate(model, proc, dev, dev_imgs, threshold, memory)
        res.update({"round": r, "labeled": len(labeled), "reviewed_this_round": len(reviewed),
                    "pool_batch_mean_conf": batch_conf, "final_train_loss": float(np.mean(losses[-10:])),
                    "train_seconds": round(train_s, 1), "round_seconds": round(time.time() - t0, 1)})
        history.append(res)
        print(json.dumps(res), flush=True)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    memory.close()
    print("done:", os.path.join(out_dir, "history.json"))


if __name__ == "__main__":
    main()
