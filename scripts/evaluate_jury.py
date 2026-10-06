"""Full jury evaluation: baselines, reranked jury, confidence routing, ablations.

  dev   : reranker + confidence model scored with GroupKFold out-of-fold
          predictions (groups = page groups), so no line is judged by a model
          that trained on its own page.
  test  : both models refit on ALL dev, then applied to test exactly once.

    python scripts/evaluate_jury.py --engines tesseract,easyocr,doctr[,trocr]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ocr_jury.cache import load_readings  # noqa: E402
from ocr_jury.candidates import Candidate, featurize, generate, rover  # noqa: E402
from ocr_jury.data import ROOT, load_lines, load_split, page_groups, page_key  # noqa: E402
from ocr_jury.jury import agreement_key  # noqa: E402
from ocr_jury.lexicon import repair_line  # noqa: E402
from ocr_jury.lm import lm_scores  # noqa: E402
from ocr_jury.memory import CorrectionMemory  # noqa: E402
from ocr_jury.metrics import cer, line_cer  # noqa: E402

ACCEPTABLE = 0.02  # line CER at or below this = "acceptable" (~1 char in 50)

ABLATIONS = {
    "full": dict(),
    "no_memory": dict(drop_sources={"memory"}, drop_feats={"memory_sim", "from_memory", "only_memory"}),
    "no_repair": dict(drop_sources={"repair"}, drop_feats={"from_repair", "n_support_repaired"}),
    "no_lm": dict(drop_feats={"lm_score"}),
    "no_rover": dict(drop_sources={"rover"}, drop_feats={"from_rover"}),
}


def build_rows(split, engines, memory):
    readings = {e: load_readings(e) for e in engines}
    lines = [l for l in load_split(split) if all(l.line_id in readings[e] for e in engines)]
    print(f"[{split}] lines with all {len(engines)} engines: {len(lines)}", flush=True)
    group_of = page_groups(load_lines())
    per_line, all_texts = [], []
    t = time.time()
    for ln in lines:
        rd = {e: readings[e][ln.line_id] for e in engines}
        cands = generate(rd, memory=memory)
        per_line.append((ln, rd, cands))
        all_texts += [c.text for c in cands]
    print(f"[{split}] {len(all_texts)} candidates generated in {time.time() - t:.0f}s; LM scoring ...", flush=True)
    t = time.time()
    scores = lm_scores(all_texts)
    print(f"[{split}] LM scored in {time.time() - t:.0f}s", flush=True)
    k = 0
    for ln, rd, cands in per_line:
        featurize(cands, rd, scores[k:k + len(cands)])
        k += len(cands)
        for c in cands:
            c.features["_cer"] = line_cer(ln.text, c.text)
    groups = [group_of[page_key(ln.line_id)] for ln, _, _ in per_line]
    return per_line, groups


def keep(c, cfg):
    drops = cfg.get("drop_sources", set())
    if not drops:
        return True
    kept = {s for s in c.sources if s.split(":")[0] not in drops}
    return bool(kept)


def matrix(per_line, cfg, feat_names=None):
    drop_feats = cfg.get("drop_feats", set())
    X, y, line_idx, cand_refs = [], [], [], []
    for i, (_, _, cands) in enumerate(per_line):
        for c in cands:
            if not keep(c, cfg):
                continue
            feats = {k: v for k, v in c.features.items() if not k.startswith("_") and k not in drop_feats}
            if feat_names is None:
                feat_names = sorted(feats)
            X.append([feats.get(f, 0.0) for f in feat_names])
            y.append(c.features["_cer"])
            line_idx.append(i)
            cand_refs.append(c)
    return np.array(X), np.array(y), np.array(line_idx), cand_refs, feat_names


def empty_candidate(line):
    """Stand-in when a line has no candidate at all (every engine returned
    nothing and the ablation removed the rest). Scored as the empty string,
    i.e. fully wrong - the system genuinely produced no reading."""
    c = Candidate(text="")
    c.features = {"n_support_raw": 0, "n_support_repaired": 0, "lm_score": -20.0, "dict_rate": 0.0,
                  "oov_count": 0, "line_n_distinct_readings": 0, "support_conf": 0.0, "n_sources": 0,
                  "_cer": line_cer(line.text, "")}
    return c


def pick(pred, line_idx, cand_refs, rows):
    """Per line: the candidate with the lowest predicted CER, plus the margin
    to the runner-up (a useful confidence signal)."""
    n_lines = len(rows)
    best = {}
    for p, i, c in zip(pred, line_idx, cand_refs):
        best.setdefault(i, []).append((p, c))
    chosen = [empty_candidate(rows[i][0]) for i in range(n_lines)]
    margin, pred_best = np.zeros(n_lines), np.ones(n_lines)
    for i, lst in best.items():
        lst.sort(key=lambda x: x[0])
        chosen[i] = lst[0][1]
        pred_best[i] = lst[0][0]
        margin[i] = (lst[1][0] - lst[0][0]) if len(lst) > 1 else 1.0
    return chosen, margin, pred_best


def conf_features(chosen, margin, pred_best):
    return np.array([[pred_best[i], margin[i], c.features["n_support_raw"], c.features["n_support_repaired"],
                      c.features["lm_score"], c.features["dict_rate"], c.features["oov_count"],
                      c.features["line_n_distinct_readings"], c.features["support_conf"], c.features["n_sources"]]
                     for i, c in enumerate(chosen)])


def run_system(dev_rows, dev_groups, test_rows, cfg, n_splits=5):
    """Returns dev (out-of-fold) and test choices + confidences."""
    Xd, yd, li_d, cr_d, feats = matrix(dev_rows, cfg)
    Xt, _, li_t, cr_t, _ = matrix(test_rows, cfg, feats) if test_rows else (None, None, None, None, None)
    line_groups = np.array(dev_groups)[li_d]
    oof = np.zeros(len(yd))
    for tr, va in GroupKFold(n_splits=n_splits).split(Xd, yd, line_groups):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42).fit(Xd[tr], yd[tr])
        oof[va] = m.predict(Xd[va])
    d_chosen, d_margin, d_pb = pick(oof, li_d, cr_d, dev_rows)

    # Confidence model, also out-of-fold on dev.
    Cd = conf_features(d_chosen, d_margin, d_pb)
    target = np.array([c.features["_cer"] <= ACCEPTABLE for c in d_chosen]).astype(int)
    d_conf = np.zeros(len(dev_rows))
    for tr, va in GroupKFold(n_splits=n_splits).split(Cd, target, dev_groups):
        lr = LogisticRegression(max_iter=2000).fit(Cd[tr], target[tr])
        d_conf[va] = lr.predict_proba(Cd[va])[:, 1]
    out = {"dev": (d_chosen, d_conf)}

    if test_rows:
        rer = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42).fit(Xd, yd)
        t_chosen, t_margin, t_pb = pick(rer.predict(Xt), li_t, cr_t, test_rows)
        lr = LogisticRegression(max_iter=2000).fit(Cd, target)
        t_conf = lr.predict_proba(conf_features(t_chosen, t_margin, t_pb))[:, 1]
        out["test"] = (t_chosen, t_conf)
    return out


def ece(conf, correct, bins=10):
    """Expected calibration error: gap between stated confidence and reality."""
    conf, correct = np.asarray(conf), np.asarray(correct)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def coverage_curve(refs, texts, conf):
    order = np.argsort(-np.asarray(conf))
    rows = []
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(cov * len(order))))
        idx = order[:k]
        rows.append({"auto_accepted": cov, "human_review": round(1 - cov, 2),
                     "accepted_cer": cer([refs[i] for i in idx], [texts[i] for i in idx])})
    return rows


def baselines(rows, engines):
    refs = [ln.text for ln, _, _ in rows]
    hyp = {e: [rd[e]["text"] for _, rd, _ in rows] for e in engines}
    res = {f"single:{e}": cer(refs, hyp[e]) for e in engines}
    res.update({f"single+repair:{e}": cer(refs, [repair_line(h) for h in hyp[e]]) for e in engines})
    best = min(engines, key=lambda e: res[f"single:{e}"])
    maj = []
    for i in range(len(rows)):
        counts = Counter(agreement_key(hyp[e][i]) for e in engines)
        key, votes = counts.most_common(1)[0]
        maj.append(next(hyp[e][i] for e in engines if agreement_key(hyp[e][i]) == key) if votes >= 2 else hyp[best][i])
    res["majority_vote"] = cer(refs, maj)
    ordered = [best] + [e for e in engines if e != best]
    res["rover"] = cer(refs, [rover([hyp[e][i] for e in ordered]) for i in range(len(rows))])
    res["rover+repair"] = cer(refs, [rover([repair_line(hyp[e][i]) for e in ordered]) for i in range(len(rows))])
    res["oracle:engines_only"] = cer(refs, [min((hyp[e][i] for e in engines), key=lambda h: line_cer(refs[i], h))
                                            for i in range(len(rows))])
    res["oracle:all_candidates"] = cer(refs, [min(cands, key=lambda c: c.features["_cer"]).text for _, _, cands in rows])
    return res, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="tesseract,easyocr,doctr")
    ap.add_argument("--test", action="store_true", help="also score the held-out test split (final run only)")
    args = ap.parse_args()
    engines = args.engines.split(",")
    tag = "+".join(engines)
    out_dir = os.path.join(ROOT, "outputs", "jury", tag)
    os.makedirs(out_dir, exist_ok=True)

    memory = CorrectionMemory()  # full pool memory, built by scripts/build_memory.py
    print(f"memory size: {memory.count()}", flush=True)
    dev_rows, dev_groups = build_rows("dev", engines, memory)
    test_rows, _ = build_rows("test", engines, memory) if args.test else (None, None)
    memory.close()

    report = {"engines": engines}
    for split, rows in (("dev", dev_rows), ("test", test_rows)):
        if not rows:
            continue
        res, best = baselines(rows, engines)
        report[f"{split}_baselines"] = res
        report[f"{split}_best_single"] = best

    refs = {"dev": [ln.text for ln, _, _ in dev_rows]}
    if test_rows:
        refs["test"] = [ln.text for ln, _, _ in test_rows]
    for name, cfg in ABLATIONS.items():
        out = run_system(dev_rows, dev_groups, test_rows, cfg)
        for split, (chosen, conf) in out.items():
            texts = [c.text for c in chosen]
            report.setdefault(f"{split}_jury", {})[name] = cer(refs[split], texts)
            if name == "full":
                correct = [line_cer(r, t) <= ACCEPTABLE for r, t in zip(refs[split], texts)]
                rows_ = dev_rows if split == "dev" else test_rows
                best = report[f"{split}_best_single"]
                eng_conf = [rd[best]["confidence"] or 0.0 for _, rd, _ in rows_]
                eng_correct = [line_cer(ln.text, rd[best]["text"]) <= ACCEPTABLE for ln, rd, _ in rows_]
                report[f"{split}_confidence"] = {
                    "jury_ece": ece(conf, correct),
                    f"{best}_self_confidence_ece": ece(eng_conf, eng_correct),
                    "coverage_curve": coverage_curve(refs[split], texts, conf),
                    "chosen_source_counts": dict(Counter(sorted(c.sources)[0].split(":")[0] for c in chosen)),
                }
                with open(os.path.join(out_dir, f"{split}_predictions.jsonl"), "w", encoding="utf-8") as f:
                    for (ln, rd, _), c, p in zip(rows_, chosen, conf):
                        f.write(json.dumps({"line_id": ln.line_id, "gt": ln.text, "jury": c.text,
                                            "sources": sorted(c.sources), "confidence": float(p),
                                            "line_cer": line_cer(ln.text, c.text),
                                            "engines": {e: rd[e]["text"] for e in engines}},
                                           ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
