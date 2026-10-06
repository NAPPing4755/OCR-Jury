"""Candidate generation, word-level ROVER, and per-candidate features.

Candidate sources for one line:
  engine        each OCR engine's raw reading
  repair        long-s/dictionary repair of each reading
  rover         word-level majority vote across readings (raw and repaired)
  memory        nearest past human corrections from the vector DB
Candidates are deduped on normalized text; every source that produced a
candidate is kept, because "several independent routes agree" is itself a
strong signal.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein

from .jury import agreement_key
from .lexicon import is_known, repair_line
from .metrics import normalize

_WORD = re.compile(r"[A-Za-z']+")


def rover(readings: list[str]) -> str:
    """Word-level ROVER: align every reading to the first (pivot) with
    difflib on token lists, then vote per pivot slot; ties keep the pivot.
    Insertions are kept only when a majority of readings insert the same
    token at the same slot."""
    if not readings:
        return ""
    pivot = readings[0].split()
    n = len(readings)
    slots = [[tok] for tok in pivot]              # votes per pivot token
    inserts: list[Counter] = [Counter() for _ in range(len(pivot) + 1)]
    for other in readings[1:]:
        toks = other.split()
        matched = [False] * len(pivot)
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, pivot, toks, autojunk=False).get_opcodes():
            if op in ("equal", "replace"):
                for k in range(min(i2 - i1, j2 - j1)):
                    slots[i1 + k].append(toks[j1 + k])
                    matched[i1 + k] = True
                if j2 - j1 > i2 - i1:
                    inserts[i2][" ".join(toks[j1 + (i2 - i1):j2])] += 1
            elif op == "insert":
                inserts[i1][" ".join(toks[j1:j2])] += 1
        for i, m in enumerate(matched):
            if not m:
                slots[i].append(None)                 # this reading deleted the token

    out = []
    for i in range(len(pivot) + 1):
        if inserts[i]:
            tok, cnt = inserts[i].most_common(1)[0]
            if cnt > n / 2:
                out.append(tok)
        if i < len(pivot):
            votes = Counter(agreement_key(t) if t is not None else None for t in slots[i])
            winner, cnt = votes.most_common(1)[0]
            if winner is None and cnt > n / 2:
                continue                              # majority deleted it
            if winner is None or cnt <= 1:
                out.append(pivot[i])                  # no agreement: keep pivot
            else:
                out.append(next(t for t in slots[i] if t is not None and agreement_key(t) == winner))
    return " ".join(out)


@dataclass
class Candidate:
    text: str
    sources: set = field(default_factory=set)     # e.g. {"engine:tesseract", "repair:doctr", "rover"}
    memory_sim: float = 0.0
    features: dict = field(default_factory=dict)


def generate(readings: dict[str, dict], memory=None, memory_k: int = 2) -> list[Candidate]:
    """readings: engine -> {"text", "confidence"}. Returns deduped candidates."""
    cands: dict[str, Candidate] = {}

    def add(text: str, source: str, sim: float = 0.0):
        key = normalize(text)
        if not key:
            return
        c = cands.setdefault(key, Candidate(text=text))
        c.sources.add(source)
        c.memory_sim = max(c.memory_sim, sim)

    engines = list(readings)
    raw = [readings[e]["text"] for e in engines]
    fixed = [repair_line(t) for t in raw]
    for e, r, f in zip(engines, raw, fixed):
        add(r, f"engine:{e}")
        add(f, f"repair:{e}")
    add(rover(raw), "rover")
    rover_fixed = rover(fixed)
    add(rover_fixed, "rover")
    if memory is not None:
        for text, sim in memory.search(rover_fixed, k=memory_k):
            add(text, "memory", sim)
    return list(cands.values())


def featurize(cands: list[Candidate], readings: dict[str, dict], lm_scores: list[float]) -> None:
    """Fills Candidate.features in place (the scoring signals from the design)."""
    engines = list(readings)
    raw_keys = {e: agreement_key(readings[e]["text"]) for e in engines}
    fixed_keys = {e: agreement_key(repair_line(readings[e]["text"])) for e in engines}
    raw_norm = [normalize(readings[e]["text"]) for e in engines]
    med_len = sorted(len(t) for t in raw_norm)[len(raw_norm) // 2] or 1
    n_distinct = len(set(raw_keys.values()))

    for c, lm in zip(cands, lm_scores):
        key = agreement_key(c.text)
        support_raw = [e for e in engines if raw_keys[e] == key]
        support_fix = [e for e in engines if fixed_keys[e] == key]
        confs = [readings[e]["confidence"] for e in support_raw if readings[e]["confidence"] is not None]
        words = _WORD.findall(normalize(c.text))
        known = sum(1 for w in words if is_known(w))
        norm = normalize(c.text)
        c.features = {
            "n_support_raw": len(support_raw),
            "n_support_repaired": len(support_fix),
            "support_conf": sum(confs) / len(confs) if confs else 0.0,
            "lm_score": lm,
            "dict_rate": known / len(words) if words else 0.0,
            "oov_count": len(words) - known,
            "mean_edit_to_engines": sum(Levenshtein.normalized_distance(norm, t) for t in raw_norm) / len(raw_norm),
            "len_ratio": len(norm) / med_len,
            "memory_sim": c.memory_sim,
            "from_repair": float(any(s.startswith("repair:") for s in c.sources)),
            "from_rover": float("rover" in c.sources),
            "from_memory": float("memory" in c.sources),
            "only_memory": float(c.sources == {"memory"}),
            "n_sources": len(c.sources),
            "line_n_distinct_readings": n_distinct,
            "line_len": med_len,
        }
        for e in engines:
            c.features[f"src_{e}"] = float(f"engine:{e}" in c.sources or f"repair:{e}" in c.sources)
