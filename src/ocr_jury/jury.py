"""The OCR Jury: deterministic policy around typed Jev decisions.

Per line:
  1. Dedupe engine readings (after normalization).
  2. Unanimous -> ACCEPT with no Jev call. (Known risk, measured separately:
     engines can agree on the same wrong reading, e.g. long-s read as 'f'.)
  3. Otherwise one Jev request: a Choice over the distinct readings plus one
     independent Noul per reading ("is this an error-free transcription?").
  4. confidence = Noul of the chosen reading. ACCEPT if >= threshold, else
     HUMAN_REVIEW. The threshold is a tunable policy value, set on dev only.

Jev decides; this module executes. Every step lands in the decision trace.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .decisions import DecisionProvider
from .metrics import normalize

POLICY_VERSION = "v1"

CHOICE_INSTRUCTIONS = (
    "Several OCR engines read the same single printed line from an 18th-century "
    "English book. Choose the reading that is the most accurate transcription. "
    "Prefer readings made of real words and plausible period spelling. Note that "
    "the long s (ſ) of this era is often misread by OCR as the letter f."
)
NOUL_INSTRUCTIONS = (
    "Is reading {key} an accurate, error-free transcription of a printed line "
    "from an 18th-century English book?"
)


@dataclass
class Verdict:
    line_id: str
    status: str                 # ACCEPT | HUMAN_REVIEW
    text: str                   # the reading the jury settled on
    route: str                  # unanimous | jev
    confidence: float | None
    jev_called: bool
    trace: dict = field(default_factory=dict)


def agreement_key(text: str) -> str:
    """Looser than the metric's normalization: also ignores spacing around
    punctuation ("prifoner ." == "prifoner."), a pure engine-formatting quirk
    that shouldn't cost a Jev call. Used only to decide agreement - scoring
    still uses metrics.normalize, so the stray space still counts as an error."""
    return re.sub(r"\s*([.,;:!?])\s*", r"\1", normalize(text))


def distinct_readings(readings: dict[str, str]) -> dict[str, dict]:
    """engine->text  ==>  {"A": {"text", "engines"}, ...}, deduped by agreement_key."""
    groups: dict[str, dict] = {}
    for engine, text in readings.items():
        key = agreement_key(text)
        if key not in groups:
            groups[key] = {"text": text, "engines": []}
        groups[key]["engines"].append(engine)
    return {chr(65 + i): g for i, g in enumerate(groups.values())}


def judge(line_id: str, readings: dict[str, str], provider: DecisionProvider, threshold: float) -> Verdict:
    cands = distinct_readings(readings)

    if len(cands) == 1:
        (only,) = cands.values()
        return Verdict(line_id, "ACCEPT", only["text"], "unanimous", None, False,
                       {"policy": POLICY_VERSION, "reason": "all engines agree", "engines": only["engines"]})

    state = {"task": "choose the correct OCR reading of one printed line",
             "candidate_readings": {k: c["text"] for k, c in cands.items()}}
    questions = {"best_reading": {"type": "choice", "instructions": CHOICE_INSTRUCTIONS,
                                  "criteria": {k: c["text"] for k, c in cands.items()}}}
    for k in cands:
        questions[f"correct_{k}"] = {"type": "noul", "instructions": NOUL_INSTRUCTIONS.format(key=k)}

    resp = provider.decide(state, questions)
    ans = resp["answers"]
    chosen = ans["best_reading"]["choice"]
    confidence = ans[f"correct_{chosen}"]["noul"]
    status = "ACCEPT" if confidence >= threshold else "HUMAN_REVIEW"

    return Verdict(line_id, status, cands[chosen]["text"], "jev", confidence, True, {
        "policy": POLICY_VERSION,
        "provider": resp.get("provider"),
        "threshold": threshold,
        "candidates": {k: {"text": c["text"], "engines": c["engines"]} for k, c in cands.items()},
        "choice": chosen,
        "choice_confidence": ans["best_reading"].get("confidence"),
        "noul": {k: ans[f"correct_{k}"]["noul"] for k in cands},
        "usage": resp.get("usage"),
        "reason": f"chosen reading confidence {confidence:.2f} {'>=' if status == 'ACCEPT' else '<'} {threshold}",
    })
