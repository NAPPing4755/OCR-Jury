"""Text normalization and character error rate.

Two views of CER are always reported side by side:
  strict     - exact ground truth, including long-s (ſ)
  normalized - the primary metric: long-s -> s, typographic quotes/dashes and
               unusual spaces folded to plain ASCII, whitespace collapsed.
Normalization is applied identically to ground truth and prediction.

It deliberately does NOT map "f" -> "s": an engine reading ſ as "f"
("Hiftory") made a real error, and catching exactly that kind of error is
part of what the jury is being tested on.
"""
from __future__ import annotations

import re

import jiwer

_FOLD = str.maketrans({
    "ſ": "s",                     # long s
    "‘": "'", "’": "'",      # curly single quotes
    "“": '"', "”": '"',      # curly double quotes
    "—": "-", "–": "-",      # em / en dash
    " ": " ", " ": " ",      # no-break / four-per-em space
})


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_FOLD)).strip()


def cer(references: list[str], hypotheses: list[str], normalized: bool = True) -> float:
    """Corpus-level CER: total character edits / total reference characters."""
    if normalized:
        references = [normalize(r) for r in references]
        hypotheses = [normalize(h) for h in hypotheses]
    return jiwer.cer(references, hypotheses)


def line_cer(reference: str, hypothesis: str, normalized: bool = True) -> float:
    if normalized:
        reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.cer(reference, hypothesis)
