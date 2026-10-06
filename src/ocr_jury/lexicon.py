"""Domain dictionary + long-s repair.

Vocabulary = modern English (wordfreq) + 18th-century spellings taken from
the training POOL's ground truth (never dev/test). Repair is deliberately
conservative: a word is only rewritten when its current form is NOT a known
word but an f->s variant IS ("prifoner" -> "prisoner"). Real words written
with f ("of", "fire") are never touched; genuinely ambiguous pairs where both
forms are real ("fame"/"same") are left for context-aware scoring.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter
from functools import lru_cache

from wordfreq import zipf_frequency

from .data import load_split
from .metrics import normalize

MIN_ZIPF = 2.5          # modern word "known" if wordfreq zipf >= this (~1 per 3M words)
MAX_F_SWAPS = 4          # cap on 2^k f/s combinations per word
_TOKEN = re.compile(r"^([^A-Za-z]*)([A-Za-z'æÆœ]+)([^A-Za-z]*)$")


@lru_cache(maxsize=1)
def period_vocab() -> Counter:
    """Word counts from the pool's ground truth, long-s folded to s."""
    counts = Counter()
    for line in load_split("pool"):
        for w in re.findall(r"[A-Za-z']+", normalize(line.text)):
            counts[w.lower()] += 1
    return counts


def is_known(word: str) -> bool:
    w = word.lower()
    return period_vocab()[w] >= 1 or zipf_frequency(w, "en") >= MIN_ZIPF


def word_score(word: str) -> float:
    """Higher = more common. Period usage outranks modern frequency."""
    w = word.lower()
    return period_vocab()[w] * 10 + zipf_frequency(w, "en")


def _match_case(template: str, word: str) -> str:
    if template.isupper():
        return word.upper()
    if template[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def repair_word(core: str) -> str:
    if "f" not in core.lower() or is_known(core):
        return core
    # Capitalized words are often proper names ("Purefoy"), which a modern
    # dictionary doesn't know - so it would happily "fix" them into
    # nonsense ("Puresoy"). For those, only accept a variant attested in
    # the period vocabulary of the same books.
    is_name_like = core[:1].isupper() and not core.isupper()
    known = (lambda w: period_vocab()[w.lower()] >= 1) if is_name_like else is_known
    positions = [i for i, c in enumerate(core) if c.lower() == "f"][:MAX_F_SWAPS]
    best, best_score = core, None
    for k in range(1, len(positions) + 1):
        for combo in itertools.combinations(positions, k):
            chars = list(core)
            for i in combo:
                chars[i] = "s" if chars[i] == "f" else "S"
            cand = "".join(chars)
            if known(cand):
                s = word_score(cand)
                if best_score is None or s > best_score:
                    best, best_score = cand, s
    return _match_case(core, best) if best_score is not None else core


def repair_line(text: str) -> str:
    """Applies long-s repair token by token, preserving punctuation/spacing."""
    out = []
    for tok in text.split(" "):
        m = _TOKEN.match(tok)
        if not m:
            out.append(tok)
            continue
        pre, core, post = m.groups()
        # A trailing hyphen marks a word split across lines ("accuf-"): the
        # fragment isn't a dictionary word, so leave it rather than guess.
        out.append(tok if post.startswith("-") else pre + repair_word(core) + post)
    return " ".join(out)
