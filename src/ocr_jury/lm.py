"""Language-likelihood scorer: how natural a candidate reading is as English.

distilgpt2 (82M params) - small enough for CPU, and it needs no prompting:
the score is simply the mean log-probability per token the model assigns to
the text. OCR garbage ("pclitics", "Englaud") gets a much lower score than
the real words. Text is normalized (long-s -> s) first, since the model was
trained on modern English.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import ROOT
from .metrics import normalize

MODEL_NAME = "distilgpt2"


@lru_cache(maxsize=1)
def _model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tok, model


_CACHE_PATH = os.path.join(ROOT, "cache", "lm_scores.json")


def lm_scores(texts: list[str], batch_size: int = 16) -> list[float]:
    """Mean log-prob per token (higher = more natural), with a disk cache
    keyed by normalized text: the jury is evaluated with and without TrOCR,
    and most candidate texts repeat between runs (~0.5 s each on CPU)."""
    cache = {}
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
    keys = [normalize(t) for t in texts]
    missing = sorted({k for k in keys if k not in cache})
    if missing:
        for k, s in zip(missing, _score(missing, batch_size)):
            cache[k] = s
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    return [cache[k] for k in keys]


@torch.no_grad()
def _score(texts: list[str], batch_size: int) -> list[float]:
    """Empty text -> -20."""
    tok, model = _model()
    out: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch = [normalize(t) for t in texts[i:i + batch_size]]
        # Prefix with EOS so the first real token is also scored.
        enc = tok([tok.eos_token + t for t in batch], return_tensors="pt", padding=True, truncation=True, max_length=64)
        logits = model(**enc).logits[:, :-1]
        targets = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].float()
        logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        per_seq = (logp * mask).sum(1) / mask.sum(1).clamp(min=1)
        for t, s in zip(batch, per_seq.tolist()):
            out.append(s if t.strip() else -20.0)
    return out
