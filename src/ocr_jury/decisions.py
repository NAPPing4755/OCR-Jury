"""Decision providers: the real TypeSafe Jev API and a mock with the same shape.

Request/response shapes follow TypeSafe's official API guide
(POST https://api.typesafe.ai/v1/systemone):
  request : {"model", "state", "questions": {name: {"type", "instructions", "criteria"}}}
  answers : choice -> {"type","choice","probabilities","confidence"}
            noul   -> {"type","noul"}
Jev only decides. Everything around it - which lines get asked, thresholds,
retries, logging - is ordinary deterministic code in this project.

MockDecisionProvider exists so the whole pipeline can be built and tested
without API access. Its outputs are tagged provider="mock" and must never be
reported as Jev results.
"""
from __future__ import annotations

import difflib
import os
import time
from typing import Protocol

import requests
from dotenv import load_dotenv

API_URL = "https://api.typesafe.ai/v1/systemone"


class DecisionProvider(Protocol):
    name: str

    def decide(self, state: dict, questions: dict) -> dict:
        """Returns {"answers": {question_name: answer}, "usage": {...}}."""
        ...


class TypeSafeJevProvider:
    name = "jev"

    def __init__(self, model: str = "jev-latest", max_retries: int = 5):
        load_dotenv()
        self._key = os.getenv("JEV_API_KEY")
        if not self._key:
            raise RuntimeError("JEV_API_KEY is not set (see .env.example)")
        self.model = model
        self.max_retries = max_retries

    def decide(self, state: dict, questions: dict) -> dict:
        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self._key}"}
        for attempt in range(self.max_retries):
            r = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            # Retry only transient overload / rate limiting, with backoff.
            if r.status_code in (429, 529) or r.status_code >= 500:
                time.sleep(min(2 ** attempt, 30))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"Jev API error {r.status_code}: {r.text[:300]}")
            body = r.json()
            body["provider"] = self.name
            return body
        raise RuntimeError(f"Jev API still failing after {self.max_retries} retries")


class MockDecisionProvider:
    """Deterministic stand-in with the same response shape as Jev.

    Choice: picks the candidate most similar to the others (the "medoid"),
    a crude consensus heuristic. Noul: that candidate's average similarity
    to the rest. Good enough to exercise the pipeline end to end; says
    nothing about how well Jev itself performs.
    """
    name = "mock"

    def decide(self, state: dict, questions: dict) -> dict:
        candidates = state["candidate_readings"]
        sim = {a: sum(difflib.SequenceMatcher(None, candidates[a], candidates[b]).ratio()
                      for b in candidates if b != a) / max(len(candidates) - 1, 1)
               for a in candidates}
        answers = {}
        for qname, q in questions.items():
            if q["type"] == "choice":
                total = sum(sim.values()) or 1.0
                probs = {k: sim[k] / total for k in q["criteria"]}
                best = max(probs, key=probs.get)
                answers[qname] = {"type": "choice", "choice": best, "probabilities": probs,
                                  "confidence": probs[best]}
            elif q["type"] == "noul":
                # Noul questions are named "correct_<candidate key>" - the
                # request itself carries only fields the real API defines.
                answers[qname] = {"type": "noul", "noul": sim[qname.removeprefix("correct_")]}
        return {"answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}, "provider": self.name}


def get_provider(prefer_real: bool = True) -> DecisionProvider:
    load_dotenv()
    if prefer_real and os.getenv("JEV_API_KEY"):
        return TypeSafeJevProvider()
    return MockDecisionProvider()
