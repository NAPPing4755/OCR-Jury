"""Unit tests for the logic where a silent bug would corrupt the results:
the metric, long-s repair, ROVER, the agreement shortcut, and the
leakage-free splits."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ocr_jury.candidates import rover  # noqa: E402
from ocr_jury.decisions import MockDecisionProvider  # noqa: E402
from ocr_jury.jury import agreement_key, judge  # noqa: E402
from ocr_jury.metrics import line_cer, normalize  # noqa: E402

HAS_DATA = os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "OCR_Testing.zip"))
needs_data = pytest.mark.skipif(not HAS_DATA, reason="dataset not downloaded")


# ---------- metric ----------
def test_long_s_folds_to_s_but_f_is_still_an_error():
    gt = "The Hiſtory of England"
    assert line_cer(gt, "The History of England") == 0
    assert line_cer(gt, "The Hiftory of England") > 0          # f-for-long-s is a real error
    assert line_cer(gt, "The History of England", normalized=False) > 0


def test_normalize_folds_typography():
    assert normalize("a—b “q” x  y") == 'a-b "q" x y'


# ---------- long-s repair ----------
@needs_data
@pytest.mark.parametrize("raw,fixed", [
    ("prifoner", "prisoner"),
    ("fuccefs", "success"),
    ("of", "of"),            # real word with f: untouched
    ("fire", "fire"),
    ("accuf-", "accuf-"),    # line-break fragment: left alone
])
def test_repair_line(raw, fixed):
    from ocr_jury.lexicon import repair_line
    assert repair_line(raw) == fixed


@needs_data
def test_repair_preserves_case_and_punctuation():
    from ocr_jury.lexicon import repair_line
    assert repair_line("PRIFONER,") == "PRISONER,"
    assert repair_line("Prifoner.") == "Prisoner."


# ---------- ROVER ----------
@pytest.mark.parametrize("readings,expected", [
    (["the cat sat on the mat", "the cat sat on tbe mat", "the cat fat on the mat"], "the cat sat on the mat"),
    (["a b c", "a x b c", "a x b c"], "a x b c"),          # majority insertion kept
    (["a b c d", "a c d", "a c d"], "a c d"),              # majority deletion applied
    (["one two three", "uno dos tres", "eins zwei drei"], "one two three"),  # no agreement: pivot
])
def test_rover(readings, expected):
    assert rover(readings) == expected


# ---------- agreement shortcut ----------
def test_agreement_ignores_spacing_around_punctuation():
    assert agreement_key("prifoner .") == agreement_key("prifoner.")
    assert agreement_key("prifoner.") != agreement_key("prisoner.")


def test_unanimous_lines_skip_the_decision_model():
    v = judge("x", {"a": "same text.", "b": "same text .", "c": "same text."}, MockDecisionProvider(), 0.8)
    assert v.status == "ACCEPT" and not v.jev_called


def test_disagreement_calls_decider_and_records_trace():
    v = judge("x", {"a": "one reading", "b": "other reading", "c": "one reading"}, MockDecisionProvider(), 0.99)
    assert v.jev_called and v.trace["provider"] == "mock" and v.status in ("ACCEPT", "HUMAN_REVIEW")


# ---------- leakage-free splits ----------
@needs_data
def test_no_page_group_shared_between_splits():
    from ocr_jury.data import load_lines, load_split, page_groups, page_key
    group_of = page_groups(load_lines())
    groups = {s: {group_of[page_key(l.line_id)] for l in load_split(s)} for s in ("dev", "test", "pool")}
    assert not groups["dev"] & groups["test"]
    assert not groups["dev"] & groups["pool"]
    assert not groups["test"] & groups["pool"]


@needs_data
def test_splits_are_disjoint_and_cover_everything():
    from ocr_jury.data import load_lines, load_split
    ids = {s: {l.line_id for l in load_split(s)} for s in ("dev", "test", "pool")}
    assert not ids["dev"] & ids["test"] and not ids["dev"] & ids["pool"] and not ids["test"] & ids["pool"]
    assert len(ids["dev"] | ids["test"] | ids["pool"]) == len(load_lines())
