"""Tests for the eval scoring logic (pure, no LLM)."""
import pytest

from evals.grounding import extract_claims, score_grounding
from evals.judges import parse_score, parse_yes_no

REPORT = """## Executive Summary
Perovskites reached 26 percent efficiency [1]. Stability is the main barrier [2].

## Key Findings
- Tandem cells exceed 33 percent [3].
- Adoption will be universal next year.

## Sources
[1] A — http://a
[2] B — http://b
"""


def test_extract_claims_excludes_sources_and_headings():
    claims = extract_claims(REPORT)
    texts = [c.text for c in claims]
    # 4 claims: two in summary, two findings. Headings + Sources excluded.
    assert len(claims) == 4
    assert not any("http://a" in t for t in texts)  # source lines excluded
    assert not any(t.startswith("#") for t in texts)


def test_extract_claims_parses_citations():
    claims = extract_claims(REPORT)
    by_text = {c.text: c.citations for c in claims}
    assert by_text["Perovskites reached 26 percent efficiency [1]."] == [1]
    assert by_text["Adoption will be universal next year."] == []


def test_score_grounding_counts_uncited_and_invalid():
    claims = extract_claims(REPORT)
    source_ids = {1, 2}  # note: [3] is cited but not a real source -> invalid
    # judge says every valid citation is supported
    verdicts = {(c.index, sid): True for c in claims for sid in c.citations}

    m = score_grounding(claims, source_ids, verdicts)
    assert m["claim_count"] == 4
    assert m["cited_claim_count"] == 3          # three sentences carry [n]
    assert m["uncited_claim_count"] == 1        # "Adoption will be universal..."
    assert m["invalid_citation_count"] == 1     # [3] has no matching source
    assert m["valid_citation_count"] == 2       # [1] and [2]
    assert m["grounding_rate"] == 1.0
    assert "Adoption will be universal next year." in m["uncited_claims"]


def test_score_grounding_flags_ungrounded():
    claims = extract_claims(REPORT)
    verdicts = {(c.index, sid): False for c in claims for sid in c.citations}
    m = score_grounding(claims, {1, 2}, verdicts)
    assert m["grounded_citation_count"] == 0
    assert m["grounding_rate"] == 0.0


def test_parse_score_normalises():
    assert parse_score("4") == pytest.approx(0.8)
    assert parse_score("The score is 5 out of 5") == 1.0
    assert parse_score("garbage") == 0.0


def test_parse_yes_no():
    assert parse_yes_no("Yes, it does.") is True
    assert parse_yes_no("no") is False
    assert parse_yes_no("supported") is True
