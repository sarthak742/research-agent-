"""Citation-grounding metrics — the core "is the output checkable?" eval.

This is pure logic (no LLM, no network) so it can be unit-tested and reused.
It answers three questions about a generated report:

1. Are claims actually cited? (uncited claims are potential hallucinations)
2. Do the cited source numbers exist? (invalid [n] = a fabricated reference)
3. Do the cited sources actually SUPPORT the claim? (the verdicts come from an
   LLM judge in eval_grounding.py; here we just aggregate them)

The LLM judgement is separated out so this scorer stays deterministic and
testable: you pass in the verdicts, you get back the metrics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Claim:
    index: int          # position among claims
    text: str
    citations: list[int] = field(default_factory=list)


def extract_claims(report: str) -> list[Claim]:
    """Pull citable claims out of a report body.

    A "claim" is any non-heading, non-empty sentence in the report *before*
    the auto-generated ``## Sources`` section (that section is references, not
    claims). Bullet markers are stripped.
    """
    # Drop the Sources section so reference lines aren't counted as claims.
    body = re.split(r"^##\s*Sources\s*$", report, flags=re.MULTILINE)[0]

    claims: list[Claim] = []
    idx = 0
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[-*]\s+", "", line)  # strip bullet markers
        # Split a line into sentences so multi-sentence bullets score fairly.
        for sentence in _split_sentences(line):
            sentence = sentence.strip()
            if not sentence:
                continue
            cites = [int(n) for n in _CITATION_RE.findall(sentence)]
            claims.append(Claim(index=idx, text=sentence, citations=cites))
            idx += 1
    return claims


def _split_sentences(text: str) -> list[str]:
    # Keep the terminator; split on . ! ? followed by space or end.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def score_grounding(
    claims: list[Claim],
    source_ids: set[int],
    verdicts: dict[tuple[int, int], bool],
) -> dict:
    """Aggregate grounding metrics.

    ``verdicts`` maps (claim_index, source_id) -> True if the source supports
    the claim. Missing entries are treated as unsupported (conservative).
    """
    total = len(claims)
    cited = [c for c in claims if c.citations]
    uncited = [c for c in claims if not c.citations]

    valid_citations = 0
    invalid_citations = 0
    grounded = 0
    for c in claims:
        for sid in c.citations:
            if sid not in source_ids:
                invalid_citations += 1
                continue
            valid_citations += 1
            if verdicts.get((c.index, sid), False):
                grounded += 1

    return {
        "claim_count": total,
        "cited_claim_count": len(cited),
        "uncited_claim_count": len(uncited),
        "cited_rate": round(len(cited) / total, 3) if total else None,
        "valid_citation_count": valid_citations,
        "invalid_citation_count": invalid_citations,
        "grounded_citation_count": grounded,
        "grounding_rate": (
            round(grounded / valid_citations, 3) if valid_citations else None
        ),
        "uncited_claims": [c.text for c in uncited],
    }
