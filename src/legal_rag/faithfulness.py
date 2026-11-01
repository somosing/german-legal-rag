"""Human-verified claim support auditing for saved RAG answer reports.

No LLM judges the model here. A reviewer must explicitly mark each claim,
provide a source-backed quotation for supported labels, and review all assertions
within a claim. Quote matching verifies provenance, *not* entailment by itself.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from legal_rag.storage import save_json

CITATIONS = re.compile(r"\[(S\d+)\]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])")
VERDICTS = {"unreviewed", "supported", "unsupported", "unclear"}
REVIEW_METHODS = {"not_recorded", "ai_assisted_preliminary", "independent_human_review"}


def _normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _claims(answer: str) -> list[str]:
    # Sentence segmentation is intentionally conservative. Reviewers must check
    # *every* factual assertion, including multiple clauses in one sentence.
    return [s.strip() for s in SENTENCE_SPLIT.split(answer.strip()) if s.strip()]


def prepare_review(answer_report: Path, output_path: Path) -> dict:
    """Create a reviewer-editable JSON file from a real eval-answers report."""
    report = json.loads(answer_report.read_text(encoding="utf-8"))
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected a nonempty eval-answers report with cases")
    claims: list[dict] = []
    for case_index, case in enumerate(cases, start=1):
        if case.get("actual_status") != "answered":
            continue
        sources = case.get("cited_sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("Answer report lacks source snapshots; rerun eval-answers")
        source_map = {s.get("source_id"): s for s in sources}
        if len(source_map) != len(sources) or None in source_map:
            raise ValueError("Duplicate or missing source ID in answer report")
        for statement in _claims(case["answer"]):
            cited = sorted(set(CITATIONS.findall(statement)))
            if not set(cited).issubset(source_map):
                raise ValueError("Answer references sources missing from its evidence snapshot")
            stable = hashlib.sha256(
                f"{case_index}\n{case['question']}\n{statement}".encode()
            ).hexdigest()[:16]
            claims.append({
                "claim_id": stable,
                "question": case["question"],
                "claim": statement,
                "cited_source_ids": cited,
                "sources": [source_map[k] for k in cited],
                "verdict": "unreviewed",
                "supporting_source_id": "",
                "evidence_quote": "",
                "reviewer_note": "",
            })
    if not claims:
        raise ValueError("No answered claims found in evaluation report")
    exported = {
        "schema_version": 1,
        "review_method": "not_recorded",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_report": str(answer_report),
        "instructions": (
            "Review every factual assertion in each claim. Mark supported only if ALL "
            "assertions follow from a cited source; provide an exact source quotation. "
            "A matching quote is NOT automatic evidence of entailment. Use unclear "
            "when the source cannot establish the claim. Never mark unreviewed as supported."
        ),
        "claims": claims,
    }
    save_json(output_path, exported)
    return exported


def score_review(annotation_path: Path, output_dir: Path) -> dict:
    """Validate manual judgments and report coverage and reviewed support rate."""
    audit = json.loads(annotation_path.read_text(encoding="utf-8"))
    claims = audit.get("claims")
    if audit.get("schema_version") != 1 or not isinstance(claims, list) or not claims:
        raise ValueError("Invalid or empty claim audit file")
    review_method = audit.get("review_method", "not_recorded")
    if review_method not in REVIEW_METHODS:
        raise ValueError(f"Invalid review_method: {review_method!r}")
    seen: set[str] = set()
    counts = {label: 0 for label in VERDICTS}
    for row in claims:
        claim_id = row.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id or claim_id in seen:
            raise ValueError("Missing or duplicate claim_id")
        seen.add(claim_id)
        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"Claim {claim_id}: invalid verdict {verdict!r}")
        citations = row.get("cited_source_ids")
        sources = row.get("sources")
        if not isinstance(citations, list) or not isinstance(sources, list):
            raise ValueError(f"Claim {claim_id}: missing citation evidence")
        by_id = {source.get("source_id"): source for source in sources}
        if len(by_id) != len(sources) or set(citations) != set(by_id):
            raise ValueError(f"Claim {claim_id}: inconsistent evidence source IDs")
        note = row.get("reviewer_note", "")
        if not isinstance(note, str):
            raise ValueError(f"Claim {claim_id}: reviewer_note must be a string")
        if verdict == "supported":
            source_id = row.get("supporting_source_id")
            quote = row.get("evidence_quote")
            if source_id not in by_id or not isinstance(quote, str) or len(quote.strip()) < 8:
                raise ValueError(f"Claim {claim_id}: supported requires cited source and quote")
            source = by_id[source_id]
            excerpts = [source.get("passage", ""), source.get("parent_excerpt", "")]
            if not any(_normalise(quote) in _normalise(e) for e in excerpts if e):
                raise ValueError(f"Claim {claim_id}: evidence quote not found in cited source")
        if verdict in {"unsupported", "unclear"} and not note.strip():
            raise ValueError(f"Claim {claim_id}: {verdict} requires reviewer_note")
        counts[verdict] += 1
    reviewed = len(claims) - counts["unreviewed"]
    summary = {
        "review_method": review_method,
        "total_claims": len(claims),
        "reviewed_claims": reviewed,
        "review_coverage": round(reviewed / len(claims), 4),
        "supported_claims": counts["supported"],
        "unsupported_claims": counts["unsupported"],
        "unclear_claims": counts["unclear"],
        "unreviewed_claims": counts["unreviewed"],
        "supported_rate_among_reviewed": (
            round(counts["supported"] / reviewed, 4) if reviewed else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "faithfulness_review_report.json", {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_annotations": str(annotation_path),
        "summary": summary,
        "limitations": [
            "Review method is self-declared; do not claim independent human verification "
            "for AI-assisted or unrecorded reviews.",
            "A matching quotation establishes text provenance, not semantic entailment.",
            "Sentence splitting is approximate; reviewers must assess every assertion.",
            "Only answered claims are included; abstention is evaluated separately.",
            "Unreviewed claims are excluded from the reviewed support rate.",
        ],
    })
    text = [
        "# Claim support annotation audit", "",
        f"Review coverage: {reviewed}/{len(claims)} ({summary['review_coverage']:.1%})", "",
        "| Metric | Value |", "| --- | ---: |",
    ]
    for key, value in summary.items():
        text.append(f"| {key.replace('_', ' ')} | {value if value is not None else 'n/a'} |")
    text.extend([
        "", "**No automatic faithfulness score is claimed.**",
        "The recorded review method is self-reported; it is not verified automatically.",
        "Reviewers must validate every assertion against the full, current legal source.",
        "The quotation check only verifies that the quoted text appears in the supplied evidence.",
    ])
    (output_dir / "faithfulness_review_report.md").write_text(
        "\n".join(text) + "\n", encoding="utf-8"
    )
    return summary
