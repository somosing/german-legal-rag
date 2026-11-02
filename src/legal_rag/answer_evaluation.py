"""Grounded-answer *contract* evaluation; no LLM judge or entailment claims.

This intentionally checks what can be proven without another model:
- status matches the curated answerable/unanswerable label;
- cited source IDs are consistent with the answer text;
- for positive questions, an expected section URL was cited.

It does NOT determine whether factual wording is true or entailed by a source.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Protocol

from legal_rag.language_quality import observed_answer_language
from legal_rag.models import Section
from legal_rag.storage import save_json

CITATION_RE = re.compile(r"\[(S\d+)\]")
STATUSES = {"answered", "insufficient_evidence"}


class AnswerService(Protocol):
    retriever: object

    def ask(self, query: str, mode: str = "dense") -> dict: ...


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, deterministic even with small samples."""
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    return sorted_values[max(0, math.ceil(p * len(sorted_values)) - 1)]


def _required_condition_checks(answer: str, requirements: list[dict]) -> list[dict]:
    """Check curated words; presence does not establish legal entailment."""
    plain = " ".join(CITATION_RE.sub("", answer).casefold().split())
    return [
        {
            "id": requirement["id"],
            "matched": any(
                " ".join(term.casefold().split()) in plain
                for term in requirement["any_of"]
            ),
        }
        for requirement in requirements
    ]


def _validate_fixture(samples: object, sections: dict[str, Section]) -> list[dict]:
    if not isinstance(samples, list) or not samples:
        raise ValueError("Answer benchmark needs a non-empty JSON list")
    known_urls = {s.url for s in sections.values()}
    validated = []
    for i, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {i}: expected object")
        question = sample.get("question")
        status = sample.get("expected_status")
        language = sample.get("language")
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 800:
            raise ValueError(f"Sample {i}: question must contain 1-800 characters")
        if status not in STATUSES:
            raise ValueError(f"Sample {i}: invalid expected_status")
        if language not in {"de", "en"}:
            raise ValueError(f"Sample {i}: language must be de or en")
        requirements = sample.get("required_answer_concepts", [])
        if not isinstance(requirements, list):
            raise ValueError(f"Sample {i}: required_answer_concepts must be a list")
        if requirements and status != "answered":
            raise ValueError(f"Sample {i}: negative cases cannot require answer concepts")
        seen: set[str] = set()
        for requirement in requirements:
            if not isinstance(requirement, dict) or set(requirement) != {"id", "any_of"}:
                raise ValueError(f"Sample {i}: invalid answer concept definition")
            identifier = requirement["id"]
            alternatives = requirement["any_of"]
            if (
                not isinstance(identifier, str)
                or not identifier.strip()
                or identifier in seen
                or not isinstance(alternatives, list)
                or not alternatives
                or any(not isinstance(t, str) or not t.strip() for t in alternatives)
            ):
                raise ValueError(f"Sample {i}: invalid required answer concept")
            seen.add(identifier)
        gold_urls = sample.get("gold_section_urls", [])
        absent_urls = sample.get("absent_section_urls", [])
        if any(
            not isinstance(urls, list)
            or any(not isinstance(url, str) or not url for url in urls)
            for urls in (gold_urls, absent_urls)
        ):
            raise ValueError(f"Sample {i}: URL fields must contain valid strings")
        if status == "answered":
            if not gold_urls or absent_urls:
                raise ValueError(f"Sample {i}: answered samples require gold_section_urls only")
            if not set(gold_urls).issubset(known_urls):
                raise ValueError(f"Sample {i}: expected source missing from current index")
        else:
            # Only make negative assertions for provisions explicitly absent from the index.
            if not absent_urls or gold_urls:
                raise ValueError(f"Sample {i}: negative samples require absent_section_urls only")
            if set(absent_urls) & known_urls:
                raise ValueError(f"Sample {i}: negative example is indexed; rebuild the fixture")
        validated.append(sample)
    return validated


def evaluate_answers(
    service: AnswerService, gold_path: Path, out_dir: Path, mode: str = "dense"
) -> dict:
    """Generate answers and score observable response properties.

    A negative label means the provision asked for is not indexed, not that the
    underlying question lacks a correct answer under German law.
    """
    samples = _validate_fixture(
        json.loads(gold_path.read_text(encoding="utf-8")), service.retriever.parents
    )
    rows = []
    for sample in samples:
        # Optional evaluation-only method captures the same retrieval that fed
        # generation, avoiding a misleading second retrieval call after failure.
        evaluate_one = getattr(service, "ask_for_evaluation", None)
        response = (
            evaluate_one(sample["question"], mode=mode)
            if callable(evaluate_one)
            else service.ask(sample["question"], mode=mode)
        )
        metrics = response["metrics"]
        abstained = metrics["abstained"]
        actual_status = "insufficient_evidence" if abstained else "answered"
        listed_ids = [str(source["source_id"]) for source in response["sources"]]
        cited_ids = set(CITATION_RE.findall(response["answer"]))
        citation_valid = (
            not cited_ids and not listed_ids
            if abstained
            else bool(cited_ids) and len(listed_ids) == len(set(listed_ids))
            and cited_ids == set(listed_ids)
        )
        returned_urls = {str(source["url"]) for source in response["sources"]}
        # Private benchmark report contains the actual passages returned to the
        # model plus available parent context for independent human review.
        # Generated reports are gitignored; do not use sensitive source data.
        cited_sources = []
        for source in response["sources"]:
            parent = service.retriever.parents.get(source.get("section_id"))
            cited_sources.append({
                "source_id": source["source_id"],
                "url": source["url"],
                "title": source.get("title", ""),
                "passage": str(source.get("passage", ""))[:1050],
                "parent_excerpt": parent.text[:1450] if parent else "",
            })
        gold_cited = bool(returned_urls & set(sample.get("gold_section_urls", [])))
        retrieved_candidates = response.get("retrieval_candidates")
        gold_rank = (
            next(
                (
                    hit["rank"]
                    for hit in retrieved_candidates
                    if hit["url"] in sample.get("gold_section_urls", [])
                ),
                None,
            )
            if retrieved_candidates is not None
            else None
        )
        condition_checks = _required_condition_checks(
            response["answer"], sample.get("required_answer_concepts", [])
        )
        status_correct = actual_status == sample["expected_status"]
        observed_language = observed_answer_language(response["answer"])
        language_match = (
            observed_language == sample["language"] if observed_language else None
        )
        case_passed = (
            status_correct
            and citation_valid
            and (gold_cited if sample["expected_status"] == "answered" else True)
            and all(check["matched"] for check in condition_checks)
        )
        rows.append(
            {
                "question": sample["question"],
                "language": sample["language"],
                "expected_status": sample["expected_status"],
                "actual_status": actual_status,
                "status_correct": status_correct,
                "citation_consistent": citation_valid,
                "gold_section_cited": (
                    gold_cited if sample["expected_status"] == "answered" else None
                ),
                "case_passed": case_passed,
                "required_condition_checks": condition_checks,
                "observed_answer_language": observed_language,
                "language_match_heuristic": language_match,
                "returned_source_urls": sorted(returned_urls),
                "cited_sources": cited_sources,
                "retrieval_candidates": retrieved_candidates,
                "retrieval_gold_rank": gold_rank,
                "answer": response["answer"],
                "abstention_reason": metrics.get("abstention_reason"),
                "generation_attempts": metrics.get("generation_attempts", 1),
                "first_pass_failure": metrics.get("first_pass_failure"),
                "first_pass_validation_error": metrics.get("first_pass_validation_error"),
                "final_validation_error": metrics.get("final_validation_error"),
                "total_ms": float(metrics["total_ms"]),
            }
        )

    positives = [r for r in rows if r["expected_status"] == "answered"]
    negatives = [r for r in rows if r["expected_status"] == "insufficient_evidence"]
    durations = [r["total_ms"] for r in rows]
    traced_positives = [
        r for r in positives if r["retrieval_candidates"] is not None
    ]
    condition_checks = [
        check for row in rows for check in row["required_condition_checks"]
    ]
    classified = [r for r in rows if r["language_match_heuristic"] is not None]
    language_mismatches = [r for r in classified if not r["language_match_heuristic"]]
    summary = {
        "question_count": len(rows),
        "expected_answerable": len(positives),
        "expected_unanswerable": len(negatives),
        "status_accuracy": round(mean(r["status_correct"] for r in rows), 4),
        "case_pass_rate": round(mean(r["case_passed"] for r in rows), 4),
        "citation_consistency_rate": round(mean(r["citation_consistent"] for r in rows), 4),
        "retrieval_traced_positive_count": len(traced_positives),
        "retrieval_gold_rate_of_traced_positives": (
            round(mean(r["retrieval_gold_rank"] is not None for r in traced_positives), 4)
            if traced_positives else None
        ),
        "gold_section_citation_rate": round(
            mean(r["gold_section_cited"] for r in positives), 4
        ) if positives else None,
        "negative_abstention_rate": round(
            mean(r["actual_status"] == "insufficient_evidence" for r in negatives), 4
        ) if negatives else None,
        "required_condition_check_count": len(condition_checks),
        "required_condition_match_rate": (
            round(mean(check["matched"] for check in condition_checks), 4)
            if condition_checks else None
        ),
        "language_checked_count": len(classified),
        "suspected_language_mismatch_count": len(language_mismatches),
        "language_match_rate_of_checked": (
            round(mean(r["language_match_heuristic"] for r in classified), 4)
            if classified else None
        ),
        "latency_p50_ms": round(_percentile(durations, 0.50), 2),
        "latency_p95_ms": round(_percentile(durations, 0.95), 2),
    }
    by_language = {}
    for language in ("de", "en"):
        group = [r for r in rows if r["language"] == language]
        if group:
            by_language[language] = {
                "questions": len(group),
                "status_accuracy": round(mean(r["status_correct"] for r in group), 4),
                "case_pass_rate": round(mean(r["case_passed"] for r in group), 4),
                "suspected_language_mismatches": sum(
                    r["language_match_heuristic"] is False for r in group
                ),
            }
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "gold_file": str(gold_path),
        "mode": mode,
        "summary": summary,
        "by_language": by_language,
        "cases": rows,
        "limitations": [
            "Citation consistency verifies labels/URLs, not whether legal claims are entailed "
            "by cited text.",
            "Required answer concepts are lexical checks, not proof of entailment; "
            "they may miss paraphrases or accept false statements containing the terms.",
            "These are curated in/out-of-corpus questions, not blinded legal QA accuracy.",
            "Negative labels are corpus-relative; such legal questions can have valid "
            "answers outside the indexed sources.",
            "Retrieval candidates are pre-generation hits, not necessarily the exact "
            "evidence sent to the model after statute scoping and context limits.",
            "Language detection uses a function-word heuristic; "
            "short or mixed answers remain unclassified.",
            "Language mismatch flags are for human review, "
            "not automatic factual quality judgments.",
            "Generated Q/A is saved; don't benchmark sensitive data without retention review.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "answer_report.json", report)
    lines = [
        "# Local legal RAG answer contract evaluation",
        "",
        f"Generated: {report['created_utc']}",
        f"Retrieval mode: `{mode}`",
        "",
        "| Observable metric | Result |",
        "| --- | ---: |",
    ]
    for key, val in summary.items():
        lines.append(f"| {key.replace('_', ' ')} | {val if val is not None else 'n/a'} |")
    lines.extend([
        "", "## Results by question language", "",
        "| Language | Questions | Status accuracy | Case pass rate | Language review flags |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for language, values in by_language.items():
        lines.append(
            f"| {language} | {values['questions']} | {values['status_accuracy']:.3f} | "
            f"{values['case_pass_rate']:.3f} | {values['suspected_language_mismatches']} |"
        )
    lines.extend(["", "## Per-question observations", ""])
    for idx, row in enumerate(rows, start=1):
        result = "PASS" if row["case_passed"] else "REVIEW"
        lines.extend(
            [
                f"### {idx}. {result} — {row['question']}",
                "",
                f"- Expected/actual: `{row['expected_status']}` / `{row['actual_status']}`",
                f"- Citation labels consistent: `{row['citation_consistent']}`",
                f"- Gold section cited: `{row['gold_section_cited']}`",
                f"- Required concepts (lexical): {row['required_condition_checks'] or '(none)'}",
                f"- Generation attempts: `{row['generation_attempts']}`",
                f"- First-pass failure: `{row['first_pass_failure']}`",
                f"- First-pass validation detail: `{row['first_pass_validation_error']}`",
                f"- Final validation detail: `{row['final_validation_error']}`",
                f"- Final abstention reason: `{row['abstention_reason']}`",
                f"- Observed language (heuristic): `{row['observed_answer_language']}`",
                f"- Language matches question (heuristic): `{row['language_match_heuristic']}`",
                f"- Returned URLs: {', '.join(row['returned_source_urls']) or '(none)'}",
                f"- Gold section in retrieved candidates (rank): "
                f"{row['retrieval_gold_rank'] or '(not found / trace unavailable)'}",
                "- Retrieved candidates: "
                + (
                    ", ".join(
                        str(hit["rank"]) + ":" + hit["url"]
                        for hit in row["retrieval_candidates"]
                    )
                    if row["retrieval_candidates"] is not None
                    else "(unavailable)"
                ),
                f"- Answer: {row['answer']}",
                "",
            ]
        )
    lines.extend(["## Language mismatches to review", ""])
    if language_mismatches:
        for row in language_mismatches:
            lines.append(f"- {row['language']}: {row['question']}")
    else:
        lines.append("No obvious mismatches detected by the heuristic.")
    lines.extend(["", "## Limitations", "", *[f"- {x}" for x in report["limitations"]], ""])
    (out_dir / "answer_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
