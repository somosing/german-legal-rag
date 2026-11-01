"""Audit workflow tests do not rely on external models or legal judgments."""

import json
from copy import deepcopy

import pytest

from legal_rag.faithfulness import prepare_review, score_review


def fixture(tmp_path):
    answer_report = {
        "cases": [
            {
                "question": "When does legal capacity begin?",
                "actual_status": "answered",
                "answer": "Legal capacity begins with completion of birth [S1].",
                "cited_sources": [{
                    "source_id": "S1",
                    "url": "https://www.gesetze-im-internet.de/bgb/__1.html",
                    "passage": (
                        "Die Rechtsfähigkeit des Menschen beginnt mit der Vollendung "
                        "der Geburt."
                    ),
                    "parent_excerpt": (
                        "Die Rechtsfähigkeit des Menschen beginnt mit der Vollendung "
                        "der Geburt."
                    ),
                }],
            },
            {
                "question": "What does 433 say?",
                "actual_status": "insufficient_evidence",
                "answer": "The supplied sources are insufficient for a reliable answer.",
                "cited_sources": [],
            },
        ],
    }
    path = tmp_path / "answer_report.json"
    path.write_text(json.dumps(answer_report), encoding="utf-8")
    exported = tmp_path / "review.json"
    result = prepare_review(path, exported)
    return result, exported


def test_export_ignores_abstentions_and_starts_unreviewed(tmp_path):
    result, exported = fixture(tmp_path)
    assert len(result["claims"]) == 1
    claim = result["claims"][0]
    assert claim["verdict"] == "unreviewed"
    assert claim["cited_source_ids"] == ["S1"]
    assert "Geburt" in claim["sources"][0]["passage"]
    assert exported.is_file()


def test_unreviewed_never_counts_as_supported(tmp_path):
    _, path = fixture(tmp_path)
    summary = score_review(path, tmp_path / "score")
    assert summary["review_coverage"] == 0
    assert summary["supported_rate_among_reviewed"] is None
    assert summary["unreviewed_claims"] == 1


def test_supported_requires_exact_source_quote(tmp_path):
    review, path = fixture(tmp_path)
    claim = review["claims"][0]
    claim.update(
        verdict="supported",
        supporting_source_id="S1",
        evidence_quote="Die Rechtsfähigkeit des Menschen beginnt mit der Vollendung der Geburt.",
        reviewer_note="Human reviewer checked the entire claim.",
    )
    path.write_text(json.dumps(review), encoding="utf-8")
    summary = score_review(path, tmp_path / "score")
    assert summary["review_coverage"] == 1
    assert summary["supported_rate_among_reviewed"] == 1
    assert (tmp_path / "score" / "faithfulness_review_report.md").exists()


@pytest.mark.parametrize("issue", ["no_quote", "wrong_source", "hallucinated_quote"])
def test_invalid_supported_annotation_fails_closed(tmp_path, issue):
    review, path = fixture(tmp_path)
    claim = review["claims"][0]
    claim.update(
        verdict="supported",
        supporting_source_id="S1",
        evidence_quote="Die Rechtsfähigkeit des Menschen beginnt mit der Vollendung der Geburt.",
    )
    if issue == "no_quote":
        claim["evidence_quote"] = ""
    elif issue == "wrong_source":
        claim["supporting_source_id"] = "S999"
    else:
        claim["evidence_quote"] = "This sentence was never supplied by the source."
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="requires cited source|quote not found"):
        score_review(path, tmp_path / "score")


def test_unsupported_counts_in_denominator(tmp_path):
    review, path = fixture(tmp_path)
    claim = review["claims"][0]
    claim.update(verdict="unsupported", reviewer_note="Not entailed by cited provision")
    path.write_text(json.dumps(review), encoding="utf-8")
    summary = score_review(path, tmp_path / "score")
    assert summary["supported_rate_among_reviewed"] == 0
    assert summary["unsupported_claims"] == 1


def test_duplicate_claim_id_rejected(tmp_path):
    review, path = fixture(tmp_path)
    review["claims"].append(deepcopy(review["claims"][0]))
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        score_review(path, tmp_path / "score")


def test_missing_source_snapshot_rejected(tmp_path):
    _, path = fixture(tmp_path)
    report = tmp_path / "answer_report.json"
    obj = json.loads(report.read_text())
    obj["cases"][0].pop("cited_sources")
    report.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="lacks source snapshots"):
        prepare_review(report, path)


def test_new_exports_have_explicit_unverified_provenance(tmp_path):
    review, _ = fixture(tmp_path)
    assert review["review_method"] == "not_recorded"


def test_ai_assisted_score_not_misrepresented_as_human_verified(tmp_path):
    review, path = fixture(tmp_path)
    claim = review["claims"][0]
    claim.update(
        verdict="supported",
        supporting_source_id="S1",
        evidence_quote="Die Rechtsfähigkeit des Menschen beginnt mit der Vollendung der Geburt.",
        reviewer_note="Preliminary AI-assisted inspection only",
    )
    review["review_method"] = "ai_assisted_preliminary"
    path.write_text(json.dumps(review), encoding="utf-8")
    score = score_review(path, tmp_path / "score")
    assert score["review_method"] == "ai_assisted_preliminary"
    report = (tmp_path / "score" / "faithfulness_review_report.md").read_text()
    assert "ai_assisted_preliminary" in report
    assert "# Claim support annotation audit" in report


def test_reject_unknown_review_method(tmp_path):
    review, path = fixture(tmp_path)
    review["review_method"] = "human_verified_by_robot"
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid review_method"):
        score_review(path, tmp_path / "score")
