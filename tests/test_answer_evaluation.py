import json
from types import SimpleNamespace

import pytest

from legal_rag.answer_evaluation import evaluate_answers
from legal_rag.models import Section

BGB1 = "https://www.gesetze-im-internet.de/bgb/__1.html"
BGB433 = "https://www.gesetze-im-internet.de/bgb/__433.html"


class FakeService:
    def __init__(self):
        self.retriever = SimpleNamespace(
            parents={"section-1": Section("section-1", "BGB", "§ 1", "Text", BGB1)}
        )

    def ask(self, query: str, mode: str = "dense") -> dict:
        if "wann" in query.lower():
            return {
                "answer": "Mit der Geburt [S1].",
                "sources": [{"source_id": "S1", "url": BGB1}],
                "metrics": {"abstained": False, "total_ms": 1200},
            }
        return {
            "answer": "Die Quellen reichen nicht aus.",
            "sources": [],
            "metrics": {
                "abstained": True,
                "abstention_reason": "model_insufficient_evidence",
                "total_ms": 1000,
            },
        }


def fixture(tmp_path, records):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def samples():
    return [
        {
            "question": "Wann beginnt die Rechtsfähigkeit?",
            "language": "de",
            "expected_status": "answered",
            "gold_section_urls": [BGB1],
        },
        {
            "question": "Welche Regel steht in § 433?",
            "language": "de",
            "expected_status": "insufficient_evidence",
            "absent_section_urls": [BGB433],
        },
    ]


def test_end_to_end_offline_answer_contract(tmp_path):
    report = evaluate_answers(FakeService(), fixture(tmp_path, samples()), tmp_path / "out")
    summary = report["summary"]
    assert summary["case_pass_rate"] == 1.0
    assert summary["citation_consistency_rate"] == 1.0
    assert summary["gold_section_citation_rate"] == 1.0
    assert summary["negative_abstention_rate"] == 1.0
    assert summary["latency_p95_ms"] == 1200
    assert report["by_language"]["de"]["case_pass_rate"] == 1.0
    assert (tmp_path / "out" / "answer_report.md").is_file()
    assert (tmp_path / "out" / "answer_report.json").is_file()


def test_negative_must_be_absent_from_index(tmp_path):
    bad = [{
        "question": "Was sagt § 1?",
        "language": "de",
        "expected_status": "insufficient_evidence",
        "absent_section_urls": [BGB1],
    }]
    with pytest.raises(ValueError, match="negative example is indexed"):
        evaluate_answers(FakeService(), fixture(tmp_path, bad), tmp_path / "out")


def test_positive_must_exist_in_index(tmp_path):
    bad = [{
        "question": "Was sagt § 433?",
        "language": "de",
        "expected_status": "answered",
        "gold_section_urls": [BGB433],
    }]
    with pytest.raises(ValueError, match="expected source missing"):
        evaluate_answers(FakeService(), fixture(tmp_path, bad), tmp_path / "out")


def test_invalid_missing_language_rejected(tmp_path):
    bad = [{"question": "Wann?", "expected_status": "answered", "gold_section_urls": [BGB1]}]
    with pytest.raises(ValueError, match="language must be"):
        evaluate_answers(FakeService(), fixture(tmp_path, bad), tmp_path / "out")


def test_response_wrong_citation_is_flagged(tmp_path):
    service = FakeService()

    def wrong(query, mode="dense"):
        return {
            "answer": "Mit Geburt [S2].",
            "sources": [{"source_id": "S1", "url": BGB1}],
            "metrics": {"abstained": False, "total_ms": 10},
        }

    service.ask = wrong
    report = evaluate_answers(service, fixture(tmp_path, samples()[:1]), tmp_path / "out")
    assert report["summary"]["case_pass_rate"] == 0
    assert report["summary"]["citation_consistency_rate"] == 0


def test_nonempty_fixture_required(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_answers(FakeService(), fixture(tmp_path, []), tmp_path / "out")


def test_english_case_flags_obviously_german_answer(tmp_path):
    service = FakeService()

    def wrong_language(query, mode="dense"):
        return {
            "answer": (
                "Eine natürliche Person qualifiziert sich als Verbraucher, wenn sie "
                "ein Rechtsgeschäft abschließt, das überwiegend nicht ihrer "
                "gewerblichen Tätigkeit zugerechnet werden kann [S1]."
            ),
            "sources": [{"source_id": "S1", "url": BGB1}],
            "metrics": {"abstained": False, "total_ms": 300},
        }

    service.ask = wrong_language
    case = {
        "question": "Who is a consumer?",
        "language": "en",
        "expected_status": "answered",
        "gold_section_urls": [BGB1],
    }
    report = evaluate_answers(service, fixture(tmp_path, [case]), tmp_path / "out")
    assert report["summary"]["suspected_language_mismatch_count"] == 1
    assert report["summary"]["language_checked_count"] == 1
    assert report["cases"][0]["language_match_heuristic"] is False
    # The established case_pass_rate remains a contract metric, not fluency quality.
    assert report["summary"]["case_pass_rate"] == 1.0
