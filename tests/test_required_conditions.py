"""Tests for the BGB § 12 required-condition regression."""

import json
from types import SimpleNamespace

import pytest

from legal_rag.answer_evaluation import evaluate_answers
from legal_rag.models import Section

URL = "https://www.gesetze-im-internet.de/bgb/__12.html"
CONDITION = {
    "id": "future_interference_condition",
    "any_of": ["weitere Beeinträchtigungen", "Wiederholungsgefahr"],
}


class Stub:
    def __init__(self, answer):
        self.answer = answer
        self.retriever = SimpleNamespace(
            parents={
                "12": Section("12", "BGB", "§ 12 Namensrecht",
                              "Sind weitere Beeinträchtigungen zu besorgen, "
                              "so kann er auf Unterlassung klagen.", URL)
            }
        )

    def ask(self, query, mode="dense"):
        return {
            "answer": self.answer,
            "sources": [{"source_id": "S1", "url": URL, "section_id": "12"}],
            "metrics": {"abstained": False, "total_ms": 30},
        }


def run(tmp_path, answer, concepts=None):
    sample = {
        "question": "Wann darf man bei Namensgebrauch Unterlassung verlangen?",
        "language": "de",
        "expected_status": "answered",
        "gold_section_urls": [URL],
    }
    if concepts is not None:
        sample["required_answer_concepts"] = concepts
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps([sample]), encoding="utf-8")
    return evaluate_answers(Stub(answer), gold, tmp_path / "result")


def test_missing_condition_fails_despite_valid_citation(tmp_path):
    report = run(
        tmp_path,
        "Bei unbefugter Namensnutzung kann Unterlassung verlangt werden [S1].",
        [CONDITION],
    )
    assert report["cases"][0]["citation_consistent"]
    assert report["cases"][0]["gold_section_cited"]
    assert report["cases"][0]["required_condition_checks"] == [
        {"id": "future_interference_condition", "matched": False}
    ]
    assert report["summary"]["case_pass_rate"] == 0
    assert report["summary"]["required_condition_match_rate"] == 0


def test_future_condition_passes(tmp_path):
    report = run(
        tmp_path,
        "Unterlassung ist möglich, wenn weitere Beeinträchtigungen "
        "zu besorgen sind [S1].",
        [CONDITION],
    )
    assert report["summary"]["case_pass_rate"] == 1
    assert report["summary"]["required_condition_match_rate"] == 1


def test_synonym_wiederholungsgefahr_matches(tmp_path):
    report = run(tmp_path, "Es muss Wiederholungsgefahr bestehen [S1].", [CONDITION])
    assert report["summary"]["case_pass_rate"] == 1


@pytest.mark.parametrize(
    "invalid",
    [
        {"id": "", "any_of": ["weitere Beeinträchtigungen"]},
        {"id": "future", "any_of": []},
        {"id": "future", "any_of": [""]},
        {"id": "future", "any_of": "weitere Beeinträchtigungen"},
    ],
)
def test_invalid_required_concept_rejected(tmp_path, invalid):
    with pytest.raises(ValueError):
        run(tmp_path, "Es drohen weitere Beeinträchtigungen [S1].", [invalid])


def test_concept_is_optional_for_other_cases(tmp_path):
    report = run(tmp_path, "Der Name ist geschützt [S1].")
    assert report["summary"]["case_pass_rate"] == 1
    assert report["summary"]["required_condition_match_rate"] is None
