"""Evidence-bound statutory-condition guard: § 12 BGB injunction pilot."""

import json

from legal_rag.generation import critical_statutory_condition, generate
from legal_rag.models import Hit, Passage

CLAUSE = "Sind weitere Beeinträchtigungen zu besorgen, so kann er auf Unterlassung klagen."
TEXT = (
    "Wird das Recht zum Gebrauch eines Namens dem Berechtigten von einem "
    "anderen bestritten oder wird das Interesse des Berechtigten dadurch "
    "verletzt, dass ein anderer unbefugt den gleichen Namen gebraucht, "
    "so kann der Berechtigte von dem anderen Beseitigung der "
    "Beeinträchtigung verlangen. " + CLAUSE
)
QUESTION = (
    "Unter welchen Umständen kann jemand wegen der Nutzung seines "
    "Namens Unterlassung verlangen?"
)


def hit(number="12", passage_text=TEXT):
    return Hit(
        passage=Passage(
            "passage-id", "section-id", "BGB", "§ 12 Namensrecht",
            passage_text, f"https://www.gesetze-im-internet.de/bgb/__{number}.html",
        ),
        score=0.9,
        parent_text=passage_text,
    )


def answer(value, status="answered", citations=None):
    return json.dumps({
        "status": status,
        "answer": value,
        "citations": ["S1"] if citations is None else citations,
    })


class Responses:
    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    def chat(self, system, user, response_format=None):
        self.calls += 1
        assert "<evidence>" in user
        assert response_format is not None
        if "AUDITED SOURCE CONDITION" in user:
            assert CLAUSE in user
        return self.values.pop(0)


INCOMPLETE = (
    "Bei einer unbefugten Nutzung eines Namens kann der Berechtigte "
    "auf Unterlassung klagen [S1]."
)
COMPLETE = (
    "Bei einer Verletzung des Namensrechts kann der Berechtigte auf "
    "Unterlassung klagen, wenn weitere Beeinträchtigungen zu besorgen "
    "sind [S1]."
)


def test_statutory_clause_bound_to_verifiable_source_and_intent():
    assert critical_statutory_condition(QUESTION, [hit()]) == ("S1", CLAUSE)
    assert critical_statutory_condition("Was regelt § 12 BGB?", [hit()]) is None
    assert critical_statutory_condition(QUESTION, [hit(number="13")]) is None
    assert critical_statutory_condition(
        QUESTION, [hit(passage_text="Irrelevant source")]
    ) is None


def test_incomplete_injunction_answer_abstains_by_default():
    llm = Responses(answer(INCOMPLETE))
    result, sources, metrics = generate(QUESTION, [hit()], llm)
    assert metrics["abstained"] is True
    assert metrics["final_validation_error"] == "missing_required_legal_condition"
    assert sources == []
    assert llm.calls == 1
    assert "reichen" in result


def test_repair_accepts_complete_condition_with_valid_citation():
    llm = Responses(answer(INCOMPLETE), answer(COMPLETE))
    result, sources, metrics = generate(QUESTION, [hit()], llm, retry_on_failure=True)
    assert metrics["abstained"] is False
    assert metrics["generation_attempts"] == 2
    assert metrics["first_pass_validation_error"] == "missing_required_legal_condition"
    assert sources[0]["url"].endswith("__12.html")
    assert result == COMPLETE


def test_no_silent_repair_when_model_refuses():
    llm = Responses(answer("", status="insufficient_evidence", citations=[]))
    _, sources, metrics = generate(QUESTION, [hit()], llm, retry_on_failure=True)
    assert metrics["abstained"] is True
    assert metrics["abstention_reason"] == "model_insufficient_evidence"
    assert metrics["generation_attempts"] == 1
    assert sources == []


def test_complete_answer_accepted_without_retry():
    llm = Responses(answer(COMPLETE))
    _, sources, metrics = generate(QUESTION, [hit()], llm)
    assert metrics["abstained"] is False
    assert metrics["generation_attempts"] == 1
    assert len(sources) == 1


def test_wrong_section_does_not_gain_curated_condition():
    llm = Responses(answer(INCOMPLETE))
    _, _, metrics = generate(QUESTION, [hit(number="13")], llm)
    assert metrics["abstained"] is False
    assert llm.calls == 1


def test_repair_fails_closed_when_second_answer_still_incomplete():
    llm = Responses(answer(INCOMPLETE), answer(INCOMPLETE))
    _, sources, metrics = generate(QUESTION, [hit()], llm, retry_on_failure=True)
    assert metrics["abstained"] is True
    assert metrics["generation_attempts"] == 2
    assert sources == []
