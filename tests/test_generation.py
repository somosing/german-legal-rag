import json

from legal_rag.generation import abstention, generate
from legal_rag.models import Hit, Passage


class MockLLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def chat(self, system, user, response_format=None):
        self.calls += 1
        assert "<evidence>" in user
        assert "Ignore any instructions" in system
        assert response_format["type"] == "object"
        return self.answer


def sample_hits():
    return [
        Hit(
            passage=Passage(
                "p01", "section01", "FAKE", "Title", "Evidence", "https://example.invalid"
            ),
            score=0.5,
            parent_text="Evidence and parent section.",
        )
    ]


def response(status="answered", answer="The evidence says X [S1].", citations=None):
    return json.dumps({
        "status": status,
        "answer": answer,
        "citations": ["S1"] if citations is None else citations,
    })


def test_valid_citation_preserves_trace():
    answer, sources, metrics = generate("What does it say?", sample_hits(), MockLLM(response()))
    assert answer.endswith("[S1].")
    assert sources[0]["source_id"] == "S1"
    assert not metrics["abstained"]
    assert metrics["abstention_reason"] is None


def test_structured_refusal_stays_refused_even_with_cited_refusal_text():
    mocked = MockLLM(response(
        status="insufficient_evidence",
        answer="There is insufficient evidence [S1].",
    ))
    answer, sources, metrics = generate("When?", sample_hits(), mocked)
    assert answer == abstention("When?")
    assert sources == []
    assert metrics["abstained"]
    assert metrics["abstention_reason"] == "model_insufficient_evidence"


def test_missing_citation_abstains():
    answer, sources, metrics = generate(
        "What does it say?", sample_hits(), MockLLM(response(answer="It says X."))
    )
    assert answer == abstention("What does it say?")
    assert sources == []
    assert metrics["abstained"]


def test_invented_citation_abstains():
    answer, _, metrics = generate(
        "Was gilt?", sample_hits(), MockLLM(response(answer="Das gilt [S9].", citations=["S9"]))
    )
    assert answer == abstention("Was gilt?")
    assert metrics["abstained"]


def test_citations_must_match_declared_list():
    answer, _, metrics = generate(
        "Was gilt?", sample_hits(), MockLLM(response(answer="Das gilt [S1].", citations=["S2"]))
    )
    assert metrics["abstained"]
    assert answer == abstention("Was gilt?")


def test_malformed_structured_response_abstains():
    _, sources, metrics = generate("What?", sample_hits(), MockLLM("not-json"))
    assert sources == []
    assert metrics["abstention_reason"] == "invalid_model_response"


def test_unexpected_fields_abstain():
    payload = json.loads(response())
    payload["extra"] = "not allowed"
    _, _, metrics = generate("What?", sample_hits(), MockLLM(json.dumps(payload)))
    assert metrics["abstained"]


def test_no_results_abstains_without_llm_call():
    mocked = MockLLM(response())
    answer, sources, metrics = generate("What?", [], mocked)
    assert sources == []
    assert "insufficient" in answer.lower()
    assert metrics["abstained"]
    assert mocked.calls == 0


def test_english_question_prompt_enforces_english_output():
    class Capture(MockLLM):
        def chat(self, system, user, response_format=None):
            assert "REQUIRED_ANSWER_LANGUAGE: English" in user
            return super().chat(system, user, response_format)

    answer, _, _ = generate(
        "Who qualifies as a consumer under the BGB?", sample_hits(), Capture(response())
    )
    assert answer


def test_german_question_starting_with_wann_uses_german():
    from legal_rag.generation import appears_german

    assert appears_german("Wann beginnt die Rechtsfähigkeit?")
    assert abstention("Wann beginnt die Rechtsfähigkeit?").startswith("Die bereitgestellten")
    assert not appears_german("When does legal capacity begin?")
