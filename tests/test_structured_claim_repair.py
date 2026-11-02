"""Opt-in quote-bound claim repair after an uncited sentence.

These tests check schema/quote/citation structure, NOT legal entailment.
"""

import json

from legal_rag.generation import (
    StructuredClaims,
    generate,
    render_grounded_claims,
)
from legal_rag.models import Hit, Passage

TEXT = (
    "Das Vermögen darf den Anfallberechtigten nicht vor dem Ablauf eines Jahres "
    "nach der Bekanntmachung der Auflösung des Vereins ausgeantwortet werden."
)
URL = "https://www.gesetze-im-internet.de/bgb/__51.html"


def hit():
    return Hit(
        Passage("p51", "s51", "BGB", "§ 51 Sperrjahr", TEXT, URL),
        0.5,
        TEXT,
    )


def standard(answer, citations=None, status="answered"):
    return json.dumps({
        "status": status,
        "answer": answer,
        "citations": ["S1"] if citations is None else citations,
    })


def claim_answer(
    quote=TEXT,
    source_id="S1",
    text="The waiting period is one year after public announcement",
    status="answered",
):
    return json.dumps({
        "status": status,
        "claims": [{
            "text": text,
            "source_id": source_id,
            "source_quote": quote,
        }] if status == "answered" else [],
    })


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, system, user, response_format=None):
        self.calls += 1
        if self.calls == 2:
            assert "SECOND-PASS STRUCTURED CLAIM REPAIR" in user
            assert "ON STRUCTURED CLAIM REPAIR" in system
            assert "source_quote" in user
        return self.replies.pop(0)


FIRST = standard(
    "The waiting period is one year after announcement. "
    "This is in the law [S1]."
)


def test_structured_claim_repair_can_render_one_cited_sentence():
    model = ScriptedLLM([FIRST, claim_answer()])
    answer, sources, metrics = generate(
        "How long is the waiting period?",
        [hit()], model, retry_on_failure=True, structured_claim_repair=True,
    )
    assert answer == "The waiting period is one year after public announcement [S1]."
    assert [s["url"] for s in sources] == [URL]
    assert metrics["first_pass_validation_error"] == "uncited_sentence"
    assert not metrics["abstained"]
    assert metrics["generation_attempts"] == 2


def test_wrong_quote_refuses_even_with_valid_source_id():
    model = ScriptedLLM([FIRST, claim_answer(quote="Invented words with no source match")])
    answer, sources, metrics = generate(
        "How long is the waiting period?",
        [hit()], model, retry_on_failure=True, structured_claim_repair=True,
    )
    assert sources == []
    assert metrics["abstained"]
    assert metrics["final_validation_error"] == "invalid_grounded_claims"
    assert "insufficient" in answer.lower()


def test_unknown_claim_source_fails_closed():
    model = ScriptedLLM([FIRST, claim_answer(source_id="S8")])
    _, sources, metrics = generate(
        "How long?", [hit()], model,
        retry_on_failure=True, structured_claim_repair=True,
    )
    assert sources == []
    assert metrics["abstained"]


def test_opt_in_never_changes_model_refusal():
    model = ScriptedLLM([
        standard("", citations=[], status="insufficient_evidence"),
        claim_answer(),
    ])
    _, sources, metrics = generate(
        "How long?", [hit()], model,
        retry_on_failure=True, structured_claim_repair=True,
    )
    assert sources == []
    assert metrics["abstained"]
    assert model.calls == 1


def test_default_retry_remains_original_schema():
    class DefaultModel(ScriptedLLM):
        def chat(self, system, user, response_format=None):
            if self.calls == 1:
                assert "answer" in response_format["properties"]
                assert "claims" not in response_format["properties"]
            self.calls += 1
            return self.replies.pop(0)

    model = DefaultModel([FIRST, standard("One year [S1].")])
    _, _, metrics = generate("How long?", [hit()], model, retry_on_failure=True)
    assert metrics["abstained"] is False
    assert model.calls == 2


def test_structured_claims_require_source_verbatim_quote():
    from legal_rag.generation import build_evidence

    evidence = build_evidence([hit()])
    valid = StructuredClaims.model_validate_json(claim_answer())
    assert render_grounded_claims(valid, evidence).citations == ["S1"]
    invalid = StructuredClaims.model_validate_json(
        claim_answer(quote="A fabricated waiting period exists here.")
    )
    import pytest
    with pytest.raises(ValueError, match="unverified_claim_quote"):
        render_grounded_claims(invalid, evidence)
