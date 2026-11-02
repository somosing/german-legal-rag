"""Statute identity must be verified before any legal answer is generated."""

import json

import pytest

from legal_rag.generation import (
    generate,
    requested_laws,
    scope_explicit_law_evidence,
)
from legal_rag.models import Hit, Passage


def hit(law: str, number: str, *, title: str = "Statutory text") -> Hit:
    return Hit(
        passage=Passage(
            id=f"{law}-{number}",
            section_id=f"section-{law}-{number}",
            law=law,
            title=title,
            text="This text is from the indexed statute.",
            url=f"https://www.gesetze-im-internet.de/{law.lower()}/__{number}.html",
        ),
        score=0.5,
        parent_text="This text is from the indexed statute.",
    )


class StubLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, system, user, response_format=None):
        self.calls += 1
        return json.dumps({
            "status": "answered",
            "answer": "The provision answers the question [S1].",
            "citations": ["S1"],
        })


@pytest.mark.parametrize(
    ("question", "source_law", "source_number"),
    [
        ("Welche Strafe droht für Diebstahl nach § 242 StGB?", "BGB", "242"),
        ("What information must be provided under Article 15 GDPR?", "BGB", "79a"),
        ("Welche Anforderungen stellt § 253 ZPO?", "BGB", "253"),
        ("What does section 3 ArbZG establish?", "BGB", "3"),
        ("Wer ist nach § 1 EStG steuerpflichtig?", "BGB", "1"),
        ("What does section 19 UStG provide?", "BGB", "19"),
    ],
)
def test_foreign_law_cannot_be_answered_with_bgb_sources(
    question, source_law, source_number
):
    llm = StubLLM()
    answer, sources, metrics = generate(
        question, [hit(source_law, source_number)], llm,
        retry_on_failure=True
    )
    assert metrics["abstained"] is True
    assert metrics["abstention_reason"] == "requested_law_not_retrieved"
    assert metrics["generation_attempts"] == 0
    assert not sources
    assert llm.calls == 0
    assert answer


def test_law_identity_not_inferred_from_passage_title_or_body():
    fake = hit("BGB", "242", title="§ 242 StGB Theft")
    assert scope_explicit_law_evidence("Was sagt § 242 StGB?", [fake]) == []


def test_law_filter_keeps_only_matching_sources():
    llm = StubLLM()
    answer, sources, metrics = generate(
        "What does section 242 StGB regulate?",
        [hit("BGB", "242"), hit("StGB", "242")],
        llm,
    )
    assert not metrics["abstained"]
    assert llm.calls == 1
    assert len(sources) == 1
    assert sources[0]["url"].endswith("/stgb/__242.html")
    assert "[S1]" in answer


def test_bgb_section_guard_still_applies_after_statute_filter():
    llm = StubLLM()
    _, sources, metrics = generate(
        "Was steht in § 433 BGB?",
        [hit("BGB", "53")],
        llm,
    )
    assert metrics["abstained"] is True
    assert metrics["abstention_reason"] == "requested_bgb_section_not_retrieved"
    assert sources == []
    assert llm.calls == 0


def test_multiple_explicit_laws_require_all_source_types():
    question = "Compare BGB section 242 with StGB section 242."
    assert requested_laws(question) == {"bgb", "stgb"}
    assert scope_explicit_law_evidence(question, [hit("BGB", "242")]) == []


def test_gdpr_dsgvo_aliases_are_equivalent():
    assert requested_laws("Was regelt die DSGVO?") == {"gdpr"}
    assert requested_laws("What is Article 15 GDPR?") == {"gdpr"}
    assert requested_laws("Was gilt nach der DS-GVO?") == {"gdpr"}
    assert scope_explicit_law_evidence(
        "Article 15 GDPR", [hit("DSGVO", "15")]
    ) is not None


def test_unscoped_question_preserves_normal_retrieval():
    assert scope_explicit_law_evidence(
        "Who represents an association?", [hit("BGB", "26")]
    ) is None
