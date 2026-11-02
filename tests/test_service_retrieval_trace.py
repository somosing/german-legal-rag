"""Evaluation-only retrieval trace must not leak through ordinary answers."""

from types import SimpleNamespace

from legal_rag.models import Hit, Passage
from legal_rag.service import RAGService


def test_service_trace_is_only_available_for_benchmark(monkeypatch):
    import legal_rag.service as service_module

    passage = Passage(
        "p1", "section-51", "BGB", "§ 51", "Eine Jahresfrist gilt.",
        "https://www.gesetze-im-internet.de/bgb/__51.html",
    )
    hit = Hit(passage, 0.4, "Section context.")

    class StubRetriever:
        def __init__(self):
            self.calls = 0

        def search(self, query, mode="dense"):
            self.calls += 1
            return [hit], {"retrieval_ms": 2.0}

    retriever = StubRetriever()
    service = RAGService.__new__(RAGService)
    service.settings = SimpleNamespace(retry_on_failure=False, llm_model="fake")
    service.retriever = retriever
    service.model = object()

    def fake_generate(query, hits, model, *, retry_on_failure=False):
        assert hits == [hit]
        return (
            "No evidence found.",
            [],
            {"abstained": True, "abstention_reason": "model_insufficient_evidence"},
        )

    monkeypatch.setattr(service_module, "generate", fake_generate)

    normal = service.ask("How long is the period?")
    assert "retrieval_candidates" not in normal

    traced = service.ask_for_evaluation("How long is the period?")
    assert retriever.calls == 2
    assert traced["metrics"]["abstained"]
    assert traced["retrieval_candidates"] == [{
        "rank": 1,
        "law": "BGB",
        "section_id": "section-51",
        "url": "https://www.gesetze-im-internet.de/bgb/__51.html",
        "title": "§ 51",
        "passage_excerpt": "Eine Jahresfrist gilt.",
    }]

