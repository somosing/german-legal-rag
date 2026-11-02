"""Orchestrate retrieval and generation with local timing metadata."""

from __future__ import annotations

from time import perf_counter

from legal_rag.config import Settings
from legal_rag.generation import generate
from legal_rag.ollama import Ollama
from legal_rag.retrieval import Retriever
from legal_rag.storage import Index


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = Ollama(settings.ollama_base_url, settings.embed_model, settings.llm_model)
        self.index = Index(settings, self.model)
        self.retriever = Retriever(self.index)

    def ask(self, query: str, mode: str = "dense") -> dict:
        return self._answer(query, mode, include_retrieval_trace=False)

    def ask_for_evaluation(self, query: str, mode: str = "dense") -> dict:
        """Keep the actual pre-generation hits in a local benchmark report only."""
        return self._answer(query, mode, include_retrieval_trace=True)

    def _answer(self, query: str, mode: str, *, include_retrieval_trace: bool) -> dict:
        if not query.strip() or len(query) > 800:
            raise ValueError("Query must contain 1 to 800 characters")
        start = perf_counter()
        hits, retrieval = self.retriever.search(query, mode=mode)
        answer, sources, generation = generate(
            query, hits, self.model, retry_on_failure=self.settings.retry_on_failure
        )
        result = {
            "answer": answer,
            "sources": sources,
            "metrics": {
                **retrieval,
                **generation,
                "total_ms": round((perf_counter() - start) * 1000, 2),
            },
            "model": self.settings.llm_model,
            "disclaimer": "Research aid only; not legal advice. Verify current official law.",
        }
        if include_retrieval_trace:
            result["retrieval_candidates"] = [
                {
                    "rank": rank,
                    "law": hit.passage.law,
                    "section_id": hit.passage.section_id,
                    "url": hit.passage.url,
                    "title": hit.passage.title,
                    "passage_excerpt": hit.passage.text[:850],
                }
                for rank, hit in enumerate(hits, start=1)
            ]
        return result
