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
        if not query.strip() or len(query) > 800:
            raise ValueError("Query must contain 1 to 800 characters")
        start = perf_counter()
        hits, retrieval = self.retriever.search(query, mode=mode)
        answer, sources, generation = generate(
            query, hits, self.model, retry_on_failure=self.settings.retry_on_failure
        )
        return {
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
