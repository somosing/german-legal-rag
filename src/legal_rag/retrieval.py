"""Independent lexical + semantic retrievers, RRF fusion, optional local cross-encoder."""

from __future__ import annotations

import math
import re
from time import perf_counter

from legal_rag.models import Hit, Passage
from legal_rag.storage import Index

TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked IDs, optionally weighting independent retrievers.

    RRF compares rank positions; its scores are not relevance probabilities.
    Keeping the default unweighted preserves the original benchmark baseline.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("weights must be finite and non-negative")
    if weights and not any(weight > 0 for weight in weights):
        raise ValueError("at least one weight must be positive")

    score: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item_id in enumerate(dict.fromkeys(ranking), start=1):
            score[item_id] = score.get(item_id, 0.0) + weight / (k + rank)
    return sorted(score.items(), key=lambda x: (-x[1], x[0]))


class Retriever:
    def __init__(self, index: Index):
        self.index = index
        self.passages, self.parents = index.load()
        from rank_bm25 import BM25Okapi
        self.lookup: dict[str, Passage] = {p.id: p for p in self.passages}
        if not self.passages:
            raise ValueError("Empty index")
        self.bm25 = BM25Okapi([tokenize(p.title + " " + p.text) for p in self.passages])
        self.reranker = None
        if self.index.settings.rerank:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("Install reranker with: uv sync --extra rerank") from exc
            self.reranker = CrossEncoder(self.index.settings.rerank_model)

    def _bm25(self, query: str, limit: int) -> list[str]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # Avoid irrelevant zero-score matches, especially for out-of-domain queries.
        ranked = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
        return [self.passages[i].id for i in ranked[:limit] if scores[i] > 0]

    def _dense(self, query: str, limit: int) -> list[str]:
        vector = self.index.embedder.embed([query])[0]
        client = self.index.client()
        try:
            result = client.query_points(
                collection_name=self.index.settings.collection,
                query=vector,
                limit=limit,
                with_payload=True,
            )
        finally:
            client.close()
        return [str(point.payload["id"]) for point in result.points if point.payload]

    def search(self, query: str, mode: str = "hybrid") -> tuple[list[Hit], dict]:
        if mode not in {"bm25", "dense", "hybrid", "hybrid_dense3"}:
            raise ValueError("mode must be bm25, dense, hybrid, or hybrid_dense3")
        start = perf_counter()
        limit = self.index.settings.candidates
        bm25_ids = self._bm25(query, limit) if mode != "dense" else []
        dense_ids = self._dense(query, limit) if mode != "bm25" else []
        # Controlled ablation: dense receives 3x BM25 weight in experimental mode.
        # Do not overwrite the original equal-weight hybrid baseline.
        weights = [1.0, 3.0] if mode == "hybrid_dense3" else None
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids], weights=weights)
        candidates = [
            Hit(
                passage=self.lookup[pid],
                score=score,
                parent_text=self.parents[self.lookup[pid].section_id].text,
            )
            for pid, score in fused
            if pid in self.lookup and self.lookup[pid].section_id in self.parents
        ]
        # Reranking only applies to hybrid queries to preserve pure baseline comparison.
        if self.reranker is not None and mode in {"hybrid", "hybrid_dense3"} and candidates:
            pairs = [(query, hit.passage.title + "\n" + hit.passage.text) for hit in candidates]
            scores = self.reranker.predict(pairs)
            candidates = sorted(
                [
                    Hit(h.passage, float(s), h.parent_text)
                    for h, s in zip(candidates, scores, strict=True)
                ],
                key=lambda h: -h.score,
            )
        # Return at most one child per parent section to reduce duplicated citations.
        seen_sections: set[str] = set()
        results = []
        for hit in candidates:
            if hit.passage.section_id not in seen_sections:
                seen_sections.add(hit.passage.section_id)
                results.append(hit)
            if len(results) >= self.index.settings.final_k:
                break
        timing = {
            "retrieval_ms": round((perf_counter() - start) * 1000, 2),
            "mode": mode,
            "reranked": self.reranker is not None and mode in {"hybrid", "hybrid_dense3"},
            "bm25_candidates": len(bm25_ids),
            "dense_candidates": len(dense_ids),
            "fusion_weights": {"bm25": weights[0], "dense": weights[1]}
            if weights is not None
            else {"bm25": 1.0, "dense": 1.0},
        }
        return results, timing
