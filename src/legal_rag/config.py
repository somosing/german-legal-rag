"""Validated local settings; .env is an example, not loaded automatically."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1 or value > 100:
        raise ValueError(f"{name} must be between 1 and 100")
    return value


@dataclass(frozen=True)
class Settings:
    state_dir: Path
    collection: str
    ollama_base_url: str
    llm_model: str
    embed_model: str
    rerank: bool
    rerank_model: str
    candidates: int
    final_k: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            state_dir=Path(os.getenv("RAG_STATE_DIR", "./state")).resolve(),
            collection=os.getenv("RAG_COLLECTION", "legal_passages"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            llm_model=os.getenv("OLLAMA_LLM_MODEL", "qwen3:4b-instruct"),
            embed_model=os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma"),
            rerank=os.getenv("RAG_RERANK", "false").lower() in {"1", "true", "yes"},
            rerank_model=os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            candidates=_env_int("RAG_CANDIDATES", 20),
            final_k=_env_int("RAG_FINAL_K", 4),
        )
