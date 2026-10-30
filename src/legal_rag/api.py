"""Local FastAPI app. No accounts or public access control: never expose to internet."""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib.resources import files
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from legal_rag.config import Settings
from legal_rag.ollama import OllamaError
from legal_rag.service import RAGService

logger = logging.getLogger(__name__)
app = FastAPI(
    title="German Legal RAG",
    version="0.1.0",
    description="Local-first research demonstrator; not legal advice.",
)


class Question(BaseModel):
    query: str = Field(min_length=1, max_length=800)
    mode: Literal["hybrid", "hybrid_dense3", "bm25", "dense"] = "dense"


@lru_cache(maxsize=1)
def service() -> RAGService:
    return RAGService(Settings.from_env())


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(str(files("legal_rag").joinpath("static", "index.html")))


@app.get("/health")
def health() -> dict:
    try:
        current = service()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        return {"status": "not_ready", "reason": type(exc).__name__}
    return {
        "status": "ready",
        "collection": current.settings.collection,
        "model": current.settings.llm_model,
    }


@app.post("/ask")
def ask(question: Question) -> dict:
    try:
        return service().ask(question.query, question.mode)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaError as exc:
        logger.warning("Local model backend unavailable: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        logger.exception("Backend request failed")
        raise HTTPException(
            status_code=503, detail=f"Backend unavailable: {type(exc).__name__}"
        ) from exc
