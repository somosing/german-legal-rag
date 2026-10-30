"""Small, explicit Ollama HTTP adapter; no cloud API or framework dependency."""

from __future__ import annotations

import httpx


class OllamaError(RuntimeError):
    """Local model service unavailable or returned an invalid response."""


class Ollama:
    def __init__(self, base_url: str, embed_model: str, llm_model: str):
        self.base_url = base_url
        self.embed_model = embed_model
        self.llm_model = llm_model

    def _post(self, route: str, payload: dict, timeout: float = 120.0) -> dict:
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                response = client.post(f"{self.base_url}{route}", json=payload)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaError(
                "Ollama request failed. Check 'ollama serve', pulled models, and OLLAMA_BASE_URL."
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._post(
            "/api/embed", {"model": self.embed_model, "input": texts, "truncate": False}
        )
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError("Invalid embedding response from Ollama")
        if any(not v or not all(isinstance(x, (float, int)) for x in v) for v in embeddings):
            raise OllamaError("Empty or nonnumeric embedding vector")
        if len({len(v) for v in embeddings}) != 1:
            raise OllamaError("Embedding dimensions differ within batch")
        return embeddings

    def chat(self, system: str, user: str, response_format: dict | None = None) -> str:
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 8192},
        }
        if response_format is not None:
            payload["format"] = response_format
        data = self._post("/api/chat", payload)
        message = data.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaError("Invalid chat response from Ollama")
        return content.strip()
