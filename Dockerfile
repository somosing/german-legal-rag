FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN uv sync --no-dev --no-install-project && uv sync --no-dev
ENV PATH="/app/.venv/bin:$PATH" RAG_STATE_DIR=/app/state
EXPOSE 8000
# Bind to a private interface on the HOST when running a Docker container.
CMD ["uvicorn", "legal_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
