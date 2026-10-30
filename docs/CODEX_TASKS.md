# Codex tasks for iterative development

Open this directory in your terminal, install [Codex CLI](https://learn.chatgpt.com/docs/codex/cli) and run `codex`. Start with `AGENTS.md` as project constraints.

## Prompt 1 — implementation audit

> Read `AGENTS.md`, `README.md` and the complete `src/legal_rag/` package. Inspect the design for correctness, privacy risk, dependency compatibility and edge cases. Do not change files yet. Produce a prioritized bug list referencing exact functions. Do not claim the end-to-end LLM was tested unless you actually run the local models.

## Prompt 2 — parsing robustness

> Improve `parse_bgb_xml` for nested lists, tables and amendment footnotes. Keep each paragraph/section linked to an accurate source URL. Add safe XML fixtures and offline tests for missing sections, nested lists and malformed documents. Keep output backwards compatible and report any structure the parser still misses. Do not invent statute text.

## Prompt 3 — index atomicity and migration

> Refactor `Index.build` to stage updates safely so a failed Ollama embedding or interrupted import does not leave the previous index unusable. Choose Qdrant aliases or another supported replacement mechanism after verifying the installed client API. Add failure-injection tests. Explain restart and concurrency assumptions for local Qdrant.

## Prompt 4 — evaluation study

> Create a reproducible evaluation harness for a reviewer-validated 30+ question BGB dataset. Keep BM25, dense and hybrid search independently measurable. Implement a result table and confidence-aware interpretation without invented claims. Compare raw retrieval and reranking with identical corpus, model and cutoff. Add exact commands to README.

## Prompt 5 — GPU performance

> On this Ubuntu laptop with NVIDIA RTX 3060 6 GB VRAM, profile end-to-end latency and peak VRAM for Ollama's Qwen3 4B generation, EmbeddingGemma vectors and optional BGE reranking. Add a benchmark runner with p50 and p95; avoid printing raw queries or documents into logs. Do not infer results without executing the benchmark.

## Prompt 6 — production-gap review

> Compare this prototype against a deployment with sensitive corporate documents. Build a concise threat model covering PII, prompt injection, auth, tenant isolation, local inference egress, retention/deletion, versioned legal sources, observability and service recovery. Implement one improvement per PR with tests and documented trade-offs. Never call it GDPR-certified without a suitable assessment.

## Development discipline

Before each Codex change: `git status --short`, checkpoint with `git add . && git commit -m "checkpoint"`, ask for one narrowly scoped change, then review `git diff` and rerun `uv run ruff check . && uv run pytest -q`. Only commit claims that logs or measurements demonstrate.
