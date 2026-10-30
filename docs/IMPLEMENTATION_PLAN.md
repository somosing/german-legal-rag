# Implementation and learning plan

A short, evidence-first roadmap for presenting this as an engineering project, not a prompt-only demo.

## Suggested reading order (first session)

1. `models.py` – know the typed Section/Passage/Hit contracts.
2. `sources.py` and `chunking.py` – know which exact source text is indexed and why.
3. `storage.py` – know how Qdrant stores dense vectors and where lexical corpus lives.
4. `retrieval.py` – explain BM25 vs semantic search, RRF, reranking and deduplication.
5. `generation.py` – explain grounding, abstention and why citations alone cannot prove truth.
6. `evaluation.py` – explain Hit@k, MRR@k, Recall@k and a paired ablation.
7. `api.py`/`static/index.html` – explain serving architecture and its authentication limitation.

## Milestone 1 — local smoke check

- `uv sync --extra dev`, `ollama pull embeddinggemma`, `ollama pull qwen3:4b-instruct`.
- Run `ollama serve` if the service isn't already running.
- `uv run legal-rag ingest --source demo`.
- `uv run legal-rag ask "Wie lange werden technische Testprotokolle gespeichert?"`.
- Validate that a known citation appears and the backing source is from the fictional test corpus.
- Run `uv run uvicorn legal_rag.api:app --host 127.0.0.1 --port 8000` and open the UI.

**Gate:** an evidenced response with at least one valid `[S#]` reference and local HTTP result.

## Milestone 2 — retrieval experiment

- `uv run legal-rag eval --gold examples/demo_questions.json`.
- Record and compare Hit@4, MRR@4 and Recall@4 for lexical, dense and hybrid modes.
- Document failed and ambiguous questions; do not claim accuracy improvements if results are tied.
- Once stable, add a separate reviewer-labelled corpus of >=30 *real* BGB questions.

**Gate:** a committed evaluation report with corpus hash, model names, cutoff and retrieval configuration.

## Milestone 3 — statutory data

- Stop API, index `uv run legal-rag ingest --source bgb --limit 150`.
- Compare section counts against source XML and inspect a few random source URLs.
- Implement robust parser handling for lists, tables, amendments and provisions without `<P>` tags.
- Record download date and effective date metadata. Discuss absence of authoritative electronic promulgation in the sampled XML.

**Gate:** cited answers can be traced back to current public source text; parsing coverage measured.

## Milestone 4 — reranking and evaluation

- `uv sync --extra dev --extra rerank` and `RAG_RERANK=true`.
- Benchmark without and with `BAAI/bge-reranker-v2-m3` on the same real gold queries.
- Measure peak RAM/VRAM, p50 and p95 latency, MRR and recall, and store environment information.
- Optionally add RAGAS with a *local* judge model, calibrated against manually labelled examples.

**Gate:** all claims reported with raw numbers, hardware, measurements, corpus IDs and limitations.

## Milestone 5 — enterprise hardening

- Server Qdrant backend, authentication and roles, access audit, input/output filters, source versioning.
- Rate limits, request size limits, max tokens, structured/redacted telemetry, red-team prompt injections.
- Threat model and a privacy impact assessment with domain experts, if appropriate.

**Gate:** can explain privacy controls, failure modes, upgrade policy and why local-only != automatically GDPR-compliant.

## Interview questions to prepare

- Why does exact legal language sometimes favor BM25 over embeddings?
- What does Reciprocal Rank Fusion solve, and what does it discard?
- How do you prevent parent context from losing legal definitions?
- How do you evaluate retrieval independently of answer generation?
- What does the system do when no relevant source exists, or when sources conflict?
- Which citation quality claims can be verified deterministically, and which require a human or local judge?
- Which controls are missing for an actual deployment handling sensitive documents?
