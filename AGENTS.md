# Implementation notes for Codex / Copilot contributors

Goal: a small, auditable, local-first bilingual legal-research RAG system.

## Constraints
- Do not add cloud LLM or telemetry services without explicit approval.
- Do not imply local inference alone establishes GDPR compliance.
- Do not claim a zero-hallucination guarantee, or report unmeasured accuracy improvements.
- Never embed private/user data into committed fixtures, logs, or tests.
- Preserve parent section IDs, source URLs, and supporting citations end-to-end.
- Preserve BM25-only, dense-only, hybrid baselines for honest ablations.
- Write offline tests for logic; avoid required model downloads inside GitHub Actions.
- Verify any official law text version and license; demo policies are fictional.

## Checks
`uv run ruff check .`
`uv run pytest -q`

## Suggested next steps
1. Add an optional Qdrant server backend and multi-worker support.
2. Add curated German BGB/DSGVO QA ground truth and failure cases.
3. Add locally judged RAGAS faithfulness only when model and judge are configured.
4. Load-test retrieval vs generation and publish measured p50/p95 latencies.
5. Add authentication, access control, retention policy, redaction, and threat-model tests before any production-like deployment.
