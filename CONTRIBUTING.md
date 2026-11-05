# Contributing to German Legal RAG

Thank you for helping improve this source-first, local-first RAG research project. Documentation, accessibility, bilingual examples, failure reproductions, tests, and careful engineering reviews are all useful contributions.

## Before opening a PR

1. Read the [README](README.md), [roadmap](docs/OPEN_SOURCE_ROADMAP.md), and [engineering rules](AGENTS.md).
2. Check existing issues and pull requests. For significant features or architecture changes, open an issue describing the user problem and trade-offs before coding.
3. Do **not** upload private legal documents, customer data, credentials, personal information, or unreviewed excerpts from proprietary corpora.
4. Avoid claiming the project provides legal advice, guarantees legal correctness, or satisfies GDPR merely because inference runs locally.

## Development environment

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and Git.

```bash
git clone https://github.com/somosing/german-legal-rag.git
cd german-legal-rag
uv sync --locked --extra dev
uv run ruff check .
uv run pytest -q
```

Those checks run without Ollama or downloading model weights. To test generation or retrieval end-to-end, install Ollama and pull the models described in the README. Keep model/network-dependent tests out of default CI. Do not replace a developer's local index with ingestion unless explicitly intended; use a separate `RAG_STATE_DIR`.

## How to submit useful changes

- Keep pull requests focused; use a descriptive title and explain **what problem** they solve.
- Add or update offline regression tests for changed behavior, including failure paths.
- Preserve law identity, original source URLs, citation contracts, and fail-closed behavior.
- For retrieval/generation claims, report corpus fingerprint, gold fixture, model, retrieval mode, cutoff, runtime environment, and measured values. Include regressions and abstentions, not only improvements.
- Label answer-quality judgements accurately: a structural citation pass is **not** legal entailment. Independent review is required to claim factual support.
- Use documentation examples from the fictional demo corpus or publicly reusable sources, with attribution and version/date notes.
- Add a PR description with a reproducible test command and known limitations. Use the provided PR template.

## Good first contributions

A typo or broken-link fix, a German/English translation correction, improved CLI help, accessibility checks for the local UI, or a small pure-Python parser regression test are good starting points.

Avoid beginning with authentication, production multi-tenancy, unsandboxed file upload, legal-advice features, or new compliance promises. These need design and threat review.

## Responsible disclosure

Potential security vulnerabilities should not be posted in public issues. See [SECURITY.md](SECURITY.md) for available private reporting guidance.

## Respectful collaboration

Discuss evidence and implementation choices respectfully; challenge conclusions and assumptions, not people. Feedback should be actionable, verifiable, and considerate of contributors' time.
