# Full-BGB answer-contract smoke benchmark

This benchmark is **separate** from the original 100-section pilot.
Use it only with the full BGB index created from the same source snapshot.
It is a preliminary engineering smoke test, **not** independent legal QA
validation, compliance evidence, or a blind benchmark.

## Corpus prerequisites

The local manifest must identify a BGB index with the expected source
fingerprint (2026-11-02 snapshot):

- source SHA-256: `34fc732d8b963eb753ed11236490d3c47c123ccb72b67c2fdda7f19544389e57`
- sections: 2517
- passages with the fixed 750-character chunker: 3624
- embedding model: `embeddinggemma`, dimension 768

The answer evaluator verifies that *positive* gold URLs exist in the loaded
index and that *negative* absent URLs do not. It does **not** verify source
version or ingestion completeness; check `manifest.json` yourself first.

## Run

```bash
cd ~/Projects/german-legal-rag
export RAG_STATE_DIR="$HOME/rag-indices/bgb-full"
cat "$RAG_STATE_DIR/manifest.json"

RAG_RETRY_ON_FAILURE=true uv run legal-rag eval-answers \
  --gold examples/bgb_full_answer_questions.json \
  --output reports/bgb_full_answers_v1 --mode dense

cat reports/bgb_full_answers_v1/answer_report.md
```

## Composition and interpretation

- **16 positive BGB questions**, in German and English: six previously
  answerable pilot cases, six formerly out-of-index cases now reclassified
  as positive, and four additional association/foundation questions.
- **Six out-of-corpus negative questions** about statutes *not* indexed:
  StGB, GDPR, ZPO, ArbZG, EStG, UStG. A negative label means the
  requested legal source is outside the index; the question has a real legal
  answer elsewhere.
- § 12 BGB requires mention of the future-interference condition when
  discussing injunctive relief.
- § 433 BGB requires **three distinct concept checks**: handover of the item,
  transfer of ownership, and freedom from material and title defects.

The `required_condition_match_rate` is **lexical**, not entailment.
It can accept negated or misleading statements and can reject correct
paraphrases. Citation consistency verifies source IDs, *not* whether
the cited text supports the actual answer. The fixture is deliberately
small and largely author-designed, so do not advertise a high pass rate
as independently verified legal accuracy.

Review the per-question report and run a claim-support audit. Record
provenance and uncertain labels. For a fair comparison, retain the
source snapshot, dataset version, embedding model, retrieval mode,
generation model, and decoding settings.

## Explicit statute identity guard

The generation step now checks named law identifiers in the query before it
constructs cited evidence. Recognized names: BGB, StGB, ZPO, ArbZG, EStG,
UStG, GDPR/DSGVO. The retrieved passage's `law` metadata, **not its
body or section number alone**, must match all named laws; otherwise the
system abstains (`requested_law_not_retrieved`) without calling Ollama.

This avoids the observed BGB §242 → StGB §242 and BGB §253 → ZPO §253
collisions and prevents answering GDPR Article 15 from BGB §79a. When
a question names multiple laws, the guard requires evidence from each.
It does not prove claim support or resolve every implicit statute reference,
nickname, cross-reference, or other unrecognized legislation.
