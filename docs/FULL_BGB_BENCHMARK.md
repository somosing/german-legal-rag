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

## Retrieval-versus-generation diagnosis

Recent versions of `eval-answers` add a benchmark-only trace of the
**actual retrieval candidates** for each query, including ones that later
abstain and emit no citations. This uses the same retrieval call that
fed answer generation, not a second potentially different query.

In `answer_report.md`, inspect the gold-section retrieved rank and
candidate URLs for failed positive cases such as §12 and §51. The JSON
also includes title, law metadata, and a short passage excerpt for
each candidate. The trace represents **pre-filter retrieval hits**,
not necessarily the evidence passed into Ollama after statute
scoping and context-window limits.

If the gold source is absent, investigate retrieval. If it is
present but the model abstained, inspect the generation contract,
evidence completeness, and any validation error separately. Never
interpret structural citations as factual entailment.

The trace can contain private retrieved text in future datasets,
so these reports are local-only, gitignored, and must not be shared
unredacted for sensitive corpora. The ordinary `ask`/API path does
**not** expose the retrieval trace.

## Coverage-oriented prompt experiment

Generation now differentiates concise one-fact questions from open-ended
requests for obligations/requirements and conditional legal remedies.
Enumerative questions explicitly request each relevant source-grounded
requirement (without mixing parties), rather than compressing the first
obligation into a one-sentence answer. On validation failure, a narrowly
audited §12 prerequisite retry highlights the actual source wording instead
of repeating the generic shortest-answer directive.

This is a **prompt-only behavior experiment**, not automatic coverage or
entailment verification. Local Qwen responses may still omit duties or
abstain. Compare the same 22-case dataset, source snapshot, model, and
retrieval mode to the v3 results, paying particular attention to §433 and
§12 and retaining all six negative abstentions. Do not reinterpret a lexical
concept match as semantic correctness.

## Experimental forward-neighbor retrieval (not baseline)

The opt-in `dense_neighbors` mode starts from the standard dense top four
sections and includes **up to two following sections in XML source order**
after each seed. This makes §51 BGB discoverable from a retrieved §50 BGB,
despite intervening §50a. The neighboring sections keep their own section
URLs and citation labels, and no existing index is rebuilt.

This is an **ablation**, not a recommended default. Adjacent provisions can
be irrelevant; the current generator may drop later hits at the 9,000-character
evidence cap. Compare retrieval trace and exact answer quality and do not
interpret a higher section-recall score as legal correctness.

```bash
export RAG_STATE_DIR="$HOME/rag-indices/bgb-full"
uv run legal-rag ask "How long is the waiting period for distributing the property of a dissolved association?" --mode dense_neighbors
RAG_RETRY_ON_FAILURE=true uv run legal-rag eval-answers \
  --gold examples/bgb_full_answer_questions.json \
  --output reports/bgb_full_dense_neighbors_v1 --mode dense_neighbors
```

## Balanced neighbor evidence experiment

The context builder previously packed each retrieval result with up to
1050 characters of child text and 1450 parent characters, and stopped at
9000 total. This meant §51 BGB was *retrieved* in the neighbor experiment
but excluded from the text sent to Ollama.

When more than four results are present, the new experimental packing
divides the 9000-character budget across all hits, prioritizing each
hit's own passage text and allocating remaining space to its parent.
The standard four-hit retrieval path retains its original rich context.

This prevents early long documents from consuming all the character budget
but may truncate important legal qualifications and introduce more
irrelevant sources. It is *not* a semantic evidence selector or token
budget guarantee. Check whether §51 is present in `build_evidence(hits)`,
whether Qwen cites the correct law, and whether the entire 22-question
benchmark holds up. This policy remains experimental until tested.
