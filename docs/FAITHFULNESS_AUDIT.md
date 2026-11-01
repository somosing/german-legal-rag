# Claim-support audit with explicit review provenance

The retrieval and answer-contract benchmarks **do not establish faithfulness**.
A valid `[S1]` token only proves that the answer references an available source ID.
This workflow exports answered claims with the cited German passages for **independent human review**, then calculates a score only on explicitly reviewed claims.

## 1. Reproduce the limited-corpus benchmark

With the API stopped and **the same 100-section BGB index** used for the published v0.1.0 pilot:

```bash
uv run legal-rag eval-answers \
  --gold examples/bgb_answer_questions.json \
  --output reports/bgb_answers_review --mode dense
```

**Do not reuse that exact fixture after indexing the full BGB.** Its six negative questions concern provisions deliberately missing from the original 100-section subset; they will no longer be valid negative examples. Create a new benchmark with genuinely unanswerable, contradictory, or out-of-scope cases and independently verified labels for the expanded corpus.

## 1a. Required-condition checks in answer benchmarks

Positive cases may define `required_answer_concepts` (each with an `id`
and nonempty `any_of` alternatives). The § 12 BGB question requires an
explicit reference to the risk of further interference, or the established
term `Wiederholungsgefahr`. A response that cites § 12 but does not mention
that condition fails the answer-contract case.

The `required_condition_match_rate` is a **curated lexical smoke test**,
not an entailment metric: it can miss valid paraphrases and can accept
negated or otherwise inaccurate statements containing the same words.
Never present this as verified legal correctness. Human review is required.

## 2. Export reviewer claims and evidence

```bash
uv run legal-rag audit-export \
  --report reports/bgb_answers_review/answer_report.json \
  --output reports/faithfulness/review.json
```

The JSON contains one or more rough sentence-level claim units per answered response, plus the cited source's URL, retrieved passage, and truncated parent context. Abstentions are excluded because their correctness is evaluated separately. **Human reviewers must check every factual assertion in each unit**; sentence boundaries may group multiple assertions.

Before scoring, set the top-level `review_method` to one of:

- `ai_assisted_preliminary` for machine-generated or AI-assisted labels (including
  labels produced by a chat assistant and copied into the file).
- `independent_human_review` **only after** a human reviewer independently
  checks the claims against the evidence.
- `not_recorded` if review provenance cannot be established (the export default).

The reported method is self-declared, not independently verified by the software.
**Review coverage of 100% is not the same as independent human review.**

For each `claims[]` entry, set:

- `verdict` to `supported`, `unsupported`, `unclear`, or leave `unreviewed`.
- For `supported`, set `supporting_source_id` to a source in `cited_source_ids` and `evidence_quote` to an **exact quotation** from its `passage` or `parent_excerpt`. Read the official provision too; a quotation alone is not proof of entailment.
- For `unsupported` or `unclear`, write a nonempty `reviewer_note` explaining why.
- Optionally use `reviewer_note` for supported judgments and include legal-version caveats.

A reviewer who can read German legal text should verify the evidence. Prefer a second reviewer for disputed claims and record their disagreements separately. Do not invent verdicts or pre-mark all claims supported.

## 3. Calculate reviewed support

```bash
uv run legal-rag audit-score \
  --annotations reports/faithfulness/review.json \
  --output reports/faithfulness/score

cat reports/faithfulness/score/faithfulness_review_report.md
```

The scorer validates cited source IDs and exact source quotations for `supported` claims. It reports `review_coverage`, counts of `supported`, `unsupported`, and `unclear`, and `supported_rate_among_reviewed` (null until at least one claim has been reviewed). **An unchanged export gives 0% coverage, not a fabricated 100% faithfulness score.** The quote check proves string provenance only, never semantic entailment; a human must make the truthfulness judgment.

Generated reports and annotations are gitignored. Do not submit sensitive material or individual question/answer logs without data-retention review. Publish aggregate scores only after inspection and clear disclosure of sample size, reviewer process, corpus snapshot, and limitations.

## Specific legal prerequisite guard (pilot)

A separate generation safeguard is scoped to an **injunction question that retrieves
§ 12 BGB**. It activates only when the actual retrieved text contains the exact
statutory clause about further interference. The answer prompt highlights that
source clause, and post-generation validation refuses or optionally repairs
answers omitting the future-interference condition or failing to cite the source.

This rule is narrow and intentionally transparent. It is not a general legal
entailment model and does not ensure that all other legal prerequisites or
exceptions are preserved. Its term check can accept a negated or otherwise
misleading statement; a human reviewer must still evaluate each assertion.
If the section cannot be retrieved, the existing exact-section guard takes
priority. Re-run the positive and negative evaluations before enabling
the optional bounded repair in deployment.

## 4. Expand the BGB corpus carefully

Stop the API, then inspect the full BGB without overwriting Qdrant:

```bash
uv run legal-rag ingest --source bgb --dry-run
```

This fetches the published BGB XML from the official public-law service, parses all supported sections, calculates a source SHA-256, and prints section/passage counts. Save a copy of the XML if you want an exact reproducible snapshot; pass it with `--input-xml path/to/bgb.xml`.

When ready to rebuild the index (this replaces the 100-section index and could take significant time on a laptop):

```bash
uv run legal-rag ingest --source bgb
```

Check `state/manifest.json` for section count, source fingerprint and model. Important limitations: the parser does not fully preserve complex statutory tables/footnotes, indexing is not atomic across all local stores, and there is no version-aware statutory effective-date handling yet. Benchmarks must be regenerated and relabelled for the expanded corpus.
