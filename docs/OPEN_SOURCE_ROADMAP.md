# Open-source roadmap: German Legal RAG

**Goal:** a trustworthy *reference implementation* for bilingual German legal research and local, evidence-oriented document retrieval. We optimize for reproducibility, inspectability, and usefulness to people, not a hosted subscription business.

Two paths share the core engine:

- **Path A — German Law Explorer**: explore public BGB provisions, ask German/English questions, inspect actual cited source text, navigate related provisions, and understand when the model abstains.
- **Path B — Local Research Workspace**: compare public laws with user-supplied policy documents entirely in a controlled local workspace. **This is not implemented yet** and should not be offered for confidential files until isolation, deletion, and threat tests exist.

## Quality gates

1. **Source provenance:** every cited passage has an unambiguous source, section identity, URL, document version/snapshot and (where available) effective date. No automatic claim that consolidation text is authoritative promulgation.
2. **Claim support:** require more than citation token validity. Record which statements are supported, unsupported, unclear, or incompletely answered, with documented independent reviewer provenance.
3. **Reproducibility:** pin the corpus fingerprint, test fixture, embedding/generation models, configuration, code commit, hardware, latency, and metric denominators.
4. **Failure safety:** out-of-corpus queries, wrong-statute matches, malformed output and prompt injections must have regression tests. No silent citations to unrelated text.
5. **Usability:** a new visitor can run a fictional example and interpret the source links without special project knowledge.

## Milestones in recommended order

### M1 — Community foundation and reproducible baseline

- [ ] Concise accurate README, working demo commands, true CI badge and clear capability-versus-plan table.
- [ ] Contributor guidance, issue/PR templates, security reporting instructions and a public roadmap.
- [ ] Tagged release documenting a reproducible, checked baseline (model-independent tests and model-dependent evaluations distinguished).
- [ ] Read-only published *reviewed* example outputs with corpus hashes; do not commit private or unreviewed prompts as quality proof.

**Acceptance:** clean checkout runs tests offline; a new developer can reproduce the demo and determine which accuracy claims are actually measured.

### M2 — Path A: source-first German Law Explorer

- [ ] Clickable exact passages, source-side highlights and primary-law text alongside AI summary; direct official links.
- [ ] Clear source/version metadata and notice when the legal snapshot may be out of date.
- [ ] Better section/cross-reference selection, bilingual paraphrase retrieval, and independent retrieval/generation measurements.
- [ ] User-facing abstention explanations and accessibility keyboard/screen-reader audit.

**Acceptance:** a user can trace each substantive answer to the displayed statute excerpt and inspect the full provision; at least ten usability sessions document pain points and successes. No implied automated legal advice.

### M3 — Evidence quality and independent evaluation

- [ ] Create a held-out, reviewer-labelled dataset spanning German/English, complex conditions, contradictions, out-of-corpus laws, and adversarial text.
- [ ] Evaluate claim-level support, omitted qualifications, citation-source match, negative abstention, latency and failure types.
- [ ] Add regression gates for known failure classes (wrong statute, missing §12 conditions, missing §433 duties, §51 source excluded from prompt).
- [ ] Compare baseline, reranking, and optional experimental repairs on the **same** frozen snapshot; publish both successes and regressions.

**Acceptance:** reproducible report with provenance and independent review; no inflated legal accuracy claims based on small author-designed fixtures.

### M4 — Path B: private local research workspace (experimental)

- [ ] Define data boundaries and minimal privacy threat model before writing an upload endpoint.
- [ ] Import explicitly selected local documents (start with non-sensitive, fictional policies) into a separate indexed collection.
- [ ] Display law-versus-policy evidence with source identity; never present a document excerpt as a legal provision.
- [ ] Add deletion tests, document limits, safe parsing, isolation and redacted logs; block public network access.
- [ ] Consider authentication, dedicated Qdrant and deployment only after the single-user workflow is validated.

**Acceptance:** deterministic import/search/delete tests with no cross-collection citation leakage; security limitations documented. Not a production compliance tool.

### M5 — Packaging, community, and ecosystem

- [ ] Package versioned CLI examples, safe Docker Compose for local-only use, troubleshooting and reproducible demos.
- [ ] Create at least three reviewed beginner-friendly issues and accept external test/doc contributions.
- [ ] Produce a short demo video, real screenshots and architecture explanation; link outside recognition only after publication.
- [ ] Explore optional APIs or plugins only when they answer a demonstrated user need.

**Acceptance:** contributors can install, test, file reproducible bugs, and improve an isolated feature without guessing project conventions.

## Explicitly out of scope for now

Subscriptions/billing; legal predictions or automated binding advice; promising GDPR certification; public multi-user hosting; unreviewed bulk expansion to many statutes; fabricated evaluation scores; arbitrary automatic citations that bypass support checks.

## How to help

Start with [CONTRIBUTING.md](../CONTRIBUTING.md). Proposed beginner-friendly areas: UI accessibility, documentation/translation, parser regression tests, benchmark schema checks and reproducibility scripts. Discuss larger architectural changes in an issue first.
