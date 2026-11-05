# Security policy

## Status and supported use

German Legal RAG is a research demonstrator without production access controls, authentication, authorization, multi-tenant isolation, or a completed security audit. It should be run only on a trusted host, bound to localhost. **Do not expose the included FastAPI instance to the public internet** or upload sensitive client documents.

Security fixes for the latest `main` branch are reviewed on a best-effort basis. No formal support or response-time guarantee is provided.

## Reporting vulnerabilities

If GitHub private vulnerability reporting is enabled for this repository, use the repository's **Security → Report a vulnerability** feature. If it is not enabled, contact the maintainer privately through the contact details available on their [GitHub profile](https://github.com/somosing) before posting exploit details. Do **not** disclose secrets, private data, working exploit payloads, or vulnerable hosts in a public issue.

Include the affected commit/version, safe reproduction steps with *synthetic* data, impact, and potential mitigation. Please allow time to evaluate a report before public disclosure.

For non-sensitive defects, open a normal GitHub issue.

## Known limitations

- The FastAPI demo is unauthenticated. Inference and persistence are local by default, but download, OS, network, telemetry, and access configuration remain the operator's responsibility.
- Local Qdrant persistence is single-process and not a multi-tenant secure storage system.
- Citation structure and strict abstention do not prove legal accuracy or resistance to prompt injection.
- Uploaded private-document workspaces, retention/deletion guarantees, and secure sharing are **not implemented**.
- Downloaded laws, model weights, and third-party dependencies have separate licensing and provenance requirements.

Security features and safe deployment requirements will be tracked in [the open-source roadmap](docs/OPEN_SOURCE_ROADMAP.md).
