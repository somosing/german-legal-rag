# Benchmark Results

## Environment
- Local NVIDIA RTX 3060 Laptop GPU (6 GB)
- Ollama with Qwen3 4B Instruct
- EmbeddingGemma embeddings (768 dimensions)
- Qdrant local storage
- 100 indexed BGB sections, 128 passages

## Retrieval: 30 bilingual BGB questions

| Method | Hit@4 | MRR@4 |
|---|---:|---:|
| BM25 | 0.600 | 0.567 |
| Dense | 1.000 | 0.961 |
| Hybrid RRF | 0.900 | 0.764 |
| Hybrid Dense 3x | 0.900 | 0.783 |

## Answer evaluation: 12 questions

- Status accuracy: 12/12
- Citation consistency: 12/12
- Correct abstention on out-of-corpus questions: 6/6
- Suspected language mismatches: 0
- Median latency: 1.81 seconds
- P95 latency: 8.06 seconds

## Limitations

These are small, curated pilot benchmarks.
Citation consistency does not prove factual faithfulness.
Language detection is heuristic.
No production security or GDPR compliance certification
is claimed.
