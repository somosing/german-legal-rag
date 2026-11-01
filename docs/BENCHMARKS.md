# Benchmark Results

## Environment
- NVIDIA RTX 3060 Laptop GPU (6 GB)
- Ollama Qwen3 4B Instruct; EmbeddingGemma (768-dimensional embeddings)
- Qdrant local index: 100 BGB sections, 128 passages

## Retrieval (30 bilingual curated BGB questions)

| Mode | Hit@4 | MRR@4 |
| --- | ---: | ---: |
| BM25 | 0.600 | 0.567 |
| Dense | 1.000 | 0.961 |
| Hybrid RRF | 0.900 | 0.764 |
| Hybrid dense 3x | 0.900 | 0.783 |

## Answer contract (12 curated questions)

- Status accuracy: 12/12
- Citation-label consistency: 12/12
- Negative abstention rate for absent sections: 6/6
- Language mismatch flags: 0 (heuristic)
- Latency p50: 1.81 seconds; p95: 8.06 seconds

## Faithfulness / factual support

**Not yet measured.** The next workflow exports claims with source passages
for independent human reviewers. See [FAITHFULNESS_AUDIT.md](FAITHFULNESS_AUDIT.md).
A citation that points to a real section is not a guarantee that the answer
is fully entailed by that section.

## Limitations

Pilot-sized, author-designed evaluation; no blind expert validation or security
certification. These numbers are not production legal-accuracy guarantees.
The six negative answer-contract samples only work against the original
100-section index; full BGB indexing requires new negative labels.
