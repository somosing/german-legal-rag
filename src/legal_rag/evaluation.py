"""Deterministic, judge-free retrieval benchmark. Report only measured results."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from legal_rag.retrieval import Retriever
from legal_rag.storage import save_json


def measure(ranks: list[list[str]], gold: list[set[str]], k: int) -> dict:
    if not ranks or len(ranks) != len(gold):
        raise ValueError("Expected equally sized non-empty ranks and gold lists")
    if k < 1:
        raise ValueError("k must be positive")
    hits = []
    reciprocal_ranks = []
    recalls = []
    for predicted, relevant in zip(ranks, gold, strict=True):
        top = predicted[:k]
        hits.append(float(bool(set(top) & relevant)))
        reciprocal_ranks.append(
            next((1.0 / (idx + 1) for idx, item in enumerate(top) if item in relevant), 0.0)
        )
        recalls.append(len(set(top) & relevant) / len(relevant) if relevant else 0.0)
    return {
        "hit_at_k": round(sum(hits) / len(hits), 4),
        "mrr_at_k": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        "recall_at_k": round(sum(recalls) / len(recalls), 4),
        "query_count": len(ranks),
    }


def resolve_gold_ids(sample: dict, sections: dict) -> set[str]:
    """Resolve human-readable official URLs or legacy section IDs to indexed IDs.

    Fail explicitly when a benchmark uses a different corpus. Otherwise, a stale
    fixture could silently produce a misleading zero-recall result.
    """
    by_url = {section.url: section_id for section_id, section in sections.items()}
    if "gold_section_urls" in sample:
        urls = sample["gold_section_urls"]
        if not isinstance(urls, list) or not urls or any(not isinstance(url, str) for url in urls):
            raise ValueError("gold_section_urls must be a non-empty list of URLs")
        missing = sorted(set(urls) - by_url.keys())
        if missing:
            raise ValueError(f"Benchmark references sections not in the current index: {missing}")
        return {by_url[url] for url in urls}
    if "gold_section_ids" in sample:
        ids = sample["gold_section_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids):
            raise ValueError("gold_section_ids must be a non-empty list of section IDs")
        missing = sorted(set(ids) - sections.keys())
        if missing:
            raise ValueError(f"Benchmark references sections not in the current index: {missing}")
        return set(ids)
    raise ValueError("Each benchmark question needs gold_section_urls or gold_section_ids")


def run_benchmark(retriever: Retriever, gold_path: Path, out_dir: Path) -> dict:
    qa = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(qa, list) or not qa:
        raise ValueError("Benchmark must contain a non-empty list")
    if any(
        not isinstance(sample, dict)
        or not isinstance(sample.get("question"), str)
        or not sample["question"].strip()
        for sample in qa
    ):
        raise ValueError("Every benchmark entry must contain a non-empty question")
    gold_by_query = [resolve_gold_ids(sample, retriever.parents) for sample in qa]
    is_official_pilot = all("gold_section_urls" in sample for sample in qa)
    modes = ["bm25", "dense", "hybrid", "hybrid_dense3"]
    metrics = {}
    details = []
    for mode in modes:
        predictions = []
        gold_ids = []
        for sample, relevant in zip(qa, gold_by_query, strict=True):
            hits, timing = retriever.search(sample["question"], mode=mode)
            predicted = [h.passage.section_id for h in hits]
            predictions.append(predicted)
            gold_ids.append(relevant)
            first_gold_rank = next(
                (
                    idx
                    for idx, section_id in enumerate(predicted, start=1)
                    if section_id in relevant
                ),
                None,
            )
            details.append(
                {
                    "question": sample["question"],
                    "mode": mode,
                    "retrieved_section_ids": predicted,
                    "retrieved_section_urls": [retriever.parents[pid].url for pid in predicted],
                    "gold_section_ids": sorted(relevant),
                    "gold_section_urls": sample.get("gold_section_urls", []),
                    "first_gold_rank": first_gold_rank,
                    **timing,
                }
            )
        metrics[mode] = measure(predictions, gold_ids, retriever.index.settings.final_k)
    if is_official_pilot:
        limitations = [
            "Measures labelled retrieval only, not answer faithfulness.",
            "Results are a small pilot, not production accuracy or legal-reliability evidence.",
            "Questions overlap legal wording; harder paraphrases and reviewer checks are needed.",
            "No statistical significance is implied by differences on this sample.",
        ]
    else:
        limitations = [
            "Small fictional demonstration set; no real-world legal quality claim.",
            "These are retrieval metrics; faithfulness and hallucinations are not measured.",
            "No statistical significance is implied by differences on this sample.",
        ]
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "gold_file": str(gold_path),
        "reranker_enabled": retriever.reranker is not None,
        "dataset_type": "real_bgb_pilot" if is_official_pilot else "synthetic_demo",
        "metrics": metrics,
        "queries": details,
        "limitations": limitations,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "retrieval_report.json", report)
    lines = [
        "# Retrieval benchmark",
        "",
        f"Generated: {report['created_utc']}",
        "",
        "| Mode | Hit@k | MRR@k | Recall@k | Queries |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, result in metrics.items():
        lines.append(
            f"| {mode} | {result['hit_at_k']:.3f} | {result['mrr_at_k']:.3f} | "
            f"{result['recall_at_k']:.3f} | {result['query_count']} |"
        )
    lines.extend(["", "## Retrieval misses (not present in top-k)", ""])
    for mode in modes:
        misses = [
            entry for entry in details
            if entry["mode"] == mode and entry["first_gold_rank"] is None
        ]
        lines.append(f"### {mode} — {len(misses)} missed questions")
        lines.append("")
        if not misses:
            lines.append("No missed sections in the evaluated top-k.")
        else:
            for entry in misses:
                lines.append(f"- **Question:** {entry['question']}")
                expected = entry["gold_section_urls"] or entry["gold_section_ids"]
                lines.append(f"  - Expected: {', '.join(expected)}")
                retrieved = entry["retrieved_section_urls"] or ["(no results)"]
                lines.append(f"  - Retrieved: {', '.join(retrieved)}")
        lines.append("")
    lines.extend(["## Limitations", "", *[f"- {s}" for s in report["limitations"]]])
    (out_dir / "retrieval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
