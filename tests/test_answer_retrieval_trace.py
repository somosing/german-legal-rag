"""Benchmark records retrieval candidates even for model abstentions."""

import json
from types import SimpleNamespace

from legal_rag.answer_evaluation import evaluate_answers
from legal_rag.models import Section

BGB51 = "https://www.gesetze-im-internet.de/bgb/__51.html"


class AbstainingWithTrace:
    def __init__(self):
        self.retriever = SimpleNamespace(
            parents={
                "s51": Section("s51", "BGB", "§ 51", "One-year waiting period", BGB51)
            }
        )
        self.calls = 0

    def ask(self, query, mode="dense"):
        raise AssertionError("Benchmark should use the evaluation-only path")

    def ask_for_evaluation(self, query, mode="dense"):
        self.calls += 1
        return {
            "answer": "The sources are insufficient.",
            "sources": [],
            "metrics": {"abstained": True, "total_ms": 12},
            "retrieval_candidates": [{
                "rank": 1,
                "law": "BGB",
                "url": BGB51,
                "title": "§ 51",
                "passage_excerpt": "One year.",
            }],
        }


def test_failed_case_retains_original_retrieval_rank(tmp_path):
    fixture = tmp_path / "benchmark.json"
    fixture.write_text(json.dumps([{
        "question": "How long must the association wait?",
        "language": "en",
        "expected_status": "answered",
        "gold_section_urls": [BGB51],
    }]), encoding="utf-8")
    service = AbstainingWithTrace()
    result = evaluate_answers(service, fixture, tmp_path / "reports")
    case = result["cases"][0]
    assert service.calls == 1
    assert case["actual_status"] == "insufficient_evidence"
    assert case["retrieval_gold_rank"] == 1
    assert case["retrieval_candidates"][0]["url"] == BGB51
    assert result["summary"]["retrieval_gold_rate_of_traced_positives"] == 1.0
    assert result["summary"]["case_pass_rate"] == 0.0
