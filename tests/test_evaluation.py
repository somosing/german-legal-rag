import pytest

from legal_rag.evaluation import measure, resolve_gold_ids
from legal_rag.models import Section


def test_hit_mrr_recall():
    result = measure([["a", "b"], ["z", "c"]], [{"a"}, {"c"}], 2)
    assert result["hit_at_k"] == 1.0
    assert result["mrr_at_k"] == 0.75
    assert result["recall_at_k"] == 1.0
    assert result["query_count"] == 2


def test_empty_benchmark_rejected():
    with pytest.raises(ValueError):
        measure([], [], 4)


def test_gold_bgb_urls_resolve_to_index_section_ids():
    indexed = {
        "hashed-bgb-1": Section(
            id="hashed-bgb-1", law="BGB", title="§ 1 Beginn der Rechtsfähigkeit",
            text="Die Rechtsfähigkeit des Menschen beginnt mit der Vollendung der Geburt.",
            url="https://www.gesetze-im-internet.de/bgb/__1.html",
        )
    }
    resolved = resolve_gold_ids(
        {"question": "Wann?", "gold_section_urls": [indexed["hashed-bgb-1"].url]},
        indexed,
    )
    assert resolved == {"hashed-bgb-1"}


def test_nonindexed_bgb_url_is_error_not_silent_zero():
    with pytest.raises(ValueError, match="not in the current index"):
        resolve_gold_ids(
            {"question": "Wann?", "gold_section_urls": ["https://example.invalid/not-indexed"]},
            {},
        )


def test_synthetic_id_can_still_be_used():
    indexed = {"SYN-1": Section("SYN-1", "SYN", "t", "v", "https://example.invalid")}
    assert resolve_gold_ids({"gold_section_ids": ["SYN-1"]}, indexed) == {"SYN-1"}


def test_synthetic_id_from_wrong_index_is_rejected():
    with pytest.raises(ValueError, match="not in the current index"):
        resolve_gold_ids({"gold_section_ids": ["DEMO-1"]}, {})


def test_real_bgb_benchmark_end_to_end_without_models(tmp_path):
    import json
    from types import SimpleNamespace

    from legal_rag.evaluation import run_benchmark
    from legal_rag.models import Hit, Passage

    section = Section(
        "hashed-bgb-1", "BGB", "§ 1 Beginn der Rechtsfähigkeit",
        "Die Rechtsfähigkeit beginnt mit Vollendung der Geburt.",
        "https://www.gesetze-im-internet.de/bgb/__1.html",
    )
    hit = Hit(
        passage=Passage(
            "passage-1", section.id, "BGB", section.title, section.text, section.url
        ),
        score=0.9,
        parent_text=section.text,
    )

    class FakeRetriever:
        parents = {section.id: section}
        index = SimpleNamespace(settings=SimpleNamespace(final_k=4))
        reranker = None

        def search(self, query, mode):
            return [hit], {"retrieval_ms": 0.1, "mode": mode}

    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps([
        {"question": "Wann beginnt Rechtsfähigkeit?", "gold_section_urls": [section.url]}
    ]), encoding="utf-8")
    report = run_benchmark(FakeRetriever(), gold_path, tmp_path / "reports")
    assert report["dataset_type"] == "real_bgb_pilot"
    assert report["metrics"]["hybrid"]["mrr_at_k"] == 1.0
    assert (tmp_path / "reports" / "retrieval_report.md").is_file()


def test_bgb_report_includes_rank_and_failure_diagnostics(tmp_path):
    import json
    from types import SimpleNamespace

    from legal_rag.evaluation import run_benchmark
    from legal_rag.models import Hit, Passage

    relevant = Section(
        "relevant", "BGB", "§ 1", "Rechtsfähigkeit beginnt mit Geburt.",
        "https://www.gesetze-im-internet.de/bgb/__1.html",
    )
    wrong = Section(
        "wrong", "BGB", "§ 2", "Volljährigkeit.",
        "https://www.gesetze-im-internet.de/bgb/__2.html",
    )
    wrong_hit = Hit(
        Passage("wrong-p", wrong.id, "BGB", wrong.title, wrong.text, wrong.url),
        0.1, wrong.text,
    )

    class FakeRetriever:
        parents = {relevant.id: relevant, wrong.id: wrong}
        index = SimpleNamespace(settings=SimpleNamespace(final_k=4))
        reranker = None

        def search(self, query, mode):
            return [wrong_hit], {"mode": mode, "retrieval_ms": 1.0}

    sample_path = tmp_path / "pilot.json"
    sample_path.write_text(json.dumps([
        {"question": "Was besagt §1?", "gold_section_urls": [relevant.url]}
    ]), encoding="utf-8")
    report = run_benchmark(FakeRetriever(), sample_path, tmp_path / "results")
    assert report["queries"][0]["first_gold_rank"] is None
    assert report["queries"][0]["retrieved_section_urls"] == [wrong.url]
    markdown = (tmp_path / "results" / "retrieval_report.md").read_text()
    assert "Retrieved: https://www.gesetze-im-internet.de/bgb/__2.html" in markdown
    assert "bm25 — 1 missed questions" in markdown
