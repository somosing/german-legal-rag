from legal_rag.retrieval import reciprocal_rank_fusion, tokenize


def test_german_tokenization():
    assert tokenize("Datenschutz-Grundverordnung (§ 13) über Daten") == [
        "datenschutz", "grundverordnung", "13", "über", "daten"
    ]


def test_rrf_favors_agreement_between_retrievers():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]])
    assert scores[0][0] == "b"
    assert scores[0][1] > scores[-1][1]


def test_rrf_removes_duplicate_ids_per_retriever():
    assert len(reciprocal_rank_fusion([["a", "a", "b"]])) == 2


def test_weighted_rrf_changes_order_without_mutating_baseline():
    rankings = [["a"], ["z"]]
    assert reciprocal_rank_fusion(rankings)[0][0] == "a"
    assert reciprocal_rank_fusion(rankings, weights=[1, 3])[0][0] == "z"
    assert reciprocal_rank_fusion(rankings)[0][0] == "a"


def test_weighted_rrf_rejects_invalid_weights():
    import pytest

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[-1])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[float("nan")])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[0])
