"""Experimental forward-neighbor mode preserves citations and source ordering."""

from legal_rag.generation import build_evidence
from legal_rag.models import Hit, Passage, Section
from legal_rag.retrieval import Retriever


def fake_retriever():
    inst = Retriever.__new__(Retriever)
    identifiers = ["49", "50", "50a", "51", "52"]
    inst.parents = {
        n: Section(n, "BGB", f"§ {n}", f"Paragraph {n}",
                   f"https://www.gesetze-im-internet.de/bgb/__{n}.html")
        for n in identifiers
    }
    inst.ordered_section_ids = list(inst.parents)
    inst.section_position = {
        sid: i for i, sid in enumerate(inst.ordered_section_ids)
    }
    inst.first_child = {
        n: Passage(n, n, "BGB", f"§ {n}", f"Paragraph {n}",
                   f"https://www.gesetze-im-internet.de/bgb/__{n}.html")
        for n in identifiers
    }
    return inst


def hit(retriever, number):
    return Hit(retriever.first_child[number], 0.3, retriever.parents[number].text)


def test_expansion_skips_50a_and_finds_51_after_two_steps():
    retriever = fake_retriever()
    original = [hit(retriever, "50")]
    results = retriever._expand_following_sections(original)
    assert [x.passage.title for x in results] == ["§ 50", "§ 50a", "§ 51"]
    assert original == [results[0]]
    assert len(original) == 1
    assert '/bgb/__51.html' in build_evidence(results)


def test_expansion_deduplicates_original_sections():
    retriever = fake_retriever()
    results = retriever._expand_following_sections([
        hit(retriever, "50"), hit(retriever, "51")
    ])
    ids = [x.passage.section_id for x in results]
    assert len(ids) == len(set(ids))


def test_neighborhood_does_not_cross_law_identity():
    retriever = fake_retriever()
    retriever.parents["52"] = Section(
        "52", "StGB", "§ 52", "Unrelated",
        "https://www.gesetze-im-internet.de/stgb/__52.html",
    )
    results = retriever._expand_following_sections([hit(retriever, "51")])
    assert [x.passage.title for x in results] == ["§ 51"]
