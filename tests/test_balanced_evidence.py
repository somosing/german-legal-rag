"""Fair-share evidence packing for experimental dense-neighbor retrieval."""

from legal_rag.generation import build_evidence
from legal_rag.models import Hit, Passage


def source(number: int, *, short: bool = False) -> Hit:
    text = (
        "The statutory waiting period is one year after publication. "
        if short
        else "Law text with many qualifying requirements. " * 35
    )
    parent = "Long parent context. " * 110
    return Hit(
        passage=Passage(
            id=f"p-{number}",
            section_id=f"s-{number}",
            law="BGB",
            title=f"§ {number} Source",
            text=text,
            url=f"https://www.gesetze-im-internet.de/bgb/__{number}.html",
        ),
        score=0.01,
        parent_text=parent,
    )


def test_twelve_sources_fit_under_cap_with_late_cited_rule():
    hits = [source(i, short=i == 51) for i in range(1, 13)]
    # Put the relevant short provision at the previously missing ninth rank.
    hits[8] = source(51, short=True)
    evidence = build_evidence(hits)

    assert len(evidence) <= 9000
    for i, hit in enumerate(hits, start=1):
        assert f'<source id="S{i}"' in evidence
        assert f'url="{hit.passage.url}"' in evidence
    assert "The statutory waiting period is one year after publication." in evidence
    assert 'id="S9"' in evidence


def test_four_hit_default_preserves_original_long_source_context():
    hits = [source(i) for i in range(1, 5)]
    evidence = build_evidence(hits)
    assert evidence.count("Long parent context.") > 50
    assert len(evidence) <= 9000


def test_small_cap_never_exceeds_bound():
    hits = [source(i) for i in range(1, 13)]
    assert len(build_evidence(hits, cap=1300)) <= 1300


def test_all_block_labels_remain_sequential_and_not_misattributed():
    hits = [source(i) for i in range(1, 13)]
    evidence = build_evidence(hits)
    for i, hit in enumerate(hits, start=1):
        begin = evidence.index(f'<source id="S{i}"')
        end = evidence.index("</source>", begin)
        block = evidence[begin:end]
        assert hit.passage.url in block
        assert f"§ {i} Source" in block
        assert f'url="{hits[i-1].passage.url}"' in block
