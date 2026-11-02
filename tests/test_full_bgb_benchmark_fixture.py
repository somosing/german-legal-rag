"""Static safeguards for the full-BGB pilot answer fixture."""

import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "examples/bgb_full_answer_questions.json"


def test_full_bgb_fixture_is_distinct_and_has_valid_splits():
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(cases) == 22
    assert len({c["question"] for c in cases}) == 22
    positives = [c for c in cases if c["expected_status"] == "answered"]
    negatives = [c for c in cases if c["expected_status"] == "insufficient_evidence"]
    assert len(positives) == 16
    assert len(negatives) == 6
    assert all(c["language"] in ("en", "de") for c in cases)
    assert all(
        c["gold_section_urls"]
        and all(url.startswith("https://www.gesetze-im-internet.de/bgb/__")
                for url in c["gold_section_urls"])
        for c in positives
    )
    assert all(
        c["absent_section_urls"]
        and all("/bgb/__" not in url for url in c["absent_section_urls"])
        for c in negatives
    )


def test_full_bgb_fixture_covers_all_seller_obligations():
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    seller = [
        c for c in cases
        if "https://www.gesetze-im-internet.de/bgb/__433.html"
        in c.get("gold_section_urls", [])
    ]
    assert len(seller) == 1
    assert {x["id"] for x in seller[0]["required_answer_concepts"]} == {
        "seller_handover",
        "seller_ownership_transfer",
        "seller_free_of_material_and_title_defects",
    }


def test_full_bgb_fixture_keeps_injunction_prerequisite():
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    name_right = [
        c for c in cases
        if "https://www.gesetze-im-internet.de/bgb/__12.html"
        in c.get("gold_section_urls", [])
    ]
    assert len(name_right) == 1
    assert {x["id"] for x in name_right[0]["required_answer_concepts"]} == {
        "future_interference_condition"
    }
