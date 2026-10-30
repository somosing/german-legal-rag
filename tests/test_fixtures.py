import json
from pathlib import Path


def test_demo_gold_ids_exist_and_have_expected_size():
    base = Path(__file__).resolve().parents[1]
    sections = json.loads((base / "examples/demo_policies.json").read_text(encoding="utf-8"))
    questions = json.loads((base / "examples/demo_questions.json").read_text(encoding="utf-8"))
    ids = {section["id"] for section in sections}
    assert len(sections) == 8
    assert len(questions) == 30
    assert all(set(row["gold_section_ids"]) <= ids for row in questions)
    assert all(row["question"].strip() for row in questions)
