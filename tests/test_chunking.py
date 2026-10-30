from legal_rag.chunking import chunk_section
from legal_rag.models import Section


def test_chunking_keeps_paragraphs_and_parent_relation():
    section = Section(
        "id01", "DEMO", "A Title", "Alpha.\n\nBeta text.\n\nGamma.", "https://example.invalid"
    )
    result = chunk_section(section, max_chars=80)
    assert len(result) == 1
    assert result[0].text == section.text
    assert result[0].section_id == "id01"
    assert chunk_section(section, max_chars=80) == result


def test_long_paragraph_is_not_arbitrarily_sliced():
    section = Section("id02", "DEMO", "A Title", "X" * 1000, "https://example.invalid")
    result = chunk_section(section, max_chars=200)
    assert len(result) == 1
    assert len(result[0].text) == 1000
