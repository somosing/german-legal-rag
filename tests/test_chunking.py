from legal_rag.chunking import chunk_section
from legal_rag.models import Section


def section(text: str, title: str = "A Title") -> Section:
    return Section("id02", "BGB", title, text, "https://example.invalid/bgb/__309.html")


def test_chunking_keeps_paragraphs_and_parent_relation():
    sample = Section(
        "id01", "DEMO", "A Title", "Alpha.\n\nBeta text.\n\nGamma.",
        "https://example.invalid",
    )
    result = chunk_section(sample, max_chars=80)
    assert len(result) == 1
    assert result[0].text == sample.text
    assert result[0].section_id == "id01"
    assert chunk_section(sample, max_chars=80) == result


def test_long_unbroken_paragraph_splits_without_losing_any_character():
    text = "X" * 1000
    result = chunk_section(section(text), max_chars=200)
    assert len(result) == 5
    assert "".join(p.text for p in result) == text
    assert all(0 < len(p.text) <= 200 for p in result)


def test_long_legal_paragraph_prefers_sentence_boundaries():
    text = (
        "Bei der Anwendung ist der konkrete Tatbestand zu prüfen. "
        "Weitere Voraussetzungen ergeben sich aus dem Gesetz. "
    ) * 95
    text = text.strip()
    result = chunk_section(section(text), max_chars=750)
    assert len(result) > 5
    assert "".join(p.text for p in result) == text
    assert all(0 < len(p.text) <= 750 for p in result)
    assert any(p.text.endswith(". ") for p in result[:-1])


def test_oversized_paragraph_without_sentences_splits_on_whitespace():
    text = ("Anwendungsbereich " * 120).strip()
    result = chunk_section(section(text), max_chars=100)
    assert "".join(p.text for p in result) == text
    assert all(len(p.text) <= 100 for p in result)
    assert all(p.text.endswith(" ") for p in result[:-1])


def test_mixed_paragraphs_preserve_words_and_order():
    original = (
        "Absatz 1.\n\n"
        + ("Unzulässige Vertragsklauseln sind sorgfältig zu prüfen. " * 75).strip()
        + "\n\nAbsatz 3."
    )
    result = chunk_section(section(original), max_chars=200)
    observed = " ".join(" ".join(p.text for p in result).split())
    expected = " ".join(original.split())
    assert observed == expected
    assert all(len(p.text) <= 200 for p in result)


def test_legal_provenance_and_ids_survive_large_section_split():
    original = ("Ein Recht entsteht nur unter bestimmten Bedingungen. " * 190).strip()
    sample = section(original, title="§ 309 Klauselverbote ohne Wertungsmöglichkeit")
    result = chunk_section(sample)
    assert "".join(p.text for p in result) == original
    assert len({p.id for p in result}) == len(result)
    assert all(p.section_id == sample.id for p in result)
    assert all(p.url == sample.url for p in result)
    assert all(p.title == sample.title for p in result)
    assert chunk_section(sample) == result


def test_rejects_unreasonably_small_limit():
    try:
        chunk_section(section("Text."), max_chars=79)
    except ValueError as exc:
        assert "max_chars" in str(exc)
    else:
        raise AssertionError("Expected invalid max_chars to raise")
