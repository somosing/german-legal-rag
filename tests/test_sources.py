from io import BytesIO
from zipfile import ZipFile

import pytest
from defusedxml.common import EntitiesForbidden

from legal_rag.sources import extract_bgb_xml, parse_bgb_xml

XML = b'''<?xml version="1.0" encoding="UTF-8"?>
<dokumente>
<norm><metadaten><enbez>\xc2\xa7 1</enbez><titel>Sample</titel></metadaten>
<textdaten><text format="XML"><Content>
<P>Paragraph A.</P><P>Paragraph B.</P></Content></text></textdaten>
</norm>
<norm><metadaten><enbez>Abschnitt A</enbez><titel>Not a law section</titel></metadaten>
<textdaten><text format="XML"><Content><P>Ignore me</P></Content></text></textdaten></norm>
</dokumente>'''


def test_parse_structure_preserves_section_and_paragraphs():
    result = parse_bgb_xml(XML)
    assert len(result) == 1
    assert result[0].title == "§ 1 Sample"
    assert result[0].text == "Paragraph A.\n\nParagraph B."
    assert result[0].url.endswith("/__1.html")


def test_rejects_xml_entities():
    payload = (
        b'<!DOCTYPE foo [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>'
        b"<dokumente>&xxe;</dokumente>"
    )
    with pytest.raises(EntitiesForbidden):
        parse_bgb_xml(payload)


def test_bgb_xml_extracted_from_zip():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("BJNR001950896.xml", XML)
        archive.writestr("README.txt", "metadata")
    extracted = extract_bgb_xml(buffer.getvalue())
    assert len(parse_bgb_xml(extracted)) == 1


def test_bgb_zip_rejects_multiple_xml():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("one.xml", XML)
        archive.writestr("two.xml", XML)
    with pytest.raises(ValueError, match="exactly one XML"):
        extract_bgb_xml(buffer.getvalue())


def test_bgb_zip_rejects_invalid_payload():
    with pytest.raises(ValueError, match="not a valid ZIP"):
        extract_bgb_xml(b"not a ZIP")


def test_bgb_zip_rejects_oversized_xml(monkeypatch):
    import legal_rag.sources as sources

    monkeypatch.setattr(sources, "MAX_BGB_XML", 5)
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("law.xml", XML)
    with pytest.raises(ValueError, match="uncompressed size"):
        extract_bgb_xml(buffer.getvalue())
