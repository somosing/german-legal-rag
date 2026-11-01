import json
import sys
from unittest.mock import patch

from legal_rag.cli import main

XML = (
    "<dokumente><norm><metadaten><enbez>§ 1</enbez><titel>Beginn</titel>"
    "</metadaten><textdaten><text><P>Die Rechtsfähigkeit beginnt mit der Geburt."
    "</P></text></textdaten></norm></dokumente>"
).encode()


def test_bgb_dry_run_reads_local_xml_and_does_not_open_ollama(tmp_path, capsys):
    xml_file = tmp_path / "bgb.xml"
    xml_file.write_bytes(XML)
    with patch.object(sys, "argv", ["legal-rag", "ingest", "--source", "bgb",
                                    "--input-xml", str(xml_file), "--dry-run"]):
        main()
    data = json.loads(capsys.readouterr().out)
    assert data["section_count"] == 1
    assert data["passage_count"] == 1
    assert data["index_modified"] is False
    assert len(data["source_sha256"]) == 64
    assert data["first_sections"] == ["§ 1 Beginn"]


def test_dry_run_does_not_require_local_models(capsys):
    with patch.object(sys, "argv", ["legal-rag", "ingest", "--source", "demo",
                                    "--dry-run"]):
        main()
    data = json.loads(capsys.readouterr().out)
    assert data["section_count"] == 8
    assert data["index_modified"] is False
