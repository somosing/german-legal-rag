"""Ingest official German statute XML and clearly labelled fictional demo policies.

The HTTPS destination for official data is fixed to avoid arbitrary-URL fetching.
"""

from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import httpx
from defusedxml import ElementTree

from legal_rag.chunking import stable_id
from legal_rag.models import Section

BGB_XML_URL = "https://www.gesetze-im-internet.de/bgb/xml.zip"
MAX_BGB_ARCHIVE = 45 * 1024 * 1024
MAX_BGB_XML = 45 * 1024 * 1024


def _normalise(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_bgb_xml(xml: bytes, limit: int | None = None) -> list[Section]:
    """Read per-§ <norm> nodes; preserves <P> paragraph boundaries.

    Source schema is maintained externally; tests cover the expected structure.
    """
    root = ElementTree.fromstring(xml)
    sections: list[Section] = []
    for norm in root.findall(".//norm"):
        meta = norm.find("metadaten")
        if meta is None:
            continue
        enbez = _normalise(meta.findtext("enbez", default=""))
        heading = _normalise(meta.findtext("titel", default=""))
        if not enbez.startswith("§"):
            continue
        content = norm.find("./textdaten/text")
        if content is None:
            continue
        paras = [
            _normalise(" ".join(p.itertext()))
            for p in content.findall(".//P")
        ]
        paras = [p for p in paras if p]
        if not paras:
            # The published XML can include non-<P> structures (e.g. tables).
            fallback = _normalise(" ".join(content.itertext()))
            if fallback:
                paras = [fallback]
        if not paras:
            continue
        # Each paragraph might include enumerated list markup; keep it together.
        section_slug = re.sub(r"[^0-9a-zA-Z]", "", enbez[1:])
        url = f"https://www.gesetze-im-internet.de/bgb/__{section_slug}.html"
        sections.append(
            Section(
                id=stable_id("BGB", enbez),
                law="BGB",
                title=f"{enbez} {heading}".strip(),
                text="\n\n".join(paras),
                url=url,
            )
        )
        if limit is not None and len(sections) >= limit:
            break
    return sections


def extract_bgb_xml(archive_bytes: bytes) -> bytes:
    """Read one XML statute from the official ZIP, without extracting paths to disk."""
    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            entries = [
                info for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".xml")
            ]
            if len(entries) != 1:
                raise ValueError(
                    f"Expected exactly one XML file in the BGB archive, found {len(entries)}"
                )
            info = entries[0]
            if info.file_size > MAX_BGB_XML:
                raise ValueError("BGB XML exceeds the allowed uncompressed size")
            with archive.open(info) as stream:
                xml = stream.read(MAX_BGB_XML + 1)
            if len(xml) > MAX_BGB_XML:
                raise ValueError("BGB XML exceeds the allowed uncompressed size")
            return xml
    except BadZipFile as exc:
        raise ValueError("The official BGB download is not a valid ZIP archive") from exc


def fetch_bgb_xml(timeout: float = 40.0) -> bytes:
    """Fetch the published ZIP and return its XML bytes for the existing parser."""
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        response = client.get(BGB_XML_URL)
        response.raise_for_status()
    if len(response.content) > MAX_BGB_ARCHIVE:
        raise ValueError("BGB archive exceeds the allowed download size")
    return extract_bgb_xml(response.content)


def load_demo(path: Path) -> list[Section]:
    items = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for item in items:
        for required in ("id", "law", "title", "text", "url"):
            if not isinstance(item.get(required), str) or not item[required].strip():
                raise ValueError(f"Invalid demo fixture: {required}")
        result.append(Section(**item))
    return result
