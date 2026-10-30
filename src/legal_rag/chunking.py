"""Structure-preserving parent/child chunks: parent = section, child = paragraphs.

Avoids arbitrary character slicing; overlong paragraphs remain whole and are flagged in docs.
"""

from __future__ import annotations

import hashlib
import re

from legal_rag.models import Passage, Section


def stable_id(*pieces: str) -> str:
    return hashlib.sha256("\x1f".join(pieces).encode("utf-8")).hexdigest()[:24]


def chunk_section(section: Section, max_chars: int = 750) -> list[Passage]:
    if max_chars < 80:
        raise ValueError("max_chars must be >= 80")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section.text) if p.strip()]
    passages: list[Passage] = []
    group: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal group, size
        if group:
            text = "\n\n".join(group)
            passages.append(
                Passage(
                    id=stable_id(section.id, str(len(passages)), text),
                    section_id=section.id,
                    law=section.law,
                    title=section.title,
                    text=text,
                    url=section.url,
                )
            )
            group, size = [], 0

    for para in paragraphs:
        if group and size + len(para) + 2 > max_chars:
            flush()
        group.append(para)
        size += len(para) + (2 if len(group) > 1 else 0)
        if size >= max_chars:
            flush()
    flush()
    return passages
