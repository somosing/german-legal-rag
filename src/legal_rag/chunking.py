"""Structure-preserving parent/child chunks for statutory passages.

Keep normal paragraphs together; split oversized paragraphs at sentence or
whitespace boundaries where possible. Never silently truncate source text.
"""

from __future__ import annotations

import hashlib
import re

from legal_rag.models import Passage, Section

SENTENCE_BREAK = re.compile(r'[.!?]["”»]?\s+')
WHITESPACE_BREAK = re.compile(r"\s+")


def stable_id(*pieces: str) -> str:
    return hashlib.sha256("\x1f".join(pieces).encode("utf-8")).hexdigest()[:24]


def _split_oversized_paragraph(paragraph: str, limit: int) -> list[str]:
    """Preserve every character while keeping individual fragments bounded.

    Prefer sentence boundaries near the end of a window, then whitespace.
    A long unbroken token must be split at the hard limit; no characters
    are dropped or added between consecutive fragments.
    """
    fragments: list[str] = []
    start = 0
    while len(paragraph) - start > limit:
        window = paragraph[start : start + limit]
        minimum = limit // 2
        sentences = [
            match.end()
            for match in SENTENCE_BREAK.finditer(window)
            if match.end() >= minimum
        ]
        words = [
            match.end()
            for match in WHITESPACE_BREAK.finditer(window)
            if match.end() >= minimum
        ]
        cut = sentences[-1] if sentences else words[-1] if words else limit
        fragments.append(paragraph[start : start + cut])
        start += cut
    if start < len(paragraph):
        fragments.append(paragraph[start:])
    return fragments


def chunk_section(section: Section, max_chars: int = 750) -> list[Passage]:
    """Split a section into passages with text length <= max_chars.

    Section identity, title, URL, and full normalized paragraph content
    are retained; no truncation is used for embedding inputs.
    """
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
        if len(para) > max_chars:
            flush()
            for fragment in _split_oversized_paragraph(para, max_chars):
                group = [fragment]
                size = len(fragment)
                flush()
            continue
        if group and size + len(para) + 2 > max_chars:
            flush()
        group.append(para)
        size += len(para) + (2 if len(group) > 1 else 0)
        if size >= max_chars:
            flush()
    flush()
    return passages
