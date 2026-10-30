"""Conservative bilingual language flag for human review, not a classifier.

Uses a handful of function words to flag clear German/English mismatches.
Returns None when the answer is too short or mixed to assess reliably.
This is not suitable for enforcing language, accuracy, or legal correctness.
"""

from __future__ import annotations

import re
from typing import Literal

Language = Literal["de", "en"]

# Common function words are more informative than domain-specific legal terms.
GERMAN = frozenset({
    "ab", "als", "auch", "bei", "das", "dem", "den", "der", "des", "die", "diese",
    "ein", "eine", "einem", "einen", "einer", "eines", "für", "im", "ist", "kann",
    "mit", "nach", "nicht", "noch", "oder", "sich", "sind", "und", "von", "wenn",
    "werden", "wird", "zu", "zur", "zum",
})
ENGLISH = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "the", "their", "this", "to",
    "under", "was", "were", "when", "which", "with", "would",
})
TOKENS = re.compile(r"[^\W\d_]+", re.UNICODE)


def observed_answer_language(answer: str) -> Language | None:
    """Return a language only for clear function-word signals; otherwise None."""
    without_citations = re.sub(r"\[S\d+\]", " ", answer)
    words = TOKENS.findall(without_citations.casefold())
    de = sum(word in GERMAN for word in words)
    en = sum(word in ENGLISH for word in words)
    if de >= 3 and de >= en + 2:
        return "de"
    if en >= 3 and en >= de + 2:
        return "en"
    return None
