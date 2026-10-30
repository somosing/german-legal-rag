"""Small typed domain layer, separate from FastAPI and Qdrant types."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Section:
    id: str
    law: str
    title: str
    text: str
    url: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Passage:
    id: str
    section_id: str
    law: str
    title: str
    text: str
    url: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float
    parent_text: str

    def to_source(self, source_id: str) -> dict:
        return {
            "source_id": source_id,
            "section_id": self.passage.section_id,
            "title": self.passage.title,
            "url": self.passage.url,
            "passage": self.passage.text,
            "score": round(self.score, 6),
        }
