"""Qdrant dense index and atomic local lexical/parent stores.

Local Qdrant is single-process. Use separate Qdrant server for concurrency in future.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from legal_rag.chunking import chunk_section
from legal_rag.config import Settings
from legal_rag.models import Passage, Section
from legal_rag.ollama import Ollama


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmpname = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            tmpname = stream.name
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(tmpname, path)
    finally:
        if tmpname and os.path.exists(tmpname):
            os.unlink(tmpname)


def load_json(path: Path) -> dict | list:
    if not path.is_file():
        raise FileNotFoundError(
            f"Index not found at {path}. Run: legal-rag ingest --source demo"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class Index:
    def __init__(self, settings: Settings, embedder: Ollama):
        self.settings = settings
        self.embedder = embedder

    @property
    def root(self) -> Path:
        return self.settings.state_dir

    def client(self):
        from qdrant_client import QdrantClient

        (self.root / "qdrant").mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(self.root / "qdrant"))

    def build(
        self,
        sections: list[Section],
        batch_size: int = 16,
        source_metadata: dict | None = None,
    ) -> dict:
        from qdrant_client import models

        if not sections:
            raise ValueError("No sections to index")
        passages = [p for section in sections for p in chunk_section(section)]
        if not passages:
            raise ValueError("No non-empty passages")
        # Avoid altering existing index until at least one embedding is available.
        first = self.embedder.embed([passages[0].title + "\n" + passages[0].text])[0]
        dimension = len(first)
        client = self.client()
        try:
            if client.collection_exists(self.settings.collection):
                client.delete_collection(self.settings.collection)
            client.create_collection(
                collection_name=self.settings.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
            for start in range(0, len(passages), batch_size):
                batch = passages[start : start + batch_size]
                vectors = (
                    [first]
                    if start == 0 and len(batch) == 1
                    else self.embedder.embed([p.title + "\n" + p.text for p in batch])
                )
                client.upsert(
                    collection_name=self.settings.collection,
                    wait=True,
                    points=[
                        models.PointStruct(
                            id=str(uuid.uuid5(uuid.NAMESPACE_URL, p.id)),
                            vector=vector,
                            payload=p.to_dict(),
                        )
                        for p, vector in zip(batch, vectors, strict=True)
                    ],
                )
        finally:
            client.close()
        # Keep local BM25 and parent/section text in portable, inspectable JSON.
        save_json(self.root / "passages.json", [p.to_dict() for p in passages])
        save_json(self.root / "sections.json", [s.to_dict() for s in sections])
        manifest = {
            "embed_model": self.settings.embed_model,
            "embedding_dim": dimension,
            "collection": self.settings.collection,
            "section_count": len(sections),
            "passage_count": len(passages),
            "indexed_utc": datetime.now(UTC).isoformat(),
            "source_disclaimer": "Source texts must be verified against authoritative versions.",
            "source_metadata": source_metadata or {},
        }
        save_json(self.root / "manifest.json", manifest)
        return manifest

    def load(self) -> tuple[list[Passage], dict[str, Section]]:
        manifest = load_json(self.root / "manifest.json")
        if manifest["embed_model"] != self.settings.embed_model:
            raise ValueError(
                f"Index uses {manifest['embed_model']}, but current config uses "
                f"{self.settings.embed_model}. Reindex or correct OLLAMA_EMBED_MODEL."
            )
        if manifest["collection"] != self.settings.collection:
            raise ValueError("Collection mismatch; reindex or correct RAG_COLLECTION")
        passages = [Passage(**p) for p in load_json(self.root / "passages.json")]
        sections = {
            s["id"]: Section(**s) for s in load_json(self.root / "sections.json")
        }
        return passages, sections
