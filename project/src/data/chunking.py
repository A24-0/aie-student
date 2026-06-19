from __future__ import annotations

from dataclasses import dataclass


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    step = max(1, chunk_size - overlap)
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += step
    return chunks


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    text: str
    title: str = ""
    topic: str = ""


def chunk_document(doc, chunk_size: int = 120, overlap: int = 20) -> list[Chunk]:
    pieces = chunk_text(doc.full_text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(
            chunk_id=f"{doc.id}#{i}",
            source_id=doc.id,
            text=piece,
            title=doc.title,
            topic=doc.topic,
        )
        for i, piece in enumerate(pieces)
    ]
