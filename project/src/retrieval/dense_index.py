from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from src.retrieval.embedder import Embedder


class DenseIndex:
    """Плотный поиск: эмбеддинги + FAISS IndexFlatIP (косинус на нормированных векторах)."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.index: faiss.Index | None = None

    def build(self, texts: list[str]) -> None:
        vectors = self.embedder.encode(texts)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index = index

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.index is None:
            raise RuntimeError("DenseIndex не построен")
        qv = self.embedder.encode([query])
        scores, idxs = self.index.search(qv, min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

    def save(self, path: Path) -> None:
        if self.index is None:
            raise RuntimeError("Нечего сохранять: индекс пуст")
        faiss.write_index(self.index, str(path))

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path))
