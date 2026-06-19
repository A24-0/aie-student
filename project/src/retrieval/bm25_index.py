from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.data.preprocess import tokenize


class BM25Index:

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None

    def build(self, texts: list[str]) -> None:
        corpus_tokens = [tokenize(t) for t in texts]
        corpus_tokens = [toks if toks else ["__empty__"] for toks in corpus_tokens]
        self._bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            raise RuntimeError("BM25Index не построен")
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(int(i), float(s)) for i, s in ranked[:k]]
