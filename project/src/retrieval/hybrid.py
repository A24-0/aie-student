from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalHit:
    chunk_id: str
    source_id: str
    text: str
    title: str
    topic: str
    score: float


def reciprocal_rank_fusion(
    rankings: dict[str, list[int]],
    weights: dict[str, float],
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for name, ranked in rankings.items():
        w = weights.get(name, 1.0)
        for rank, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + w * (1.0 / (rrf_k + rank + 1))
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:

    def __init__(self, chunks, dense_index, bm25_index, config) -> None:
        self.chunks = chunks
        self.dense = dense_index
        self.bm25 = bm25_index
        self.cfg = config

    def _to_hits(self, scored: list[tuple[int, float]], top_k: int) -> list[RetrievalHit]:
        hits = []
        for idx, score in scored[:top_k]:
            ch = self.chunks[idx]
            hits.append(
                RetrievalHit(
                    chunk_id=ch.chunk_id,
                    source_id=ch.source_id,
                    text=ch.text,
                    title=ch.title,
                    topic=ch.topic,
                    score=round(float(score), 6),
                )
            )
        return hits

    def retrieve(
        self,
        query: str,
        method: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        method = method or self.cfg.method
        top_k = top_k or self.cfg.top_k
        cand = max(self.cfg.candidate_k, top_k)

        if method == "dense":
            return self._to_hits(self.dense.search(query, cand), top_k)
        if method == "bm25":
            return self._to_hits(self.bm25.search(query, cand), top_k)
        if method == "hybrid":
            dense_ranked = [i for i, _ in self.dense.search(query, cand)]
            bm25_ranked = [i for i, _ in self.bm25.search(query, cand)]
            fused = reciprocal_rank_fusion(
                {"dense": dense_ranked, "bm25": bm25_ranked},
                {"dense": self.cfg.dense_weight, "bm25": self.cfg.bm25_weight},
                rrf_k=self.cfg.rrf_k,
            )
            return self._to_hits(fused, top_k)
        raise ValueError(f"Неизвестный метод retrieval: {method}")
