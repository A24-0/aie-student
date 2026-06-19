from src.config import RetrievalConfig
from src.data.chunking import Chunk
from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def test_rrf_prefers_docs_ranked_high_by_both():
    rankings = {"dense": [2, 0, 1], "bm25": [2, 1, 0]}
    fused = reciprocal_rank_fusion(rankings, {"dense": 1.0, "bm25": 1.0}, rrf_k=60)
    assert fused[0][0] == 2


def test_rrf_weight_shifts_result():
    rankings = {"dense": [0, 1], "bm25": [1, 0]}
    fused_dense = reciprocal_rank_fusion(rankings, {"dense": 5.0, "bm25": 1.0}, rrf_k=60)
    assert fused_dense[0][0] == 0


class _FakeIndex:
    def __init__(self, order):
        self.order = order

    def search(self, query, k):
        return [(i, 1.0 / (r + 1)) for r, i in enumerate(self.order[:k])]


def _chunks(n):
    return [Chunk(chunk_id=f"c{i}", source_id=f"d{i}", text=f"text {i}", title="", topic="t") for i in range(n)]


def test_hybrid_retriever_modes():
    chunks = _chunks(4)
    dense = _FakeIndex([3, 2, 1, 0])
    bm25 = _FakeIndex([3, 0, 1, 2])
    cfg = RetrievalConfig(method="hybrid", top_k=2, candidate_k=4, rrf_k=60)
    retr = HybridRetriever(chunks, dense, bm25, cfg)

    hybrid_hits = retr.retrieve("q", method="hybrid", top_k=2)
    assert hybrid_hits[0].source_id == "d3"

    dense_hits = retr.retrieve("q", method="dense", top_k=1)
    assert dense_hits[0].source_id == "d3"
