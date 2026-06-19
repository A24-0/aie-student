from __future__ import annotations

import math


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    found = len(set(retrieved[:k]) & relevant)
    return found / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_run(
    predictions: list[tuple[list[str], set[str]]],
    k: int,
) -> dict[str, float]:
    if not predictions:
        return {}
    n = len(predictions)
    agg = {
        f"hit@{k}": 0.0,
        f"recall@{k}": 0.0,
        f"precision@{k}": 0.0,
        f"mrr@{k}": 0.0,
        f"ndcg@{k}": 0.0,
    }
    for retrieved, relevant in predictions:
        agg[f"hit@{k}"] += hit_at_k(retrieved, relevant, k)
        agg[f"recall@{k}"] += recall_at_k(retrieved, relevant, k)
        agg[f"precision@{k}"] += precision_at_k(retrieved, relevant, k)
        agg[f"mrr@{k}"] += mrr_at_k(retrieved, relevant, k)
        agg[f"ndcg@{k}"] += ndcg_at_k(retrieved, relevant, k)
    return {key: round(val / n, 4) for key, val in agg.items()}
