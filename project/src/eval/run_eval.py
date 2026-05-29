from __future__ import annotations

from src.eval.metrics import evaluate_run
from src.rag.engine import RagEngine


def evaluate_method(engine: RagEngine, eval_set: list[dict], method: str, k: int) -> dict:
    predictions = []
    per_query = []
    for item in eval_set:
        hits = engine.retrieve(item["query"], method=method, top_k=k)
        # один source может дать несколько чанков — схлопываем, сохраняя порядок
        seen = []
        for h in hits:
            if h.source_id not in seen:
                seen.append(h.source_id)
        relevant = set(item["relevant"])
        predictions.append((seen, relevant))
        per_query.append(
            {
                "query": item["query"],
                "relevant": ";".join(item["relevant"]),
                "retrieved": ";".join(seen[:k]),
                "hit": int(bool(set(seen[:k]) & relevant)),
            }
        )
    metrics = evaluate_run(predictions, k=k)
    return {"metrics": metrics, "per_query": per_query}
