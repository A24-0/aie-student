"""Сборка индексов и прогон экспериментов по retrieval.

Запуск:  python -m src.train
Результат: индексы в artifacts/ + таблицы метрик + лог экспериментов.
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import load_config
from src.data.loader import load_eval_set
from src.eval.run_eval import evaluate_method
from src.logging_utils import get_logger, setup_logging
from src.rag.engine import RagEngine
from src.tracking import ExperimentTracker

logger = get_logger("train")


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.service.log_level)
    art = cfg.artifacts
    art.mkdir(parents=True, exist_ok=True)

    logger.info("Строю индексы из %s", cfg.paths.corpus)
    engine = RagEngine(cfg)
    engine.build()
    engine.save()

    eval_set = load_eval_set(cfg.path(cfg.paths.eval_set))
    tracker = ExperimentTracker(art)
    tracker.reset()

    k = cfg.retrieval.top_k
    summary_rows = []
    detailed_best = None

    # Эксперимент 1: сравнение трёх стратегий retrieval при одинаковом top_k.
    for method in ("bm25", "dense", "hybrid"):
        res = evaluate_method(engine, eval_set, method=method, k=k)
        metrics = res["metrics"]
        params = {
            "method": method,
            "top_k": k,
            "embedding_model": cfg.retrieval.embedding_model.split("/")[-1],
            "rrf_k": cfg.retrieval.rrf_k,
        }
        tracker.log_run(f"retriever={method}", params, metrics)
        summary_rows.append({**params, **metrics})
        logger.info("%-7s -> %s", method, metrics)
        if method == cfg.retrieval.method:
            detailed_best = res["per_query"]

    # Эксперимент 2: влияние top_k на гибридный режим.
    for k_try in (1, 3, 5, 10):
        res = evaluate_method(engine, eval_set, method="hybrid", k=k_try)
        params = {
            "method": "hybrid",
            "top_k": k_try,
            "embedding_model": cfg.retrieval.embedding_model.split("/")[-1],
            "rrf_k": cfg.retrieval.rrf_k,
        }
        tracker.log_run(f"hybrid_topk={k_try}", params, res["metrics"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(art / "retrieval_comparison.csv", index=False)

    if detailed_best is not None:
        pd.DataFrame(detailed_best).to_csv(art / "eval_per_query.csv", index=False)

    best = max(summary_rows, key=lambda r: (r.get(f"mrr@{k}", 0), r.get(f"hit@{k}", 0)))
    (art / "training_summary.json").write_text(
        json.dumps(
            {
                "final_method": cfg.retrieval.method,
                "top_k": k,
                "embedding_model": cfg.retrieval.embedding_model,
                "best_by_mrr": best,
                "num_eval_queries": len(eval_set),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Лучшая стратегия по MRR: %s", best["method"])
    logger.info("Готово. Артефакты: %s", art)


if __name__ == "__main__":
    main()
