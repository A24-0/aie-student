# Тесты

Запуск: `pytest` из папки `project`.

- `test_chunking.py` — чанкинг и препроцессинг (токенизация, стоп-слова).
- `test_metrics.py` — метрики retrieval (hit/recall/mrr/ndcg).
- `test_hybrid.py` — Reciprocal Rank Fusion и режимы `HybridRetriever`.
- `test_api.py` — endpoints сервиса через `TestClient` (поднимает индекс).
