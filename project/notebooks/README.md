# Ноутбуки

- `01_eda.ipynb` — разведочный анализ корпуса: документы по темам, длина текстов, чанкинг.
- `02_retrieval_experiments.ipynb` — сравнение BM25 / dense / hybrid, влияние `top_k`, анализ ошибок.

Ноутбуки читают артефакты из `../artifacts`, поэтому перед запуском стоит выполнить
`python -m src.train`. Для запуска нужен Jupyter (`pip install jupyter`).
