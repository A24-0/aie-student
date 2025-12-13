# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Просмотр первых строк

```bash
uv run eda-cli head data/example.csv --n 10
```

Команда выводит первые `n` строк CSV-файла.

Параметры:

- `--n` – количество строк для вывода (по умолчанию `5`);
- `--sep` – разделитель в CSV (по умолчанию `,`);
- `--encoding` – кодировка файла (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

**Параметры команды `report`:**

- `--out-dir` – каталог для сохранения отчёта (по умолчанию `reports`);
- `--sep` – разделитель в CSV (по умолчанию `,`);
- `--encoding` – кодировка файла (по умолчанию `utf-8`);
- `--max-hist-columns` – максимум числовых колонок для гистограмм (по умолчанию `6`);
- `--top-k-categories` – сколько top-значений выводить для категориальных признаков (по умолчанию `5`);
- `--title` – заголовок отчёта в Markdown (по умолчанию `EDA-отчёт`);
- `--min-missing-share` – порог доли пропусков, выше которого колонка считается проблемной (по умолчанию `0.1`);
- `--json-summary` – сохранить компактную JSON-сводку по датасету (флаг, по умолчанию `False`);

**Пример с новыми параметрами:**

```bash
uv run eda-cli report data/example.csv \
  --out-dir reports_example \
  --max-hist-columns 10 \
  --top-k-categories 8 \
  --title "Анализ данных пользователей" \
  --min-missing-share 0.15 \
  --json-summary
```

В результате в каталоге `reports/` (или указанном `--out-dir`) появятся:

- `report.md` – основной отчёт в Markdown с заголовком из `--title`;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам (количество определяется `--top-k-categories`);
- `hist_*.png` – гистограммы числовых колонок (максимум `--max-hist-columns`);
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций;
- `summary.json` – компактная JSON-сводка (только при использовании `--json-summary`).

**Новые эвристики качества данных:**

Отчёт теперь включает дополнительные проверки качества:

- `has_constant_columns` – обнаружение колонок, где все значения одинаковые;
- `has_high_cardinality_categoricals` – категориальные признаки с очень большим числом уникальных значений (порог: 1000);
- `has_suspicious_id_duplicates` – проверка уникальности идентификаторов (колонки с "id" в названии);
- `has_many_zero_values` – числовые колонки с большой долей нулевых значений (порог: 50%).

Все эти флаги влияют на интегральный показатель `quality_score` и отображаются в разделе "Качество данных" отчёта.

**JSON-сводка (`--json-summary`):**

При использовании флага `--json-summary` дополнительно создаётся файл `summary.json` с компактной сводкой:

- размеры датасета (`n_rows`, `n_cols`);
- интегральный `quality_score`;
- список проблемных колонок (по пропускам и эвристикам качества);
- все флаги качества данных.

Пример содержимого `summary.json`:

```json
{
  "n_rows": 36,
  "n_cols": 14,
  "quality_score": 0.49,
  "problematic_columns": ["user_id", "churned", "city"],
  "quality_flags": {
    "too_few_rows": true,
    "too_many_columns": false,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": false,
    "has_suspicious_id_duplicates": true,
    "has_many_zero_values": true
  }
}
```

## Тесты

```bash
uv run pytest -q
```