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

## HTTP-сервис (FastAPI)

Проект включает HTTP-сервис на FastAPI для анализа качества данных через REST API.

### Запуск сервиса

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

После запуска сервис будет доступен по адресу `http://localhost:8000`.

Интерактивная документация API доступна по адресу:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Эндпоинты API

#### GET /health

Health check эндпоинт для проверки работоспособности сервиса.

**Ответ:**
```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}
```

#### POST /quality

Оценка качества данных на основе параметров (без загрузки файла).

**Тело запроса:**
```json
{
  "n_rows": 1000,
  "n_cols": 10,
  "max_missing_share": 0.1,
  "numeric_cols": 5,
  "categorical_cols": 5
}
```

**Ответ:**
```json
{
  "ok_for_model": true,
  "quality_score": 0.9,
  "message": "Данных достаточно, модель можно обучать (по текущим эвристикам).",
  "latency_ms": 0.5,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "no_numeric_columns": false,
    "no_categorical_columns": false
  },
  "dataset_shape": {
    "n_rows": 1000,
    "n_cols": 10
  }
}
```

#### POST /quality-from-csv

Оценка качества данных из загруженного CSV-файла. Использует полный EDA-анализ.

**Параметры:**
- `file` (form-data): CSV-файл для анализа

**Ответ:**
```json
{
  "ok_for_model": false,
  "quality_score": 0.49,
  "message": "CSV требует доработки перед обучением модели (по текущим эвристикам).",
  "latency_ms": 15.2,
  "flags": {
    "too_few_rows": true,
    "too_many_columns": false,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": false,
    "has_suspicious_id_duplicates": true,
    "has_many_zero_values": true
  },
  "dataset_shape": {
    "n_rows": 36,
    "n_cols": 14
  }
}
```

#### POST /quality-flags-from-csv

Возвращает полный набор флагов качества из загруженного CSV-файла. Включает все эвристики качества данных, реализованные в HW03.

**Параметры:**
- `file` (form-data): CSV-файл для анализа

**Ответ:**
```json
{
  "flags": {
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

### Примеры использования

#### Через curl

```bash
# Health check
curl http://localhost:8000/health

# Оценка качества по параметрам
curl -X POST http://localhost:8000/quality \
  -H "Content-Type: application/json" \
  -d '{"n_rows": 1000, "n_cols": 10, "max_missing_share": 0.1, "numeric_cols": 5, "categorical_cols": 5}'

# Оценка качества из CSV
curl -X POST http://localhost:8000/quality-from-csv \
  -F "file=@data/example.csv"

# Получение флагов качества из CSV
curl -X POST http://localhost:8000/quality-flags-from-csv \
  -F "file=@data/example.csv"
```

#### Через Python (httpx)

```python
import httpx

# Health check
response = httpx.get("http://localhost:8000/health")
print(response.json())

# Оценка качества из CSV
with open("data/example.csv", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/quality-from-csv",
        files={"file": f}
    )
    print(response.json())
```

## Тесты

```bash
uv run pytest -q
```