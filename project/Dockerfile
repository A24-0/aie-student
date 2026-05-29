FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Скачиваем модель эмбеддингов и собираем индексы на этапе сборки образа,
# чтобы первый запрос к сервису не ждал загрузку из сети.
RUN python -m src.train

EXPOSE 8000

CMD ["python", "-m", "src.service.main"]
