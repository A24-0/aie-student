from __future__ import annotations

import re

import numpy as np

from src.config import AppConfig, get_llm_api_key
from src.logging_utils import get_logger

logger = get_logger("rag.generator")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


class AnswerGenerator:
    """Формирует ответ по найденным фрагментам.

    По умолчанию работает экстрактивно (без внешних сервисов): выбирает из
    топ-документов предложения, наиболее близкие к запросу по эмбеддингам.
    Если включён режим llm и задан ключ в .env — пробует сгенерировать ответ
    через OpenAI-совместимый API, но при любой ошибке откатывается к экстракции.
    """

    def __init__(self, config: AppConfig, embedder) -> None:
        self.cfg = config
        self.embedder = embedder

    def generate(self, query: str, hits) -> tuple[str, str]:
        if not hits:
            return ("По этому вопросу в базе знаний ничего не нашлось.", "empty")

        if self.cfg.generation.mode == "llm":
            answer = self._try_llm(query, hits)
            if answer is not None:
                return (answer, "llm")
            logger.warning("LLM-режим недоступен, использую экстрактивный ответ")

        return (self._extractive(query, hits), "extractive")

    def _extractive(self, query: str, hits) -> str:
        sentences: list[str] = []
        for h in hits:
            sentences.extend(split_sentences(h.text))
        if not sentences:
            return hits[0].text

        q_vec = self.embedder.encode([query])
        s_vec = self.embedder.encode(sentences)
        sims = (s_vec @ q_vec.T).ravel()
        n = min(self.cfg.generation.max_sentences, len(sentences))
        top_idx = np.argsort(-sims)[:n]
        # сохраняем исходный порядок предложений для читаемости
        chosen = [sentences[i] for i in sorted(top_idx)]
        return " ".join(chosen)

    def _try_llm(self, query: str, hits) -> str | None:
        api_key = get_llm_api_key()
        if not api_key:
            return None
        try:
            import httpx

            context = "\n".join(f"[{h.source_id}] {h.text}" for h in hits)
            prompt = (
                "Ты ассистент по личным финансам. Ответь кратко на вопрос, "
                "опираясь только на приведённые фрагменты. Если данных нет — так и скажи.\n\n"
                f"Фрагменты:\n{context}\n\nВопрос: {query}\nОтвет:"
            )
            resp = httpx.post(
                f"{self.cfg.generation.llm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.cfg.generation.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # сеть/ключ/лимиты — не валим сервис
            logger.warning("LLM-запрос не удался: %s", exc)
            return None
