from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import AppConfig
from src.data.chunking import Chunk, chunk_document
from src.data.loader import load_corpus
from src.logging_utils import get_logger
from src.rag.generator import AnswerGenerator
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid import HybridRetriever, RetrievalHit

logger = get_logger("rag.engine")


@dataclass
class RagResult:
    query: str
    answer: str
    answer_type: str
    method: str
    sources: list[RetrievalHit]


class RagEngine:
    """Связывает данные, retrieval и генерацию. Один объект на жизнь сервиса."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.embedder = Embedder(config.retrieval.embedding_model)
        self.chunks: list[Chunk] = []
        self.dense = DenseIndex(self.embedder)
        self.bm25 = BM25Index()
        self.retriever: HybridRetriever | None = None
        self.generator = AnswerGenerator(config, self.embedder)

    # ---- построение / сохранение / загрузка ----

    def build(self) -> None:
        docs = load_corpus(self.cfg.path(self.cfg.paths.corpus))
        self.chunks = []
        for doc in docs:
            self.chunks.extend(
                chunk_document(
                    doc,
                    chunk_size=self.cfg.chunking.chunk_size,
                    overlap=self.cfg.chunking.overlap,
                )
            )
        texts = [c.text for c in self.chunks]
        logger.info("Документов: %d, чанков: %d", len(docs), len(self.chunks))
        self.dense.build(texts)
        self.bm25.build(texts)
        self.retriever = HybridRetriever(self.chunks, self.dense, self.bm25, self.cfg.retrieval)

    def save(self) -> None:
        art = self.cfg.artifacts
        art.mkdir(parents=True, exist_ok=True)
        self.dense.save(art / "dense.faiss")
        chunks_payload = [asdict(c) for c in self.chunks]
        (art / "chunks.json").write_text(
            json.dumps(chunks_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (art / "index_meta.json").write_text(
            json.dumps(
                {
                    "embedding_model": self.cfg.retrieval.embedding_model,
                    "num_chunks": len(self.chunks),
                    "chunk_size": self.cfg.chunking.chunk_size,
                    "overlap": self.cfg.chunking.overlap,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> None:
        art = self.cfg.artifacts
        chunks_path = art / "chunks.json"
        faiss_path = art / "dense.faiss"
        if not chunks_path.is_file() or not faiss_path.is_file():
            raise FileNotFoundError(
                f"Артефакты не найдены в {art}. Сначала: python -m src.train"
            )
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**c) for c in payload]
        self.dense.load(faiss_path)
        # BM25 дёшево пересобрать из чанков, отдельно сохранять смысла нет
        self.bm25.build([c.text for c in self.chunks])
        self.retriever = HybridRetriever(self.chunks, self.dense, self.bm25, self.cfg.retrieval)

    # ---- запросы ----

    def retrieve(self, query: str, method: str | None = None, top_k: int | None = None):
        if self.retriever is None:
            raise RuntimeError("Движок не инициализирован: вызовите build() или load()")
        return self.retriever.retrieve(query, method=method, top_k=top_k)

    def answer(self, query: str, method: str | None = None, top_k: int | None = None) -> RagResult:
        hits = self.retrieve(query, method=method, top_k=top_k)
        text, answer_type = self.generator.generate(query, hits)
        return RagResult(
            query=query,
            answer=text,
            answer_type=answer_type,
            method=method or self.cfg.retrieval.method,
            sources=hits,
        )
