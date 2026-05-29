from __future__ import annotations

import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.config import load_config
from src.logging_utils import get_logger, setup_logging
from src.rag.engine import RagEngine
from src.service.schemas import (
    PredictRequest,
    PredictResponse,
    SearchRequest,
    SearchResponse,
    SourceItem,
)

logger = get_logger("service")

_metrics = {
    "requests_total": defaultdict(int),
    "errors_total": 0,
    "latency_ms_sum": 0.0,
    "latency_count": 0,
}


def _hits_to_items(hits) -> list[SourceItem]:
    return [
        SourceItem(
            source_id=h.source_id,
            title=h.title,
            topic=h.topic,
            text=h.text,
            score=h.score,
        )
        for h in hits
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    setup_logging(cfg.service.log_level)
    engine = RagEngine(cfg)
    try:
        engine.load()
        logger.info("Индексы загружены из %s", cfg.paths.artifacts_dir)
    except FileNotFoundError as exc:
        logger.warning("%s — собираю индекс на старте", exc)
        engine.build()
        engine.save()
    app.state.engine = engine
    app.state.cfg = cfg
    yield


app = FastAPI(
    title="ФинФакт — RAG-ассистент по личным финансам",
    description="Гибридный поиск (BM25 + плотные эмбеддинги + RRF) по базе знаний с генерацией ответа.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def track_latency(request, call_next):
    start = time.perf_counter()
    _metrics["requests_total"][request.url.path] += 1
    response = await call_next(request)
    _metrics["latency_ms_sum"] += (time.perf_counter() - start) * 1000
    _metrics["latency_count"] += 1
    return response


@app.get("/health")
def health() -> dict:
    engine: RagEngine = app.state.engine
    return {"status": "ok", "chunks": len(engine.chunks)}


@app.get("/metrics")
def metrics() -> dict:
    count = _metrics["latency_count"] or 1
    return {
        "requests_total": dict(_metrics["requests_total"]),
        "errors_total": _metrics["errors_total"],
        "latency_ms_avg": round(_metrics["latency_ms_sum"] / count, 2),
    }


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    engine: RagEngine = app.state.engine
    try:
        hits = engine.retrieve(body.query, method=body.method, top_k=body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _metrics["errors_total"] += 1
        logger.exception("search failed")
        raise HTTPException(status_code=500, detail="search failed") from exc
    return SearchResponse(
        query=body.query,
        method=body.method or engine.cfg.retrieval.method,
        results=_hits_to_items(hits),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    engine: RagEngine = app.state.engine
    try:
        result = engine.answer(body.question, method=body.method, top_k=body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _metrics["errors_total"] += 1
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail="predict failed") from exc
    logger.info("predict q=%r method=%s sources=%d", body.question, result.method, len(result.sources))
    return PredictResponse(
        question=result.query,
        answer=result.answer,
        answer_type=result.answer_type,
        method=result.method,
        sources=_hits_to_items(result.sources),
    )


@app.post("/reindex")
def reindex() -> dict:
    engine: RagEngine = app.state.engine
    engine.build()
    engine.save()
    logger.info("Индекс пересобран по запросу")
    return {"status": "reindexed", "chunks": len(engine.chunks)}
