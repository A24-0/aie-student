from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    method: Literal["bm25", "dense", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=10)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    method: Literal["bm25", "dense", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceItem(BaseModel):
    source_id: str
    title: str
    topic: str
    text: str
    score: float


class PredictResponse(BaseModel):
    question: str
    answer: str
    answer_type: str
    method: str
    sources: list[SourceItem]


class SearchResponse(BaseModel):
    query: str
    method: str
    results: list[SourceItem]
