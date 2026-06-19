from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PathsConfig(BaseModel):
    corpus: str = "data/corpus.json"
    eval_set: str = "data/eval.json"
    artifacts_dir: str = "artifacts"


class ChunkingConfig(BaseModel):
    chunk_size: int = 120
    overlap: int = 20


class RetrievalConfig(BaseModel):
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    method: str = "hybrid"
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0


class GenerationConfig(BaseModel):
    mode: str = "extractive"
    max_sentences: int = 3
    llm_model: str = "openai/gpt-4o-mini"
    llm_base_url: str = "https://openrouter.ai/api/v1"


class ServiceConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)

    def path(self, relative: str) -> Path:
        return (PROJECT_ROOT / relative).resolve()

    @property
    def artifacts(self) -> Path:
        return self.path(self.paths.artifacts_dir)


class EnvSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="AIE_",
        extra="ignore",
    )

    config_path: str = "configs/config.yaml"
    log_level: str | None = None
    service_host: str | None = None
    service_port: int | None = None
    generation_mode: str | None = None

    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None


def load_config() -> AppConfig:
    env = EnvSettings()
    cfg_file = PROJECT_ROOT / env.config_path
    raw: dict[str, Any] = {}
    if cfg_file.is_file():
        raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}

    cfg = AppConfig.model_validate(raw)

    if env.log_level:
        cfg.service.log_level = env.log_level
    if env.service_host:
        cfg.service.host = env.service_host
    if env.service_port is not None:
        cfg.service.port = env.service_port
    if env.generation_mode:
        cfg.generation.mode = env.generation_mode
    if env.llm_model:
        cfg.generation.llm_model = env.llm_model
    if env.llm_base_url:
        cfg.generation.llm_base_url = env.llm_base_url
    return cfg


def get_llm_api_key() -> str | None:
    return EnvSettings().llm_api_key
