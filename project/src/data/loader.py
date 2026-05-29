from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    id: str
    text: str
    title: str = ""
    topic: str = ""

    @property
    def full_text(self) -> str:
        # заголовок помогает и BM25, и эмбеддингам зацепиться за тему
        return f"{self.title}. {self.text}".strip(". ") if self.title else self.text


def _read_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_corpus(path: Path) -> list[Document]:
    raw = _read_json(path)
    docs = [
        Document(
            id=item["id"],
            text=item["text"],
            title=item.get("title", ""),
            topic=item.get("topic", ""),
        )
        for item in raw
    ]
    ids = [d.id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("В корпусе есть документы с одинаковым id")
    return docs


def load_eval_set(path: Path) -> list[dict]:
    raw = _read_json(path)
    for item in raw:
        if "query" not in item or "relevant" not in item:
            raise ValueError("Каждый элемент eval должен содержать query и relevant")
    return raw
