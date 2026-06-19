import pytest
from fastapi.testclient import TestClient

from src.service.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0


def test_predict_returns_answer_and_sources(client):
    r = client.post("/predict", json={"question": "застрахованы ли вклады"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["answer"]) > 10
    assert data["sources"]
    assert any(s["source_id"] == "dep-04" for s in data["sources"])


def test_search_endpoint(client):
    r = client.post("/search", json={"query": "что такое etf", "method": "hybrid"})
    assert r.status_code == 200
    assert r.json()["results"]


def test_predict_validation(client):
    r = client.post("/predict", json={"question": "a"})
    assert r.status_code == 422
