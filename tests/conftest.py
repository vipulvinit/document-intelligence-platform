import os
from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch, tmp_path):
    """Every test runs offline: mock LLM, hashing embeddings, a fresh SQLite database."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("DATABASE_URL", os.environ.get("TEST_DATABASE_URL", f"sqlite:///{tmp_path}/test.db"))
    from app.core.config import get_settings
    from app.llm.embeddings import get_embedder

    get_settings.cache_clear()
    get_embedder.cache_clear()
    yield
    get_settings.cache_clear()
    get_embedder.cache_clear()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.db.base import Base
    from app.db.session import init_engine
    from app.main import app

    engine = init_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(engine)


def upload(client, name, sample):
    path = SAMPLES / sample
    with open(path, "rb") as f:
        return client.post("/documents", data={"name": name}, files={"file": (path.name, f)})
