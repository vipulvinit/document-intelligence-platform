import os
from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch, tmp_path):
    """Every test runs offline: mock LLM, hashing embeddings, fresh SQLite database and storage."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("VISION_MODEL", "")
    monkeypatch.setenv("SEED_DEFAULT_SOURCES", "False")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("DATABASE_URL", os.environ.get("TEST_DATABASE_URL", f"sqlite:///{tmp_path}/test.db"))
    from app.core.config import get_settings
    from app.llm.embeddings import get_embedder
    from app.services.retrieval_service import reset_index
    from app.storage import get_storage

    def clear():
        get_settings.cache_clear()
        get_embedder.cache_clear()
        get_storage.cache_clear()
        reset_index()

    clear()
    yield
    clear()


@pytest.fixture
def db():
    from app.db.session import init_engine, new_session

    init_engine()
    session = new_session()
    yield session
    session.close()


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


def upload(client, title, sample, data: bytes | None = None):
    path = SAMPLES / sample
    content = data if data is not None else path.read_bytes()
    return client.post("/articles/upload", data={"title": title}, files={"file": (path.name, content)})
