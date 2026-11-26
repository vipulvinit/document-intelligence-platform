"""Database engine and session factory.

SQLite is the default (free, no install). WAL mode lets the API read while the scheduler writes,
and the busy timeout makes a second writer wait instead of failing. Tables are created on start;
there are no migrations, so after a model change delete the database file and start fresh.
"""

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

_engine = None
_SessionLocal = None


def _sqlite_pragmas(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def init_engine(url: str | None = None):
    global _engine, _SessionLocal
    url = url or get_settings().database_url
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        path = url.split("///", 1)[-1]
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(url, pool_pre_ping=True, **kwargs)
    if url.startswith("sqlite"):
        event.listen(_engine, "connect", _sqlite_pragmas)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    import app.models  # noqa: F401  (registers the tables)
    from app.db.base import Base

    Base.metadata.create_all(_engine)
    return _engine


def new_session() -> Session:
    if _SessionLocal is None:
        init_engine()
    return _SessionLocal()


def get_session() -> Iterator[Session]:
    session = new_session()
    try:
        yield session
    finally:
        session.close()
