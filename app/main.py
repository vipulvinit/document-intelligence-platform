from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import articles, qa, sources
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import init_engine, new_session


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    init_engine()
    if get_settings().seed_default_sources:
        from app.services.source_service import seed_default_sources

        db = new_session()
        try:
            seed_default_sources(db)
        finally:
            db.close()
    yield


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Automated medical helpline: ingests open-access medical literature on a schedule "
    "(text, figures and tables) and answers questions with verified citations.",
    lifespan=lifespan,
)
app.include_router(qa.router)
app.include_router(articles.router)
app.include_router(sources.router)
app.include_router(qa.objects)


@app.get("/", tags=["Health"])
async def root() -> dict[str, str]:
    return {"status": "running", "message": f"Welcome to the {settings.app_name}"}
