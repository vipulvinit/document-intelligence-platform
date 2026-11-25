from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import documents, generation, qa
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import init_engine


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    init_engine()
    yield


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Parse documents into validated trees, track versions semantically, answer questions "
    "with citations, and generate traceable QA test cases with an LLM.",
    lifespan=lifespan,
)
app.include_router(documents.router)
app.include_router(generation.router)
app.include_router(qa.router)


@app.get("/", tags=["Health"])
async def root() -> dict[str, str]:
    return {"status": "running", "message": f"Welcome to the {settings.app_name}"}
