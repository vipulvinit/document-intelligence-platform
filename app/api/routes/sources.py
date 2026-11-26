from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.constants import SourceKind
from app.db.session import new_session
from app.models import IngestionRun, Source
from app.schemas.api import LiteratureSearchIn, RunOut, SourceIn, SourceOut, SourcePatch
from app.services import source_service

router = APIRouter(tags=["Literature sources"])


@router.get("/sources", response_model=list[SourceOut])
def list_sources(db: Session = Depends(get_db)):
    return db.scalars(select(Source).order_by(Source.id)).all()


@router.post("/sources", response_model=SourceOut, status_code=201, summary="Add a literature source to poll")
def create_source(body: SourceIn, db: Session = Depends(get_db)):
    if body.kind not in {k.value for k in SourceKind}:
        raise HTTPException(
            422, f"Unsupported kind {body.kind!r}; supported: {[k.value for k in SourceKind]}"
        )
    source = Source(**body.model_dump())
    db.add(source)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, f"A source named {body.name!r} already exists") from exc
    return source


@router.patch("/sources/{source_id}", response_model=SourceOut)
def update_source(source_id: int, body: SourcePatch, db: Session = Depends(get_db)):
    source = db.get(Source, source_id)
    if not source:
        raise HTTPException(404, "Source not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(source, field, value)
    db.commit()
    return source


def _poll_in_background(source_id: int) -> None:
    db = new_session()
    try:
        source = db.get(Source, source_id)
        if source:
            source_service.poll_source(db, source, trigger="manual")
    finally:
        db.close()


@router.post(
    "/sources/{source_id}/poll", status_code=202, summary="Poll a source now (runs in the background)"
)
def poll_now(source_id: int, tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if not db.get(Source, source_id):
        raise HTTPException(404, "Source not found")
    tasks.add_task(_poll_in_background, source_id)
    return {"status": "started", "see": "/ingestion-runs"}


@router.post(
    "/literature/search",
    response_model=RunOut,
    summary="Search Europe PMC now and ingest the top results (waits until done)",
)
def search_and_ingest(body: LiteratureSearchIn, db: Session = Depends(get_db)):
    run, _, _ = source_service.ingest_query(db, body.query, limit=body.limit, trigger="manual")
    return run


@router.get("/ingestion-runs", response_model=list[RunOut], summary="Recent polls and searches, newest first")
def runs(source_id: int | None = None, limit: int = 50, db: Session = Depends(get_db)):
    q = select(IngestionRun).order_by(IngestionRun.id.desc()).limit(min(limit, 500))
    if source_id:
        q = q.where(IngestionRun.source_id == source_id)
    return db.scalars(q).all()
