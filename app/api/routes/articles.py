from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models import Article
from app.parsers.extractor import UnsupportedFileType
from app.schemas.api import ArticleOut, IngestOut
from app.schemas.node import NodeOut
from app.services import ingestion_service, version_service

router = APIRouter(prefix="/articles", tags=["Articles"])
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


def ingest_out(result: ingestion_service.IngestResult) -> IngestOut:
    s = result.summary or {}
    return IngestOut(
        status=result.status,
        article_id=result.article.id,
        external_id=result.article.external_id,
        title=result.article.title,
        version=result.version.label if result.version else None,
        previous_version=s.get("previous_version"),
        bump=s.get("bump"),
        counts=s.get("counts", {}),
        changes=s.get("changes", []),
        stale_answer_ids=s.get("stale_answer_ids", []),
        injection_flags=s.get("injection_flags", {}),
        figures_stored=s.get("figures_stored", 0),
        media_nodes=s.get("media_nodes", 0),
    )


def article_out(a: Article) -> ArticleOut:
    out = ArticleOut.model_validate(a)
    if a.versions:
        out.latest_version, out.content_kind = a.versions[-1].label, a.versions[-1].content_kind
    return out


@router.post("/upload", response_model=IngestOut, summary="Upload a guideline or article (PDF, JATS XML, MD)")
async def upload(title: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 30 MB)")
    try:
        result = ingestion_service.ingest_upload(db, title, file.filename or "upload", data)
    except UnsupportedFileType as exc:
        raise HTTPException(415, str(exc)) from exc
    except ingestion_service.EmptyDocumentError as exc:
        raise HTTPException(422, str(exc)) from exc
    return ingest_out(result)


@router.get("", response_model=list[ArticleOut], summary="Articles in the knowledge base, newest first")
def list_articles(q: str | None = None, limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    query = select(Article).order_by(Article.id.desc()).limit(min(limit, 200)).offset(offset)
    if q:
        query = query.where(Article.title.ilike(f"%{q}%"))
    return [article_out(a) for a in db.scalars(query).all()]


@router.get("/{article_id}", response_model=ArticleOut)
def get_article(article_id: int, db: Session = Depends(get_db)):
    a = db.get(Article, article_id)
    if not a:
        raise HTTPException(404, "Article not found")
    return article_out(a)


@router.get("/{article_id}/tree", response_model=list[NodeOut], summary="Parsed section tree of a version")
def tree(article_id: int, version: str | None = None, db: Session = Depends(get_db)):
    v = ingestion_service.get_version(db, article_id, version)
    if not v:
        raise HTTPException(404, "Article or version not found")
    return v.nodes


@router.get("/{article_id}/versions", summary="Version history with semantic version labels")
def versions(article_id: int, db: Session = Depends(get_db)):
    result = version_service.list_versions(db, article_id)
    if not result:
        raise HTTPException(404, "Article not found")
    return result


@router.get("/{article_id}/diff", summary="What changed between two versions")
def diff(
    article_id: int,
    from_version: str | None = None,
    to_version: str | None = None,
    db: Session = Depends(get_db),
):
    result = version_service.diff_versions(db, article_id, from_version, to_version)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post(
    "/{article_id}/retract", summary="Mark as retracted: excluded from answers, citing answers go stale"
)
def retract(article_id: int, db: Session = Depends(get_db)):
    if not db.get(Article, article_id):
        raise HTTPException(404, "Article not found")
    return {"stale_answer_ids": ingestion_service.set_retracted(db, article_id)}
