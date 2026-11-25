from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.parsers.extractor import UnsupportedFileType
from app.schemas.document import VersionOut
from app.schemas.node import NodeOut
from app.services import document_service, version_service

router = APIRouter(prefix="/documents", tags=["Documents"])
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@router.post("", response_model=VersionOut, summary="Upload a document or a new version of it")
async def upload(name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 20 MB)")
    try:
        version, summary = document_service.ingest_document(db, name, file.filename or "upload", data)
    except UnsupportedFileType as exc:
        raise HTTPException(415, str(exc)) from exc
    except (document_service.DuplicateVersionError, document_service.EmptyDocumentError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return VersionOut(
        document=name,
        version=version.label,
        previous_version=summary.get("previous_version"),
        bump=summary.get("bump"),
        counts=summary.get("counts", {}),
        changes=summary.get("changes", []),
        stale_artifact_ids=summary.get("stale_artifact_ids", []),
        injection_flags=summary.get("injection_flags", {}),
    )


@router.get("/{name}/tree", response_model=list[NodeOut], summary="Parsed section tree of a version")
def tree(name: str, version: str | None = None, db: Session = Depends(get_db)):
    v = document_service.get_version(db, name, version)
    if not v:
        raise HTTPException(404, "Document or version not found")
    return v.nodes


@router.get("/{name}/versions", summary="Version history with semantic version labels")
def versions(name: str, db: Session = Depends(get_db)):
    result = version_service.list_versions(db, name)
    if not result:
        raise HTTPException(404, "Document not found")
    return result


@router.get("/{name}/diff", summary="What changed between two versions")
def diff(
    name: str, from_version: str | None = None, to_version: str | None = None, db: Session = Depends(get_db)
):
    result = version_service.diff_versions(db, name, from_version, to_version)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result
