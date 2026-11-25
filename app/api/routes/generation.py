from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_llm
from app.schemas.generation import ArtifactOut, GenerateRequest
from app.schemas.selection import SelectionRequest
from app.services import (
    document_service,
    generation_service,
    selection_service,
    staleness_service,
)

router = APIRouter(prefix="/documents/{name}", tags=["Generation & traceability"])


@router.post("/selections", summary="Save a set of sections for later generation")
def save_selection(name: str, req: SelectionRequest, db: Session = Depends(get_db)):
    doc = document_service.get_document(db, name)
    if not doc:
        raise HTTPException(404, "Document not found")
    sel = selection_service.save_selection(db, doc.id, req.name, req.node_keys)
    return {"id": sel.id, "name": sel.name, "node_keys": sel.node_keys}


@router.post("/test-cases", response_model=list[ArtifactOut], summary="Generate QA test cases from sections")
def generate(name: str, req: GenerateRequest, db: Session = Depends(get_db), llm=Depends(get_llm)):
    version = document_service.get_version(db, name, req.version)
    if not version:
        raise HTTPException(404, "Document or version not found")
    try:
        return generation_service.generate_for_selection(db, llm, version, req.node_keys, req.max_cases)
    except generation_service.GenerationFailed as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get(
    "/test-cases", response_model=list[ArtifactOut], summary="List test cases (filter: fresh or stale)"
)
def list_cases(name: str, status: str | None = None, db: Session = Depends(get_db)):
    doc = document_service.get_document(db, name)
    if not doc:
        raise HTTPException(404, "Document not found")
    return staleness_service.list_artifacts(db, doc.id, status)
