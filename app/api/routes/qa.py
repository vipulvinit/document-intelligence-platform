from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_llm
from app.llm.agent import run_agent
from app.llm.client import LLMUnavailable
from app.schemas.api import AgentRequest, AnswerOut, AskRequest
from app.services import qa_service, staleness_service
from app.storage import content_type, get_storage

router = APIRouter(tags=["Questions & answers"])


@router.post("/ask", summary="Ask a medical question; answered only from the literature, with citations")
def ask(req: AskRequest, db: Session = Depends(get_db), llm=Depends(get_llm)):
    try:
        return qa_service.ask(db, llm, req.question, req.article_ids)
    except LLMUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/agent", summary="Give the agent a goal; it calls tools and returns a trace")
def agent(req: AgentRequest, db: Session = Depends(get_db), llm=Depends(get_llm)):
    try:
        return run_agent(db, llm, req.goal)
    except LLMUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/answers", response_model=list[AnswerOut], summary="Past answers (filter: fresh or stale)")
def answers(status: str | None = None, limit: int = 100, db: Session = Depends(get_db)):
    return staleness_service.list_answers(db, status, min(limit, 500))


objects = APIRouter(tags=["Objects"])


@objects.get("/objects/{key:path}", summary="A stored figure image or uploaded file")
def get_object(key: str):
    data = get_storage().get(key)
    if data is None:
        raise HTTPException(404, "Object not found")
    return Response(content=data, media_type=content_type(key))
