from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_llm
from app.llm.agent import run_agent
from app.schemas.document import AgentRequest, AskRequest
from app.services import qa_service

router = APIRouter(tags=["Retrieval & agent"])


@router.post("/ask", summary="Ask a question; answered only from the documents, with citations")
def ask(req: AskRequest, db: Session = Depends(get_db), llm=Depends(get_llm)):
    return qa_service.ask(db, llm, req.question, req.document_name)


@router.post("/agent", summary="Give the agent a goal; it calls read-only tools and returns a trace")
def agent(req: AgentRequest, db: Session = Depends(get_db), llm=Depends(get_llm)):
    return run_agent(db, llm, req.goal)
