"""Question answering over documents (RAG): retrieve -> gate on relevance -> generate -> validate."""

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.safety import detect_injection
from app.llm.client import LLMClient
from app.llm.generator import answer_question
from app.services.retrieval_service import context_for_hits, search

NO_ANSWER = "I don't know based on the uploaded documents."


def ask(db: Session, llm: LLMClient, question: str, document_name: str | None = None) -> dict:
    hits = search(db, question, document_name)
    relevant = [h for h in hits if h.bm25 > 0 or h.cosine >= get_settings().min_retrieval_score]
    if not relevant:
        return {"answer": NO_ANSWER, "citations": [], "sources": [], "grounded": False}

    context = context_for_hits(db, relevant)
    # Quarantine: passages that look like prompt injection are reported, never sent to the model.
    flags = {k: f for k, v in context.items() if (f := detect_injection(v))}
    context = {k: v for k, v in context.items() if k not in flags}
    if not context:
        return {
            "answer": NO_ANSWER,
            "citations": [],
            "sources": [],
            "grounded": False,
            "injection_flags": flags,
        }
    answer, attempts, errors = answer_question(llm, question, context)
    if answer is None:
        return {
            "answer": NO_ANSWER,
            "citations": [],
            "sources": [],
            "grounded": False,
            "validation_errors": errors,
        }
    return {
        "answer": answer.answer,
        "citations": answer.citations,
        "quotes": answer.quotes,
        "grounded": bool(answer.citations),
        "attempts": attempts,
        "injection_flags": flags,
        "sources": [
            {"document": h.document, "version": h.version, "node_keys": h.node_keys, "score": h.score}
            for h in relevant
        ],
    }
