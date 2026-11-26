"""Question answering over the medical literature (RAG).

  emergency check -> retrieve -> gate on relevance -> quarantine injected passages
  -> generate -> validate citations and quotes (1 retry) -> store the answer with what it cited

Every answer is saved with the content hash of each passage it cites, so it can be marked stale
when an article is revised or retracted, and so the judge agent can score it later.
"""

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.safety import DISCLAIMER, detect_emergency, detect_injection, emergency_message, redact_pii
from app.llm import prompts
from app.llm.client import LLMClient
from app.llm.generator import answer_question
from app.models import Answer, AnswerSource, Article
from app.services.retrieval_service import ContextItem, context_for_hits, search

NO_ANSWER = "I don't know: the medical literature in the knowledge base does not answer this question."


def _no_answer(reason: str, **extra) -> dict:
    return {
        "answer": NO_ANSWER,
        "grounded": False,
        "citations": [],
        "reason": reason,
        # Hook for the on-demand agent: nothing relevant is stored, so the literature should be searched.
        "needs_literature_search": reason == "no_relevant_passages",
        "disclaimer": DISCLAIMER,
    } | extra


def _citation(db: Session, alias: str, item: ContextItem) -> dict:
    article = db.get(Article, item.article_id)
    return {
        "id": alias,
        "key": item.key,
        "section": item.section,
        "kind": item.kind,
        "figure_url": item.figure_url,
        **{k: v for k, v in article.citation().items() if k != "article_id"},
        "article_id": article.id,
    }


def ask(db: Session, llm: LLMClient, question: str, article_ids: list[int] | None = None) -> dict:
    settings = get_settings()
    if category := detect_emergency(question):
        return {
            "answer": emergency_message(category, settings.emergency_number, settings.mental_health_helpline),
            "emergency": category,
            "grounded": False,
            "citations": [],
        }

    hits = search(db, question, article_ids)
    relevant = [h for h in hits if h.bm25 > 0 or h.cosine >= settings.min_retrieval_score]
    if not relevant:
        return _no_answer("no_relevant_passages")

    items = context_for_hits(db, relevant)
    # Quarantine: passages that look like prompt injection are reported, never sent to the model.
    flags = {i.key: f for i in items if (f := detect_injection(i.text))}
    items = [i for i in items if i.key not in flags]
    if not items:
        return _no_answer("only_quarantined_passages", injection_flags=flags)

    by_key = {i.key: i for i in items}
    answer, attempts, errors = answer_question(llm, question, {i.key: i.text for i in items})
    if answer is None:
        return _no_answer("validation_failed", validation_errors=errors, injection_flags=flags)
    if not answer.citations:
        return _no_answer("model_found_no_answer", attempts=attempts, injection_flags=flags)

    record = Answer(
        question=redact_pii(question),
        answer=answer.answer,
        payload={
            "quotes": answer.quotes,
            "attempts": attempts,
            "model": settings.llm_model if settings.llm_provider != "mock" else "mock",
            "prompt_version": prompts.PROMPT_VERSION,
            "alias_map": answer.alias_map,
            "retrieved": [i.key for i in items],
        },
    )
    for key in answer.citations:
        item = by_key[key]
        record.sources.append(
            AnswerSource(article_id=item.article_id, node_key=item.node_key, content_hash=item.content_hash)
        )
    db.add(record)
    db.commit()

    citations = [_citation(db, alias, by_key[key]) for alias, key in answer.alias_map.items()]
    return {
        "answer_id": record.id,
        "answer": answer.answer,
        "grounded": True,
        "citations": citations,
        "quotes": answer.quotes,
        "attempts": attempts,
        "injection_flags": flags,
        "disclaimer": DISCLAIMER,
    }
