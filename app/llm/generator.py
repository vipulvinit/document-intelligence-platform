"""Runs a prompt, validates the output, and retries once with the validation errors.

Citation ids: the model sees short ids ([S1], [S2]...) instead of long keys like
"PMC1234567#results/p2". Small local models copy short ids far more reliably; the ids are
mapped back to the real keys after validation.
"""

from app.core.logging import logger
from app.core.safety import redact_pii, wrap_untrusted
from app.llm import prompts
from app.llm.client import LLMClient
from app.llm.validator import Answer, validate_answer

MAX_ATTEMPTS = 2


def redact_context(context: dict[str, str]) -> dict[str, str]:
    """Mask personal data. The model sees, and is validated against, only the masked text."""
    return {k: redact_pii(v) for k, v in context.items()}


def format_context(context: dict[str, str]) -> str:
    """One line per excerpt, `[id] text`, wrapped as untrusted data."""
    lines = "\n".join(f"[{k}] {' '.join(v.split())}" for k, v in context.items())
    return wrap_untrusted(lines, label="excerpts")


def _run(llm: LLMClient, system: str, user: str, validate, context: dict[str, str]):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    errors: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if errors:
            messages[-1]["content"] = user + prompts.RETRY_SUFFIX.format(errors="\n".join(errors))
        resp = llm.chat(messages, json_mode=True)
        result, errors = validate(resp.content, context)
        if result is not None:
            return result, attempt, []
        logger.warning(f"Validation failed on attempt {attempt}: {errors}")
    return None, MAX_ATTEMPTS, errors


def answer_question(
    llm: LLMClient, question: str, context: dict[str, str]
) -> tuple[Answer | None, int, list[str]]:
    """`context` maps real citation keys to excerpt text. Citations in the result use the real keys."""
    aliases = {f"S{i}": key for i, key in enumerate(context, start=1)}
    aliased = redact_context({alias: context[key] for alias, key in aliases.items()})
    user = prompts.ANSWER_USER.format(context=format_context(aliased), question=redact_pii(question))
    answer, attempts, errors = _run(llm, prompts.ANSWER_SYSTEM, user, validate_answer, aliased)
    if answer is not None:
        text = answer.answer
        # "[S2]" in the prose stays as is: the API returns the S-id -> source mapping alongside.
        answer = Answer(answer=text, citations=[aliases[c] for c in answer.citations], quotes=answer.quotes)
        answer.alias_map = {a: k for a, k in aliases.items() if k in answer.citations}
    return answer, attempts, errors
