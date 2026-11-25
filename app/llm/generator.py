"""Runs a prompt, validates the output, and retries once with the validation errors."""

from app.core.logging import logger
from app.core.safety import redact_pii, wrap_untrusted
from app.llm import prompts
from app.llm.client import LLMClient
from app.llm.validator import Answer, TestCaseSet, validate_answer, validate_test_cases

MAX_ATTEMPTS = 2


def redact_context(context: dict[str, str]) -> dict[str, str]:
    """Mask personal data. The model sees, and is validated against, only the masked text."""
    return {k: redact_pii(v) for k, v in context.items()}


def format_context(context: dict[str, str]) -> str:
    """One line per excerpt, `[node_key] text`, wrapped as untrusted data."""
    lines = "\n".join(f"[{k}] {v}" for k, v in context.items())
    return wrap_untrusted(lines)


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
    context = redact_context(context)
    user = prompts.ANSWER_USER.format(context=format_context(context), question=redact_pii(question))
    return _run(llm, prompts.ANSWER_SYSTEM, user, validate_answer, context)


def generate_test_cases(
    llm: LLMClient, context: dict[str, str], max_cases: int = 8
) -> tuple[TestCaseSet | None, int, list[str]]:
    context = redact_context(context)
    user = prompts.TEST_CASE_USER.format(context=format_context(context), max_cases=max_cases)
    return _run(llm, prompts.TEST_CASE_SYSTEM, user, validate_test_cases, context)
