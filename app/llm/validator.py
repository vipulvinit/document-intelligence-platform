"""Validation of LLM output: schema checks (Pydantic) plus grounding checks.

Grounding means every citation points at an excerpt the model was actually given, and
every quoted sentence really appears in that excerpt. This is the main defence against
hallucinated sources.
"""

import json
import re
from difflib import SequenceMatcher

from pydantic import BaseModel, Field, ValidationError

from app.versioning.hashing import normalize


class Answer(BaseModel):
    answer: str = Field(min_length=1)
    citations: list[str] = []
    quotes: list[str] = []
    alias_map: dict[str, str] = Field(default_factory=dict, exclude=True)  # "S1" -> real citation key


def _letters(text: str) -> str:
    """Lowercase words only: punctuation and spacing differences are ignored."""
    return normalize(re.sub(r"[^\w\s]", " ", text))


def quote_supported(quote: str, source: str, threshold: float = 0.85) -> bool:
    """True if `quote` appears in `source`, allowing small differences (punctuation, case)."""
    q, s = _letters(quote), _letters(source)
    if q in s:
        return True
    m = SequenceMatcher(None, q, s).find_longest_match(0, len(q), 0, len(s))
    return len(q) > 0 and m.size / len(q) >= threshold


def parse_json(raw: str | None) -> tuple[dict | None, list[str]]:
    if not raw:
        return None, ["empty response"]
    text = raw.strip()
    if text.startswith("```"):  # tolerate fenced output
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text), []
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON: {exc}"]


def validate_answer(raw: str | None, context: dict[str, str]) -> tuple[Answer | None, list[str]]:
    data, errors = parse_json(raw)
    if errors:
        return None, errors
    try:
        ans = Answer.model_validate(data)
    except ValidationError as exc:
        return None, [e["msg"] for e in exc.errors()]
    unknown = [c for c in ans.citations if c not in context]
    if unknown:
        errors.append(f"citations not in the provided excerpts: {unknown}")
    cited_text = " ".join(context[c] for c in ans.citations if c in context)
    for q in ans.quotes:
        if not quote_supported(q, cited_text):
            errors.append(f"quote not found in cited excerpts: {q[:80]!r}")
    return (ans if not errors else None), errors
