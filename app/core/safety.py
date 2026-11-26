"""Safety helpers: PII redaction and prompt-injection defence.

Two rules the platform follows:
1. Personal data is masked before text reaches an LLM or a log line.
2. Document text is untrusted data. It is wrapped in delimiters, scanned for
   injection patterns, and the model is told to treat it as data, never as instructions.
"""

import re

_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?\d{10}(?!\d)")),
]

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules)",
    r"disregard (the |all )?(system|previous) (prompt|instructions)",
    r"you are now",
    r"reveal (the |your )?(system prompt|instructions|api key)",
    r"act as (an? )?(unrestricted|jailbroken)",
    r"</?(system|assistant)>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def redact_pii(text: str) -> str:
    for label, pattern in _PII_PATTERNS:
        text = pattern.sub(f"[{label}]", text)
    return text


def detect_injection(text: str) -> list[str]:
    """Return the suspicious phrases found (empty list = looks clean)."""
    return [m.group(0) for m in _INJECTION_RE.finditer(text)]


def wrap_untrusted(text: str, label: str = "document") -> str:
    """Delimit untrusted content so the model can tell data apart from instructions."""
    cleaned = text.replace("<<<", "").replace(">>>", "")  # stop content from closing the fence
    return f"<<<{label}\n{cleaned}\n{label}>>>"


# ── Medical emergencies ──────────────────────────────────────────────────────
# A helpline must not answer "crushing chest pain right now" with a literature summary.
# These patterns catch common phrasings of emergencies; a match returns emergency guidance
# instead of a RAG answer. Pattern matching misses some phrasings: it is a safety net, not triage.

_EMERGENCY_PATTERNS: dict[str, list[str]] = {
    "self_harm": [
        r"\b(kill|hurt|harm)(ing)? myself\b",
        r"\bsuicid",
        r"\bend (my|it all|my life)\b",
        r"\bwant to die\b",
        r"\b(took|taken|swallowed|had) (an? )?overdose\b",
        r"\boverdosed\b",
    ],
    "cardiac": [
        r"\bchest (pain|tightness|pressure)\b.*\b(now|right now|sudden|severe|spreading|arm|jaw|sweat)",
        r"\b(severe|sudden|crushing) chest (pain|tightness|pressure)\b",
        r"\bheart attack\b.*\b(now|having|think)",
    ],
    "stroke": [
        r"\b(face|mouth) (is )?(drooping|droops)\b",
        r"\bslurred speech\b",
        r"\bsudden(ly)? (weakness|numbness|can'?t (move|speak|see))\b",
        r"\bstroke\b.*\b(now|having|think)",
    ],
    "breathing": [
        r"\b(can'?t|cannot|unable to|struggling to) breathe?\b",
        r"\b(lips|face) (turning |are |is )?(blue|grey|gray)\b",
        r"\bchok(ing|ed)\b",
    ],
    "other": [
        r"\bunconscious\b",
        r"\b(not|isn'?t|won'?t) (waking up|breathing|responding)\b",
        r"\bseizure\b.*\b(now|won'?t stop|still)\b",
        r"\b(heavy|severe|uncontrolled) bleeding\b",
        r"\b(swallowed|drank|ate|took)\b.*\b(poison|bleach|pesticide|rat killer|kerosene)\b",
        r"\b(i|he|she|they|my \w+) (was|were|got|is|am) poisoned\b",
        r"\banaphyla\w*\b.*\b(now|having)\b",
        r"\bthroat (is )?(closing|swelling)\b",
    ],
}
_EMERGENCY_RE = {k: re.compile("|".join(v), re.IGNORECASE) for k, v in _EMERGENCY_PATTERNS.items()}


def detect_emergency(text: str) -> str | None:
    """The emergency category the message looks like, or None."""
    for category, pattern in _EMERGENCY_RE.items():
        if pattern.search(text):
            return category
    return None


def emergency_message(category: str, emergency_number: str, mental_health_line: str) -> str:
    if category == "self_harm":
        return (
            "It sounds like you may be going through something very difficult. You don't have to face "
            f"it alone. Please call {mental_health_line} now to talk to a trained counsellor, or "
            f"{emergency_number} if you are in immediate danger. If you can, reach out to someone you "
            "trust and stay with them."
        )
    return (
        f"This may be a medical emergency. Call {emergency_number} or go to the nearest emergency "
        "department now. Do not wait for an online answer. If someone is unresponsive or not "
        "breathing, call for help immediately."
    )


DISCLAIMER = (
    "This information comes from published medical literature and is not a diagnosis or medical "
    "advice. Please consult a qualified doctor about your situation."
)
