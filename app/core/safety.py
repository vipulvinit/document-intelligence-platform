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
