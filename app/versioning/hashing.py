"""Content hashing. Text is normalised first, so whitespace or case changes don't count as edits."""

import hashlib
import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
