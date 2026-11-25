"""Logging that never writes personal data: every message passes through PII redaction."""

import sys

from loguru import logger

from app.core.safety import redact_pii


def _redacting_patcher(record):
    record["message"] = redact_pii(record["message"])


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.configure(patcher=_redacting_patcher)
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level} | {message}")


__all__ = ["logger", "setup_logging"]
