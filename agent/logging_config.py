"""Structured JSON logging with request ids and secret redaction."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

SECRET_KEY_RE = re.compile(
    r"(pass(word)?|secret|token|api[_-]?key|authorization|bearer|credential)", re.I
)
REDACTED = "***REDACTED***"
SECRET_VALUE_RES = [
    (re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.I), r"\1" + REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_\-*]{4,}"), REDACTED),  # API keys, incl. provider-masked
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"), REDACTED),  # JWT
    (re.compile(r"((?:password|token|api_key|secret)\s*[=:]\s*)\S+", re.I), r"\1" + REDACTED),
]


def redact(value: Any) -> Any:
    """Recursively redact secret-looking keys and values."""
    if isinstance(value, dict):
        return {
            k: (REDACTED if SECRET_KEY_RE.search(str(k)) else redact(v)) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for rx, repl in SECRET_VALUE_RES:
            value = rx.sub(repl, value)
        return value
    return value


class JsonFormatter(logging.Formatter):
    RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "msg": redact(record.getMessage()),
        }
        for key, val in record.__dict__.items():
            if key not in self.RESERVED and key not in payload:
                payload[key] = redact(val)
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "urllib3", "kubernetes"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
