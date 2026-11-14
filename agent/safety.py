"""Safety controls: confirmation tokens for state-changing tools and namespace / DAG allow-lists."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

from agent.errors import PermissionDeniedError


def _fingerprint(tool: str, args: dict[str, Any]) -> str:
    blob = json.dumps({"tool": tool, "args": args}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class PendingAction:
    tool: str
    args: dict[str, Any]
    fingerprint: str
    expires_at: float
    summary: str


class ConfirmationStore:
    """One-time, expiring tokens bound to an exact (tool, arguments) pair.

    A write tool is never executed on the LLM's say-so: the first call returns a token, and only a
    human calling ``/confirm`` with that token executes the *identical* action.
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl = ttl_seconds
        self._pending: dict[str, PendingAction] = {}
        self._lock = threading.Lock()

    def issue(self, tool: str, args: dict[str, Any], summary: str) -> str:
        token = secrets.token_urlsafe(16)
        with self._lock:
            self._purge()
            self._pending[token] = PendingAction(
                tool, args, _fingerprint(tool, args), time.time() + self.ttl, summary
            )
        return token

    def consume(self, token: str) -> PendingAction:
        with self._lock:
            self._purge()
            action = self._pending.pop(token, None)
        if action is None:
            raise PermissionDeniedError("confirmation token is invalid, expired or already used")
        if action.fingerprint != _fingerprint(action.tool, action.args):  # pragma: no cover
            raise PermissionDeniedError("confirmation token does not match the requested action")
        return action

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            self._purge()
            return [
                {"tool": a.tool, "args": a.args, "summary": a.summary}
                for a in self._pending.values()
            ]

    def _purge(self) -> None:
        now = time.time()
        for tok in [t for t, a in self._pending.items() if a.expires_at < now]:
            del self._pending[tok]


def require_namespace_allowed(namespace: str, allowed: list[str]) -> None:
    if namespace not in allowed:
        raise PermissionDeniedError(
            f"namespace {namespace!r} is not in the approved list {allowed}; "
            "state-changing Kubernetes operations are only permitted there"
        )


def require_dag_allowed(dag_id: str, allowed: list[str]) -> None:
    if dag_id not in allowed:
        raise PermissionDeniedError(f"DAG {dag_id!r} is not approved for triggering ({allowed})")
