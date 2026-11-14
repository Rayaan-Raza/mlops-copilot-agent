"""Tool registry: schema generation, validation, permission checks, timeouts and tracing."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from agent.errors import (
    InvalidRequestError,
    PermissionDeniedError,
    ServiceUnavailableError,
    ToolError,
)
from agent.logging_config import redact
from agent.safety import ConfirmationStore
from agent.schemas import Permission, ToolCall, ToolResult, ToolStatus

log = logging.getLogger("agent.tools")


class NoArgs(BaseModel):
    """Input model for tools that take no arguments."""


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[Any], Any]
    input_model: type[BaseModel] = NoArgs
    permission: Permission = Permission.READ
    module: str = "core"
    timeout: float | None = None
    # Optional safety check run *before* a confirmation token is issued (e.g. namespace allow-list)
    precheck: Callable[[Any], None] | None = None
    # Human-readable summary of a write action shown when asking for confirmation
    summarize: Callable[[Any], str] | None = None

    def spec(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        schema.setdefault("properties", {})
        desc = self.description
        if self.permission == Permission.WRITE:
            desc += (
                " [STATE-CHANGING. Calling it does NOT run it: the system holds the call and asks"
                " the human to confirm with a token. If the user asked for this action, call it"
                " directly instead of asking for confirmation in text.]"
            )
        return {"name": self.name, "description": desc, "input_schema": schema}


class ToolRegistry:
    def __init__(
        self,
        default_timeout: float = 30.0,
        confirmations: ConfirmationStore | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self.default_timeout = default_timeout
        self.confirmations = confirmations or ConfirmationStore()
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")

    # ------------------------------------------------------------------ registration
    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str, description: str, **kwargs: Any):
        def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            self.register(Tool(name=name, description=description, func=func, **kwargs))
            return func

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    # ------------------------------------------------------------------ execution
    def execute(self, call: ToolCall, *, confirmed: bool = False) -> ToolResult:
        start = time.perf_counter()
        tool = self._tools.get(call.name)
        if tool is None:
            return self._finish(
                call, ToolStatus.INVALID_REQUEST, start, error=f"unknown tool {call.name!r}"
            )

        try:
            args = tool.input_model.model_validate(call.arguments or {})
        except ValidationError as exc:
            msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return self._finish(
                call, ToolStatus.INVALID_REQUEST, start, error=f"invalid arguments: {msg}"
            )

        if tool.permission == Permission.WRITE and not confirmed:
            try:
                if tool.precheck:
                    tool.precheck(args)
            except PermissionDeniedError as exc:
                return self._finish(call, ToolStatus.DENIED, start, error=str(exc))
            summary = (
                tool.summarize(args) if tool.summarize else f"{tool.name}({args.model_dump()})"
            )
            token = self.confirmations.issue(tool.name, args.model_dump(), summary)
            return self._finish(
                call,
                ToolStatus.CONFIRMATION_REQUIRED,
                start,
                data={
                    "action": summary,
                    "message": "Not executed. A human must confirm this action.",
                },
                token=token,
            )

        timeout = tool.timeout or self.default_timeout
        future = self._pool.submit(tool.func, args)
        try:
            data = future.result(timeout=timeout)
        except FutureTimeout:
            future.cancel()
            return self._finish(
                call, ToolStatus.TIMEOUT, start, error=f"tool exceeded {timeout}s timeout"
            )
        except PermissionDeniedError as exc:
            return self._finish(call, ToolStatus.DENIED, start, error=str(exc))
        except ServiceUnavailableError as exc:
            return self._finish(call, ToolStatus.UNAVAILABLE, start, error=str(exc))
        except InvalidRequestError as exc:
            return self._finish(call, ToolStatus.INVALID_REQUEST, start, error=str(exc))
        except (ConnectionError, OSError) as exc:
            return self._finish(
                call, ToolStatus.UNAVAILABLE, start, error=f"{type(exc).__name__}: {exc}"
            )
        except ToolError as exc:
            return self._finish(call, ToolStatus.ERROR, start, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - never let a tool crash the agent
            log.exception("unhandled tool exception", extra={"tool": call.name})
            return self._finish(call, ToolStatus.ERROR, start, error=f"{type(exc).__name__}: {exc}")
        return self._finish(call, ToolStatus.OK, start, data=data)

    def execute_confirmed(self, token: str) -> ToolResult:
        """Execute a previously requested write action after a human confirmed it."""
        try:
            action = self.confirmations.consume(token)
        except PermissionDeniedError as exc:
            call = ToolCall(id=f"confirm-{uuid.uuid4().hex[:8]}", name="confirm")
            return self._finish(call, ToolStatus.DENIED, time.perf_counter(), error=str(exc))
        call = ToolCall(
            id=f"confirm-{uuid.uuid4().hex[:8]}", name=action.tool, arguments=action.args
        )
        return self.execute(call, confirmed=True)

    # ------------------------------------------------------------------ helpers
    def _finish(
        self,
        call: ToolCall,
        status: ToolStatus,
        start: float,
        *,
        data: Any = None,
        error: str | None = None,
        token: str | None = None,
    ) -> ToolResult:
        duration = round((time.perf_counter() - start) * 1000, 2)
        tool = self._tools.get(call.name)
        log.info(
            "tool_call",
            extra={
                "event": "tool_call",
                "tool": call.name,
                "tool_module": tool.module if tool else None,
                "call_id": call.id,
                "status": status.value,
                "duration_ms": duration,
                "arguments": redact(call.arguments),
                "error": redact(error) if error else None,
            },
        )
        return ToolResult(
            tool=call.name,
            call_id=call.id,
            status=status,
            data=data,
            error=error,
            duration_ms=duration,
            confirmation_token=token,
        )
