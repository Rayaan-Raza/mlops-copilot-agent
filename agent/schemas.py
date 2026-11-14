"""Structured inputs/outputs shared by tools, the orchestrator and the API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID_REQUEST = "invalid_request"
    DENIED = "denied"
    CONFIRMATION_REQUIRED = "confirmation_required"


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"  # changes system state -> needs explicit user confirmation


class ToolResult(BaseModel):
    """Envelope returned by every tool call. The LLM only ever sees this shape."""

    tool: str
    call_id: str
    status: ToolStatus
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    confirmation_token: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.OK

    def for_llm(self, max_chars: int = 6000) -> str:
        text = self.model_dump_json(exclude_none=True)
        if len(text) > max_chars:
            text = text[:max_chars] + '..."(truncated)"'
        return text


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    tool: str
    call_id: str
    status: ToolStatus


class AgentResponse(BaseModel):
    request_id: str
    answer: str
    facts: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    pending_confirmations: list[dict[str, Any]] = Field(default_factory=list)
    steps: int = 0


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None


class ConfirmRequest(BaseModel):
    confirmation_token: str = Field(..., min_length=8)
