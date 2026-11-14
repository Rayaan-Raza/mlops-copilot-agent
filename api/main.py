"""FastAPI interface for the MLOps agent.

    uvicorn api.main:app --port 8001

Endpoints:
* ``POST /chat``    - ask the agent a question; returns facts, uncertainties, evidence
* ``POST /confirm`` - a human confirms a pending state-changing action by token
* ``GET  /tools``   - tool catalogue with permission levels
* ``GET  /health``
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from agent.logging_config import configure_logging, request_id_var
from agent.orchestrator import Agent
from agent.schemas import AgentResponse, ChatRequest, ConfirmRequest, ToolResult, ToolStatus

log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    if getattr(app.state, "agent", None) is None:
        app.state.agent = Agent.from_settings()
    yield


app = FastAPI(title="MLOps Copilot Agent", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


def _agent(request: Request) -> Agent:
    return request.app.state.agent


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tools")
def tools(request: Request) -> dict:
    reg = _agent(request).registry
    return {
        "tools": [
            {"name": n, "module": reg.get(n).module, "permission": reg.get(n).permission.value}
            for n in reg.names
        ]
    }


@app.post("/chat", response_model=AgentResponse)
async def chat(body: ChatRequest, request: Request) -> AgentResponse:
    rid = request_id_var.get()
    return await run_in_threadpool(_agent(request).run, body.message, rid)


@app.post("/confirm", response_model=ToolResult)
async def confirm(body: ConfirmRequest, request: Request) -> ToolResult:
    result = await run_in_threadpool(_agent(request).confirm, body.confirmation_token)
    if result.status == ToolStatus.DENIED:
        raise HTTPException(status_code=403, detail=result.error)
    return result
