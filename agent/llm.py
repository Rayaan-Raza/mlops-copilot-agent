"""LLM provider abstraction with native tool calling.

Conversation messages use a provider-neutral shape::

    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
    {"role": "tool", "call_id": str, "name": str, "content": str}

Providers: ``anthropic`` (Claude), ``openai`` (OpenAI or any OpenAI-compatible endpoint such as
Groq, Gemini or Ollama via ``OPENAI_BASE_URL``) and ``scripted`` (deterministic, used in tests).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.schemas import ToolCall


@dataclass
class LLMTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn: ...


# --------------------------------------------------------------------------- Anthropic
class AnthropicProvider:
    def __init__(self, model: str, max_tokens: int = 4096, client: Any = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.client, self.model, self.max_tokens = client, model, max_tokens

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m.get("tool_calls", []):
                    blocks.append(
                        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    )
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m["call_id"],
                    "content": m["content"],
                }
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=self._convert(messages),
        )
        turn = LLMTurn()
        for block in resp.content:
            if block.type == "text":
                turn.text += block.text
            elif block.type == "tool_use":
                turn.tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                )
        return turn


# --------------------------------------------------------------------------- OpenAI-compatible
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def openai_compatible_credentials(provider: str) -> tuple[str | None, str | None]:
    """(api_key, base_url) for an OpenAI-compatible provider.

    For ``groq`` only the Groq key and endpoint are used, so an unrelated ``OPENAI_API_KEY`` in
    the shell environment is never sent to Groq (or a Groq key to OpenAI).
    """
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY")
        return key, os.getenv("GROQ_BASE_URL") or GROQ_BASE_URL
    return os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL") or None


class OpenAIProvider:
    def __init__(self, model: str, client: Any = None, provider: str = "openai") -> None:
        if client is None:
            import openai

            api_key, base_url = openai_compatible_credentials(provider)
            # Free tiers (e.g. Groq) rate-limit per minute; the SDK honours `retry-after` on 429.
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=int(os.getenv("LLM_MAX_RETRIES", "6")),
            )
        self.client, self.model = client, model

    @staticmethod
    def _convert(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in m["tool_calls"]
                    ]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["call_id"], "content": m["content"]})
        return out

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn:
        resp = self.client.chat.completions.create(
            model=self.model,
            # Explicit cap: without it some gateways (OpenRouter) reserve the model's full output
            # window and reject low-credit accounts with HTTP 402.
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            messages=self._convert(system, messages),
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ],
        )
        msg = resp.choices[0].message
        turn = LLMTurn(text=msg.content or "")
        for c in msg.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__unparseable_arguments__": c.function.arguments}
            turn.tool_calls.append(ToolCall(id=c.id, name=c.function.name, arguments=args))
        return turn


# --------------------------------------------------------------------------- scripted
class ScriptedLLM:
    """Deterministic provider: each entry is an ``LLMTurn`` or a callable producing one.

    Used by the test-suite (no network, no API key) to exercise the orchestrator.
    """

    def __init__(self, script: list[LLMTurn | Callable[[list[dict]], LLMTurn]]) -> None:
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn:
        self.calls.append(list(messages))
        if not self.script:
            return LLMTurn(text="(script exhausted)")
        step = self.script.pop(0)
        return step(messages) if callable(step) else step


def build_provider(provider: str, model: str) -> LLMProvider:
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicProvider(model)
    if provider in ("openai", "groq", "gemini", "ollama"):
        return OpenAIProvider(model, provider=provider)
    raise ValueError(f"unknown LLM_PROVIDER {provider!r} (expected anthropic, openai or groq)")
