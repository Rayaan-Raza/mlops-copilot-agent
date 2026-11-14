"""The MLOps agent: LLM tool-calling loop with evidence tracking and anti-hallucination guards."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from pydantic import BaseModel, Field

from agent.config import Settings, get_settings
from agent.llm import LLMProvider, build_provider
from agent.logging_config import request_id_var
from agent.policy import assess_retraining
from agent.registry import Tool, ToolRegistry
from agent.safety import ConfirmationStore
from agent.schemas import AgentResponse, Evidence, ToolCall, ToolResult, ToolStatus

log = logging.getLogger("agent.orchestrator")

SYSTEM_PROMPT = """You are MLOps Copilot, an operations agent for an ML system built on DVC \
(dataset versioning), MLflow (experiments + model registry), Apache Airflow 3 (training \
pipeline `training_pipeline`) and Kubernetes (FastAPI inference service `ml-inference` in \
namespace `{namespace}`).

RULES - follow strictly:
1. Never state an operational fact (a version, metric, status, run id, pod state, log line) \
unless it came from a tool result in THIS conversation. Your training knowledge is not evidence.
2. If a tool returns status unavailable / timeout / error / denied, say the information could not \
be verified. Do NOT guess or fill in plausible values.
3. Use as many tools as the question needs; for multi-part questions call tools for every part. \
To link a model to its data: read the run's `dvc.data_md5` / `dvc.dataset_version` tags \
(mlflow_get_run_details) and resolve them with dvc_resolve_dataset_version.
4. The best metric does not mean production-ready: always mention quality gate, overfitting gap, \
data provenance and deployment status caveats returned by tools.
5. State-changing tools (trigger DAG, restart/scale, dataset checkout) only ever produce a \
confirmation request; never call them unless the user explicitly asked for that action. Never \
trigger retraining on your own - you may only recommend it, using agent_assess_retraining, and \
only on evidence (freshness policy, failed quality gate, below-threshold metric, changed data, \
user-reported degradation).

FINAL ANSWER FORMAT (markdown, exactly these headings):
## Summary
<2-4 sentences>
## Facts
- <fact> [evidence: <call_id>]   <- every fact MUST cite the call_id of the tool result it came from
## Uncertainties
- <what could not be verified and why>
## Recommended actions
- <action and the evidence that motivates it>
"""

FACT_CITATION_RE = re.compile(r"\[evidence:\s*([^\]]+)\]", re.I)
FAILED = {
    ToolStatus.UNAVAILABLE,
    ToolStatus.TIMEOUT,
    ToolStatus.ERROR,
    ToolStatus.DENIED,
    ToolStatus.INVALID_REQUEST,
}


class RetrainingArgs(BaseModel):
    model_trained_at_ms: int | None = Field(
        None, description="best/deployed run start_time in epoch ms"
    )
    latest_f1: float | None = None
    last_quality_gate: str | None = Field(None, description="'passed' / 'failed' from MLflow tags")
    last_pipeline_state: str | None = Field(None, description="latest Airflow run state")
    data_changed_since_training: bool | None = Field(
        None, description="True if current DVC md5 differs from the run's dvc.data_md5"
    )
    reported_degradation: str | None = Field(None, description="only if the USER supplied it")


def build_default_registry(settings: Settings | None = None) -> ToolRegistry:
    from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools

    settings = settings or get_settings()
    registry = ToolRegistry(
        default_timeout=settings.tool_timeout_seconds,
        confirmations=ConfirmationStore(settings.confirmation_ttl_seconds),
    )
    dvc_tools.register(registry, settings)
    mlflow_tools.register(registry, settings)
    airflow_tools.register(registry, settings)
    kubernetes_tools.register(registry, settings)
    register_policy_tool(registry, settings)
    return registry


def register_policy_tool(registry: ToolRegistry, settings: Settings) -> None:
    def _assess(a: RetrainingArgs):
        return assess_retraining(
            **a.model_dump(),
            freshness_days=settings.retrain_freshness_days,
            min_f1=settings.retrain_min_f1,
        ).model_dump()

    registry.register(
        Tool(
            "agent_assess_retraining",
            "Apply the retraining policy (freshness, min F1, quality gate, data change) to evidence "
            "gathered from other tools. Returns a recommendation only; never starts retraining.",
            _assess,
            RetrainingArgs,
            module="policy",
        )
    )


def parse_sections(text: str) -> dict[str, list[str] | str]:
    sections: dict[str, list[str] | str] = {
        "summary": "",
        "facts": [],
        "uncertainties": [],
        "recommendations": [],
    }
    key_map = {
        "summary": "summary",
        "facts": "facts",
        "uncertainties": "uncertainties",
        "recommended actions": "recommendations",
        "recommendations": "recommendations",
    }
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            current = key_map.get(line.lstrip("#").strip().lower())
            continue
        if not line or current is None:
            continue
        if current == "summary":
            sections["summary"] = (str(sections["summary"]) + " " + line).strip()
        elif line[:1] in "-*":
            sections[current].append(line[1:].strip())  # type: ignore[union-attr]
        elif sections[current]:
            sections[current][-1] += " " + line  # type: ignore[index]
    return sections


class Agent:
    def __init__(
        self,
        registry: ToolRegistry,
        llm: LLMProvider,
        settings: Settings | None = None,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.settings = settings or get_settings()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Agent:
        settings = settings or get_settings()
        return cls(
            build_default_registry(settings),
            build_provider(settings.llm_provider, settings.llm_model),
            settings,
        )

    # ------------------------------------------------------------------ main loop
    def run(self, message: str, request_id: str | None = None) -> AgentResponse:
        request_id = request_id or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            return self._run(message, request_id, started)
        finally:
            request_id_var.reset(token)

    def _run(self, message: str, request_id: str, started: float) -> AgentResponse:
        system = SYSTEM_PROMPT.format(namespace=self.settings.k8s_namespace)
        specs = self.registry.specs()
        messages: list[dict] = [{"role": "user", "content": message}]
        results: list[ToolResult] = []
        final_text = ""
        steps = 0  # updated by the loop below

        log.info("agent_request", extra={"event": "agent_request", "chars": len(message)})
        for steps in range(1, self.settings.max_steps + 1):  # noqa: B007 - used after loop
            try:
                turn = self.llm.complete(system, messages, specs)
            except Exception as exc:  # noqa: BLE001
                log.exception("llm_failure")
                final_text = (
                    "## Summary\nThe language model could not be reached, so no analysis was produced.\n"
                    f"## Uncertainties\n- LLM call failed: {type(exc).__name__}"
                )
                break
            if not turn.tool_calls:
                final_text = turn.text
                break
            messages.append(
                {"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls}
            )
            for call in turn.tool_calls:
                result = self.registry.execute(call)
                results.append(result)
                messages.append(
                    {
                        "role": "tool",
                        "call_id": call.id,
                        "name": call.name,
                        "content": result.for_llm(),
                    }
                )
        else:
            final_text = (
                "## Summary\nStopped after reaching the maximum number of reasoning steps.\n"
                "## Uncertainties\n- The investigation is incomplete (step limit reached)."
            )

        response = self._build_response(request_id, final_text, results, steps)
        log.info(
            "agent_response",
            extra={
                "event": "agent_response",
                "steps": steps,
                "tool_calls": len(results),
                "failed_tool_calls": sum(r.status in FAILED for r in results),
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return response

    # ------------------------------------------------------------------ guards
    def _build_response(
        self, request_id: str, text: str, results: list[ToolResult], steps: int
    ) -> AgentResponse:
        sections = parse_sections(text)
        ok_ids = {r.call_id for r in results if r.status == ToolStatus.OK}

        facts: list[str] = []
        llm_uncertainties: list[str] = list(sections["uncertainties"])  # type: ignore[arg-type]
        uncertainties: list[str] = list(llm_uncertainties)
        removed = 0
        for fact in sections["facts"]:  # type: ignore[union-attr]
            cited = {c.strip() for m in FACT_CITATION_RE.findall(fact) for c in m.split(",")}
            if cited and cited <= ok_ids:
                facts.append(fact)
            else:
                # Guard: a "fact" without a successful tool result behind it is not a fact.
                reason = (
                    "no evidence cited"
                    if not cited
                    else "cited evidence "
                    + ", ".join(sorted(cited - ok_ids))
                    + " is missing or failed"
                )
                uncertainties.append(
                    f"Unverified claim removed from facts ({reason}): "
                    f"{FACT_CITATION_RE.sub('', fact).strip()}"
                )
                removed += 1

        # Guard: every failed tool must be surfaced, whatever the LLM wrote.
        failed = [r for r in results if r.status in FAILED]
        for r in failed:
            if not any(r.tool in u for u in llm_uncertainties):
                uncertainties.append(
                    f"{r.tool} returned {r.status.value}: {r.error or 'no detail'} "
                    f"[evidence: {r.call_id}]"
                )

        # Guard: the free-text answer is flagged when evidence was missing or claims were removed.
        if failed or removed:
            banner = (
                f"> WARNING: {len(failed)} tool call(s) failed and {removed} unsupported claim(s) "
                "were removed. Only the Facts list is verified; treat anything else as unverified.\n\n"
            )
            text = banner + text

        pending = [
            {
                "tool": r.tool,
                "action": (r.data or {}).get("action"),
                "confirmation_token": r.confirmation_token,
            }
            for r in results
            if r.status == ToolStatus.CONFIRMATION_REQUIRED
        ]
        if not results:
            uncertainties.append(
                "No tools were called, so nothing in this answer is backed by live system evidence."
            )

        return AgentResponse(
            request_id=request_id,
            answer=text.strip() or "(no answer produced)",
            facts=facts,
            uncertainties=uncertainties,
            recommendations=list(sections["recommendations"]),  # type: ignore[arg-type]
            evidence=[Evidence(tool=r.tool, call_id=r.call_id, status=r.status) for r in results],
            pending_confirmations=pending,
            steps=steps,
        )

    def confirm(self, token: str) -> ToolResult:
        return self.registry.execute_confirmed(token)


def evidence_dump(results: list[ToolResult]) -> str:  # pragma: no cover - debugging helper
    return json.dumps([r.model_dump(mode="json") for r in results], indent=2, default=str)


__all__ = ["Agent", "ToolCall", "build_default_registry", "SYSTEM_PROMPT"]
