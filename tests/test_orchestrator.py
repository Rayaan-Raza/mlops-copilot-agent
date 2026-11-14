"""End-to-end agent tests: scripted LLM + all four tool modules backed by fakes."""

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.config import Settings
from agent.llm import AnthropicProvider, LLMTurn, ScriptedLLM
from agent.orchestrator import Agent, parse_sections, register_policy_tool
from agent.registry import ToolRegistry
from agent.schemas import ToolCall, ToolStatus
from agent.tools import airflow_tools, kubernetes_tools, mlflow_tools
from agent.tools.airflow_tools import AirflowClient
from agent.tools.kubernetes_tools import KubeGateway
from tests.fakes import FakeMlflowClient
from tests.k8s_fakes import FakeApps, FakeCore, pod
from tests.test_airflow_tools import handler_factory

SETTINGS = Settings(
    k8s_namespace="mlops-agent",
    k8s_allowed_namespaces=["mlops-agent"],
    airflow_allowed_dags=["training_pipeline"],
    max_steps=6,
)


def stub_dvc(registry: ToolRegistry, md5_to_tag: dict[str, str]):
    """Minimal DVC tool stand-in (the real module is tested against a git repo elsewhere)."""
    from agent.registry import Tool
    from agent.tools.dvc_tools import ResolveArgs

    def resolve(a: ResolveArgs):
        tag = md5_to_tag.get(a.md5)
        return {
            "md5": a.md5,
            "matching_version_tags": [{"tag": tag}] if tag else [],
            "resolved": bool(tag),
        }

    registry.register(
        Tool("dvc_resolve_dataset_version", "resolve md5", resolve, ResolveArgs, module="dvc")
    )


def full_registry(mlflow_client=None, airflow_calls=None):
    reg = ToolRegistry(default_timeout=5)
    mlflow_tools.register(reg, SETTINGS, mlflow_client or FakeMlflowClient())
    http = httpx.Client(
        transport=httpx.MockTransport(
            handler_factory(airflow_calls if airflow_calls is not None else [])
        )
    )
    airflow_tools.register(reg, SETTINGS, AirflowClient("http://af", "u", "p", http=http))
    kubernetes_tools.register(
        reg,
        SETTINGS,
        KubeGateway(FakeCore([pod("ml-inference-a"), pod("ml-inference-b")]), FakeApps()),
    )
    stub_dvc(reg, {"b" * 32: "data-v2.0"})
    register_policy_tool(reg, SETTINGS)
    return reg


def tc(id_, name, **args):
    return ToolCall(id=id_, name=name, arguments=args)


MANDATORY = (
    "Find the best recent model, identify the dataset version used to train it, check the "
    "deployment status, and explain whether retraining is necessary."
)


def test_mandatory_multi_tool_scenario_uses_all_four_systems_without_retraining():
    airflow_calls: list = []
    final = """## Summary
Run 222 (random forest) is the strongest gate-passing candidate; it was trained on data-v2.0 and the service is healthy.
## Facts
- Best F1 run overall is 333 but it failed the quality gate [evidence: c1]
- Run 222 used dataset md5 bbbb (data-v2.0) [evidence: c2, c3]
- Both ml-inference pods are ready, version 1.0.0 [evidence: c4, c5]
- Latest training_pipeline run failed [evidence: c6]
## Uncertainties
- No production performance data was available.
## Recommended actions
- Investigate the failed pipeline run before retraining [evidence: c6, c7]
"""
    llm = ScriptedLLM(
        [
            LLMTurn(tool_calls=[tc("c1", "mlflow_find_best_run", metric="f1")]),
            LLMTurn(
                tool_calls=[
                    tc("c2", "mlflow_get_run_details", run_id="2" * 32),
                    tc("c3", "dvc_resolve_dataset_version", md5="b" * 32),
                ]
            ),
            LLMTurn(
                tool_calls=[
                    tc("c4", "k8s_find_unhealthy_pods"),
                    tc("c5", "k8s_get_deployed_version"),
                    tc("c6", "airflow_get_dag_runs", dag_id="training_pipeline"),
                ]
            ),
            LLMTurn(
                tool_calls=[
                    tc(
                        "c7",
                        "agent_assess_retraining",
                        latest_f1=0.982,
                        last_quality_gate="passed",
                        last_pipeline_state="failed",
                    )
                ]
            ),
            LLMTurn(text=final),
        ]
    )
    agent = Agent(full_registry(airflow_calls=airflow_calls), llm, SETTINGS)
    resp = agent.run(MANDATORY)

    assert all(e.status == ToolStatus.OK for e in resp.evidence), resp.evidence
    modules = {agent.registry.get(e.tool).module for e in resp.evidence}
    assert {"mlflow", "dvc", "kubernetes", "airflow", "policy"} <= modules
    assert len(resp.facts) == 4 and resp.recommendations
    # retraining was NOT initiated: no write tool executed, no POST to Airflow
    assert not any(m == "POST" and "dagRuns" in p for m, p in airflow_calls)
    assert resp.pending_confirmations == []
    # tool results were fed back to the LLM
    assert any(m["role"] == "tool" and m["call_id"] == "c6" for m in llm.calls[-1])


def test_agent_does_not_invent_results_when_tool_unavailable():
    reg = full_registry(mlflow_client=FakeMlflowClient(fail=ConnectionError("Connection refused")))
    hallucinated = """## Summary
The best model has F1 0.97.
## Facts
- The best run is abc123 with F1 0.97 [evidence: c1]
- Accuracy is 0.96
## Uncertainties
"""
    llm = ScriptedLLM(
        [LLMTurn(tool_calls=[tc("c1", "mlflow_find_best_run")]), LLMTurn(text=hallucinated)]
    )
    resp = Agent(reg, llm, SETTINGS).run("what is the best model?")

    assert resp.facts == []  # nothing survives without successful evidence
    joined = " ".join(resp.uncertainties)
    assert "mlflow_find_best_run returned unavailable" in joined
    assert "Unverified claim removed" in joined and "abc123" in joined
    assert resp.evidence[0].status == ToolStatus.UNAVAILABLE


def test_answer_without_any_tool_call_is_flagged():
    llm = ScriptedLLM([LLMTurn(text="## Facts\n- Everything is healthy\n")])
    resp = Agent(full_registry(), llm, SETTINGS).run("is everything ok?")
    assert resp.facts == []
    assert any("No tools were called" in u for u in resp.uncertainties)


def test_write_request_only_yields_confirmation():
    llm = ScriptedLLM(
        [
            LLMTurn(tool_calls=[tc("w1", "airflow_trigger_dag", dag_id="training_pipeline")]),
            LLMTurn(text="## Summary\nTriggering requires your confirmation."),
        ]
    )
    calls: list = []
    agent = Agent(full_registry(airflow_calls=calls), llm, SETTINGS)
    resp = agent.run("trigger the training pipeline")
    assert resp.pending_confirmations[0]["tool"] == "airflow_trigger_dag"
    assert not any(m == "POST" and "dagRuns" in p for m, p in calls)
    result = agent.confirm(resp.pending_confirmations[0]["confirmation_token"])
    assert result.status == ToolStatus.OK
    assert any(m == "POST" and "dagRuns" in p for m, p in calls)


def test_llm_outage_is_handled_gracefully():
    class Boom:
        def complete(self, *a, **k):
            raise TimeoutError("llm timeout")

    resp = Agent(full_registry(), Boom(), SETTINGS).run("hello")
    assert "could not be reached" in resp.answer and resp.facts == []


def test_step_limit_is_enforced():
    loop = [LLMTurn(tool_calls=[tc(f"c{i}", "k8s_list_pods")]) for i in range(20)]
    resp = Agent(full_registry(), ScriptedLLM(loop), SETTINGS).run("loop forever")
    assert resp.steps == SETTINGS.max_steps and "maximum number" in resp.answer


def test_parse_sections():
    s = parse_sections(
        "## Summary\nok\n## Facts\n- a [evidence: x]\n- b\n## Recommended actions\n- do\n"
    )
    assert s["summary"] == "ok" and len(s["facts"]) == 2 and s["recommendations"] == ["do"]


def test_anthropic_message_conversion_groups_tool_results():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [tc("a", "x"), tc("b", "y")]},
        {"role": "tool", "call_id": "a", "name": "x", "content": "{}"},
        {"role": "tool", "call_id": "b", "name": "y", "content": "{}"},
    ]
    out = AnthropicProvider._convert(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["a", "b"]


@pytest.fixture()
def api_client():
    from api.main import app

    llm = ScriptedLLM(
        [
            LLMTurn(tool_calls=[tc("c1", "k8s_list_pods")]),
            LLMTurn(text="## Facts\n- 2 pods [evidence: c1]"),
        ]
    )
    app.state.agent = Agent(full_registry(), llm, SETTINGS)
    with TestClient(app) as client:
        yield client
    app.state.agent = None


def test_api_chat_confirm_and_validation(api_client):
    r = api_client.post(
        "/chat", json={"message": "how many pods?"}, headers={"X-Request-ID": "req-42"}
    )
    assert r.status_code == 200 and r.headers["X-Request-ID"] == "req-42"
    body = r.json()
    assert body["request_id"] == "req-42" and body["facts"] == ["2 pods [evidence: c1]"]
    assert api_client.post("/chat", json={"message": ""}).status_code == 422
    assert (
        api_client.post("/confirm", json={"confirmation_token": "not-a-real-token"}).status_code
        == 403
    )
    tools = api_client.get("/tools").json()["tools"]
    assert {
        "name": "k8s_restart_deployment",
        "module": "kubernetes",
        "permission": "write",
    } in tools


def test_retraining_policy_fetches_facts_instead_of_trusting_llm_numbers():
    """Regression (live run): the LLM passed a bogus epoch-ms value (model "4748 days old")
    and omitted the pipeline state. The tool must look these up from the systems themselves."""
    from datetime import datetime, timedelta, timezone

    from pydantic import BaseModel

    from agent.registry import Tool

    class RunArgs(BaseModel):
        run_id: str

    class PathArgs(BaseModel):
        path: str = "data/raw/dataset.csv"

    class DagArgs(BaseModel):
        dag_id: str
        limit: int = 5

    started = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    reg = ToolRegistry(default_timeout=5)
    reg.register(
        Tool(
            "mlflow_get_run_details",
            "run",
            lambda a: {
                "start_time": started,
                "metrics": {"f1": 0.97},
                "provenance": {"quality_gate": "passed", "dvc.data_md5": "a" * 32},
            },
            RunArgs,
            module="mlflow",
        )
    )
    reg.register(
        Tool("dvc_get_dataset_version", "v", lambda a: {"md5": "b" * 32}, PathArgs, module="dvc")
    )
    reg.register(
        Tool(
            "airflow_get_dag_runs",
            "runs",
            lambda a: {"latest": {"dag_run_id": "r1", "state": "success"}},
            DagArgs,
            module="airflow",
        )
    )
    register_policy_tool(reg, SETTINGS)

    res = reg.execute(
        tc(
            "p",
            "agent_assess_retraining",
            run_id="abc",
            dag_id="training_pipeline",
            model_trained_at_ms=1,  # bogus value from the LLM: must be ignored
        )
    )
    assert res.status == ToolStatus.OK, res.error
    checks = res.data["checks"]
    assert checks["freshness"].startswith("model age 2.0 days")
    assert checks["last_pipeline_run"] == "success"
    assert checks["data_drift"] == "tracked data changed"
    assert res.data["input_sources"]["model_trained_at_ms"] == "mlflow run abc"
    assert res.data["recommend_retraining"] is True  # only because the data changed
    assert all("days old" not in r for r in res.data["reasons"])
