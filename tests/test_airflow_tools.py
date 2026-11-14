"""Airflow 3 tools against a mocked REST API (httpx.MockTransport)."""

import httpx

from agent.config import Settings
from agent.registry import ToolRegistry
from agent.schemas import ToolCall, ToolStatus
from agent.tools import airflow_tools
from agent.tools.airflow_tools import AirflowClient

BASE = "http://airflow.test"
RUN_OK = "manual__2026-11-13T10:00:00+00:00"
RUN_BAD = "manual__2026-11-14T09:00:00+00:00"


def handler_factory(calls: list):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        p = request.url.path
        if p == "/auth/token":
            return httpx.Response(201, json={"access_token": "jwt-token-value"})
        assert request.headers["Authorization"] == "Bearer jwt-token-value"
        if p == "/api/v2/dags":
            return httpx.Response(
                200,
                json={
                    "total_entries": 2,
                    "dags": [
                        {
                            "dag_id": "training_pipeline",
                            "is_paused": False,
                            "tags": [{"name": "ml"}],
                        },
                        {"dag_id": "cleanup", "is_paused": True, "tags": []},
                    ],
                },
            )
        if p == "/api/v2/dags/training_pipeline/dagRuns" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "dag_runs": [
                        {"dag_run_id": RUN_BAD, "state": "failed", "run_type": "manual"},
                        {"dag_run_id": RUN_OK, "state": "success", "run_type": "manual"},
                    ]
                },
            )
        if p == "/api/v2/dags/training_pipeline/dagRuns" and request.method == "POST":
            return httpx.Response(200, json={"dag_run_id": "manual__new", "state": "queued"})
        if p.endswith("/taskInstances"):
            return httpx.Response(
                200,
                json={
                    "task_instances": [
                        {"task_id": "validate_dataset", "state": "failed", "try_number": 1},
                        {"task_id": "preprocess_data", "state": "upstream_failed", "try_number": 0},
                        {"task_id": "promote_model", "state": "upstream_failed", "try_number": 0},
                    ]
                },
            )
        if "/logs/" in p:
            return httpx.Response(
                200,
                json={
                    "content": [
                        {"timestamp": "t1", "level": "info", "event": "starting validation"},
                        {
                            "timestamp": "t2",
                            "level": "error",
                            "event": "AirflowFailException: dataset validation failed: missing required columns: ['mean_smoothness']",
                        },
                    ]
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    return handler


def make(calls=None, allowed=("training_pipeline",)):
    calls = calls if calls is not None else []
    http = httpx.Client(transport=httpx.MockTransport(handler_factory(calls)))
    client = AirflowClient(BASE, "airflow", "s3cret", http=http)
    reg = ToolRegistry(default_timeout=5)
    settings = Settings(airflow_allowed_dags=list(allowed))
    airflow_tools.register(reg, settings, client)
    return reg, calls


def call(reg, name, **args):
    return reg.execute(ToolCall(id="t", name=name, arguments=args))


def test_list_dags_marks_approved():
    reg, _ = make()
    res = call(reg, "airflow_list_dags")
    dags = {d["dag_id"]: d for d in res.data["dags"]}
    assert dags["training_pipeline"]["approved_for_agent_trigger"] is True
    assert dags["cleanup"]["approved_for_agent_trigger"] is False


def test_failed_tasks_identified_with_log_diagnosis():
    reg, _ = make()
    res = call(reg, "airflow_get_failed_tasks", dag_id="training_pipeline")
    assert res.status == ToolStatus.OK, res.error
    assert res.data["run_id"] == RUN_BAD
    assert [t["task_id"] for t in res.data["failed_tasks"]] == ["validate_dataset"]
    assert "promote_model" in res.data["blocked_downstream_tasks"]
    assert "mean_smoothness" in res.data["diagnostics"][0]["error_lines"][0]


def test_trigger_requires_confirmation_then_posts_once():
    reg, calls = make()
    first = call(
        reg, "airflow_trigger_dag", dag_id="training_pipeline", conf={"model_type": "random_forest"}
    )
    assert first.status == ToolStatus.CONFIRMATION_REQUIRED
    assert not any(m == "POST" and "dagRuns" in path for m, path in calls)  # nothing triggered yet
    done = reg.execute_confirmed(first.confirmation_token)
    assert done.status == ToolStatus.OK and done.data["dag_run_id"] == "manual__new"


def test_trigger_of_unapproved_dag_is_denied():
    reg, calls = make()
    res = call(reg, "airflow_trigger_dag", dag_id="cleanup")
    assert res.status == ToolStatus.DENIED and res.confirmation_token is None


def test_airflow_down_is_unavailable():
    def boom(request):
        raise httpx.ConnectError("connection refused")

    client = AirflowClient(BASE, "u", "p", http=httpx.Client(transport=httpx.MockTransport(boom)))
    reg = ToolRegistry()
    airflow_tools.register(reg, Settings(), client)
    res = call(reg, "airflow_get_dag_runs", dag_id="training_pipeline")
    assert res.status == ToolStatus.UNAVAILABLE


def test_run_ids_with_special_chars_are_url_encoded():
    reg, calls = make()
    call(reg, "airflow_get_task_instances", dag_id="training_pipeline", run_id=RUN_OK)
    assert any("manual__2026-11-13T10:00:00+00:00" in path for _, path in calls)
