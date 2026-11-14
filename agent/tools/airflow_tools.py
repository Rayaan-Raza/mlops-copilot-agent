"""Apache Airflow 3 workflow tools (REST API ``/api/v2`` with JWT authentication)."""

from __future__ import annotations

import re
import threading
import time
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from agent.config import Settings
from agent.errors import InvalidRequestError, ServiceUnavailableError, ToolError
from agent.registry import Tool, ToolRegistry
from agent.safety import require_dag_allowed
from agent.schemas import Permission

TERMINAL_STATES = {"success", "failed"}
FAILED_TASK_STATES = {"failed", "upstream_failed"}
ERROR_LINE_RE = re.compile(r"(error|exception|traceback|failed|AirflowFailException)", re.I)


def _seg(value: str) -> str:
    return quote(value, safe="")


class AirflowClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        *,
        timeout: float = 15.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._http = http or httpx.Client(timeout=timeout)
        self._token: str | None = None
        self._lock = threading.Lock()

    def _authenticate(self) -> str:
        try:
            resp = self._http.post(
                f"{self.base_url}/auth/token",
                json={"username": self._username, "password": self._password},
            )
        except httpx.TransportError as exc:
            raise ServiceUnavailableError(
                f"Airflow API unreachable at {self.base_url}: {type(exc).__name__}"
            ) from exc
        if resp.status_code >= 400:
            raise ToolError(f"Airflow authentication failed (HTTP {resp.status_code})")
        token = resp.json().get("access_token")
        if not token:
            raise ToolError("Airflow authentication response contained no access_token")
        return token

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        with self._lock:
            if self._token is None:
                self._token = self._authenticate()
            token = self._token
        for attempt in (1, 2):
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            try:
                resp = self._http.request(
                    method, f"{self.base_url}{path}", headers=headers, **kwargs
                )
            except httpx.TimeoutException as exc:
                raise ServiceUnavailableError(f"Airflow API timed out on {path}") from exc
            except httpx.TransportError as exc:
                raise ServiceUnavailableError(
                    f"Airflow API unreachable: {type(exc).__name__}"
                ) from exc
            if resp.status_code == 401 and attempt == 1:  # token expired -> refresh once
                with self._lock:
                    self._token = token = self._authenticate()
                continue
            break
        if resp.status_code == 404:
            raise InvalidRequestError(f"Airflow resource not found: {path}")
        if resp.status_code >= 500:
            raise ServiceUnavailableError(f"Airflow API error HTTP {resp.status_code} on {path}")
        if resp.status_code >= 400:
            detail = resp.text[:300]
            raise ToolError(f"Airflow API HTTP {resp.status_code} on {path}: {detail}")
        ctype = resp.headers.get("content-type", "")
        return resp.json() if "json" in ctype else resp.text


# --------------------------------------------------------------------------- inputs
DAG_ID = Field(..., pattern=r"^[A-Za-z0-9_.\-]{1,250}$")


class DagArgs(BaseModel):
    dag_id: str = DAG_ID


class DagRunsArgs(DagArgs):
    limit: int = Field(5, ge=1, le=50)


class DagRunArgs(DagArgs):
    run_id: str = Field(..., min_length=1, max_length=250)


class FailedTasksArgs(DagArgs):
    run_id: str | None = Field(None, description="DAG run id; default = most recent failed run")
    include_logs: bool = True


class TaskLogArgs(DagRunArgs):
    task_id: str = Field(..., pattern=r"^[A-Za-z0-9_.\-]{1,250}$")
    try_number: int | None = Field(None, ge=1)
    tail_lines: int = Field(60, ge=5, le=400)


class TriggerArgs(DagArgs):
    conf: dict[str, Any] = Field(
        default_factory=dict, description="DAG params, e.g. {'model_type': 'random_forest'}"
    )


class WaitArgs(DagRunArgs):
    timeout_seconds: int = Field(300, ge=10, le=900)
    poll_seconds: float = Field(10, ge=0, le=60)


# --------------------------------------------------------------------------- helpers
def _summ_run(r: dict) -> dict:
    return {
        k: r.get(k)
        for k in (
            "dag_run_id",
            "state",
            "run_type",
            "logical_date",
            "start_date",
            "end_date",
            "conf",
        )
    }


def _summ_ti(t: dict) -> dict:
    return {
        k: t.get(k)
        for k in (
            "task_id",
            "state",
            "try_number",
            "max_tries",
            "duration",
            "start_date",
            "end_date",
            "operator",
        )
    }


def _log_text(payload: Any) -> str:
    """Airflow 3 returns structured log events; older/other paths return plain text."""
    if isinstance(payload, str):
        return payload
    content = payload.get("content", payload) if isinstance(payload, dict) else payload
    if isinstance(content, str):
        return content
    lines = []
    for item in content or []:
        if isinstance(item, dict):
            lvl = item.get("level", "")
            lines.append(
                f"{item.get('timestamp', '')} {lvl.upper()} {item.get('event', '')}".strip()
            )
            if item.get("error_detail"):
                lines.append(str(item["error_detail"])[:1500])
        else:
            lines.append(str(item))
    return "\n".join(lines)


def build_tools(client: AirflowClient, allowed_dags: list[str]) -> list[Tool]:
    def list_dags(_: BaseModel) -> dict:
        data = client.request("GET", "/api/v2/dags", params={"limit": 100})
        return {
            "total": data.get("total_entries"),
            "dags": [
                {
                    "dag_id": d["dag_id"],
                    "is_paused": d.get("is_paused"),
                    "tags": [
                        t.get("name") if isinstance(t, dict) else t for t in d.get("tags", [])
                    ],
                    "schedule": d.get("timetable_summary") or d.get("timetable_description"),
                    "approved_for_agent_trigger": d["dag_id"] in allowed_dags,
                }
                for d in data.get("dags", [])
            ],
        }

    def dag_runs(a: DagRunsArgs) -> dict:
        data = client.request(
            "GET",
            f"/api/v2/dags/{_seg(a.dag_id)}/dagRuns",
            params={"limit": a.limit, "order_by": "-start_date"},
        )
        runs = [_summ_run(r) for r in data.get("dag_runs", [])]
        states: dict[str, int] = {}
        for r in runs:
            states[r["state"]] = states.get(r["state"], 0) + 1
        return {
            "dag_id": a.dag_id,
            "runs": runs,
            "state_counts": states,
            "latest": runs[0] if runs else None,
        }

    def task_instances(a: DagRunArgs) -> dict:
        data = client.request(
            "GET", f"/api/v2/dags/{_seg(a.dag_id)}/dagRuns/{_seg(a.run_id)}/taskInstances"
        )
        tis = [_summ_ti(t) for t in data.get("task_instances", [])]
        return {"dag_id": a.dag_id, "run_id": a.run_id, "task_instances": tis}

    def task_logs(a: TaskLogArgs) -> dict:
        try_number = a.try_number
        if try_number is None:
            ti = client.request(
                "GET",
                f"/api/v2/dags/{_seg(a.dag_id)}/dagRuns/{_seg(a.run_id)}/taskInstances/{_seg(a.task_id)}",
            )
            try_number = max(int(ti.get("try_number") or 1), 1)
        payload = client.request(
            "GET",
            f"/api/v2/dags/{_seg(a.dag_id)}/dagRuns/{_seg(a.run_id)}/taskInstances/"
            f"{_seg(a.task_id)}/logs/{try_number}",
            params={"full_content": "true"},
        )
        lines = _log_text(payload).splitlines()
        return {
            "task_id": a.task_id,
            "try_number": try_number,
            "error_lines": [ln for ln in lines if ERROR_LINE_RE.search(ln)][-15:],
            "tail": lines[-a.tail_lines :],
        }

    def failed_tasks(a: FailedTasksArgs) -> dict:
        run_id = a.run_id
        if run_id is None:
            runs = client.request(
                "GET",
                f"/api/v2/dags/{_seg(a.dag_id)}/dagRuns",
                params={"limit": 25, "order_by": "-start_date", "state": "failed"},
            ).get("dag_runs", [])
            runs = [r for r in runs if r.get("state") == "failed"]
            if not runs:
                return {"dag_id": a.dag_id, "failed_run": None, "message": "no failed runs found"}
            run_id = runs[0]["dag_run_id"]
        tis = task_instances(DagRunArgs(dag_id=a.dag_id, run_id=run_id))["task_instances"]
        failed = [t for t in tis if t["state"] == "failed"]
        blocked = [t["task_id"] for t in tis if t["state"] == "upstream_failed"]
        diagnostics = []
        if a.include_logs:
            for t in failed:
                try:
                    lg = task_logs(
                        TaskLogArgs(
                            dag_id=a.dag_id,
                            run_id=run_id,
                            task_id=t["task_id"],
                            try_number=t.get("try_number") or None,
                            tail_lines=30,
                        )
                    )
                    diagnostics.append({"task_id": t["task_id"], "error_lines": lg["error_lines"]})
                except ToolError as exc:
                    diagnostics.append({"task_id": t["task_id"], "log_error": str(exc)})
        return {
            "dag_id": a.dag_id,
            "run_id": run_id,
            "failed_tasks": failed,
            "blocked_downstream_tasks": blocked,
            "diagnostics": diagnostics,
        }

    def trigger(a: TriggerArgs) -> dict:
        require_dag_allowed(a.dag_id, allowed_dags)
        run = client.request(
            "POST",
            f"/api/v2/dags/{_seg(a.dag_id)}/dagRuns",
            json={"logical_date": None, "conf": a.conf},
        )
        return {"triggered": True, **_summ_run(run)}

    def wait(a: WaitArgs) -> dict:
        deadline = time.monotonic() + a.timeout_seconds
        while True:
            run = client.request("GET", f"/api/v2/dags/{_seg(a.dag_id)}/dagRuns/{_seg(a.run_id)}")
            if run.get("state") in TERMINAL_STATES or time.monotonic() >= deadline:
                break
            time.sleep(a.poll_seconds)
        tis = task_instances(DagRunArgs(dag_id=a.dag_id, run_id=a.run_id))["task_instances"]
        return {
            "finished": run.get("state") in TERMINAL_STATES,
            "run": _summ_run(run),
            "task_states": {t["task_id"]: t["state"] for t in tis},
        }

    def _precheck(a: TriggerArgs) -> None:
        require_dag_allowed(a.dag_id, allowed_dags)

    return [
        Tool(
            "airflow_list_dags",
            "List Airflow DAGs, whether paused, and whether the agent may trigger them.",
            list_dags,
            module="airflow",
        ),
        Tool(
            "airflow_get_dag_runs",
            "Get recent runs of a DAG with their state (success/failed/running).",
            dag_runs,
            DagRunsArgs,
            module="airflow",
        ),
        Tool(
            "airflow_get_task_instances",
            "Get every task's state, tries and duration for a DAG run.",
            task_instances,
            DagRunArgs,
            module="airflow",
        ),
        Tool(
            "airflow_get_failed_tasks",
            "Identify failed and blocked tasks in a DAG run (default: latest failed run) with log diagnostics.",
            failed_tasks,
            FailedTasksArgs,
            module="airflow",
        ),
        Tool(
            "airflow_get_task_logs",
            "Fetch a task attempt's log tail and error lines.",
            task_logs,
            TaskLogArgs,
            module="airflow",
        ),
        Tool(
            "airflow_trigger_dag",
            f"Trigger an approved training DAG run (approved: {', '.join(allowed_dags)}).",
            trigger,
            TriggerArgs,
            permission=Permission.WRITE,
            module="airflow",
            precheck=_precheck,
            summarize=lambda a: f"Trigger Airflow DAG '{a.dag_id}' with conf {a.conf}",
        ),
        Tool(
            "airflow_wait_for_dag_run",
            "Poll a DAG run until it finishes (or times out) and report final task states.",
            wait,
            WaitArgs,
            module="airflow",
            timeout=930,
        ),
    ]


def register(
    registry: ToolRegistry, settings: Settings, client: AirflowClient | None = None
) -> AirflowClient:
    client = client or AirflowClient(
        settings.airflow_base_url, settings.airflow_username, settings.airflow_password
    )
    for tool in build_tools(client, settings.airflow_allowed_dags):
        registry.register(tool)
    return client
