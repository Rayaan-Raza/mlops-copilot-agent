"""Registry, safety, logging and policy tests."""

import json
import logging
import time
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from agent.errors import ServiceUnavailableError
from agent.logging_config import JsonFormatter, redact
from agent.policy import assess_retraining
from agent.registry import Tool, ToolRegistry
from agent.safety import require_namespace_allowed
from agent.schemas import Permission, ToolCall, ToolStatus


class EchoArgs(BaseModel):
    text: str
    times: int = 1


class NsArgs(BaseModel):
    namespace: str
    deployment: str


def make_registry(timeout: float = 2.0) -> ToolRegistry:
    reg = ToolRegistry(default_timeout=timeout)
    reg.register(Tool("echo", "echo text", lambda a: a.text * a.times, EchoArgs))
    reg.register(Tool("slow", "sleeps", lambda _: time.sleep(5), timeout=0.2))
    reg.register(
        Tool(
            "down",
            "service down",
            lambda _: (_ for _ in ()).throw(ServiceUnavailableError("mlflow unreachable")),
        )
    )
    reg.register(
        Tool(
            "restart",
            "restart deployment",
            lambda a: {"restarted": a.deployment},
            NsArgs,
            permission=Permission.WRITE,
            precheck=lambda a: require_namespace_allowed(a.namespace, ["mlops-agent"]),
        )
    )
    return reg


def test_successful_tool_execution():
    res = make_registry().execute(
        ToolCall(id="1", name="echo", arguments={"text": "ab", "times": 2})
    )
    assert res.status == ToolStatus.OK and res.data == "abab" and res.duration_ms >= 0


def test_invalid_arguments_are_rejected_before_execution():
    res = make_registry().execute(ToolCall(id="1", name="echo", arguments={"times": "many"}))
    assert res.status == ToolStatus.INVALID_REQUEST
    assert "text" in res.error and "times" in res.error


def test_unknown_tool_is_invalid_request():
    res = make_registry().execute(ToolCall(id="1", name="rm_rf", arguments={}))
    assert res.status == ToolStatus.INVALID_REQUEST


def test_tool_timeout_is_reported_not_raised():
    res = make_registry().execute(ToolCall(id="1", name="slow"))
    assert res.status == ToolStatus.TIMEOUT
    assert res.data is None


def test_service_failure_maps_to_unavailable():
    res = make_registry().execute(ToolCall(id="1", name="down"))
    assert res.status == ToolStatus.UNAVAILABLE and "unreachable" in res.error


def test_write_tool_requires_confirmation_and_token_is_single_use():
    reg = make_registry()
    args = {"namespace": "mlops-agent", "deployment": "ml-inference"}
    first = reg.execute(ToolCall(id="1", name="restart", arguments=args))
    assert first.status == ToolStatus.CONFIRMATION_REQUIRED
    assert first.data["message"].startswith("Not executed")
    token = first.confirmation_token

    done = reg.execute_confirmed(token)
    assert done.status == ToolStatus.OK and done.data == {"restarted": "ml-inference"}
    assert reg.execute_confirmed(token).status == ToolStatus.DENIED  # replay rejected


def test_write_outside_approved_namespace_is_denied_without_token():
    reg = make_registry()
    res = reg.execute(
        ToolCall(
            id="1", name="restart", arguments={"namespace": "kube-system", "deployment": "coredns"}
        )
    )
    assert res.status == ToolStatus.DENIED
    assert res.confirmation_token is None
    assert reg.confirmations.pending() == []


def test_redaction_of_secret_keys_and_values():
    out = redact(
        {
            "password": "hunter2",
            "headers": {"Authorization": "Bearer abc.def"},
            "note": "token=xyz123 and key sk-abcdefghijklmnop",
            "safe": "f1=0.97",
        }
    )
    blob = json.dumps(out)
    assert "hunter2" not in blob and "xyz123" not in blob and "sk-abcdefghijklmnop" not in blob
    assert out["safe"] == "f1=0.97"


def test_structured_tool_log_has_required_fields(caplog):
    reg = make_registry()
    with caplog.at_level(logging.INFO, logger="agent.tools"):
        reg.execute(
            ToolCall(id="c-1", name="echo", arguments={"text": "x", "api_key": "sk-secret12345"})
        )
    record = next(r for r in caplog.records if getattr(r, "event", None) == "tool_call")
    payload = json.loads(JsonFormatter().format(record))
    for field in ("tool", "status", "duration_ms", "request_id", "call_id"):
        assert field in payload
    assert "sk-secret12345" not in json.dumps(payload)


def test_retraining_policy_requires_evidence():
    now = datetime(2026, 11, 14, tzinfo=timezone.utc)
    fresh = int((now - timedelta(days=3)).timestamp() * 1000)
    ok = assess_retraining(
        model_trained_at_ms=fresh,
        latest_f1=0.97,
        last_quality_gate="passed",
        last_pipeline_state="success",
        data_changed_since_training=False,
        freshness_days=30,
        min_f1=0.9,
        now=now,
    )
    assert not ok.recommend_retraining

    stale = int((now - timedelta(days=45)).timestamp() * 1000)
    bad = assess_retraining(
        model_trained_at_ms=stale,
        latest_f1=0.97,
        last_quality_gate="failed",
        last_pipeline_state="failed",
        data_changed_since_training=None,
        freshness_days=30,
        min_f1=0.9,
        now=now,
    )
    assert bad.recommend_retraining and len(bad.reasons) == 2
    assert bad.checks["data_drift"] == "unknown"


def test_masked_provider_key_in_error_is_redacted():
    """Regression (live run): OpenAI 401 errors echo a masked key that ends in real characters."""
    msg = "Incorrect API key provided: sk-proj-****************wwEA. See docs."
    out = redact(msg)
    assert "wwEA" not in out and "sk-proj" not in out and "See docs." in out
