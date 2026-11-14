"""MLflow tools against a fake MlflowClient."""

from agent.config import Settings
from agent.registry import ToolRegistry
from agent.schemas import ToolCall, ToolStatus
from agent.tools import mlflow_tools
from tests.fakes import FakeMlflowClient


def make(client=None) -> ToolRegistry:
    reg = ToolRegistry(default_timeout=5)
    mlflow_tools.register(reg, Settings(), client or FakeMlflowClient())
    return reg


def call(reg, name, **args):
    return reg.execute(ToolCall(id="t", name=name, arguments=args))


def test_find_best_run_by_f1_reports_caveats():
    res = call(make(), "mlflow_find_best_run", metric="f1")
    assert res.status == ToolStatus.OK
    best = res.data["best_run"]
    # highest raw F1 belongs to gradient boosting, but it failed the quality gate -> caveats say so
    assert best["run_id"] == "3" * 32
    caveats = " ".join(res.data["production_readiness_caveats"])
    assert "quality gate failed" in caveats and "overfitting" in caveats
    assert res.data["runs_with_metric"] == 4  # FAILED run excluded


def test_compare_last_five_runs():
    res = call(make(), "mlflow_compare_runs", last_n=5, rank_by="f1")
    assert res.data["compared"] == 5
    assert res.data["best_by_metric"]["train_test_f1_gap"]["value"] == 0.02  # lower gap is better
    assert res.data["recommended_candidate"] == "3" * 32


def test_run_details_expose_dataset_provenance():
    res = call(make(), "mlflow_get_run_details", run_id="4" * 32)
    assert res.data["provenance"]["dvc.dataset_version"] == "data-v1.0"
    assert res.data["provenance"]["dvc.data_md5"] == "a" * 32
    assert "model" in res.data["artifacts"]


def test_registry_lists_champion_alias():
    res = call(make(), "mlflow_get_model_registry")
    v2 = next(v for v in res.data["versions"] if v["version"] == "2")
    assert v2["aliases"] == ["champion"] and v2["run_id"] == "2" * 32


def test_compare_experiments():
    res = call(
        make(), "mlflow_compare_experiments", experiment_names=["exp-a", "exp-b"], metric="f1"
    )
    exps = {e["experiment"]: e for e in res.data["experiments"]}
    assert exps["exp-a"]["best_f1"] == 0.99 and exps["exp-b"]["runs"] == 2


def test_unreachable_tracking_server_is_unavailable():
    reg = make(FakeMlflowClient(fail=ConnectionError("Connection refused: localhost:5000")))
    res = call(reg, "mlflow_find_best_run")
    assert res.status == ToolStatus.UNAVAILABLE
    assert res.data is None


def test_unknown_run_and_bad_run_id():
    assert (
        call(make(), "mlflow_get_run_details", run_id="f" * 32).status == ToolStatus.INVALID_REQUEST
    )
    assert call(make(), "mlflow_get_run_details", run_id="25").status == ToolStatus.INVALID_REQUEST
