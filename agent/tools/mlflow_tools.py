"""MLflow experiment-intelligence tools (uses ``MlflowClient``; never scrapes the web UI)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agent.config import Settings
from agent.errors import InvalidRequestError, ServiceUnavailableError, ToolError
from agent.registry import Tool, ToolRegistry

PROVENANCE_TAGS = (
    "dvc.dataset_version",
    "dvc.data_md5",
    "dvc.data_path",
    "git.commit",
    "model_type",
    "quality_gate",
    "quality_gate.reasons",
    "airflow.dag_run_id",
)
SUMMARY_METRICS = (
    "f1",
    "accuracy",
    "precision",
    "recall",
    "roc_auc",
    "train_f1",
    "train_test_f1_gap",
)


def _iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat() if ms else None


def _wrap_errors(exc: Exception) -> Exception:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if any(
        s in lowered for s in ("connection", "max retries", "refused", "timed out", "unreachable")
    ):
        return ServiceUnavailableError(f"MLflow tracking server unreachable: {text[:300]}")
    if "resource_does_not_exist" in lowered or "not found" in lowered:
        return InvalidRequestError(text[:300])
    return ToolError(text[:300])


class MLflowGateway:
    """Thin wrapper so tests can inject a fake ``MlflowClient``."""

    def __init__(self, tracking_uri: str, default_experiment: str, client: Any = None) -> None:
        self.tracking_uri = tracking_uri
        self.default_experiment = default_experiment
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from mlflow.tracking import MlflowClient

            self._client = MlflowClient(tracking_uri=self.tracking_uri)
        return self._client

    def call(self, fn_name: str, *args, **kwargs):
        try:
            return getattr(self.client, fn_name)(*args, **kwargs)
        except (ServiceUnavailableError, InvalidRequestError, ToolError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalise MLflow/requests errors
            raise _wrap_errors(exc) from exc

    def experiment(self, name: str | None):
        name = name or self.default_experiment
        exp = self.call("get_experiment_by_name", name)
        if exp is None:
            raise InvalidRequestError(f"experiment {name!r} does not exist")
        return exp

    def recent_runs(self, experiment_names: list[str] | None, limit: int, filter_string: str = ""):
        exps = [self.experiment(n) for n in (experiment_names or [None])]
        return self.call(
            "search_runs",
            [e.experiment_id for e in exps],
            filter_string=filter_string,
            max_results=limit,
            order_by=["attributes.start_time DESC"],
        )


def summarize_run(run) -> dict[str, Any]:
    m, tags = run.data.metrics, run.data.tags
    return {
        "run_id": run.info.run_id,
        "run_name": tags.get("mlflow.runName") or getattr(run.info, "run_name", None),
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": _iso(run.info.start_time),
        "metrics": {k: round(m[k], 4) for k in SUMMARY_METRICS if k in m},
        "provenance": {k: tags[k] for k in PROVENANCE_TAGS if k in tags},
    }


def readiness_caveats(summary: dict[str, Any], max_gap: float = 0.08) -> list[str]:
    """Reasons the top-metric run might still not be production-ready."""
    caveats = []
    prov, metrics = summary["provenance"], summary["metrics"]
    gate = prov.get("quality_gate")
    if gate is None:
        caveats.append("no quality-gate result recorded for this run")
    elif gate != "passed":
        caveats.append(
            f"quality gate {gate}: {prov.get('quality_gate.reasons', 'no reason recorded')}"
        )
    gap = metrics.get("train_test_f1_gap")
    if gap is not None and gap > max_gap:
        caveats.append(f"train/test F1 gap {gap:.3f} suggests overfitting")
    if "dvc.data_md5" not in prov:
        caveats.append("no DVC data hash recorded: dataset provenance cannot be verified")
    if summary["status"] != "FINISHED":
        caveats.append(f"run status is {summary['status']}")
    caveats.append("offline test metrics only: no live/production performance evidence was checked")
    return caveats


# --------------------------------------------------------------------------- inputs
class ExperimentArg(BaseModel):
    experiment_name: str | None = Field(None, description="Experiment name; default from config")


class ListRunsArgs(ExperimentArg):
    limit: int = Field(10, ge=1, le=200)


class RunIdArgs(BaseModel):
    run_id: str = Field(..., pattern=r"^[0-9a-f]{32}$")


class BestRunArgs(ExperimentArg):
    metric: str = Field("f1", pattern=r"^[A-Za-z0-9_.\-]+$")
    last_n: int = Field(50, ge=1, le=500, description="only consider the N most recent runs")
    higher_is_better: bool = True


class CompareRunsArgs(ExperimentArg):
    run_ids: list[str] | None = Field(
        None, description="explicit run ids; otherwise the last_n most recent"
    )
    last_n: int = Field(5, ge=2, le=50)
    rank_by: str = Field("f1", pattern=r"^[A-Za-z0-9_.\-]+$")


class CompareExperimentsArgs(BaseModel):
    experiment_names: list[str] = Field(..., min_length=2, max_length=10)
    metric: str = Field("f1", pattern=r"^[A-Za-z0-9_.\-]+$")


class RegistryArgs(BaseModel):
    model_name: str | None = Field(None, description="registered model; default from config")


# --------------------------------------------------------------------------- tools
def build_tools(gw: MLflowGateway, registered_model: str) -> list[Tool]:
    def list_experiments(_: BaseModel) -> dict:
        exps = gw.call("search_experiments")
        return {
            "experiments": [
                {
                    "experiment_id": e.experiment_id,
                    "name": e.name,
                    "lifecycle_stage": e.lifecycle_stage,
                    "created": _iso(getattr(e, "creation_time", None)),
                }
                for e in exps
            ]
        }

    def list_runs(a: ListRunsArgs) -> dict:
        runs = gw.recent_runs([a.experiment_name] if a.experiment_name else None, a.limit)
        return {
            "experiment": a.experiment_name or gw.default_experiment,
            "runs": [summarize_run(r) for r in runs],
        }

    def get_run_details(a: RunIdArgs) -> dict:
        run = gw.call("get_run", a.run_id)
        artifacts = [f.path for f in gw.call("list_artifacts", a.run_id)]
        return {
            **summarize_run(run),
            "all_metrics": run.data.metrics,
            "params": run.data.params,
            "tags": {
                k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.log-model")
            },
            "artifacts": artifacts,
            "end_time": _iso(run.info.end_time),
        }

    def find_best_run(a: BestRunArgs) -> dict:
        runs = gw.recent_runs([a.experiment_name] if a.experiment_name else None, a.last_n)
        scored = [r for r in runs if a.metric in r.data.metrics and r.info.status == "FINISHED"]
        if not scored:
            return {
                "best_run": None,
                "considered": len(runs),
                "message": f"no finished runs with metric {a.metric!r}",
            }
        scored.sort(key=lambda r: r.data.metrics[a.metric], reverse=a.higher_is_better)
        best = summarize_run(scored[0])
        ranking = [
            {
                "run_id": r.info.run_id,
                "run_name": r.data.tags.get("mlflow.runName"),
                a.metric: round(r.data.metrics[a.metric], 4),
            }
            for r in scored[:10]
        ]
        margin = (
            round(abs(scored[0].data.metrics[a.metric] - scored[1].data.metrics[a.metric]), 4)
            if len(scored) > 1
            else None
        )
        return {
            "metric": a.metric,
            "considered_runs": len(runs),
            "runs_with_metric": len(scored),
            "best_run": best,
            "margin_over_runner_up": margin,
            "ranking": ranking,
            "production_readiness_caveats": readiness_caveats(best),
        }

    def compare_runs(a: CompareRunsArgs) -> dict:
        if a.run_ids:
            runs = [gw.call("get_run", rid) for rid in a.run_ids]
        else:
            runs = gw.recent_runs([a.experiment_name] if a.experiment_name else None, a.last_n)
        rows = [summarize_run(r) for r in runs]
        best_by_metric = {}
        for metric in SUMMARY_METRICS:
            vals = [(r["metrics"][metric], r["run_id"]) for r in rows if metric in r["metrics"]]
            if vals:
                pick = min(vals) if metric == "train_test_f1_gap" else max(vals)
                best_by_metric[metric] = {"run_id": pick[1], "value": pick[0]}
        ranked = sorted(
            (r for r in rows if a.rank_by in r["metrics"]),
            key=lambda r: r["metrics"][a.rank_by],
            reverse=True,
        )
        rec = ranked[0] if ranked else None
        return {
            "compared": len(rows),
            "runs": rows,
            "best_by_metric": best_by_metric,
            "recommended_candidate": rec["run_id"] if rec else None,
            "recommendation_basis": f"highest {a.rank_by}"
            if rec
            else "no run has the ranking metric",
            "caveats": readiness_caveats(rec) if rec else [],
        }

    def compare_experiments(a: CompareExperimentsArgs) -> dict:
        out = []
        for name in a.experiment_names:
            runs = gw.recent_runs([name], 200)
            vals = [r.data.metrics[a.metric] for r in runs if a.metric in r.data.metrics]
            best = (
                max(runs, key=lambda r: r.data.metrics.get(a.metric, float("-inf")))
                if vals
                else None
            )
            out.append(
                {
                    "experiment": name,
                    "runs": len(runs),
                    "runs_with_metric": len(vals),
                    f"best_{a.metric}": round(max(vals), 4) if vals else None,
                    f"mean_{a.metric}": round(sum(vals) / len(vals), 4) if vals else None,
                    "best_run_id": best.info.run_id if best else None,
                }
            )
        return {"metric": a.metric, "experiments": out}

    def model_registry(a: RegistryArgs) -> dict:
        name = a.model_name or registered_model
        try:
            model = gw.call("get_registered_model", name)
        except InvalidRequestError:
            return {"model": name, "registered": False, "versions": []}
        versions = gw.call("search_model_versions", f"name='{name}'")
        aliases = dict(getattr(model, "aliases", {}) or {})
        return {
            "model": name,
            "registered": True,
            "aliases": aliases,
            "versions": [
                {
                    "version": v.version,
                    "run_id": v.run_id,
                    "status": v.status,
                    "created": _iso(v.creation_timestamp),
                    "aliases": [k for k, ver in aliases.items() if str(ver) == str(v.version)],
                    "tags": dict(v.tags or {}),
                }
                for v in sorted(versions, key=lambda v: int(v.version), reverse=True)
            ],
        }

    return [
        Tool(
            "mlflow_list_experiments", "List MLflow experiments.", list_experiments, module="mlflow"
        ),
        Tool(
            "mlflow_list_runs",
            "List the most recent runs of an experiment with key metrics and data-provenance tags.",
            list_runs,
            ListRunsArgs,
            module="mlflow",
        ),
        Tool(
            "mlflow_get_run_details",
            "Get params, all metrics, tags (incl. DVC dataset version/md5) and artifacts for one run.",
            get_run_details,
            RunIdArgs,
            module="mlflow",
        ),
        Tool(
            "mlflow_find_best_run",
            "Find the best recent run by a metric, with ranking, margin and production-readiness caveats.",
            find_best_run,
            BestRunArgs,
            module="mlflow",
        ),
        Tool(
            "mlflow_compare_runs",
            "Compare several runs (explicit ids or last N) side by side and recommend a candidate.",
            compare_runs,
            CompareRunsArgs,
            module="mlflow",
        ),
        Tool(
            "mlflow_compare_experiments",
            "Compare two or more experiments by best and mean value of a metric.",
            compare_experiments,
            CompareExperimentsArgs,
            module="mlflow",
        ),
        Tool(
            "mlflow_get_model_registry",
            "Get registered model versions, aliases (e.g. champion) and their source runs.",
            model_registry,
            RegistryArgs,
            module="mlflow",
        ),
    ]


def register(registry: ToolRegistry, settings: Settings, client: Any = None) -> MLflowGateway:
    gw = MLflowGateway(settings.mlflow_tracking_uri, settings.mlflow_experiment, client)
    for tool in build_tools(gw, settings.mlflow_registered_model):
        registry.register(tool)
    return gw
