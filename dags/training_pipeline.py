"""Airflow 3 training pipeline.

validate_dataset -> preprocess_data -> train_model -> evaluate_and_log_mlflow
    -> apply_quality_gate -> promote_model

* Every task retries (``retries=2``) except where retrying cannot help: validation and quality
  gate failures raise ``AirflowFailException`` so they fail fast and loudly.
* ``promote_model`` uses the default ``all_success`` trigger rule, so a failed validation or
  quality gate makes it ``upstream_failed`` and nothing is promoted.
* The training code lives in the ``training`` package, mounted into the Airflow containers at
  ``$REPO_ROOT`` (see infra/docker-compose.yaml).
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta

from airflow.exceptions import AirflowFailException
from airflow.sdk import dag, task

REPO_ROOT = os.getenv("REPO_ROOT", "/opt/mlops")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

log = logging.getLogger("training_pipeline")

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(seconds=20),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(minutes=15),
}


@dag(
    dag_id="training_pipeline",
    description="Validate -> preprocess -> train -> evaluate/MLflow -> quality gate -> promote",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["ml", "training", "approved-for-agent"],
    params={"model_type": "random_forest"},
)
def training_pipeline():
    @task(retries=0)
    def validate_dataset() -> dict:
        from training import validate

        try:
            report = validate.run()
        except validate.DataValidationError as exc:
            log.error("dataset validation failed: %s", exc)
            raise AirflowFailException(f"dataset validation failed: {exc}") from exc
        log.info("dataset validation passed rows=%s", report.rows)
        return {"rows": report.rows, "columns": report.columns}

    @task
    def preprocess_data(_: dict) -> dict:
        from training import preprocess

        train_path, test_path = preprocess.run()
        return {"train": train_path, "test": test_path}

    @task
    def train_model(_: dict, **context) -> dict:
        from training import train

        model_type = context["params"].get("model_type")
        info = train.run(model_type)
        log.info("trained run_id=%s dataset=%s", info["run_id"], info.get("dvc.dataset_version"))
        return {"run_id": info["run_id"], "dataset_version": info.get("dvc.dataset_version")}

    @task
    def evaluate_and_log_mlflow(train_info: dict) -> dict:
        from training import evaluate

        metrics = evaluate.run()
        log.info("evaluation run_id=%s f1=%.4f", train_info["run_id"], metrics["f1"])
        return {"run_id": train_info["run_id"], **metrics}

    @task(retries=0)
    def apply_quality_gate(metrics: dict) -> dict:
        from training import quality_gate

        try:
            return quality_gate.run(raise_on_fail=True)
        except quality_gate.QualityGateError as exc:
            log.error("quality gate failed for run %s: %s", metrics["run_id"], exc)
            raise AirflowFailException(f"quality gate failed: {exc}") from exc

    @task
    def promote_model(gate: dict) -> dict:
        from training import promote

        result = promote.run()
        log.info("promoted model=%s version=%s", result["model"], result["version"])
        return result

    validated = validate_dataset()
    processed = preprocess_data(validated)
    trained = train_model(processed)
    evaluated = evaluate_and_log_mlflow(trained)
    gate = apply_quality_gate(evaluated)
    promote_model(gate)


training_pipeline()
