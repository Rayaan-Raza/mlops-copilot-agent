"""Model quality gate: blocks promotion of models that do not meet the thresholds."""

from __future__ import annotations

import json

import mlflow

from training.config import load_params, repo_path


class QualityGateError(Exception):
    """Raised when a model fails the quality gate."""


def evaluate_gate(metrics: dict[str, float], rules: dict[str, float]) -> list[str]:
    """Return a list of human-readable failures (empty list == gate passed)."""
    failures: list[str] = []
    if metrics["f1"] < rules["min_f1"]:
        failures.append(f"f1 {metrics['f1']:.4f} < min_f1 {rules['min_f1']}")
    if metrics["recall"] < rules["min_recall"]:
        failures.append(f"recall {metrics['recall']:.4f} < min_recall {rules['min_recall']}")
    gap = metrics.get("train_test_f1_gap", 0.0)
    if gap > rules["max_train_test_f1_gap"]:
        failures.append(
            f"train/test f1 gap {gap:.4f} > {rules['max_train_test_f1_gap']} (overfitting)"
        )
    return failures


def run(raise_on_fail: bool = True) -> dict:
    params = load_params()
    metrics = json.loads(repo_path(params["paths"]["metrics"]).read_text(encoding="utf-8"))
    failures = evaluate_gate(metrics, params["quality_gate"])
    status = "failed" if failures else "passed"

    with mlflow.start_run(run_id=metrics["run_id"]):
        mlflow.set_tag("quality_gate", status)
        mlflow.set_tag("quality_gate.reasons", "; ".join(failures) or "all thresholds met")

    result = {"run_id": metrics["run_id"], "status": status, "failures": failures}
    print(json.dumps(result, indent=2))
    if failures and raise_on_fail:
        raise QualityGateError("; ".join(failures))
    return result


if __name__ == "__main__":
    run()
