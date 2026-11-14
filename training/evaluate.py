"""Evaluate the trained model on the held-out split and log metrics to the same MLflow run."""

from __future__ import annotations

import json
import pickle

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from training.config import load_params, repo_path


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def run() -> dict[str, float]:
    params = load_params()
    dcfg, paths = params["data"], params["paths"]
    with open(repo_path(paths["run_info"]), encoding="utf-8") as fh:
        info = json.load(fh)
    with open(repo_path(paths["model_dir"]) / "model.pkl", "rb") as fh:
        model = pickle.load(fh)  # noqa: S301 - artefact produced by our own train stage

    test_df = pd.read_csv(repo_path(dcfg["processed_dir"]) / "test.csv")
    x_test, y_test = test_df.drop(columns=[dcfg["target"]]), test_df[dcfg["target"]]
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_score)
    metrics["train_test_f1_gap"] = float(info["train_f1"] - metrics["f1"])
    cm = confusion_matrix(y_test, y_pred).tolist()

    with mlflow.start_run(run_id=info["run_id"]):
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"labels": [0, 1], "matrix": cm}, "evaluation/confusion_matrix.json")
        mlflow.log_param("n_test", len(test_df))

    out = repo_path(paths["metrics"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_id": info["run_id"], **metrics}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run()
