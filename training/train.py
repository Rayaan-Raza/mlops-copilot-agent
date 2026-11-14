"""Train a classifier and log it to MLflow together with data provenance tags."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import f1_score

from training import provenance
from training.config import load_params, repo_path
from training.models import build_model

__all__ = ["build_model", "run"]


# Types we create ourselves and therefore trust when MLflow serialises with skops (MLflow >= 3.x).
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree", "sklearn._loss._loss.CyHalfBinomialLoss"]


def _log_model(model) -> str:
    """Log the sklearn model; supports MLflow 2.x (artifact_path) and 3.x (name, skops)."""
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    kwargs: dict[str, Any] = {}
    if "skops_trusted_types" in params:
        kwargs["skops_trusted_types"] = SKOPS_TRUSTED_TYPES
    if "name" in params:
        info = mlflow.sklearn.log_model(model, name="model", **kwargs)
    else:
        info = mlflow.sklearn.log_model(model, artifact_path="model", **kwargs)
    return info.model_uri


def run(model_type: str | None = None, hyperparams: dict[str, Any] | None = None) -> dict:
    params = load_params()
    dcfg, tcfg, mcfg = params["data"], params["train"], params["mlflow"]
    model_type = model_type or tcfg["model_type"]
    hp = {**tcfg.get(model_type, {}), **(hyperparams or {})}

    train_df = pd.read_csv(repo_path(dcfg["processed_dir"]) / "train.csv")
    x_train = train_df.drop(columns=[dcfg["target"]])
    y_train = train_df[dcfg["target"]]

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", mcfg["experiment_name"]))
    prov = provenance.collect(dcfg["raw_path"])
    run_name = f"{model_type}-{prov.get('dvc.dataset_version', 'untagged')}"

    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags(prov)
        mlflow.set_tag("model_type", model_type)
        if os.getenv("AIRFLOW_CTX_DAG_RUN_ID"):
            mlflow.set_tag("airflow.dag_run_id", os.environ["AIRFLOW_CTX_DAG_RUN_ID"])
            mlflow.set_tag("airflow.dag_id", os.getenv("AIRFLOW_CTX_DAG_ID", ""))
        mlflow.log_params({"model_type": model_type, **hp})
        mlflow.log_params({"n_train": len(train_df), "n_features": x_train.shape[1]})

        model = build_model(model_type, hp, tcfg["random_state"])
        model.fit(x_train, y_train)
        train_f1 = f1_score(y_train, model.predict(x_train))
        mlflow.log_metric("train_f1", train_f1)
        model_uri = _log_model(model)

        model_dir = repo_path(params["paths"]["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "model.pkl", "wb") as fh:
            pickle.dump(model, fh)

        info = {
            "run_id": active.info.run_id,
            "experiment_id": active.info.experiment_id,
            "model_type": model_type,
            "model_uri": model_uri,
            "train_f1": train_f1,
            "feature_names": list(x_train.columns),
            **prov,
        }
    with open(repo_path(params["paths"]["run_info"]), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    print(f"trained {model_type} run_id={info['run_id']} train_f1={train_f1:.4f}")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default=None)
    args = parser.parse_args()
    run(args.model_type)
