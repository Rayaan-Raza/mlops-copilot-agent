"""Register the gated model in the MLflow Model Registry and point the champion alias at it.

Only ever executed after ``quality_gate`` succeeds (enforced by the Airflow DAG dependencies).
"""

from __future__ import annotations

import json

import mlflow
from mlflow.tracking import MlflowClient

from training.config import load_params, repo_path


def run() -> dict:
    params = load_params()
    mcfg = params["mlflow"]
    info = json.loads(repo_path(params["paths"]["run_info"]).read_text(encoding="utf-8"))
    client = MlflowClient()

    run = client.get_run(info["run_id"])
    if run.data.tags.get("quality_gate") != "passed":
        raise RuntimeError(f"refusing to promote run {info['run_id']}: quality gate not passed")

    mv = mlflow.register_model(info["model_uri"], mcfg["registered_model"])
    for key in ("dvc.dataset_version", "dvc.data_md5", "git.commit"):
        if key in run.data.tags:
            client.set_model_version_tag(
                mcfg["registered_model"], mv.version, key, run.data.tags[key]
            )
    client.set_model_version_tag(
        mcfg["registered_model"], mv.version, "source_run_id", run.info.run_id
    )
    client.set_registered_model_alias(mcfg["registered_model"], mcfg["champion_alias"], mv.version)

    result = {"model": mcfg["registered_model"], "version": mv.version, "run_id": run.info.run_id}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
