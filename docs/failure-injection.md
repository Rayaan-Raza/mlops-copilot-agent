# Failure injection runbook

Use these scenarios in the demo and in the report's two failure investigations. For each one: inject the failure, ask the agent, save the transcript (`mlops-agent --json "..." > docs/conversations/<name>.json`), then restore.

## F1: Bad image tag (ImagePullBackOff)

```bash
kubectl -n mlops-agent set image deployment/ml-inference ml-inference=ml-inference:9.9.9
```

Ask: *"Why is my prediction API unavailable?"*

Expected evidence chain:
1. `k8s_diagnose_deployment`: the new pod is `ImagePullBackOff`, `updated_replicas` is behind, and an event reads `Failed to pull image "ml-inference:9.9.9"`.
2. `k8s_get_deployed_version`: the image tag `9.9.9` doesn't match the ConfigMap `APP_VERSION 1.0.0`, so `version_sources_consistent=false`.
3. Because `maxUnavailable: 0`, the old pods keep serving. The agent should note that the API is degraded rather than down.

Root cause: the deployment references an image tag that doesn't exist.

Resolve:

```bash
kubectl -n mlops-agent rollout undo deployment/ml-inference
```

## F2: Crash on startup (CrashLoopBackOff)

```bash
kubectl -n mlops-agent patch configmap ml-inference-config --type merge -p '{"data":{"SIMULATE_STARTUP_FAILURE":"true"}}'
kubectl -n mlops-agent rollout restart deployment/ml-inference
```

Ask: *"Is the prediction API healthy? If not, why?"*

Expected evidence chain:
1. `k8s_find_unhealthy_pods`: the pods show `CrashLoopBackOff` with a rising restart count.
2. `k8s_get_pod_logs(previous=true)`: `injected startup failure: model store unreachable`.
3. `k8s_list_services`: `ready_endpoints` drops while the rollout stalls.

Root cause: the application fails during startup, so the readiness probe never passes.

Resolve: set the value back to `"false"`, then run `rollout restart`. The agent may propose `k8s_restart_deployment`, but it only runs after `/confirm`.

## F3: Corrupt dataset blocks the pipeline (Airflow)

```bash
python -m training.make_dataset --version corrupt     # workspace now differs from the DVC pointer
```

Trigger `training_pipeline`, then ask: *"Did the last training run succeed? If not, what failed?"*

Expected evidence chain:
1. `dvc_check_data_status`: `workspace_in_sync=false`, because the dataset was modified.
2. `airflow_get_failed_tasks`: `validate_dataset` failed. Its log shows `missing required columns: ['mean_smoothness']` and a null fraction above the limit. The remaining 5 tasks are `upstream_failed`, so the promotion was blocked.

Root cause: an unversioned change to the training data.

Resolve:

```bash
dvc checkout data/raw/dataset.csv.dvc
```

## F4: Quality gate failure

Set `quality_gate.min_f1: 0.995` in `params.yaml` and trigger the DAG. The `apply_quality_gate` task fails, `promote_model` becomes `upstream_failed`, and the MLflow run gets the tag `quality_gate=failed`. `mlflow_find_best_run` then reports this as a production-readiness caveat.

## F5: Service outage (no invented answers)

```bash
docker compose -f infra/docker-compose.yaml stop mlflow
```

Ask: *"What is the best model?"*

The `mlflow_*` tools return `unavailable`, `facts` is empty, and `uncertainties` explains that MLflow could not be reached. The agent does not produce any metric values.
