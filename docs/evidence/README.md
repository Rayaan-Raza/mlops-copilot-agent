# Live-run evidence (local stack, 2026-11-14)

The stack was Docker Compose (MLflow 3.18 + Airflow 3.1, LocalExecutor) and a kind v1.34 cluster (`mlops`).
Every JSON file in this folder is the raw `ToolResult` output from the agent's own tools.

## Failure investigation 1: training DAG fails at `train_model` (genuine, unplanned)

File: [`failure-airflow-missing-skops.json`](failure-airflow-missing-skops.json)

| | |
|---|---|
| Symptom | `training_pipeline` run `…ZZEFL8Uy` finished `failed` |
| Agent trace | `airflow_wait_for_dag_run` → `airflow_get_failed_tasks` |
| Evidence | `train_model` failed after 3 tries (`retries=2`). `evaluate_and_log_mlflow`, `apply_quality_gate` and `promote_model` were `upstream_failed`. The log error line was `ModuleNotFoundError: No module named 'skops'` |
| Root cause | MLflow 3 serialises sklearn models with **skops**. The Airflow image installed `mlflow-skinny`, which does not depend on skops |
| Resolution | Added `skops` to `infra/airflow/requirements.txt`, pinned `mlflow-skinny==3.18.0` to match the server, rebuilt the image and re-triggered. Run `…shje8dn2` passed all 6 tasks and registered `breast-cancer-classifier` v1 with alias `champion` |
| Safety observed | Retries ran, then the dependency chain blocked promotion. No unvalidated model reached the registry |

## Failure investigation 2: bad image tag (deliberately injected)

Files: [`failure-f1-imagepullbackoff.json`](failure-f1-imagepullbackoff.json), [`failure-f1-resolved.json`](failure-f1-resolved.json)

| | |
|---|---|
| Injection | `kubectl -n mlops-agent set image deployment/ml-inference ml-inference=ml-inference:9.9.9` |
| Agent trace | `k8s_diagnose_deployment`, `k8s_get_deployed_version` |
| Evidence | The new pod was `Pending` and not Ready, waiting with `ImagePullBackOff`. Warning events read `Failed to pull image "ml-inference:9.9.9" … pull access denied`. Version check: image tag `9.9.9` vs ConfigMap `APP_VERSION 1.0.0`, `version_sources_consistent: false` |
| Diagnosis | The API was **degraded, not down**. `maxUnavailable: 0` kept both old pods serving (2/2 ready, Service endpoints healthy) |
| Resolution | `kubectl -n mlops-agent rollout undo deployment/ml-inference`. Afterwards: tag `1.0.0`, consistent, 2/2 ready |

## Failure investigation 3: API down with CrashLoopBackOff (injected, F2)

Files: [`failure-f2-crashloop.json`](failure-f2-crashloop.json), [`failure-f2-restart-confirmed.json`](failure-f2-restart-confirmed.json), [`failure-f2-resolved.json`](failure-f2-resolved.json). Transcript: [`../conversations/04-diagnosis.json`](../conversations/04-diagnosis.json)

| | |
|---|---|
| Injection | Set `SIMULATE_STARTUP_FAILURE=true` in the ConfigMap, then deleted all pods |
| Agent trace | "Why is my prediction API unavailable?" → `k8s_diagnose_deployment` |
| Evidence | 0/2 ready, `CrashLoopBackOff` after 4 restarts, readiness probe refused, and `RuntimeError: injected startup failure: model store unreachable` in the previous container's logs. `api_reachable_via_service=false` |
| Resolution | Set the flag back to `false`, then ran `k8s_restart_deployment`. It returned `confirmation_required` and ran once confirmed. Afterwards: 2/2 ready and reachable |

## Other live checks (all `ok`)

- **DVC**
  - the current dataset resolves to `data-v2.0` (md5 `75953f06…`)
  - the v1→v2 diff shows `modified`, +54,206 bytes
  - the best MLflow run's `dvc.data_md5` resolves back to the `data-v2.0` tag
- **MLflow**
  - 9 runs found
  - best by F1 is `logistic_regression-data-v2.0` (F1 0.9861, margin 0.0071), with readiness caveats
  - the registry shows the `champion` alias
- **Airflow**
  - JWT auth works and the DAG is listed as approved
  - the trigger returned `confirmation_required`, and the run only started after the token was confirmed
- **Kubernetes**
  - 2/2 pods healthy and the ClusterIP service has 2 ready endpoints
  - a restart in `kube-system` was **denied** with no token issued

## Bugs found by the live run and fixed

1. `dvc_list_tracked_datasets` crashed on Windows because the `.dvc/` folder matched `*.dvc`.
2. The `dvc` executable wasn't found when the venv was not activated.
3. An empty `KUBECONFIG=` (as in `.env.example`) broke Kubernetes config loading.
4. `dvc_check_data_status` reported an unrun `dvc.yaml` pipeline as if it were a data change.
5. The Airflow image was missing `skops` (investigation 1 above).
6. The log redaction missed provider-masked API keys echoed in error messages.
