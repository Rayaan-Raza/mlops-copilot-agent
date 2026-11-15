# Technical Report: MLOps Copilot Agent

All evidence in this report comes from a live local environment:
- Docker Compose with MLflow 3.18, Airflow 3.1 (LocalExecutor) and Postgres
- a kind v1.34 Kubernetes cluster running the inference service
- DVC with two tagged dataset versions

The LLM is `openai/gpt-4.1-mini` through OpenRouter. Screenshots are in [`../screenshots/`](../screenshots/), agent transcripts in [`conversations/`](conversations/) and raw tool output in [`evidence/`](evidence/).

## 1. System architecture

See [../architecture.md](../architecture.md). It covers the component diagram, the provenance chain, the guards and the permissions.

In short, the agent is an LLM tool-calling loop (`agent/orchestrator.py`) over a registry of 33 typed tools in four modules: DVC, MLflow, Airflow and Kubernetes. One policy tool sits on top. Every tool call goes through the registry, which enforces:
- argument validation
- timeouts
- error classification
- permission checks
- single-use confirmation tokens for state-changing actions

After the LLM answers, an evidence check removes any "fact" that doesn't cite a successful tool call.

**Provenance chain.** Each training run is tagged in MLflow with `dvc.data_md5`, `dvc.dataset_version`, `dvc.data_path` and `git.commit`. `dvc_resolve_dataset_version` maps the md5 back to a git tag (`data-v1.0` / `data-v2.0`). The inference service reports what it serves on `/version`, which the agent reads through the Kubernetes API proxy.

## 2. Tool definitions and permissions

Generate the live list with `curl localhost:8001/tools`.

| Module | Tool | Permission |
|---|---|---|
| DVC | dvc_list_tracked_datasets, dvc_get_dataset_version, dvc_list_dataset_versions, dvc_resolve_dataset_version, dvc_compare_dataset_versions, dvc_check_data_status, dvc_get_reproduction_steps | read |
| DVC | dvc_checkout_dataset_version | **write (confirm)** |
| MLflow | mlflow_list_experiments, mlflow_list_runs, mlflow_get_run_details, mlflow_find_best_run, mlflow_compare_runs, mlflow_compare_experiments, mlflow_get_model_registry | read |
| Airflow | airflow_list_dags, airflow_get_dag_runs, airflow_get_task_instances, airflow_get_failed_tasks, airflow_get_task_logs, airflow_wait_for_dag_run | read |
| Airflow | airflow_trigger_dag (approved DAGs only) | **write (confirm)** |
| Kubernetes | k8s_list_deployments, k8s_list_pods, k8s_get_pod_logs, k8s_list_services, k8s_get_events, k8s_find_unhealthy_pods, k8s_get_deployed_version, k8s_diagnose_deployment | read (approved namespaces) |
| Kubernetes | k8s_restart_deployment, k8s_scale_deployment | **write (confirm, approved namespaces)** |
| Policy | agent_assess_retraining | read (recommendation only; it looks up the run, data hash and DAG state itself) |

Safety is enforced at two independent layers:
1. **The agent.** It keeps a namespace and DAG allow-list, and requires a confirmation token for every write.
2. **Kubernetes RBAC.** The `mlops-agent` ServiceAccount has only read access, `services/proxy` GET, and `patch` on deployments, all in `mlops-agent` only (see screenshot 8: `can-i delete pods -n kube-system` → `no`).

## 3. Evidence from each system

| # | System | Screenshot | What it shows |
|---|---|---|---|
| 1 | DVC | [01-dvc-versions.png](../screenshots/01-dvc-versions.png) | Tags `data-v1.0` (398 rows, 30 features) and `data-v2.0` (569 rows, 32 features), the `.dvc` pointer (md5 `75953f06…`, 139,720 bytes), `dvc diff data-v1.0 data-v2.0` → `modified`, and `dvc status` → up to date |
| 2 | MLflow | [02-mlflow-runs.png](../screenshots/02-mlflow-runs.png) | 13 runs in `breast-cancer-classifier`: sweep runs (LR, RF and GB), Airflow runs, and the two failed Airflow runs from the skops incident |
| 3 | MLflow | [03-mlflow-compare.png](../screenshots/03-mlflow-compare.png) | Side-by-side comparison of the top five runs by F1 |
| 4 | MLflow | [04-mlflow-run-dvc-tags.png](../screenshots/04-mlflow-run-dvc-tags.png) | The best run (`logistic_regression-data-v2.0`, F1 0.9861). Tags `dvc.data_md5`, `dvc.dataset_version=data-v2.0`, `git.commit`, `quality_gate=passed` |
| 5 | MLflow | [05-mlflow-registry.png](../screenshots/05-mlflow-registry.png) | Registered model `breast-cancer-classifier`: version 2 holds the `champion` alias, and every version is tagged with its dataset version, md5, commit and source run |
| 6 | Airflow | [06-airflow-graph-success.png](../screenshots/06-airflow-graph-success.png) | Graph of the run the agent triggered (conversation 3): all six tasks succeeded |
| 7 | Airflow | [07-airflow-graph-failed.png](../screenshots/07-airflow-graph-failed.png) | Graph of the failed run: `train_model` failed, while evaluate, gate and promote are `upstream_failed` (investigation 1) |
| 7b | Airflow | [07b-airflow-runs-grid.png](../screenshots/07b-airflow-runs-grid.png) | Runs grid: the failed run next to the successful ones |
| 8 | Kubernetes | [08-k8s-healthy.png](../screenshots/08-k8s-healthy.png) | Deployment 2/2, ClusterIP service with 2 endpoints, image `ml-inference:1.0.0`, and the RBAC checks |
| 9 | Kubernetes | [09-k8s-failure-crashloop.png](../screenshots/09-k8s-failure-crashloop.png) | During investigation 2: 0/2 ready, `CrashLoopBackOff`, no endpoints, readiness probe refused, and the crash line in the logs |

## 4. Five tested agent conversations

Each transcript is the full structured response: answer, cited facts, tool evidence, uncertainties and pending confirmations. Details are in [conversations/README.md](conversations/README.md).

| # | Scenario | Prompt | Tools called | Result | Transcript |
|---|---|---|---|---|---|
| 1 | Dataset provenance | "Which dataset version was used for the best run?" | `mlflow_find_best_run` | `data-v2.0`, md5 `75953f06…`, logistic regression with F1 0.9861 and the gate passed. It notes that only offline metrics are available | [01-provenance.json](conversations/01-provenance.json) |
| 2 | Model selection | "Compare the last five training runs and recommend the best candidate based on F1 score." | `mlflow_compare_runs` | Recommends `gradient_boosting` run `0058987c` (F1 0.9589, gate passed) as the best of the five most recent runs | [02-model-selection.json](conversations/02-model-selection.json) |
| 3 | Pipeline execution | "Trigger the model training pipeline and report whether execution succeeded." | turn 1 `airflow_trigger_dag` → `confirmation_required`; human confirms; turn 2 `airflow_wait_for_dag_run` | Nothing ran until the token was confirmed. Run `…hpHJ4bi0` then **succeeded** and all six tasks were reported. The run registered model version 2 (screenshots 5 and 6) | [03-pipeline.json](conversations/03-pipeline.json) |
| 4 | Deployment diagnosis | "Why is my prediction API unavailable?" (asked during injected failure F2) | `k8s_diagnose_deployment` | 0/2 ready, `CrashLoopBackOff` after 4 restarts, readiness probe refused, and the log line `injected startup failure: model store unreachable`. Recommends removing the flag and restarting | [04-diagnosis.json](conversations/04-diagnosis.json) |
| 5 | Multi-tool reasoning | "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary." | `mlflow_find_best_run`, `k8s_get_deployed_version`, `agent_assess_retraining` | The best run is LR on `data-v2.0` (F1 0.9861). The **deployed** model is a *different* one, a random forest (test F1 0.951), served 2/2 ready. Retraining is not needed: the model is 0.6 days old, the gate passed, the last pipeline run succeeded and the data is unchanged. It flags that the deployment is not linked to an MLflow run id | [05-multi-tool.json](conversations/05-multi-tool.json) |

Notes on accuracy:
- **Conversation 2.** Two of the five most recent runs were the failed Airflow runs, which have no metrics. The recommendation is correct among the runs that do have F1, but the answer doesn't call out that two runs had no metrics.
- **Conversation 3.** It was recorded with a small script (`agent.run` → `agent.confirm(token)` → `agent.run`). The script does exactly what a user does by typing `yes` in the CLI or calling `POST /confirm`. The confirmation step is recorded in the transcript as a `human_confirmation` turn.
- **Additional recordings** are kept in `conversations/extra-*.json`: dataset lineage, a pipeline-failure diagnosis, a safety test, a health check, and the first take of conversation 5.

## 5. Failure investigations

### Investigation 1: training DAG fails at `train_model` (genuine, unplanned)

| | |
|---|---|
| Symptom | `training_pipeline` run `…ZZEFL8Uy` finished `failed` ([screenshot 7](../screenshots/07-airflow-graph-failed.png)) |
| Agent trace | `airflow_wait_for_dag_run` → `airflow_get_failed_tasks` (raw: [evidence/failure-airflow-missing-skops.json](evidence/failure-airflow-missing-skops.json)). The agent reproduced the diagnosis in [extra-pipeline-failure.json](conversations/extra-pipeline-failure.json) |
| Evidence | `train_model` failed after 3 tries (`retries=2`), and the task log reads `ModuleNotFoundError: No module named 'skops'`. `evaluate_and_log_mlflow`, `apply_quality_gate` and `promote_model` are `upstream_failed` |
| Root cause | MLflow 3 serialises scikit-learn models with **skops**. The Airflow image installed `mlflow-skinny`, which does not depend on it |
| Resolution | Added `skops` to `infra/airflow/requirements.txt` and pinned `mlflow-skinny==3.18.0` to match the server. Rebuilt the image and re-triggered: all 6 tasks succeeded |
| What the guards did | Retries ran first, then the task dependencies stopped the pipeline before promotion, so no unvalidated model reached the registry. Asked about it, the agent confirmed this from task states instead of assuming it |

### Investigation 2: inference API down (injected CrashLoopBackOff, F2)

| | |
|---|---|
| Injection | `SIMULATE_STARTUP_FAILURE=true` in the ConfigMap. Then all pods were deleted, so none stayed healthy ([runbook F2](failure-injection.md)) |
| Symptom | Deployment 0/2 available and no Service endpoints ([screenshot 9](../screenshots/09-k8s-failure-crashloop.png)) |
| Agent trace | "Why is my prediction API unavailable?" → `k8s_diagnose_deployment` ([04-diagnosis.json](conversations/04-diagnosis.json); raw: [evidence/failure-f2-crashloop.json](evidence/failure-f2-crashloop.json)) |
| Evidence | Both pods are not Ready and in `CrashLoopBackOff` after 4 restarts. Warning events show `Readiness probe failed … connection refused` and `Back-off restarting failed container`. The previous container's logs read `RuntimeError: injected startup failure: model store unreachable`. `api_reachable_via_service=false` |
| Root cause | The app fails during startup, so the readiness probe never passes and the Service has no endpoints |
| Resolution | Set the flag back to `false`, then ran the agent's `k8s_restart_deployment`. It returned `confirmation_required` and executed only after the token was confirmed ([evidence/failure-f2-restart-confirmed.json](evidence/failure-f2-restart-confirmed.json)). Afterwards: 2/2 ready and the API reachable ([evidence/failure-f2-resolved.json](evidence/failure-f2-resolved.json)) |
| What the guards did | Even for an obvious fix, the restart went through the confirmation gate. A restart outside `mlops-agent` is denied outright ([extra-safety-actions.json](conversations/extra-safety-actions.json)) |

A further investigation, F1 (a bad image tag causing `ImagePullBackOff`, where the API stayed degraded but not down because `maxUnavailable: 0`), is documented as investigation 2 in [evidence/README.md](evidence/README.md).

## 6. Testing, reliability and safety

The suite has 68 automated tests (`pytest -m "not integration"`), and they all pass in CI on Python 3.10 and 3.11. They cover:
- tool selection and execution across all four modules
- the mandatory multi-tool scenario, including a check that no retraining is triggered
- timeouts, invalid arguments, unknown tools and service outages for MLflow, Airflow, Kubernetes and DVC
- the agent not inventing results when a tool is unavailable
- unauthorized Kubernetes changes being rejected, plus RBAC 403 mapping
- confirmation tokens being single-use
- structured logs that include request_id, tool, status and duration, with secrets redacted (including provider-masked keys)
- dataset validation and quality-gate logic
- the inference app, including injected startup failure
- regression tests for each bug found in the live runs (below)

CI also covers ruff lint, Airflow DAG import integrity, Kubernetes manifest and Compose validation, image builds, and a training-pipeline smoke test.

**Bugs found only by running against the live stack**, each fixed with a regression test:
1. The DVC pointer scan crashed on Windows, and `dvc` wasn't found unless the virtual environment was activated.
2. An empty `KUBECONFIG=` broke Kubernetes config loading.
3. `dvc_check_data_status` reported a pipeline that had never been run as a data change.
4. The Airflow image was missing `skops` (investigation 1).
5. The retraining policy trusted numbers the LLM typed. The model converted a timestamp wrongly ("model is 4,748 days old") and left out the pipeline state. The tool now fetches these itself from MLflow, DVC and Airflow, and reports where each input came from.
6. The LLM asked for confirmation in prose instead of calling the tool, so the real token gate never ran. The prompt and the descriptions of state-changing tools were fixed.
7. The agent claimed "the best model is deployed", but the service was serving a different model. `k8s_get_deployed_version` now reads the service's `/version` and states that the deployment isn't linked to an MLflow run id.

## 7. Known limitations and proposed improvements

- **Limitation:** the deployed model isn't linked to an MLflow run id. The inference image bakes in a model identified only by type, dataset, test F1 and hash. The agent can therefore say *what* is served, but can't prove *which run* it came from. In this environment, the serving model (RF, F1 0.951) is not the best run (LR, F1 0.9861). **Improvement:** serve `models:/breast-cancer-classifier@champion` through `MODEL_URI`, and expose the run id on `/version`.
- **Limitation:** the evidence check verifies that a fact *cites* a successful tool call, not that its *content* matches the tool output (see the accuracy note on conversation 2). **Improvement:** structured fact extraction, checked against the tool JSON.
- **Limitation:** the DVC remote is a local folder. **Improvement:** move to S3 or MinIO with credentials from a secret manager.
- **Limitation:** the API is single-tenant with no authentication. **Improvement:** OIDC in front of `/chat` and `/confirm`, and a per-user audit log of confirmations.
- **Limitation:** metrics are offline only. **Improvement:** live prediction monitoring (Prometheus plus drift metrics) as another evidence source for the retraining policy.
- **Limitation:** confirmation tokens live in process memory. **Improvement:** store them in Redis for API deployments with several replicas.
- **Limitation:** staging is an ephemeral kind cluster. **Improvement:** a persistent staging cluster with GitOps (Argo CD) promotion to production.
