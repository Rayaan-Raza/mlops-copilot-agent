# Technical Report: MLOps Copilot Agent

> Fill in the sections marked TODO with evidence from your own environment (screenshots in `screenshots/`, transcripts in `docs/conversations/`).

## 1. System architecture

See [../architecture.md](../architecture.md). It covers the component diagram, the provenance chain, the guards and the permissions.

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
| Policy | agent_assess_retraining | read (recommendation only) |

## 3. Evidence from each system (TODO: screenshots)

| System | Screenshot |
|---|---|
| DVC | `git tag -n data-v*`, `dvc status`, `dvc diff data-v1.0 data-v2.0` |
| MLflow | experiment run comparison, run tags showing `dvc.*`, model registry with the `champion` alias |
| Airflow | DAG graph view, a successful run, a failed run with `validate_dataset` red |
| Kubernetes | `kubectl -n mlops-agent get deploy,pods,svc`, `describe pod` during a failure |

## 4. Five tested agent conversations (TODO: transcripts)

| # | Scenario | Prompt | Transcript |
|---|---|---|---|
| 1 | Dataset provenance | "Which dataset version was used for the best run?" | `docs/conversations/01-provenance.json` |
| 2 | Model selection | "Compare the last five training runs and recommend the best candidate based on F1 score." | `02-model-selection.json` |
| 3 | Pipeline execution | "Trigger the model training pipeline and report whether execution succeeded." | `03-pipeline.json` |
| 4 | Deployment diagnosis | "Why is my prediction API unavailable?" | `04-diagnosis.json` |
| 5 | Multi-tool reasoning | the mandatory integration prompt | `05-multi-tool.json` |

## 5. Failure investigations

Use F1 and F3 (or F2) from [failure-injection.md](failure-injection.md). For each one, give:
- the symptom
- the agent's tool trace
- the root cause
- the resolution
- what the guards did

## 6. Testing, reliability and safety

The suite has 59 automated tests (`pytest -m "not integration"`). They cover:
- tool selection and execution across all four modules
- the mandatory multi-tool scenario, including a check that no retraining is triggered
- timeouts, invalid arguments, unknown tools and service outages for MLflow, Airflow, Kubernetes and DVC
- the agent not inventing results when a tool is unavailable
- unauthorized Kubernetes changes being rejected, plus RBAC 403 mapping
- confirmation tokens being single-use
- structured logs that include request_id, tool, status and duration, with secrets redacted
- dataset validation and quality-gate logic
- the inference app, including injected startup failure

Live integration tests run against a kind cluster in CD.

## 7. Known limitations and proposed improvements

- **Limitation:** the DVC remote is a local folder. **Improvement:** move to S3 or MinIO with credentials from a secret manager.
- **Limitation:** single-tenant API without authentication. **Improvement:** OIDC in front of `/chat` and `/confirm`, and per-user audit of confirmations.
- **Limitation:** offline metrics only. **Improvement:** add live prediction monitoring (Prometheus plus drift metrics) as another evidence source for the retraining policy.
- **Limitation:** the LLM can still misread a tool result. The guards only check that a fact cites successful evidence, not that it matches what the evidence says. **Improvement:** structured fact extraction checked against tool JSON.
- **Limitation:** confirmation tokens live in process memory. **Improvement:** store them in Redis for multi-replica API deployments.
- **Limitation:** staging is an ephemeral kind cluster. **Improvement:** a persistent staging cluster with GitOps (Argo CD) promotion to production.
