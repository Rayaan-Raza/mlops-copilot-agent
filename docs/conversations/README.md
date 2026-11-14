# Recorded agent conversations

Real runs of `mlops-agent --json "<question>"` against the live local stack: Compose (MLflow 3.18, Airflow 3.1) and a kind cluster.
The LLM is `openai/gpt-4.1-mini` through OpenRouter. These runs used `LLM_PROVIDER=openai` with `OPENAI_BASE_URL=https://openrouter.ai/api/v1`. The dedicated `LLM_PROVIDER=openrouter` added since sends exactly the same requests.
Each JSON file is the full `AgentResponse`: answer, verified facts, evidence (the tool calls), uncertainties and pending confirmations.

| # | File | Question | Tools called | What it shows |
|---|---|---|---|---|
| 1 | [01-multi-tool.json](01-multi-tool.json) | **Mandatory:** best recent model → dataset version → deployment status → is retraining needed? | `mlflow_find_best_run`, `k8s_get_deployed_version`, `mlflow_get_run_details`, `agent_assess_retraining` | Cross-system reasoning across MLflow, DVC provenance, Kubernetes and the policy. Every fact cites a tool call. Retraining is **not** recommended: the model is fresh, F1 0.9861 ≥ 0.90, the gate passed, the last pipeline run succeeded and the data is unchanged |
| 2 | [02-dataset-lineage.json](02-dataset-lineage.json) | What changed between `data-v1.0` and `data-v2.0`, and which runs used each? | `dvc_compare_dataset_versions`, `mlflow_list_runs` | DVC diff (+54,206 bytes). All current runs trace to v2.0, and the agent says it did not check older experiments |
| 3 | [03-pipeline-failure.json](03-pipeline-failure.json) | Why did a `training_pipeline` run fail, and did an unvalidated model get through? | `airflow_get_failed_tasks`, … | Root cause from the task log: `ModuleNotFoundError: skops`. Evaluate, gate and promote were `upstream_failed`, so nothing reached the registry. A tool call with a wrong experiment name was rejected (`invalid_request`) and reported, not hidden |
| 4 | [04-safety-actions.json](04-safety-actions.json) | Trigger the DAG and restart CoreDNS in `kube-system` | `airflow_trigger_dag` → `confirmation_required`; `k8s_restart_deployment` → `denied` | Safety: the trigger only produced a confirmation token and did not run. The out-of-scope namespace was refused outright. Unsupported claims were stripped by the evidence check |
| 5 | [05-deployment-health.json](05-deployment-health.json) | Is the inference API healthy? | `k8s_find_unhealthy_pods`, `k8s_list_services`, `k8s_get_events`, `k8s_get_deployed_version` | 2/2 pods ready, 2 ready endpoints, no warning events, and the version is consistent across image tag, label and ConfigMap |

## Issues these runs exposed, now fixed

- **The LLM transcribed numbers wrongly.** In the first version of conversation 1, the model converted an ISO timestamp to epoch milliseconds incorrectly ("model is 4,748 days old") and left out the pipeline state. `agent_assess_retraining` now takes `run_id` and `dag_id` and fetches the training time, F1, quality gate, DVC hash and last DAG state itself. It reports the source of each input in `input_sources`.
- **Confirmation was asked in text.** The model asked for confirmation in prose instead of calling the tool, which skipped the real token-based gate. The system prompt and the descriptions of state-changing tools now say that calling the tool *is* the confirmation step.
- **Provider limits.** OpenRouter returned HTTP 402 when no output cap was set, so `LLM_MAX_TOKENS` (default 4096) is now sent. Groq's free tier allows 8,000 tokens per minute, so there is a configurable `AGENT_TOOL_RESULT_MAX_CHARS` and SDK retries that honour `retry-after`.

Confirmation tokens in these files have expired: they are single-use and live 5 minutes.
