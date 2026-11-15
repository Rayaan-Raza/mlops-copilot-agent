# Recorded agent conversations

These are real runs against the live local stack: Compose (MLflow 3.18, Airflow 3.1) and a kind cluster.
The LLM is `openai/gpt-4.1-mini` through OpenRouter (`LLM_PROVIDER=openrouter`).
Each JSON file is the full `AgentResponse`: answer, verified facts, evidence (the tool calls), uncertainties and pending confirmations.
Single-question runs use `mlops-agent --json "<question>"`.

## The five required scenarios

| # | File | Prompt | Tools called | What it shows |
|---|---|---|---|---|
| 1 | [01-provenance.json](01-provenance.json) | "Which dataset version was used for the best run?" | `mlflow_find_best_run` | `data-v2.0` and md5 `75953f06…` from the run's DVC tags, with the offline-only caveat |
| 2 | [02-model-selection.json](02-model-selection.json) | "Compare the last five training runs and recommend the best candidate based on F1 score." | `mlflow_compare_runs` | Recommends gradient boosting `0058987c` (F1 0.9589). It doesn't mention that two of the five runs were failed Airflow runs with no metrics |
| 3 | [03-pipeline.json](03-pipeline.json) | "Trigger the model training pipeline and report whether execution succeeded." | `airflow_trigger_dag` → `confirmation_required`, then a human confirmation, then `airflow_wait_for_dag_run` | Multi-turn. Nothing ran until the token was confirmed. Run `…hpHJ4bi0` then succeeded and all six task states were reported |
| 4 | [04-diagnosis.json](04-diagnosis.json) | "Why is my prediction API unavailable?" (during injected failure F2) | `k8s_diagnose_deployment` | 0/2 ready, `CrashLoopBackOff`, readiness probe refused, and the crash line from the previous container's logs |
| 5 | [05-multi-tool.json](05-multi-tool.json) | Mandatory: best model → dataset version → deployment → retraining? | `mlflow_find_best_run`, `k8s_get_deployed_version`, `agent_assess_retraining` | The best run is LR on `data-v2.0`. The deployed model is a different one (RF, F1 0.951). Retraining is not needed, and every policy input was fetched from the real systems |

**How conversation 3 was recorded.** A small script calls `agent.run(prompt)`, then `agent.confirm(token)`, then `agent.run(follow_up)`. That is exactly what a user does by typing `yes` in the interactive CLI, or by calling `POST /confirm` on the API. The transcript records the approval as a `human_confirmation` turn.

## Additional recordings

| File | Prompt | What it shows |
|---|---|---|
| [extra-dataset-lineage.json](extra-dataset-lineage.json) | What changed between `data-v1.0` and `data-v2.0`, and which runs used each? | DVC diff (+54,206 bytes), and all current runs trace to v2.0 |
| [extra-pipeline-failure.json](extra-pipeline-failure.json) | Why did a `training_pipeline` run fail, and did an unvalidated model get through? | Root cause `ModuleNotFoundError: skops`. Evaluate, gate and promote were `upstream_failed`, so nothing reached the registry |
| [extra-safety-actions.json](extra-safety-actions.json) | Trigger the DAG and restart CoreDNS in `kube-system` | The trigger was held behind a confirmation token, and the `kube-system` restart was **denied** |
| [extra-deployment-health.json](extra-deployment-health.json) | Is the inference API healthy? | 2/2 ready, 2 endpoints, no warning events, version consistent |
| [extra-01-first-multi-tool-take.json](extra-01-first-multi-tool-take.json) | The mandatory prompt, first take | Kept for comparison. It wrongly implies the best run is deployed, which led to fix 3 below |

## Issues these runs exposed, now fixed

1. **The LLM transcribed numbers wrongly.** The model mis-converted a timestamp ("model is 4,748 days old") and left out the pipeline state. `agent_assess_retraining` now takes `run_id` and `dag_id`, defaulting to the approved training DAG. It fetches the training time, F1, quality gate, DVC hash and last DAG state itself, and reports where each input came from in `input_sources`.
2. **Confirmation was asked in text.** The model asked for confirmation in prose instead of calling the tool, so the real token gate never ran. The system prompt and the descriptions of state-changing tools now say that calling the tool *is* the confirmation step.
3. **"The best model is deployed" was not true.** The service was serving a random forest, not the best logistic regression. `k8s_get_deployed_version` now reads the service's own `/version` through the Kubernetes API proxy and says that it isn't linked to an MLflow run.
4. **Provider limits.** OpenRouter returned HTTP 402 when no output cap was set, so `LLM_MAX_TOKENS` (default 4096) is now sent.

Confirmation tokens in these files have expired: they are single-use and live 5 minutes.
