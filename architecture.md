# Architecture

```
            User / Engineer (CLI or HTTP)
                        │  natural-language request
                        ▼
        ┌───────────────────────────────────┐
        │ FastAPI  api/main.py              │  validation, X-Request-ID, /confirm
        └───────────────┬───────────────────┘
                        ▼
        ┌───────────────────────────────────┐
        │ Agent  agent/orchestrator.py      │
        │  LLM tool-calling loop (≤ N steps)│──► LLM provider (Anthropic / OpenAI-compatible)
        │  ToolRegistry: schema validation, │
        │   permissions, timeouts, tracing  │
        │  Evidence guard + retrain policy  │
        └──┬─────────┬──────────┬────────┬──┘
           ▼         ▼          ▼        ▼
         DVC      MLflow     Airflow 3  Kubernetes
     git+dvc CLI MlflowClient /api/v2   python client
    (allow-list)            (JWT)      (RBAC-scoped)
           └─────────┴──────────┴────────┘
                        ▼
     Evidence-based response: facts [evidence: call_id] · uncertainties · recommendations
```

## Components

| Component | File | Responsibility |
|---|---|---|
| API | `api/main.py` | `/chat`, `/confirm`, `/tools`, `/health`; request ids; runs the agent in a threadpool |
| Orchestrator | `agent/orchestrator.py` | prompt, tool loop, final-answer parsing, anti-hallucination guards |
| LLM layer | `agent/llm.py` | provider-neutral messages ↔ Anthropic / OpenAI tool-calling formats; scripted provider for tests |
| Registry | `agent/registry.py` | tool specs (JSON Schema from Pydantic), validation, timeouts, error→status mapping, structured logs |
| Safety | `agent/safety.py` | confirmation tokens, namespace and DAG allow-lists |
| Policy | `agent/policy.py` | evidence-based retraining recommendation (never triggers) |
| Tools | `agent/tools/*.py` | 8 DVC, 7 MLflow, 7 Airflow and 10 Kubernetes tools, plus 1 policy tool |

## Data provenance chain

Provenance is the key design decision. It lets the agent answer *"which data trained this model?"* from records instead of guessing.

1. `data/raw/dataset.csv` is tracked with `dvc add`. Each version is a git tag (`data-v1.0`, `data-v2.0`), and the `.dvc` pointer holds the content md5.
2. `training/train.py` tags every MLflow run with `dvc.data_md5`, `dvc.dataset_version`, `git.commit` and `airflow.dag_run_id`.
3. `promote.py` copies those tags onto the registered model version.
4. The agent reads the tags (`mlflow_get_run_details`), then calls `dvc_resolve_dataset_version(md5)`. That call checks the md5 against the pointer file at every version tag, so the version is verified rather than taken from the label.

## Tool result envelope

Every tool returns a `ToolResult`:

```json
{"tool": "...", "call_id": "...", "status": "ok|error|timeout|unavailable|invalid_request|denied|confirmation_required",
 "data": {...}, "error": "...", "duration_ms": 12.3, "confirmation_token": "..."}
```

Tools never raise into the LLM loop. Exceptions are mapped to statuses:
- `ServiceUnavailableError` and connection errors → `unavailable`
- `PermissionDeniedError` and RBAC 403 → `denied`
- Pydantic validation errors → `invalid_request`
- tool timeout → `timeout`

## Anti-hallucination guards (deterministic, outside the LLM)

1. **Cited facts only.** Every line under `## Facts` must end with `[evidence: <call_id>]`, and every cited call must have returned `ok`. Any other "fact" is moved to `uncertainties` as *"Unverified claim removed"*.
2. **Failures are always surfaced.** Each failed, timed-out, denied or unavailable tool call is added to `uncertainties`, even if the LLM didn't mention it.
3. **Warning banner.** If any evidence is missing, the free-text answer is prefixed with a warning that only the Facts list is verified.
4. **No-tool answers are flagged.** An answer produced without any tool call says that nothing is backed by live evidence.

## Permissions

| Permission | Tools | Control |
|---|---|---|
| read | all list/get/find/compare/diagnose tools | namespace allow-list (K8s), DVC command allow-list |
| write | `airflow_trigger_dag`, `k8s_restart_deployment`, `k8s_scale_deployment`, `dvc_checkout_dataset_version` | precheck (approved DAG or namespace) → single-use token, expires after 5 min, bound to tool+args → human `POST /confirm` |

Kubernetes RBAC (`kubernetes/rbac.yaml`) enforces the same scope at the cluster level. Even a compromised agent process can only read, plus patch deployments, in `mlops-agent`.

## Training DAG

```
validate_dataset ─► preprocess_data ─► train_model ─► evaluate_and_log_mlflow ─► apply_quality_gate ─► promote_model
   (retries=0,                          retries=2, exponential backoff              (retries=0,          (all_success)
    AirflowFailException)                                                            AirflowFailException)
```

A validation or gate failure leaves `promote_model` in `upstream_failed`, so nothing reaches the registry alias `champion`.

## Kubernetes deployment

- Deployment `ml-inference`: 2 replicas, RollingUpdate (`maxUnavailable: 0`), readiness probe `/health/ready` (only passes once the model is loaded), liveness probe `/health/live`, requests of 100m/256Mi and limits of 500m/512Mi, non-root, read-only root filesystem.
- Service `ml-inference`: ClusterIP on port 80 → 8000.
- ConfigMap `ml-inference-config`: `APP_VERSION`, `MODEL_PATH`, threshold, log level, and the `SIMULATE_STARTUP_FAILURE` failure-injection switch.

## Why these choices

- **Direct tool calling instead of a framework.** It keeps the loop small and auditable, and makes it easy to put safety checks *outside* the LLM. Providers are swappable.
- **Pydantic schemas** give one source of truth for LLM tool specs, input validation and API models.
- **Subprocess with an allow-list for DVC.** The DVC Python API covers little of what's needed (`status`, `diff`, git tags). An allow-list with no shell, revision regex and timeouts keeps it safe.
- **MlflowClient**, as the brief requires. It also gives model registry aliases.
- **Airflow 3 REST `/api/v2` with JWT.** This is the supported external interface in Airflow 3. The token is refreshed once on a 401.
