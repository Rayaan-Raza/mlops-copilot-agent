# MLOps Copilot Agent

[![CI](https://github.com/Rayaan-Raza/mlops-copilot-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Rayaan-Raza/mlops-copilot-agent/actions/workflows/ci.yml)
[![CD](https://github.com/Rayaan-Raza/mlops-copilot-agent/actions/workflows/cd.yml/badge.svg)](https://github.com/Rayaan-Raza/mlops-copilot-agent/actions/workflows/cd.yml)

An AI agent that answers natural-language questions about an ML system and operates it safely.
It uses four tool integrations, and every operational claim it makes is backed by a live tool result:

| Layer | Technology | What the agent can do |
|---|---|---|
| Data | **DVC** | list tracked data, version and commit, map an MLflow run's md5 to a version tag, diff versions, detect changes, reproduction steps, confirmed checkout |
| Experiments | **MLflow** | list runs, find the best run by a metric (with production-readiness caveats), compare runs and experiments, read the model registry |
| Orchestration | **Airflow 3** | list DAGs, check run status, inspect task instances, diagnose failed tasks from their logs, confirmed trigger of approved DAGs |
| Serving | **Kubernetes** | pods, deployments, services and endpoints, events, logs, unhealthy-pod detection, deployed version, one-shot diagnosis, confirmed restart and scale |

See [architecture.md](architecture.md) for the design and [docs/report.md](docs/report.md) for the technical report.

---

## 1. Prerequisites

| Tool | Notes |
|---|---|
| Python 3.10 or 3.11 | 3.12+ may lack wheels for some ML dependencies |
| Docker Desktop | runs MLflow and Airflow (docker compose) and builds images |
| Kubernetes | **Docker Desktop → Settings → Kubernetes → Enable** (simplest), or `kind` |
| git | DVC provenance reads git tags |
| An LLM API key | Anthropic, or any OpenAI-compatible provider (OpenAI, Groq, Gemini, Ollama) |

## 2. Install

```bash
git clone https://github.com/Rayaan-Raza/mlops-copilot-agent.git
cd mlops-copilot-agent
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -e ".[tools,llm,ml,dev]"
cp .env.example .env              # then edit .env (never commit it)
```

Key settings in `.env`:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER`, `LLM_MODEL`, `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` (+ `OPENAI_BASE_URL`) | LLM used for tool calling |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` |
| `AIRFLOW_BASE_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD`, `AIRFLOW_ALLOWED_DAGS` | Airflow 3 API and the DAGs the agent may trigger |
| `K8S_NAMESPACE`, `K8S_ALLOWED_NAMESPACES` | namespaces the agent may read and modify |
| `RETRAIN_FRESHNESS_DAYS`, `RETRAIN_MIN_F1` | retraining policy |

## 3. Launch each service

### 3.1 Data (DVC)

The dataset has two tagged versions: `data-v1.0` (398 rows) and `data-v2.0` (569 rows + 2 features).

```bash
dvc pull                                   # fetch data for the current version from the remote
git tag --list "data-v*"                   # data-v1.0, data-v2.0
dvc status                                 # verify the workspace matches the recorded hashes
```

To reproduce an old version:

```bash
git checkout data-v1.0 -- data/raw/dataset.csv.dvc
dvc checkout data/raw/dataset.csv.dvc
```

> The default remote `localremote` is a local folder (`../../dvc-remote` relative to `.dvc/`).
> Point it at S3, GCS or MinIO for a shared setup with `dvc remote modify localremote url s3://...`.

### 3.2 MLflow + Airflow 3

```bash
docker compose -f infra/docker-compose.yaml up -d --build
```

- MLflow UI: <http://localhost:5000>
- Airflow UI: <http://localhost:8080> (user and password come from `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD`, default `airflow`/`airflow`)

The repository is mounted into the Airflow containers, so the `training_pipeline` DAG trains on the DVC-tracked data in your workspace.

Seed MLflow with comparable runs, then trigger the DAG:

```bash
set MLFLOW_TRACKING_URI=http://localhost:5000     # bash: export MLFLOW_TRACKING_URI=...
python -m training.sweep                          # 8 runs: LR / RF / GB variants
```

Then trigger `training_pipeline` from the Airflow UI, or ask the agent to do it (it will ask for confirmation).

### 3.3 Inference service on Kubernetes

```bash
docker build -t ml-inference:1.0.0 --build-arg APP_VERSION=1.0.0 .
# kind only:  kind load docker-image ml-inference:1.0.0
kubectl apply -k kubernetes
kubectl -n mlops-agent rollout status deployment/ml-inference
kubectl -n mlops-agent port-forward svc/ml-inference 8000:80   # then GET http://localhost:8000/version
```

`kubernetes/rbac.yaml` creates a least-privilege `mlops-agent` ServiceAccount scoped to that namespace.

### 3.4 The agent

```bash
# CLI (asks y/n before any state-changing action)
mlops-agent "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."

# HTTP API
uvicorn api.main:app --port 8001
curl -X POST localhost:8001/chat -H "Content-Type: application/json" \
     -d '{"message": "Why is my prediction API unavailable?"}'
curl -X POST localhost:8001/confirm -H "Content-Type: application/json" \
     -d '{"confirmation_token": "<token from pending_confirmations>"}'
```

Every `/chat` response contains:
- `facts`: each fact cites the tool call it came from
- `uncertainties`: failed tools are always listed here
- `recommendations`
- `evidence`: tool, call id and status
- `pending_confirmations`

## 4. Tests

```bash
pytest -m "not integration"        # 59 tests, all external services mocked
ruff check . && ruff format --check .
# live checks against your current kube context:
RUN_LIVE_K8S_TESTS=1 pytest -m integration tests/integration
```

## 5. CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | every PR and push to `main` | ruff; pytest on 3.10/3.11 with coverage; training-pipeline smoke test (including corrupt-data rejection); Airflow DagBag integrity; kubeconform and manifest policy; compose validation; image builds and container smoke test |
| [`cd.yml`](.github/workflows/cd.yml) | push to `main`, `v*` tags | push `ml-inference` and `mlops-agent` images to GHCR; deploy to an ephemeral **kind** staging cluster; rollout and smoke test through the Service; live agent Kubernetes tests; GitHub release on tags |
| [`train.yml`](.github/workflows/train.yml) | manual or weekly | continuous training: validate → preprocess → train → evaluate → quality gate → promote, with MLflow store and metrics as artifacts |

## 6. Repository layout

```
agent/            orchestrator, LLM providers, registry, safety, logging, policy, CLI
agent/tools/      dvc_tools, mlflow_tools, airflow_tools, kubernetes_tools
api/              FastAPI interface (/chat, /confirm, /tools, /health)
dags/             Airflow 3 training_pipeline DAG
training/         dataset versions, validation, preprocessing, training, evaluation, gate, promotion
inference/        FastAPI model-serving app deployed to Kubernetes
kubernetes/       namespace, configmap, deployment, service, rbac, kustomization
infra/            docker-compose for MLflow + Airflow 3
tests/            unit tests with fakes + live integration tests
docs/             technical report and failure-injection runbook
```

## 7. Security

- No secrets in git: `.env` and kubeconfig files are ignored, and `.env.example` documents the variables.
- State-changing tools need a single-use, expiring confirmation token bound to the exact action.
- Kubernetes writes are limited to approved namespaces. This is enforced by the agent and by RBAC.
- DVC and git commands go through an allow-list, never use a shell, and validate revisions.
- Logs are JSON with request ids, and tokens, passwords and API keys are redacted.
