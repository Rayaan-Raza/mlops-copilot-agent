# MLOps Copilot Agent

A production-style AI agent that answers natural-language questions about an ML system and
operates it safely through four tool integrations:

| Layer | Technology | What the agent does |
|---|---|---|
| Data | DVC | dataset versions, provenance, diffs, reproduction steps |
| Experiments | MLflow | runs, metrics, best-run selection, model registry |
| Orchestration | Apache Airflow 3 | DAG status, triggering, failed-task diagnostics |
| Serving | Kubernetes | pods, services, health, logs, deployed version |

> Full setup instructions are added as each component lands (see `docs/`).

## Quick start (development)

```bash
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[tools,llm,dev]"
cp .env.example .env        # then fill in values
pytest
```
