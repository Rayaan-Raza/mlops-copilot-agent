"""Agent settings loaded from environment variables (``.env`` supported)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

try:  # optional at import time so tests don't need a .env
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _csv(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


class Settings(BaseModel):
    # LLM
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-5-5"))

    # MLflow
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow_experiment: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "breast-cancer-classifier")
    )
    mlflow_registered_model: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_REGISTERED_MODEL", "breast-cancer-classifier")
    )

    # Airflow
    airflow_base_url: str = Field(
        default_factory=lambda: os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")
    )
    airflow_username: str = Field(default_factory=lambda: os.getenv("AIRFLOW_USERNAME", ""))
    airflow_password: str = Field(default_factory=lambda: os.getenv("AIRFLOW_PASSWORD", ""))
    airflow_allowed_dags: list[str] = Field(
        default_factory=lambda: _csv("AIRFLOW_ALLOWED_DAGS", "training_pipeline")
    )

    # Kubernetes
    k8s_namespace: str = Field(default_factory=lambda: os.getenv("K8S_NAMESPACE", "mlops-agent"))
    k8s_allowed_namespaces: list[str] = Field(
        default_factory=lambda: _csv("K8S_ALLOWED_NAMESPACES", "mlops-agent")
    )
    k8s_app_label: str = Field(
        default_factory=lambda: os.getenv("K8S_APP_LABEL", "app=ml-inference")
    )

    # DVC / repo
    repo_root: Path = Field(default_factory=lambda: Path(os.getenv("REPO_ROOT", ".")).resolve())

    # Agent behaviour
    tool_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("AGENT_TOOL_TIMEOUT_SECONDS", "30"))
    )
    max_steps: int = Field(default_factory=lambda: int(os.getenv("AGENT_MAX_STEPS", "10")))
    # Per tool result sent back to the LLM. Lower it for small context / rate budgets (Groq free).
    tool_result_max_chars: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_TOOL_RESULT_MAX_CHARS", "6000"))
    )
    confirmation_ttl_seconds: int = 300
    retrain_freshness_days: int = Field(
        default_factory=lambda: int(os.getenv("RETRAIN_FRESHNESS_DAYS", "30"))
    )
    retrain_min_f1: float = Field(
        default_factory=lambda: float(os.getenv("RETRAIN_MIN_F1", "0.90"))
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
