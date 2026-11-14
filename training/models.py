"""Model factory (kept free of MLflow so the serving image can import it)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(model_type: str, hp: dict[str, Any], seed: int):
    if model_type == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **hp))
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **hp)
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed, **hp)
    raise ValueError(f"unknown model_type {model_type!r}")
