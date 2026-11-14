"""Retraining policy.

The agent may only *recommend* retraining, and only when backed by evidence: an explicit
freshness policy breach, a failed quality gate, below-threshold metrics, or user-supplied
degradation results. It never triggers retraining on its own.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class RetrainingAssessment(BaseModel):
    recommend_retraining: bool
    reasons: list[str] = Field(default_factory=list)
    checks: dict[str, str] = Field(default_factory=dict)
    note: str = "Recommendation only. Retraining is never started automatically."


def assess_retraining(
    *,
    model_trained_at_ms: int | None,
    latest_f1: float | None,
    last_quality_gate: str | None,
    last_pipeline_state: str | None,
    data_changed_since_training: bool | None,
    freshness_days: int,
    min_f1: float,
    reported_degradation: str | None = None,
    now: datetime | None = None,
) -> RetrainingAssessment:
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []
    checks: dict[str, str] = {}

    if model_trained_at_ms is None:
        checks["freshness"] = "unknown (no training timestamp available)"
    else:
        age_days = (now.timestamp() * 1000 - model_trained_at_ms) / 86_400_000
        checks["freshness"] = f"model age {age_days:.1f} days (policy: {freshness_days} days)"
        if age_days > freshness_days:
            reasons.append(
                f"model is {age_days:.0f} days old, exceeding the {freshness_days}-day freshness policy"
            )

    if latest_f1 is None:
        checks["f1"] = "unknown"
    else:
        checks["f1"] = f"{latest_f1:.4f} (policy minimum {min_f1})"
        if latest_f1 < min_f1:
            reasons.append(f"F1 {latest_f1:.4f} is below the {min_f1} minimum")

    checks["quality_gate"] = last_quality_gate or "unknown"
    if last_quality_gate == "failed":
        reasons.append("the most recent quality gate failed")

    checks["last_pipeline_run"] = last_pipeline_state or "unknown"

    if data_changed_since_training is None:
        checks["data_drift"] = "unknown"
    else:
        checks["data_drift"] = (
            "tracked data changed" if data_changed_since_training else "unchanged"
        )
        if data_changed_since_training:
            reasons.append("the DVC-tracked dataset has changed since the model was trained")

    if reported_degradation:
        reasons.append(f"user-reported degradation: {reported_degradation}")

    return RetrainingAssessment(recommend_retraining=bool(reasons), reasons=reasons, checks=checks)
