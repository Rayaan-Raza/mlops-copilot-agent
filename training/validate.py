"""Dataset validation (first stage of the training DAG)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from training.config import load_params, repo_path

# Columns every dataset version must provide (v2 adds engineered features on top).
REQUIRED_COLUMNS = (
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_concavity",
    "target",
)


class DataValidationError(Exception):
    """Raised when the dataset violates the validation contract."""


@dataclass
class ValidationReport:
    rows: int
    columns: int
    null_fraction: float
    minority_class_fraction: float
    missing_columns: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


def validate_frame(df: pd.DataFrame, rules: dict) -> ValidationReport:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    null_fraction = float(df.isna().mean().max()) if len(df) else 1.0
    if "target" in df.columns and len(df):
        minority = float(df["target"].value_counts(normalize=True).min())
    else:
        minority = 0.0

    errors: list[str] = []
    if missing:
        errors.append(f"missing required columns: {missing}")
    if len(df) < rules["min_rows"]:
        errors.append(f"row count {len(df)} < min_rows {rules['min_rows']}")
    if null_fraction > rules["max_null_fraction"]:
        errors.append(
            f"null fraction {null_fraction:.3f} > max_null_fraction {rules['max_null_fraction']}"
        )
    if minority < rules["min_minority_class_fraction"]:
        errors.append(
            f"minority class fraction {minority:.3f} < {rules['min_minority_class_fraction']}"
        )

    return ValidationReport(
        rows=len(df),
        columns=df.shape[1],
        null_fraction=round(null_fraction, 4),
        minority_class_fraction=round(minority, 4),
        missing_columns=missing,
        errors=errors,
    )


def run(raw_path: str | None = None) -> ValidationReport:
    params = load_params()
    path = repo_path(raw_path or params["data"]["raw_path"])
    df = pd.read_csv(path)
    report = validate_frame(df, params["validation"])
    print(json.dumps(asdict(report) | {"passed": report.passed}, indent=2))
    if not report.passed:
        raise DataValidationError("; ".join(report.errors))
    return report


if __name__ == "__main__":
    run()
