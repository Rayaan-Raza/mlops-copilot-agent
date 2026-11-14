"""Populate MLflow with a set of comparable runs (used to seed the demo experiments).

Runs preprocess once, then train -> evaluate -> quality gate for each configuration. Gate
failures are recorded as tags rather than raised, so weak candidates still appear in MLflow.

    python -m training.sweep
"""

from __future__ import annotations

from training import evaluate, preprocess, quality_gate, train, validate

CONFIGS: list[tuple[str, dict]] = [
    ("logistic_regression", {"C": 0.01}),
    ("logistic_regression", {"C": 1.0}),
    ("logistic_regression", {"C": 10.0}),
    ("random_forest", {"n_estimators": 50, "max_depth": 3}),
    ("random_forest", {"n_estimators": 200, "max_depth": 8}),
    ("random_forest", {"n_estimators": 300, "max_depth": None}),
    ("gradient_boosting", {"n_estimators": 100, "learning_rate": 0.05}),
    ("gradient_boosting", {"n_estimators": 300, "learning_rate": 0.2, "max_depth": 5}),
]


def main() -> None:
    validate.run()
    preprocess.run()
    for model_type, hp in CONFIGS:
        train.run(model_type, hp)
        evaluate.run()
        quality_gate.run(raise_on_fail=False)


if __name__ == "__main__":
    main()
