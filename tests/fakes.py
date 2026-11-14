"""Lightweight fakes for external services used across the test-suite."""

from __future__ import annotations

from types import SimpleNamespace

NOW_MS = 1_794_000_000_000  # fixed "now" (~Nov 2026)
DAY_MS = 86_400_000


def fake_run(
    run_id: str,
    f1: float,
    *,
    name: str,
    md5: str = "b" * 32,
    version: str = "data-v2.0",
    gate: str = "passed",
    gap: float = 0.02,
    status: str = "FINISHED",
    age_days: float = 2,
):
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id,
            experiment_id="1",
            status=status,
            start_time=NOW_MS - int(age_days * DAY_MS),
            end_time=NOW_MS - int(age_days * DAY_MS) + 5000,
            run_name=name,
        ),
        data=SimpleNamespace(
            metrics={
                "f1": f1,
                "accuracy": f1 - 0.01,
                "recall": f1 + 0.005,
                "precision": f1 - 0.005,
                "roc_auc": 0.99,
                "train_f1": f1 + gap,
                "train_test_f1_gap": gap,
            },
            params={"model_type": name.split("-")[0]},
            tags={
                "mlflow.runName": name,
                "dvc.data_md5": md5,
                "dvc.dataset_version": version,
                "git.commit": "c0ffee" * 6 + "abcd",
                "quality_gate": gate,
                "model_type": name.split("-")[0],
            },
        ),
    )


RUNS = [
    fake_run("1" * 32, 0.951, name="logistic_regression-data-v2.0", age_days=1),
    fake_run("2" * 32, 0.982, name="random_forest-data-v2.0", age_days=2),
    fake_run(
        "3" * 32, 0.990, name="gradient_boosting-data-v2.0", gate="failed", gap=0.10, age_days=3
    ),
    fake_run(
        "4" * 32,
        0.940,
        name="random_forest-data-v1.0",
        md5="a" * 32,
        version="data-v1.0",
        age_days=10,
    ),
    fake_run("5" * 32, 0.0, name="crashed-run", status="FAILED", age_days=11),
]


class FakeMlflowClient:
    def __init__(self, runs=None, fail: Exception | None = None):
        self.runs = list(RUNS if runs is None else runs)
        self.fail = fail

    def _check(self):
        if self.fail:
            raise self.fail

    def get_experiment_by_name(self, name):
        self._check()
        if name == "missing":
            return None
        return SimpleNamespace(
            experiment_id="1" if name != "exp-b" else "2", name=name, lifecycle_stage="active"
        )

    def search_experiments(self):
        self._check()
        return [
            SimpleNamespace(
                experiment_id="1",
                name="breast-cancer-classifier",
                lifecycle_stage="active",
                creation_time=NOW_MS,
            )
        ]

    def search_runs(self, experiment_ids, filter_string="", max_results=100, order_by=None):
        self._check()
        runs = sorted(self.runs, key=lambda r: r.info.start_time, reverse=True)
        if experiment_ids == ["2"]:
            runs = runs[3:]
        return runs[:max_results]

    def get_run(self, run_id):
        self._check()
        for r in self.runs:
            if r.info.run_id == run_id:
                return r
        raise RuntimeError(f"RESOURCE_DOES_NOT_EXIST: run {run_id} not found")

    def list_artifacts(self, run_id):
        return [SimpleNamespace(path="model"), SimpleNamespace(path="evaluation")]

    def get_registered_model(self, name):
        self._check()
        return SimpleNamespace(name=name, aliases={"champion": "2"})

    def search_model_versions(self, filter_string):
        return [
            SimpleNamespace(
                version="1",
                run_id="4" * 32,
                status="READY",
                creation_timestamp=NOW_MS - 10 * DAY_MS,
                tags={"dvc.dataset_version": "data-v1.0"},
            ),
            SimpleNamespace(
                version="2",
                run_id="2" * 32,
                status="READY",
                creation_timestamp=NOW_MS - 2 * DAY_MS,
                tags={"dvc.dataset_version": "data-v2.0"},
            ),
        ]
