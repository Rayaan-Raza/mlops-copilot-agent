"""Unit tests for the training pipeline building blocks (no MLflow server needed)."""

import pytest

pytest.importorskip("sklearn")

from training import make_dataset  # noqa: E402
from training.provenance import dvc_file_info  # noqa: E402
from training.quality_gate import evaluate_gate  # noqa: E402
from training.validate import validate_frame  # noqa: E402

RULES = {"min_rows": 200, "max_null_fraction": 0.0, "min_minority_class_fraction": 0.2}
GATE = {"min_f1": 0.9, "min_recall": 0.9, "max_train_test_f1_gap": 0.08}


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_clean_dataset_versions_pass_validation(version):
    report = validate_frame(make_dataset.build(version), RULES)
    assert report.passed, report.errors


def test_dataset_versions_are_distinct_and_deterministic():
    v1, v2 = make_dataset.build("v1"), make_dataset.build("v2")
    assert len(v1) < len(v2)
    assert v2.shape[1] == v1.shape[1] + 2
    assert make_dataset.build("v1").equals(v1)


def test_corrupt_dataset_fails_validation_with_reasons():
    report = validate_frame(make_dataset.build("corrupt"), RULES)
    assert not report.passed
    assert "mean_smoothness" in report.missing_columns
    assert any("null fraction" in e for e in report.errors)


def test_quality_gate_passes_good_model():
    metrics = {"f1": 0.96, "recall": 0.97, "train_test_f1_gap": 0.02}
    assert evaluate_gate(metrics, GATE) == []


def test_quality_gate_blocks_overfit_and_weak_models():
    failures = evaluate_gate({"f1": 0.85, "recall": 0.95, "train_test_f1_gap": 0.15}, GATE)
    assert any("min_f1" in f for f in failures)
    assert any("overfitting" in f for f in failures)


def test_dvc_pointer_file_is_parsed(tmp_path):
    pointer = tmp_path / "dataset.csv.dvc"
    pointer.write_text(
        "outs:\n- md5: abc123\n  size: 42\n  hash: md5\n  path: dataset.csv\n", encoding="utf-8"
    )
    assert dvc_file_info(pointer) == {"md5": "abc123", "size": "42", "path": "dataset.csv"}
    assert dvc_file_info(tmp_path / "missing.dvc")["md5"] is None
