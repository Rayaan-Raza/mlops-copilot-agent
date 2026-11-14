"""Inference service: probes, version and prediction against a freshly trained model."""

import importlib
import pickle

import pytest

pytest.importorskip("sklearn")

from fastapi.testclient import TestClient  # noqa: E402

from training import make_dataset  # noqa: E402
from training.models import build_model  # noqa: E402


@pytest.fixture()
def model_path(tmp_path):
    df = make_dataset.build("v2")
    x, y = df.drop(columns=["target"]), df["target"]
    model = build_model("logistic_regression", {"max_iter": 2000}, 42).fit(x, y)
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(model))
    return path, x.iloc[:3].to_dict(orient="records")


def _client(monkeypatch, path, **env):
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("APP_VERSION", "1.2.3")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import inference.app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_probes_version_and_predict(monkeypatch, model_path):
    path, records = model_path
    with _client(monkeypatch, path) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        assert client.get("/health/ready").status_code == 200
        assert client.get("/version").json()["app_version"] == "1.2.3"
        r = client.post("/predict", json={"records": records})
        assert r.status_code == 200
        body = r.json()
        assert len(body["predictions"]) == 3 and body["model_version"] == "1.2.3"
        bad = client.post("/predict", json={"records": [{"mean_radius": 1.0}]})
        assert bad.status_code == 422


def test_injected_startup_failure(monkeypatch, model_path):
    path, _ = model_path
    client = _client(monkeypatch, path, SIMULATE_STARTUP_FAILURE="true")
    with pytest.raises(RuntimeError, match="injected startup failure"), client:
        pass
