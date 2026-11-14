"""Prediction API served on Kubernetes.

Endpoints: ``POST /predict``, ``GET /health/live``, ``GET /health/ready``, ``GET /version``.

Configuration (from the ConfigMap) is read from environment variables:

* ``APP_VERSION``        - reported by /version and used by the agent's version check
* ``MODEL_PATH``         - pickled model baked into the image (default)
* ``MODEL_URI``          - optional MLflow URI, e.g. ``models:/breast-cancer-classifier@champion``
* ``PREDICTION_THRESHOLD``
* ``SIMULATE_STARTUP_FAILURE`` - "true" crashes on start (failure-injection demo)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    stream=sys.stdout,
)
log = logging.getLogger("inference")

STATE: dict[str, Any] = {"model": None, "meta": {}, "started": time.time()}


def _load_model() -> None:
    if os.getenv("SIMULATE_STARTUP_FAILURE", "false").lower() == "true":
        log.error("SIMULATE_STARTUP_FAILURE=true -> refusing to start (injected failure)")
        raise RuntimeError("injected startup failure: model store unreachable")

    uri = os.getenv("MODEL_URI")
    if uri:
        import mlflow.sklearn  # optional dependency

        log.info("loading model from MLflow uri=%s", uri)
        STATE["model"] = mlflow.sklearn.load_model(uri)
        STATE["meta"] = {"source": uri}
        return

    path = Path(os.getenv("MODEL_PATH", "/app/models/model.pkl"))
    log.info("loading model from path=%s", path)
    with open(path, "rb") as fh:
        STATE["model"] = pickle.load(fh)  # noqa: S301 - artefact baked into our own image
    meta_path = path.with_name("model_meta.json")
    if meta_path.exists():
        STATE["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    STATE["meta"]["source"] = str(path)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    log.info("model loaded; service ready version=%s", os.getenv("APP_VERSION", "unknown"))
    yield


app = FastAPI(title="ML Inference Service", lifespan=lifespan)


class PredictRequest(BaseModel):
    records: list[dict[str, float]] = Field(..., min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
    model_version: str


@app.get("/health/live")
def live() -> dict:
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> dict:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    return {
        "app_version": os.getenv("APP_VERSION", "unknown"),
        "model": STATE["meta"],
        "uptime_seconds": round(time.time() - STATE["started"], 1),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    features = getattr(model, "feature_names_in_", None)
    df = pd.DataFrame(req.records)
    if features is not None:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"missing features: {missing[:5]}")
        df = df[list(features)]
    threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
    proba = model.predict_proba(df)[:, 1]
    return PredictResponse(
        predictions=[int(p >= threshold) for p in proba],
        probabilities=[round(float(p), 4) for p in proba],
        model_version=os.getenv("APP_VERSION", "unknown"),
    )
