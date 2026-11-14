"""Train a model at image-build time so the serving image is self-contained.

Writes ``models/model.pkl`` and ``models/model_meta.json``. At runtime the service can instead
load the registry champion from MLflow (set ``MODEL_URI``), see ``inference/app.py``.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from training import make_dataset
from training.config import load_params
from training.train import build_model

DATASET_VERSION = "v2"


def main(out_dir: str = "models") -> None:
    params = load_params()
    tcfg = params["train"]
    df = make_dataset.build(DATASET_VERSION)
    x, y = df.drop(columns=["target"]), df["target"]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = build_model(tcfg["model_type"], tcfg[tcfg["model_type"]], tcfg["random_state"])
    model.fit(x_tr, y_tr)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(model)
    (out / "model.pkl").write_bytes(blob)
    meta = {
        "model_type": tcfg["model_type"],
        "dataset_version": DATASET_VERSION,
        "test_f1": round(float(f1_score(y_te, model.predict(x_te))), 4),
        "sha256": hashlib.sha256(blob).hexdigest()[:12],
        "feature_names": list(x.columns),
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
