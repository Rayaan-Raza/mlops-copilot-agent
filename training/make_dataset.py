"""Materialise a versioned dataset at data/raw/dataset.csv.

The source is scikit-learn's Wisconsin breast-cancer dataset (no download needed), so every
version is exactly reproducible. Each version is then tracked with ``dvc add`` and tagged in git:

* ``v1``      - a stratified 70% sample (the "early" data drop)
* ``v2``      - the full dataset plus two engineered features (the "current" data)
* ``corrupt`` - v2 with injected nulls and a dropped column; used to demonstrate that the
  Airflow ``validate_dataset`` task blocks the pipeline (deliberate failure injection)

Usage::

    python -m training.make_dataset --version v1
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from training.config import load_params, repo_path

VERSIONS = ("v1", "v2", "corrupt")


def _base_frame() -> pd.DataFrame:
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def build(version: str, seed: int = 42) -> pd.DataFrame:
    if version not in VERSIONS:
        raise ValueError(f"unknown dataset version {version!r}; expected one of {VERSIONS}")
    df = _base_frame()
    if version == "v1":
        return (
            df.groupby("target", group_keys=False)
            .sample(frac=0.7, random_state=seed)
            .sort_index()
            .reset_index(drop=True)
        )

    df["radius_to_texture"] = df["mean_radius"] / df["mean_texture"]
    df["area_x_concavity"] = df["mean_area"] * df["mean_concavity"]
    if version == "v2":
        return df

    # corrupt: nulls + missing column -> must fail validation
    rng = np.random.default_rng(seed)
    corrupt = df.drop(columns=["mean_smoothness"])
    idx = rng.choice(len(corrupt), size=len(corrupt) // 10, replace=False)
    corrupt.loc[idx, "mean_radius"] = np.nan
    return corrupt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=VERSIONS, required=True)
    args = parser.parse_args()

    params = load_params()
    out = repo_path(params["data"]["raw_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build(args.version)
    df.to_csv(out, index=False)
    print(f"wrote dataset version={args.version} rows={len(df)} cols={df.shape[1]} -> {out}")


if __name__ == "__main__":
    main()
