"""Split the raw dataset into train/test sets (second stage of the training DAG)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from training.config import load_params, repo_path


def run() -> tuple[str, str]:
    params = load_params()
    dcfg = params["data"]
    df = pd.read_csv(repo_path(dcfg["raw_path"]))
    train_df, test_df = train_test_split(
        df,
        test_size=dcfg["test_size"],
        random_state=dcfg["random_state"],
        stratify=df[dcfg["target"]],
    )
    out_dir = repo_path(dcfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = out_dir / "train.csv", out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"train={len(train_df)} test={len(test_df)} -> {out_dir}")
    return str(train_path), str(test_path)


if __name__ == "__main__":
    run()
