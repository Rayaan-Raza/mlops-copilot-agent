"""Shared helpers for loading params.yaml and resolving repository paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(os.getenv("REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    params_path = Path(path) if path else REPO_ROOT / "params.yaml"
    with open(params_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def repo_path(relative: str | Path) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p
