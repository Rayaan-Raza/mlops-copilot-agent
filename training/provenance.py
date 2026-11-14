"""Collect data provenance (DVC hash, dataset version tag, git commit) for MLflow tagging.

These tags are what let the agent answer "which dataset version trained experiment X?" from
recorded evidence instead of guessing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from training.config import REPO_ROOT


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def dvc_file_info(dvc_file: Path) -> dict[str, str | None]:
    """Read md5/size/path from a ``*.dvc`` pointer file."""
    if not dvc_file.exists():
        return {"md5": None, "size": None, "path": None}
    meta = yaml.safe_load(dvc_file.read_text(encoding="utf-8")) or {}
    out = (meta.get("outs") or [{}])[0]
    return {
        "md5": out.get("md5"),
        "size": str(out.get("size")) if out.get("size") is not None else None,
        "path": out.get("path"),
    }


def collect(raw_path: str = "data/raw/dataset.csv") -> dict[str, str]:
    info = dvc_file_info(REPO_ROOT / f"{raw_path}.dvc")
    tags = {
        "dvc.data_path": raw_path,
        "dvc.data_md5": info["md5"],
        "dvc.data_size": info["size"],
        "dvc.dataset_version": _git("describe", "--tags", "--abbrev=0", "--match", "data-v*"),
        "git.commit": _git("rev-parse", "HEAD"),
        "git.branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git.dirty": "true" if _git("status", "--porcelain", "--", raw_path + ".dvc") else "false",
    }
    return {k: v for k, v in tags.items() if v is not None}
