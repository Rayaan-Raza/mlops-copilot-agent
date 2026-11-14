"""DVC data-management tools.

All commands go through :class:`DVCClient`, which only runs allow-listed ``dvc``/``git``
sub-commands, never uses a shell, validates revisions against option injection and applies a
timeout. Version provenance is read from ``*.dvc`` pointer files at a given git revision.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from agent.config import Settings
from agent.errors import (
    InvalidRequestError,
    PermissionDeniedError,
    ServiceUnavailableError,
    ToolError,
)
from agent.registry import Tool, ToolRegistry
from agent.schemas import Permission

# (program, subcommand...) prefixes the agent may run. Anything else is rejected.
ALLOWED_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("dvc", "status"),
    ("dvc", "diff"),
    ("dvc", "data", "status"),
    ("dvc", "checkout"),
    ("dvc", "pull"),
    ("dvc", "remote", "list"),
    ("git", "log"),
    ("git", "show"),
    ("git", "tag"),
    ("git", "describe"),
    ("git", "rev-parse"),
    ("git", "checkout"),
)
WRITE_COMMANDS = {("dvc", "checkout"), ("dvc", "pull"), ("git", "checkout")}
REV_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,99}$")
DEFAULT_DATASET = "data/raw/dataset.csv"
VERSION_TAG_GLOB = "data-v*"


def _validate_rev(rev: str) -> str:
    if not REV_RE.match(rev) or ".." in rev:
        raise ValueError(f"invalid git revision {rev!r}")
    return rev


def _validate_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts or path.startswith("-"):
        raise ValueError(f"invalid dataset path {path!r}")
    return p.as_posix()


SKIP_DIRS = {".dvc", ".git", ".venv", "venv", "node_modules", "__pycache__", "dvc-remote"}


def resolve_executable(name: str) -> str:
    """Find ``dvc``/``git`` on PATH, or next to the running interpreter (an inactive venv)."""
    found = shutil.which(name)
    if found:
        return found
    for candidate in (name, f"{name}.exe"):
        local = Path(sys.executable).parent / candidate
        if local.is_file():
            return str(local)
    return name


def find_dvc_pointers(root: Path) -> list[Path]:
    """All ``*.dvc`` pointer *files* under root, skipping VCS/env/cache directories."""
    pointers = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        pointers += [Path(dirpath) / f for f in sorted(filenames) if f.endswith(".dvc")]
    return pointers


class DVCClient:
    def __init__(
        self,
        repo_root: Path,
        timeout: float = 60.0,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.timeout = timeout
        self._runner = runner

    @staticmethod
    def is_allowed(cmd: list[str]) -> bool:
        return any(tuple(cmd[: len(p)]) == p for p in ALLOWED_COMMANDS)

    def run(self, cmd: list[str], *, allow_write: bool = False, check: bool = True) -> str:
        if not self.is_allowed(cmd):
            raise PermissionDeniedError(f"command not allow-listed: {' '.join(cmd[:3])}")
        if not allow_write and any(tuple(cmd[: len(p)]) == p for p in WRITE_COMMANDS):
            raise PermissionDeniedError(
                f"{' '.join(cmd[:2])} changes the workspace; confirmation needed"
            )
        try:
            proc = self._runner(
                [resolve_executable(cmd[0]), *cmd[1:]],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise ServiceUnavailableError(f"{cmd[0]} executable not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"{' '.join(cmd[:2])} timed out after {self.timeout}s") from exc
        if check and proc.returncode != 0:
            raise ToolError(
                f"{' '.join(cmd[:3])} failed: {(proc.stderr or proc.stdout).strip()[:500]}"
            )
        return proc.stdout

    # --------------------------------------------------------------- helpers
    def pointer_at(self, path: str, rev: str | None = None) -> dict[str, Any]:
        """Return the parsed ``<path>.dvc`` pointer at a revision (working tree if rev is None)."""
        pointer = f"{path}.dvc"
        if rev is None:
            file = self.repo_root / pointer
            if not file.exists():
                raise InvalidRequestError(f"{pointer} not found: dataset is not tracked by DVC")
            text = file.read_text(encoding="utf-8")
        else:
            try:
                text = self.run(["git", "show", f"{rev}:{pointer}"])
            except ToolError as exc:
                raise InvalidRequestError(f"{pointer} does not exist at revision {rev}") from exc
        meta = yaml.safe_load(text) or {}
        out = (meta.get("outs") or [{}])[0]
        return {"md5": out.get("md5"), "size": out.get("size"), "nfiles": out.get("nfiles")}

    def version_tags(self) -> list[dict[str, str]]:
        raw = self.run(
            [
                "git",
                "tag",
                "--list",
                VERSION_TAG_GLOB,
                "--sort=creatordate",
                "--format=%(refname:short)|%(objectname:short)|%(creatordate:iso-strict)|%(subject)",
            ]
        )
        tags = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            name, commit, date, subject = (line.split("|", 3) + ["", "", ""])[:4]
            tags.append({"tag": name, "commit": commit, "date": date, "message": subject})
        return tags


# --------------------------------------------------------------------------- input models
class DatasetArgs(BaseModel):
    path: str = Field(DEFAULT_DATASET, description="DVC-tracked dataset path relative to repo root")

    @field_validator("path")
    @classmethod
    def _v_path(cls, v: str) -> str:
        return _validate_path(v)


class DatasetAtRevArgs(DatasetArgs):
    rev: str | None = Field(
        None, description="git revision / tag (e.g. data-v1.0). Omit for the current workspace."
    )

    @field_validator("rev")
    @classmethod
    def _v_rev(cls, v: str | None) -> str | None:
        return _validate_rev(v) if v else v


class CompareArgs(DatasetArgs):
    rev_a: str = Field(..., description="older revision/tag, e.g. data-v1.0")
    rev_b: str = Field("HEAD", description="newer revision/tag, default HEAD")

    @field_validator("rev_a", "rev_b")
    @classmethod
    def _v_revs(cls, v: str) -> str:
        return _validate_rev(v)


class ResolveArgs(DatasetArgs):
    md5: str = Field(..., pattern=r"^[0-9a-f]{32}$", description="DVC md5 recorded in MLflow tags")


class VersionArgs(DatasetArgs):
    version: str = Field(..., description="dataset version tag, e.g. data-v1.0")

    @field_validator("version")
    @classmethod
    def _v_version(cls, v: str) -> str:
        return _validate_rev(v)


# --------------------------------------------------------------------------- tools
def build_tools(client: DVCClient) -> list[Tool]:
    def list_tracked_datasets(_: BaseModel) -> dict:
        root = client.repo_root
        tracked = []
        for f in find_dvc_pointers(root):
            meta = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            for out in meta.get("outs", []):
                rel = (f.parent.relative_to(root) / out.get("path", "")).as_posix()
                tracked.append(
                    {
                        "path": rel,
                        "md5": out.get("md5"),
                        "size": out.get("size"),
                        "source": "dvc add",
                    }
                )
        lock = root / "dvc.lock"
        if lock.exists():
            stages = (yaml.safe_load(lock.read_text(encoding="utf-8")) or {}).get("stages", {})
            for stage, body in stages.items():
                for out in body.get("outs", []):
                    tracked.append(
                        {
                            "path": out["path"],
                            "md5": out.get("md5"),
                            "source": f"pipeline stage {stage}",
                        }
                    )
        return {"repo_root": str(root), "tracked": tracked, "count": len(tracked)}

    def get_dataset_version(a: DatasetAtRevArgs) -> dict:
        rev = a.rev or "HEAD"
        pointer = client.pointer_at(a.path, a.rev)
        log_line = client.run(
            ["git", "log", "-1", "--format=%H|%cI|%s", rev, "--", f"{a.path}.dvc"]
        ).strip()
        commit, date, subject = (
            (log_line.split("|", 2) + ["", "", ""])[:3] if log_line else ("", "", "")
        )
        tag = (
            client.run(
                ["git", "describe", "--tags", "--abbrev=0", "--match", VERSION_TAG_GLOB, rev],
                check=False,
            ).strip()
            or None
        )
        exact = (
            client.run(
                [
                    "git",
                    "describe",
                    "--tags",
                    "--exact-match",
                    "--match",
                    VERSION_TAG_GLOB,
                    commit or rev,
                ],
                check=False,
            ).strip()
            or None
        )
        return {
            "path": a.path,
            "revision": rev,
            "md5": pointer["md5"],
            "size_bytes": pointer["size"],
            "dataset_version": exact or tag,
            "version_tag_is_exact": bool(exact),
            "pointer_last_changed_commit": commit or None,
            "pointer_last_changed_at": date or None,
            "commit_message": subject or None,
        }

    def list_dataset_versions(a: DatasetArgs) -> dict:
        versions = []
        for t in client.version_tags():
            try:
                md5 = client.pointer_at(a.path, t["tag"])["md5"]
            except InvalidRequestError:
                md5 = None
            versions.append({**t, "md5": md5})
        return {"path": a.path, "versions": versions}

    def resolve_dataset_version(a: ResolveArgs) -> dict:
        matches = []
        for t in client.version_tags():
            try:
                if client.pointer_at(a.path, t["tag"])["md5"] == a.md5:
                    matches.append(t)
            except InvalidRequestError:
                continue
        current = client.pointer_at(a.path)["md5"]
        return {
            "md5": a.md5,
            "matching_version_tags": matches,
            "matches_current_workspace": current == a.md5,
            "resolved": bool(matches) or current == a.md5,
        }

    def compare_dataset_versions(a: CompareArgs) -> dict:
        pa, pb = client.pointer_at(a.path, a.rev_a), client.pointer_at(a.path, a.rev_b)
        diff_raw = client.run(["dvc", "diff", "--json", a.rev_a, a.rev_b], check=False)
        try:
            diff = json.loads(diff_raw) if diff_raw.strip() else {}
        except json.JSONDecodeError:
            diff = {"raw": diff_raw[:2000]}
        return {
            "path": a.path,
            a.rev_a: pa,
            a.rev_b: pb,
            "content_changed": pa["md5"] != pb["md5"],
            "size_delta_bytes": (pb["size"] or 0) - (pa["size"] or 0),
            "dvc_diff": diff,
        }

    def check_data_status(_: BaseModel) -> dict:
        raw = client.run(["dvc", "status", "--json"], check=False)
        try:
            status = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            status = {"raw": raw[:2000]}
        # Separate versioned *data* (``*.dvc`` pointers) from pipeline *stages* (dvc.yaml), so a
        # pipeline that was never run/locked is not mistaken for drift in the dataset itself.
        data_changes = {k: v for k, v in status.items() if k.endswith(".dvc")}
        stage_changes = {k: v for k, v in status.items() if not k.endswith(".dvc")}
        return {
            "workspace_in_sync": status == {},
            "tracked_data_changed": bool(data_changes),
            "data_changes": data_changes,
            "pipeline_stages_out_of_date": sorted(stage_changes),
            "pipeline_details": stage_changes,
            "interpretation": (
                "tracked datasets match their recorded DVC hashes"
                if not data_changes
                else "a DVC-tracked dataset differs from its recorded hash (unversioned change)"
            )
            + (
                "; pipeline stages are out of date or not yet recorded in dvc.lock "
                "(this alone is not a data change)"
                if stage_changes
                else ""
            ),
        }

    def reproduction_steps(a: VersionArgs) -> dict:
        pointer = client.pointer_at(a.path, a.version)
        return {
            "version": a.version,
            "expected_md5": pointer["md5"],
            "steps": [
                f"git checkout {a.version} -- {a.path}.dvc   # restore the pointer for that version",
                f"dvc pull {a.path}.dvc                    # fetch content from the DVC remote (if not cached)",
                f"dvc checkout {a.path}.dvc                # materialise the file in the workspace",
                f"dvc status {a.path}.dvc                  # verify: should report no changes",
                "python -m training.validate               # optional: re-run validation on that version",
            ],
            "note": "No commands were executed; these are instructions. "
            "Use dvc_checkout_dataset_version to run them with confirmation.",
        }

    def checkout_dataset_version(a: VersionArgs) -> dict:
        client.run(["git", "checkout", a.version, "--", f"{a.path}.dvc"], allow_write=True)
        client.run(["dvc", "checkout", f"{a.path}.dvc"], allow_write=True)
        pointer = client.pointer_at(a.path)
        return {"checked_out": a.version, "path": a.path, "md5": pointer["md5"]}

    return [
        Tool(
            "dvc_list_tracked_datasets",
            "List all datasets and pipeline outputs tracked by DVC with their md5 hashes.",
            list_tracked_datasets,
            module="dvc",
        ),
        Tool(
            "dvc_get_dataset_version",
            "Get the DVC md5, dataset version tag and git commit of a dataset at a revision (default: current).",
            get_dataset_version,
            DatasetAtRevArgs,
            module="dvc",
        ),
        Tool(
            "dvc_list_dataset_versions",
            "List every tagged dataset version (data-v*) with commit, date and md5.",
            list_dataset_versions,
            DatasetArgs,
            module="dvc",
        ),
        Tool(
            "dvc_resolve_dataset_version",
            "Map a DVC md5 (e.g. from an MLflow run's dvc.data_md5 tag) back to its dataset version tag.",
            resolve_dataset_version,
            ResolveArgs,
            module="dvc",
        ),
        Tool(
            "dvc_compare_dataset_versions",
            "Compare a dataset between two revisions/tags (hash, size, dvc diff).",
            compare_dataset_versions,
            CompareArgs,
            module="dvc",
        ),
        Tool(
            "dvc_check_data_status",
            "Detect whether DVC-tracked data or pipeline outputs changed versus their recorded hashes (dvc status).",
            check_data_status,
            module="dvc",
        ),
        Tool(
            "dvc_get_reproduction_steps",
            "Explain the exact commands to reproduce a previous dataset version (does not execute them).",
            reproduction_steps,
            VersionArgs,
            module="dvc",
        ),
        Tool(
            "dvc_checkout_dataset_version",
            "Check out a previous dataset version into the workspace (git checkout pointer + dvc checkout).",
            checkout_dataset_version,
            VersionArgs,
            permission=Permission.WRITE,
            module="dvc",
            summarize=lambda a: (
                f"Replace workspace copy of {a.path} with dataset version {a.version}"
            ),
        ),
    ]


def register(
    registry: ToolRegistry, settings: Settings, client: DVCClient | None = None
) -> DVCClient:
    client = client or DVCClient(settings.repo_root)
    for tool in build_tools(client):
        registry.register(tool)
    return client
