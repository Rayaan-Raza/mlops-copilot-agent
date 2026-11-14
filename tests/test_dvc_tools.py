"""DVC tools: exercised against a real temporary git repo; `dvc` itself is faked."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.config import Settings
from agent.registry import ToolRegistry
from agent.schemas import ToolCall, ToolStatus
from agent.tools import dvc_tools
from agent.tools.dvc_tools import DVCClient

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

MD5_V1 = "a" * 32
MD5_V2 = "b" * 32


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _pointer(md5: str, size: int) -> str:
    return f"outs:\n- md5: {md5}\n  size: {size}\n  hash: md5\n  path: dataset.csv\n"


@pytest.fixture()
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "ci@example.com")
    _git(tmp_path, "config", "user.name", "ci")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    ptr = tmp_path / "data" / "raw" / "dataset.csv.dvc"
    ptr.write_text(_pointer(MD5_V1, 1000), encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "data v1")
    _git(tmp_path, "tag", "-a", "data-v1.0", "-m", "dataset v1")
    ptr.write_text(_pointer(MD5_V2, 1500), encoding="utf-8")
    _git(tmp_path, "commit", "-q", "-am", "data v2")
    _git(tmp_path, "tag", "-a", "data-v2.0", "-m", "dataset v2")
    return tmp_path


def fake_runner(dvc_outputs: dict):
    """Run git for real; answer dvc sub-commands from a dict."""

    def run(cmd, **kwargs):
        if Path(cmd[0]).stem.lower() == "dvc":  # executable may be resolved to a full path
            out = dvc_outputs.get(cmd[1], "{}")
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        return subprocess.run(cmd, **kwargs)

    return run


def make(repo, dvc_outputs=None) -> ToolRegistry:
    reg = ToolRegistry(default_timeout=10)
    client = DVCClient(repo, runner=fake_runner(dvc_outputs or {}))
    dvc_tools.register(reg, Settings(repo_root=repo), client)
    return reg


def call(reg, name, **args):
    return reg.execute(ToolCall(id="t", name=name, arguments=args))


def test_non_allowlisted_commands_are_blocked(repo):
    client = DVCClient(repo)
    assert not client.is_allowed(["dvc", "gc", "-w"])
    assert not client.is_allowed(["rm", "-rf", "/"])
    with pytest.raises(Exception, match="not allow-listed"):
        client.run(["dvc", "remove", "data.dvc"])
    with pytest.raises(Exception, match="confirmation"):
        client.run(["dvc", "checkout"])  # write op without allow_write


def test_get_current_dataset_version(repo):
    res = call(make(repo), "dvc_get_dataset_version")
    assert res.status == ToolStatus.OK, res.error
    assert res.data["md5"] == MD5_V2
    assert res.data["dataset_version"] == "data-v2.0"
    assert res.data["version_tag_is_exact"] is True
    assert len(res.data["pointer_last_changed_commit"]) == 40


def test_get_dataset_version_at_old_tag(repo):
    res = call(make(repo), "dvc_get_dataset_version", rev="data-v1.0")
    assert res.data["md5"] == MD5_V1 and res.data["dataset_version"] == "data-v1.0"


def test_resolve_mlflow_md5_to_version_tag(repo):
    res = call(make(repo), "dvc_resolve_dataset_version", md5=MD5_V1)
    assert res.status == ToolStatus.OK
    assert [t["tag"] for t in res.data["matching_version_tags"]] == ["data-v1.0"]
    assert res.data["matches_current_workspace"] is False


def test_compare_versions_detects_change(repo):
    diff = json.dumps({"modified": [{"path": "data/raw/dataset.csv"}]})
    res = call(
        make(repo, {"diff": diff}),
        "dvc_compare_dataset_versions",
        rev_a="data-v1.0",
        rev_b="data-v2.0",
    )
    assert res.data["content_changed"] is True
    assert res.data["size_delta_bytes"] == 500
    assert res.data["dvc_diff"]["modified"][0]["path"] == "data/raw/dataset.csv"


def test_data_status_reports_changes(repo):
    changed = json.dumps(
        {"data/raw/dataset.csv.dvc": [{"changed outs": {"data/raw/dataset.csv": "modified"}}]}
    )
    res = call(make(repo, {"status": changed}), "dvc_check_data_status")
    assert res.data["workspace_in_sync"] is False
    assert res.data["tracked_data_changed"] is True
    clean = call(make(repo, {"status": "{}"}), "dvc_check_data_status")
    assert clean.data["workspace_in_sync"] is True
    assert clean.data["tracked_data_changed"] is False


def test_unrun_pipeline_is_not_reported_as_data_change(repo):
    """Regression (live run): stages missing from dvc.lock looked like dataset drift."""
    stages_only = json.dumps(
        {"train": [{"changed deps": {"params.yaml": "new"}}], "evaluate": [{"changed outs": {}}]}
    )
    res = call(make(repo, {"status": stages_only}), "dvc_check_data_status")
    assert res.data["tracked_data_changed"] is False
    assert res.data["pipeline_stages_out_of_date"] == ["evaluate", "train"]
    assert "not a data change" in res.data["interpretation"]


def test_option_injection_in_revision_is_rejected(repo):
    res = call(make(repo), "dvc_get_dataset_version", rev="--upload-pack=touch /tmp/pwned")
    assert res.status == ToolStatus.INVALID_REQUEST


def test_reproduction_steps_and_checkout_requires_confirmation(repo):
    reg = make(repo)
    steps = call(reg, "dvc_get_reproduction_steps", version="data-v1.0")
    assert steps.data["expected_md5"] == MD5_V1
    assert any("dvc checkout" in s for s in steps.data["steps"])
    checkout = call(reg, "dvc_checkout_dataset_version", version="data-v1.0")
    assert checkout.status == ToolStatus.CONFIRMATION_REQUIRED


def test_missing_dvc_binary_is_unavailable(repo):
    def runner(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    reg = ToolRegistry()
    dvc_tools.register(reg, Settings(repo_root=repo), DVCClient(repo, runner=runner))
    res = call(reg, "dvc_check_data_status")
    assert res.status == ToolStatus.UNAVAILABLE


def test_list_tracked_skips_dvc_dir_and_envs(repo):
    """Regression (live run on Windows): the `.dvc/` folder itself matched `*.dvc`."""
    (repo / ".dvc" / "cache").mkdir(parents=True)
    (repo / ".venv" / "lib").mkdir(parents=True)
    (repo / ".venv" / "lib" / "junk.dvc").write_text("outs: []", encoding="utf-8")
    res = call(make(repo), "dvc_list_tracked_datasets")
    assert res.status == ToolStatus.OK, res.error
    assert [t["path"] for t in res.data["tracked"]] == ["data/raw/dataset.csv"]


def test_dvc_executable_resolved_next_to_interpreter(tmp_path, monkeypatch):
    """Regression: `dvc` lives in the venv's Scripts/bin, which may not be on PATH."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    exe = fake_bin / "dvc.exe"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(dvc_tools.shutil, "which", lambda name: None)
    monkeypatch.setattr(dvc_tools.sys, "executable", str(fake_bin / "python.exe"))
    assert dvc_tools.resolve_executable("dvc") == str(exe)
    assert dvc_tools.resolve_executable("nonexistent-tool") == "nonexistent-tool"
