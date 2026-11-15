"""Kubernetes tools against fake CoreV1Api / AppsV1Api objects."""

from agent.config import Settings
from agent.registry import ToolRegistry
from agent.schemas import ToolCall, ToolStatus
from agent.tools import kubernetes_tools
from agent.tools.kubernetes_tools import KubeGateway
from tests.k8s_fakes import (
    ApiException,
    FakeApps,
    FakeCore,
    container_status,
    deployment,
    event,
    pod,
)

SETTINGS = Settings(k8s_namespace="mlops-agent", k8s_allowed_namespaces=["mlops-agent"])


def make(core=None, apps=None):
    core = core or FakeCore([pod("ml-inference-a"), pod("ml-inference-b")])
    apps = apps or FakeApps()
    reg = ToolRegistry(default_timeout=5)
    kubernetes_tools.register(reg, SETTINGS, KubeGateway(core, apps))
    return reg, core, apps


def call(reg, name, **args):
    return reg.execute(ToolCall(id="t", name=name, arguments=args))


def test_healthy_deployment_reports_no_issues_and_version():
    reg, *_ = make()
    assert call(reg, "k8s_find_unhealthy_pods").data["healthy"] is True
    ver = call(reg, "k8s_get_deployed_version").data
    assert ver["image_tags"] == ["1.0.0"] and ver["configmap_app_version"] == "1.0.0"
    assert ver["version_sources_consistent"] is True and ver["ready_replicas"] == "2/2"


def test_image_pull_backoff_is_diagnosed():
    bad = container_status(
        ready=False,
        waiting="ImagePullBackOff",
        msg='Back-off pulling image "ml-inference:9.9.9"',
        image="ml-inference:9.9.9",
    )
    core = FakeCore(
        [pod("ml-inference-a"), pod("ml-inference-x", ready=False, phase="Pending", cs=bad)],
        events=[
            event(
                "Failed",
                'Failed to pull image "ml-inference:9.9.9": not found',
                name="ml-inference-x",
            )
        ],
        ready_endpoints=1,
    )
    reg, *_ = make(core, FakeApps(deployment(ready=1, image="ml-inference:9.9.9")))
    res = call(reg, "k8s_diagnose_deployment", deployment="ml-inference")
    assert res.status == ToolStatus.OK, res.error
    issues = " ".join(res.data["unhealthy_pods"][0]["issues"])
    assert "ImagePullBackOff" in issues
    assert "9.9.9" in res.data["warning_events"][0]["message"]


def test_crashloop_with_oom_kill_and_previous_logs():
    crash = container_status(
        ready=False, restarts=5, waiting="CrashLoopBackOff", last=("OOMKilled", 137)
    )
    core = FakeCore(
        [pod("ml-inference-a", ready=False, cs=crash)],
        logs="INFO loading model\nERROR MemoryError: cannot allocate\n",
        ready_endpoints=0,
    )
    reg, *_ = make(core, FakeApps(deployment(ready=0)))
    res = call(reg, "k8s_diagnose_deployment")
    pod_issue = " ".join(res.data["unhealthy_pods"][0]["issues"])
    assert "CrashLoopBackOff" in pod_issue and "OOMKilled" in pod_issue
    assert res.data["logs"][0]["previous_instance"] is True
    assert "MemoryError" in res.data["logs"][0]["error_lines"][0]
    assert res.data["api_reachable_via_service"] is False


def test_unauthorized_namespace_modification_is_rejected():
    reg, _, apps = make()
    res = call(reg, "k8s_restart_deployment", deployment="coredns", namespace="kube-system")
    assert res.status == ToolStatus.DENIED
    assert res.confirmation_token is None and apps.patches == []
    res = call(
        reg, "k8s_scale_deployment", deployment="ml-inference", namespace="default", replicas=0
    )
    assert res.status == ToolStatus.DENIED and apps.patches == []


def test_reads_outside_approved_namespace_are_denied():
    reg, *_ = make()
    assert call(reg, "k8s_list_pods", namespace="kube-system").status == ToolStatus.DENIED


def test_restart_executes_only_after_confirmation():
    reg, _, apps = make()
    pending = call(reg, "k8s_restart_deployment", deployment="ml-inference")
    assert pending.status == ToolStatus.CONFIRMATION_REQUIRED and apps.patches == []
    done = reg.execute_confirmed(pending.confirmation_token)
    assert done.status == ToolStatus.OK and apps.patches[0][:3] == (
        "restart",
        "ml-inference",
        "mlops-agent",
    )


def test_rbac_forbidden_and_cluster_down_are_mapped():
    reg, *_ = make(apps=FakeApps(fail=ApiException(403, "Forbidden")))
    assert call(reg, "k8s_list_deployments").status == ToolStatus.DENIED
    reg, *_ = make(core=FakeCore([], fail=ConnectionRefusedError("connection refused")))
    assert call(reg, "k8s_list_pods").status == ToolStatus.UNAVAILABLE


def test_services_report_endpoints():
    reg, *_ = make()
    svc = call(reg, "k8s_list_services").data["services"][0]
    assert svc["type"] == "ClusterIP" and svc["ready_endpoints"] == 2


def test_empty_kubeconfig_env_falls_back_to_default(monkeypatch):
    """Regression: `KUBECONFIG=` (empty, as in .env.example) broke config loading."""
    import os

    monkeypatch.setenv("KUBECONFIG", "")
    assert kubernetes_tools.kubeconfig_path() == os.path.expanduser("~/.kube/config")
    monkeypatch.setenv("KUBECONFIG", "/tmp/a.yaml" + os.pathsep + "/tmp/b.yaml")
    assert kubernetes_tools.kubeconfig_path() == "/tmp/a.yaml"


def test_deployed_version_reports_what_the_service_actually_serves():
    """Regression (live run): the agent claimed the best MLflow run was deployed, but the
    service was serving a different model. The tool now reads /version via the API proxy."""
    reg, *_ = make()
    serving = call(reg, "k8s_get_deployed_version").data["serving_model"]
    assert serving["available"] is True
    assert serving["model_type"] == "random_forest" and serving["test_f1"] == 0.951
    assert "feature_names" not in serving
