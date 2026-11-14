"""Live tests against a real cluster (run by the CD workflow on a kind cluster).

pytest -m integration tests/integration
"""

import os

import pytest

from agent.config import Settings
from agent.registry import ToolRegistry
from agent.schemas import ToolCall, ToolStatus
from agent.tools import kubernetes_tools

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def registry():
    if not os.getenv("RUN_LIVE_K8S_TESTS"):
        pytest.skip("set RUN_LIVE_K8S_TESTS=1 to run against the current kube context")
    reg = ToolRegistry(default_timeout=30)
    kubernetes_tools.register(reg, Settings())
    return reg


def call(reg, name, **args):
    return reg.execute(ToolCall(id="live", name=name, arguments=args))


def test_live_deployment_is_healthy(registry):
    res = call(registry, "k8s_find_unhealthy_pods")
    assert res.status == ToolStatus.OK, res.error
    assert res.data["total_pods"] >= 2
    assert res.data["healthy"], res.data["unhealthy_pods"]


def test_live_version_and_service(registry):
    ver = call(registry, "k8s_get_deployed_version").data
    assert ver["ready_replicas"].split("/")[0] == ver["ready_replicas"].split("/")[1]
    svc = call(registry, "k8s_list_services").data["services"][0]
    assert svc["type"] == "ClusterIP" and svc["ready_endpoints"] >= 2


def test_live_write_outside_namespace_denied(registry):
    res = call(registry, "k8s_restart_deployment", deployment="coredns", namespace="kube-system")
    assert res.status == ToolStatus.DENIED
