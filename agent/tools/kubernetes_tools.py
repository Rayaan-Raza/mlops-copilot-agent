"""Kubernetes operations tools for the ML inference service.

Read tools are restricted to approved namespaces; the two state-changing tools (restart, scale)
additionally require explicit human confirmation, and RBAC (kubernetes/rbac.yaml) limits the
agent's service account to the same namespace.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agent.config import Settings
from agent.errors import (
    InvalidRequestError,
    PermissionDeniedError,
    ServiceUnavailableError,
    ToolError,
)
from agent.registry import Tool, ToolRegistry
from agent.safety import require_namespace_allowed
from agent.schemas import Permission

BAD_WAITING_REASONS = {
    "CrashLoopBackOff",
    "ImagePullBackOff",
    "ErrImagePull",
    "CreateContainerConfigError",
    "CreateContainerError",
    "InvalidImageName",
    "RunContainerError",
}
ERROR_LINE_RE = re.compile(r"(error|exception|traceback|fail|refus|denied|killed)", re.I)
NAME = r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$"


def _iso(dt) -> str | None:
    return dt.isoformat() if isinstance(dt, datetime) else (str(dt) if dt else None)


def kubeconfig_path() -> str:
    """``$KUBECONFIG`` if set to something, else ``~/.kube/config``.

    An empty ``KUBECONFIG=`` (as in .env.example) must not be treated as a path: the kubernetes
    client would otherwise fail with "Invalid kube-config file".
    """
    value = (os.environ.get("KUBECONFIG") or "").strip()
    if value:
        return value.split(os.pathsep)[0]
    return os.path.expanduser("~/.kube/config")


class KubeGateway:
    """Lazily builds kubernetes API clients; tests inject fakes via ``core``/``apps``."""

    def __init__(self, core: Any = None, apps: Any = None) -> None:
        self._core, self._apps = core, apps

    def _load(self) -> None:
        try:
            from kubernetes import client, config
        except ImportError as exc:  # pragma: no cover
            raise ServiceUnavailableError("kubernetes python client not installed") from exc
        try:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config(config_file=kubeconfig_path())
        except Exception as exc:  # noqa: BLE001
            raise ServiceUnavailableError(f"no usable kubeconfig: {exc}") from exc
        self._core = self._core or client.CoreV1Api()
        self._apps = self._apps or client.AppsV1Api()

    @property
    def core(self):
        if self._core is None:
            self._load()
        return self._core

    @property
    def apps(self):
        if self._apps is None:
            self._load()
        return self._apps

    @staticmethod
    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ToolError, ServiceUnavailableError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalise ApiException / urllib3 errors
            status = getattr(exc, "status", None)
            reason = getattr(exc, "reason", None) or str(exc)
            if status == 404:
                raise InvalidRequestError(f"Kubernetes resource not found: {reason}") from exc
            if status in (401, 403):
                raise PermissionDeniedError(
                    f"Kubernetes RBAC denied the request: {reason}"
                ) from exc
            if status is None or status >= 500:
                raise ServiceUnavailableError(
                    f"Kubernetes API unreachable: {str(exc)[:200]}"
                ) from exc
            raise ToolError(f"Kubernetes API error {status}: {reason}") from exc


# --------------------------------------------------------------------------- inputs
class NsArgs(BaseModel):
    namespace: str | None = Field(None, description="namespace; default from config")


class PodsArgs(NsArgs):
    label_selector: str | None = Field(
        None, description="e.g. app=ml-inference; default from config"
    )


class PodLogArgs(NsArgs):
    pod: str = Field(..., pattern=NAME)
    container: str | None = None
    tail_lines: int = Field(100, ge=1, le=1000)
    previous: bool = Field(False, description="logs of the previous (crashed) container instance")


class EventsArgs(NsArgs):
    warnings_only: bool = True
    limit: int = Field(30, ge=1, le=200)


class DeploymentArgs(NsArgs):
    deployment: str = Field("ml-inference", pattern=NAME)


class ScaleArgs(DeploymentArgs):
    replicas: int = Field(..., ge=0, le=10)


# --------------------------------------------------------------------------- summaries
def pod_summary(p) -> dict[str, Any]:
    statuses = p.status.container_statuses or []
    containers = []
    for cs in statuses:
        state = cs.state
        waiting = state.waiting.reason if state and state.waiting else None
        terminated = (
            cs.last_state.terminated if cs.last_state and cs.last_state.terminated else None
        )
        containers.append(
            {
                "name": cs.name,
                "ready": bool(cs.ready),
                "restart_count": cs.restart_count or 0,
                "image": cs.image,
                "waiting_reason": waiting,
                "waiting_message": (state.waiting.message if waiting else None),
                "last_termination": {
                    "reason": terminated.reason,
                    "exit_code": terminated.exit_code,
                    "finished_at": _iso(terminated.finished_at),
                }
                if terminated
                else None,
            }
        )
    conds = {c.type: c.status for c in (p.status.conditions or [])}
    return {
        "name": p.metadata.name,
        "phase": p.status.phase,
        "ready": conds.get("Ready") == "True",
        "restarts": sum(c["restart_count"] for c in containers),
        "node": p.spec.node_name,
        "started": _iso(p.status.start_time),
        "version_label": (p.metadata.labels or {}).get("app.kubernetes.io/version"),
        "containers": containers,
    }


def pod_issues(s: dict[str, Any]) -> list[str]:
    issues = []
    if s["phase"] not in ("Running", "Succeeded"):
        issues.append(f"phase is {s['phase']}")
    if not s["ready"]:
        issues.append("not Ready (readiness probe failing or container not started)")
    for c in s["containers"]:
        if c["waiting_reason"] in BAD_WAITING_REASONS or (c["waiting_reason"] and not c["ready"]):
            issues.append(
                f"container {c['name']} waiting: {c['waiting_reason']} {c['waiting_message'] or ''}".strip()
            )
        if c["restart_count"]:
            issues.append(f"container {c['name']} restarted {c['restart_count']}x")
        lt = c["last_termination"]
        if lt and lt["reason"]:
            issues.append(f"last termination: {lt['reason']} (exit {lt['exit_code']})")
    return issues


def deployment_summary(d) -> dict[str, Any]:
    containers = d.spec.template.spec.containers
    return {
        "name": d.metadata.name,
        "desired_replicas": d.spec.replicas,
        "ready_replicas": d.status.ready_replicas or 0,
        "available_replicas": d.status.available_replicas or 0,
        "updated_replicas": d.status.updated_replicas or 0,
        "images": [c.image for c in containers],
        "version_label": (d.metadata.labels or {}).get("app.kubernetes.io/version"),
        "revision": (d.metadata.annotations or {}).get("deployment.kubernetes.io/revision"),
        "conditions": [
            {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
            for c in (d.status.conditions or [])
        ],
    }


# --------------------------------------------------------------------------- tools
def build_tools(kube: KubeGateway, settings: Settings) -> list[Tool]:
    allowed = settings.k8s_allowed_namespaces

    def ns(value: str | None) -> str:
        namespace = value or settings.k8s_namespace
        require_namespace_allowed(namespace, allowed)
        return namespace

    def list_deployments(a: NsArgs) -> dict:
        n = ns(a.namespace)
        items = kube.call(kube.apps.list_namespaced_deployment, n).items
        return {"namespace": n, "deployments": [deployment_summary(d) for d in items]}

    def list_pods(a: PodsArgs) -> dict:
        n = ns(a.namespace)
        sel = a.label_selector or settings.k8s_app_label
        items = kube.call(kube.core.list_namespaced_pod, n, label_selector=sel).items
        return {"namespace": n, "label_selector": sel, "pods": [pod_summary(p) for p in items]}

    def pod_logs(a: PodLogArgs) -> dict:
        n = ns(a.namespace)
        kwargs: dict[str, Any] = {"tail_lines": a.tail_lines, "previous": a.previous}
        if a.container:
            kwargs["container"] = a.container
        text = kube.call(kube.core.read_namespaced_pod_log, a.pod, n, **kwargs) or ""
        lines = text.splitlines()
        return {
            "pod": a.pod,
            "previous_instance": a.previous,
            "error_lines": [ln for ln in lines if ERROR_LINE_RE.search(ln)][-20:],
            "tail": lines[-a.tail_lines :],
        }

    def list_services(a: NsArgs) -> dict:
        n = ns(a.namespace)
        out = []
        for s in kube.call(kube.core.list_namespaced_service, n).items:
            try:
                ep = kube.call(kube.core.read_namespaced_endpoints, s.metadata.name, n)
                ready = sum(len(sub.addresses or []) for sub in (ep.subsets or []))
                not_ready = sum(len(sub.not_ready_addresses or []) for sub in (ep.subsets or []))
            except InvalidRequestError:
                ready, not_ready = 0, 0
            out.append(
                {
                    "name": s.metadata.name,
                    "type": s.spec.type,
                    "cluster_ip": s.spec.cluster_ip,
                    "ports": [
                        {"port": p.port, "target_port": str(p.target_port), "protocol": p.protocol}
                        for p in s.spec.ports or []
                    ],
                    "selector": s.spec.selector,
                    "ready_endpoints": ready,
                    "not_ready_endpoints": not_ready,
                }
            )
        return {"namespace": n, "services": out}

    def events(a: EventsArgs) -> dict:
        n = ns(a.namespace)
        items = kube.call(kube.core.list_namespaced_event, n).items
        if a.warnings_only:
            items = [e for e in items if e.type == "Warning"]

        def ts(e):
            t = e.last_timestamp or getattr(e, "event_time", None) or e.metadata.creation_timestamp
            return t or datetime.min.replace(tzinfo=timezone.utc)

        items = sorted(items, key=ts, reverse=True)[: a.limit]
        return {
            "namespace": n,
            "events": [
                {
                    "type": e.type,
                    "reason": e.reason,
                    "object": f"{e.involved_object.kind}/{e.involved_object.name}",
                    "message": e.message,
                    "count": e.count,
                    "last_seen": _iso(ts(e)),
                }
                for e in items
            ],
        }

    def unhealthy(a: PodsArgs) -> dict:
        pods = list_pods(a)
        flagged = []
        for p in pods["pods"]:
            issues = pod_issues(p)
            if issues:
                flagged.append({"pod": p["name"], "issues": issues, "restarts": p["restarts"]})
        return {
            "namespace": pods["namespace"],
            "total_pods": len(pods["pods"]),
            "unhealthy_pods": flagged,
            "healthy": not flagged and bool(pods["pods"]),
        }

    def serving_model(service: str, namespace: str) -> dict:
        """Ask the running service what it serves (GET /version via the API-server proxy).

        Read-only; needs ``services/proxy: get`` in the namespace. Any failure is reported in
        the result rather than raised, because the rest of the version check is still useful.
        """
        try:
            raw = kube.call(
                kube.core.connect_get_namespaced_service_proxy_with_path,
                f"{service}:http",
                namespace,
                "version",
                # Raw bytes: with preloading the client returns a Python repr, not JSON.
                _preload_content=False,
            )
            raw = raw.data.decode("utf-8") if hasattr(raw, "data") else raw
            body = json.loads(raw) if isinstance(raw, str) else raw
            model = dict(body.get("model") or {})
            model.pop("feature_names", None)  # long and not useful to the LLM
            return {"available": True, "app_version": body.get("app_version"), **model}
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    def deployed_version(a: DeploymentArgs) -> dict:
        n = ns(a.namespace)
        d = deployment_summary(kube.call(kube.apps.read_namespaced_deployment, a.deployment, n))
        try:
            cm = kube.call(kube.core.read_namespaced_config_map, f"{a.deployment}-config", n)
            cfg_version = (cm.data or {}).get("APP_VERSION")
        except InvalidRequestError:
            cfg_version = None
        tags = [img.rsplit(":", 1)[-1] if ":" in img else "latest" for img in d["images"]]
        consistent = len({v for v in [d["version_label"], cfg_version, *tags] if v}) <= 1
        serving = serving_model(a.deployment, n)
        return {
            "serving_model": serving,
            "model_link_note": (
                "The service identifies its model only by type, dataset, test F1 and hash, not by "
                "MLflow run id. Compare these with MLflow before claiming a specific run is deployed."
            ),
            "deployment": a.deployment,
            "namespace": n,
            "image_tags": tags,
            "images": d["images"],
            "version_label": d["version_label"],
            "configmap_app_version": cfg_version,
            "revision": d["revision"],
            "version_sources_consistent": consistent,
            "ready_replicas": f"{d['ready_replicas']}/{d['desired_replicas']}",
        }

    def diagnose(a: DeploymentArgs) -> dict:
        n = ns(a.namespace)
        dep = deployment_summary(kube.call(kube.apps.read_namespaced_deployment, a.deployment, n))
        health = unhealthy(PodsArgs(namespace=n))
        svc = [s for s in list_services(NsArgs(namespace=n))["services"] if s["selector"]]
        evts = events(EventsArgs(namespace=n, warnings_only=True, limit=15))["events"]
        logs = []
        for p in health["unhealthy_pods"][:2]:
            prev = any("restarted" in i for i in p["issues"])
            try:
                lg = pod_logs(PodLogArgs(namespace=n, pod=p["pod"], tail_lines=40, previous=prev))
                logs.append(
                    {
                        "pod": p["pod"],
                        "previous_instance": prev,
                        "error_lines": lg["error_lines"],
                        "tail": lg["tail"][-10:],
                    }
                )
            except (InvalidRequestError, ToolError) as exc:
                logs.append({"pod": p["pod"], "log_error": str(exc)})
        available = dep["available_replicas"] > 0 and any(s["ready_endpoints"] > 0 for s in svc)
        return {
            "deployment": dep,
            "service_endpoints": svc,
            "unhealthy_pods": health["unhealthy_pods"],
            "warning_events": evts,
            "logs": logs,
            "api_reachable_via_service": available,
        }

    def restart(a: DeploymentArgs) -> dict:
        n = ns(a.namespace)
        stamp = datetime.now(timezone.utc).isoformat()
        body = {
            "spec": {
                "template": {
                    "metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": stamp}}
                }
            }
        }
        kube.call(kube.apps.patch_namespaced_deployment, a.deployment, n, body)
        return {"restarted": a.deployment, "namespace": n, "restarted_at": stamp}

    def scale(a: ScaleArgs) -> dict:
        n = ns(a.namespace)
        kube.call(
            kube.apps.patch_namespaced_deployment_scale,
            a.deployment,
            n,
            {"spec": {"replicas": a.replicas}},
        )
        return {"scaled": a.deployment, "namespace": n, "replicas": a.replicas}

    def _write_precheck(a: DeploymentArgs) -> None:
        require_namespace_allowed(a.namespace or settings.k8s_namespace, allowed)

    return [
        Tool(
            "k8s_list_deployments",
            "List Deployments with desired/ready replicas, images, version label and conditions.",
            list_deployments,
            NsArgs,
            module="kubernetes",
        ),
        Tool(
            "k8s_list_pods",
            "List application Pods with phase, readiness, restarts and container states.",
            list_pods,
            PodsArgs,
            module="kubernetes",
        ),
        Tool(
            "k8s_get_pod_logs",
            "Get a Pod's recent logs (optionally of the previous crashed instance) and error lines.",
            pod_logs,
            PodLogArgs,
            module="kubernetes",
        ),
        Tool(
            "k8s_list_services",
            "List Services with type, ports, selector and ready/not-ready endpoint counts.",
            list_services,
            NsArgs,
            module="kubernetes",
        ),
        Tool(
            "k8s_get_events",
            "Get recent (warning) events in the namespace, e.g. probe failures, image pull errors, OOM kills.",
            events,
            EventsArgs,
            module="kubernetes",
        ),
        Tool(
            "k8s_find_unhealthy_pods",
            "Identify unhealthy Pods: not ready, CrashLoopBackOff, ImagePullBackOff, OOMKilled, restarts.",
            unhealthy,
            PodsArgs,
            module="kubernetes",
        ),
        Tool(
            "k8s_get_deployed_version",
            "Report the currently deployed application version (image tag, version label, ConfigMap APP_VERSION).",
            deployed_version,
            DeploymentArgs,
            module="kubernetes",
        ),
        Tool(
            "k8s_diagnose_deployment",
            "One-shot health investigation of a deployment: status, unhealthy pods, events, service endpoints and logs.",
            diagnose,
            DeploymentArgs,
            module="kubernetes",
        ),
        Tool(
            "k8s_restart_deployment",
            "Rolling-restart a Deployment in an approved namespace.",
            restart,
            DeploymentArgs,
            permission=Permission.WRITE,
            module="kubernetes",
            precheck=_write_precheck,
            summarize=lambda a: (
                f"Rolling restart of deployment/{a.deployment} in namespace {a.namespace or settings.k8s_namespace}"
            ),
        ),
        Tool(
            "k8s_scale_deployment",
            "Scale a Deployment (0-10 replicas) in an approved namespace.",
            scale,
            ScaleArgs,
            permission=Permission.WRITE,
            module="kubernetes",
            precheck=_write_precheck,
            summarize=lambda a: (
                f"Scale deployment/{a.deployment} in {a.namespace or settings.k8s_namespace} to {a.replicas} replicas"
            ),
        ),
    ]


def register(
    registry: ToolRegistry, settings: Settings, gateway: KubeGateway | None = None
) -> KubeGateway:
    gateway = gateway or KubeGateway()
    for tool in build_tools(gateway, settings):
        registry.register(tool)
    return gateway
