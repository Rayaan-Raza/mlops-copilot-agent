"""Fake Kubernetes API objects shaped like the official python client's models."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace as NS

T0 = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)


class ApiException(Exception):
    def __init__(self, status, reason):
        super().__init__(f"({status}) {reason}")
        self.status, self.reason = status, reason


def container_status(
    ready=True, restarts=0, waiting=None, msg=None, last=None, image="ml-inference:1.0.0"
):
    return NS(
        name="ml-inference",
        ready=ready,
        restart_count=restarts,
        image=image,
        state=NS(waiting=NS(reason=waiting, message=msg) if waiting else None),
        last_state=NS(
            terminated=NS(reason=last[0], exit_code=last[1], finished_at=T0) if last else None
        ),
    )


def pod(name, *, ready=True, phase="Running", cs=None):
    return NS(
        metadata=NS(
            name=name, labels={"app": "ml-inference", "app.kubernetes.io/version": "1.0.0"}
        ),
        spec=NS(node_name="kind-control-plane"),
        status=NS(
            phase=phase,
            start_time=T0,
            container_statuses=[cs or container_status()],
            conditions=[NS(type="Ready", status="True" if ready else "False")],
        ),
    )


def deployment(ready=2, image="ml-inference:1.0.0"):
    return NS(
        metadata=NS(
            name="ml-inference",
            labels={"app.kubernetes.io/version": "1.0.0"},
            annotations={"deployment.kubernetes.io/revision": "3"},
        ),
        spec=NS(replicas=2, template=NS(spec=NS(containers=[NS(image=image)]))),
        status=NS(
            ready_replicas=ready,
            available_replicas=ready,
            updated_replicas=2,
            conditions=[
                NS(
                    type="Available",
                    status="True" if ready else "False",
                    reason="MinimumReplicasAvailable",
                    message="",
                )
            ],
        ),
    )


def event(reason, message, kind="Pod", name="ml-inference-abc", etype="Warning"):
    return NS(
        type=etype,
        reason=reason,
        message=message,
        count=4,
        last_timestamp=T0,
        involved_object=NS(kind=kind, name=name),
        metadata=NS(creation_timestamp=T0),
    )


class FakeCore:
    def __init__(self, pods, events=(), logs="", ready_endpoints=2, fail=None):
        self.pods, self.events, self.logs = list(pods), list(events), logs
        self.ready_endpoints, self.fail = ready_endpoints, fail

    def _chk(self):
        if self.fail:
            raise self.fail

    def list_namespaced_pod(self, ns, label_selector=None):
        self._chk()
        return NS(items=self.pods)

    def read_namespaced_pod_log(self, name, ns, **kw):
        self._chk()
        return self.logs

    def list_namespaced_service(self, ns):
        self._chk()
        return NS(
            items=[
                NS(
                    metadata=NS(name="ml-inference"),
                    spec=NS(
                        type="ClusterIP",
                        cluster_ip="10.96.0.10",
                        selector={"app": "ml-inference"},
                        ports=[NS(port=80, target_port="http", protocol="TCP")],
                    ),
                )
            ]
        )

    def read_namespaced_endpoints(self, name, ns):
        addrs = [NS(ip=f"10.0.0.{i}") for i in range(self.ready_endpoints)]
        nr = [NS(ip="10.0.0.99")] * (2 - self.ready_endpoints)
        return NS(subsets=[NS(addresses=addrs, not_ready_addresses=nr)])

    def list_namespaced_event(self, ns):
        self._chk()
        return NS(items=self.events)

    def read_namespaced_config_map(self, name, ns):
        return NS(data={"APP_VERSION": "1.0.0"})


class FakeApps:
    def __init__(self, dep=None, fail=None):
        self.dep, self.fail, self.patches = dep or deployment(), fail, []

    def list_namespaced_deployment(self, ns):
        if self.fail:
            raise self.fail
        return NS(items=[self.dep])

    def read_namespaced_deployment(self, name, ns):
        if self.fail:
            raise self.fail
        if name != "ml-inference":
            raise ApiException(404, "Not Found")
        return self.dep

    def patch_namespaced_deployment(self, name, ns, body):
        self.patches.append(("restart", name, ns, body))

    def patch_namespaced_deployment_scale(self, name, ns, body):
        self.patches.append(("scale", name, ns, body))
