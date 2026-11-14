"""Exception types tools raise; the registry maps them onto ``ToolStatus`` values."""


class ToolError(Exception):
    """Generic tool failure (maps to status=error)."""


class ServiceUnavailableError(ToolError):
    """The backing system (MLflow, Airflow, Kubernetes, DVC) cannot be reached."""


class InvalidRequestError(ToolError):
    """Arguments are syntactically valid but semantically wrong (e.g. unknown run id)."""


class PermissionDeniedError(ToolError):
    """The operation is not allowed by the agent's safety policy."""
