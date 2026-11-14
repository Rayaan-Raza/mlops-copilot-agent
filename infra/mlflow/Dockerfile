FROM python:3.11-slim

ARG MLFLOW_VERSION=3.18.0
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}"

EXPOSE 5000
VOLUME ["/mlflow"]

# Artifacts are proxied through the server (--serve-artifacts) so the host-side agent and the
# Airflow containers can both read/write them via http://.../api/2.0/mlflow-artifacts.
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 \
     --backend-store-uri sqlite:////mlflow/mlflow.db \
     --artifacts-destination /mlflow/artifacts --serve-artifacts \
     --allowed-hosts 'localhost,localhost:*,127.0.0.1:*,mlflow,mlflow:*,host.docker.internal:*'"]
