# Inference service image (deployed to Kubernetes).
# Stage 1 trains/bakes a model so the image is self-contained; stage 2 is the slim runtime.

FROM python:3.11-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir "scikit-learn>=1.4,<2" "pandas>=2.1,<3" "pyyaml>=6"
COPY params.yaml ./
COPY training ./training
COPY inference ./inference
RUN REPO_ROOT=/build python -m inference.bake_model

FROM python:3.11-slim
ARG APP_VERSION=dev
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_VERSION=${APP_VERSION} \
    MODEL_PATH=/app/models/model.pkl
WORKDIR /app
RUN pip install --no-cache-dir "scikit-learn>=1.4,<2" "pandas>=2.1,<3" \
        "fastapi>=0.110,<1" "uvicorn>=0.29" "pydantic>=2.6,<3" \
    && useradd --uid 10001 --create-home appuser
COPY inference ./inference
COPY --from=builder /build/models ./models
USER 10001
EXPOSE 8000
LABEL org.opencontainers.image.title="ml-inference" \
      org.opencontainers.image.version="${APP_VERSION}"
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
