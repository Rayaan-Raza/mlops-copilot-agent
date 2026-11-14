#!/usr/bin/env bash
# Open one PR per feature branch against main, in merge order.
# Requires the GitHub CLI, authenticated:  gh auth login
# Usage (Git Bash):  bash scripts/open_prs.sh
set -euo pipefail

REPO="Rayaan-Raza/mlops-copilot-agent"
BRANCHES=(
  "chore/project-scaffold-ci|chore: project scaffold, packaging and CI"
  "feat/data-training-dvc|feat(data): DVC-versioned dataset and training pipeline with MLflow provenance"
  "feat/airflow-training-dag|feat(airflow): Airflow 3 training DAG and local MLflow/Airflow stack"
  "feat/k8s-inference-service|feat(k8s): FastAPI inference service and Kubernetes manifests"
  "feat/agent-core|feat(agent): tool registry, safety controls and structured logging"
  "feat/dvc-tools|feat(tools): DVC data-management tools"
  "feat/mlflow-tools|feat(tools): MLflow experiment-intelligence tools"
  "feat/airflow-tools|feat(tools): Airflow 3 workflow tools"
  "feat/kubernetes-tools|feat(tools): Kubernetes operations tools"
  "feat/agent-orchestrator-api|feat(agent): multi-tool orchestrator, LLM providers, FastAPI and CLI"
  "ci/cd-pipeline|ci: CI/CD and continuous-training pipelines"
  "docs/readme-architecture|docs: README, architecture, failure-injection runbook, report"
)

n=0
for entry in "${BRANCHES[@]}"; do
  n=$((n + 1))
  branch="${entry%%|*}"
  title="${entry#*|}"
  if gh pr view "$branch" --repo "$REPO" >/dev/null 2>&1; then
    echo "[$n] PR already exists for $branch: $(gh pr view "$branch" --repo "$REPO" --json url -q .url)"
    continue
  fi
  body="Merge order: **${n} of ${#BRANCHES[@]}**. Branches are stacked; merge the previous PR first.

Use **Create a merge commit** (not squash/rebase) so the \`data-v*\` dataset tags stay reachable from \`main\`.

Commits: see the commit list for details."
  url=$(gh pr create --repo "$REPO" --base main --head "$branch" --title "$title" --body "$body")
  echo "[$n] $url"
done
