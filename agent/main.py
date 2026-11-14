"""Command-line chat with the MLOps agent.

mlops-agent                      # interactive
mlops-agent "is the prediction API healthy?"
python -m agent.main --json "find the best recent model"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from agent.logging_config import configure_logging
from agent.orchestrator import Agent
from agent.schemas import AgentResponse


def render(resp: AgentResponse, agent: Agent, interactive: bool) -> None:
    print("\n" + resp.answer + "\n")
    if resp.evidence:
        print(
            "Evidence: "
            + ", ".join(f"{e.tool}[{e.call_id}]={e.status.value}" for e in resp.evidence)
        )
    guard_notes = [u for u in resp.uncertainties if u.startswith(("Unverified claim", "No tools"))]
    for note in guard_notes:
        print(f"  ! {note}")
    for p in resp.pending_confirmations:
        print(f"\nPENDING ACTION: {p['action']}")
        if not interactive:
            print(
                f'  confirm via API: POST /confirm {{"confirmation_token": "{p["confirmation_token"]}"}}'
            )
            continue
        answer = input("  Type 'yes' to execute this action, anything else to cancel: ").strip()
        if answer.lower() == "yes":
            result = agent.confirm(p["confirmation_token"])
            print(
                f"  -> {result.status.value}: {json.dumps(result.data, default=str)[:800] if result.data else result.error}"
            )
        else:
            print("  -> cancelled")


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLOps Copilot agent")
    parser.add_argument("question", nargs="*", help="question to ask (omit for interactive mode)")
    parser.add_argument("--json", action="store_true", help="print the structured response as JSON")
    args = parser.parse_args(argv)

    configure_logging(os.getenv("LOG_LEVEL", "WARNING"))
    agent = Agent.from_settings()

    if args.question:
        resp = agent.run(" ".join(args.question))
        if args.json:
            print(resp.model_dump_json(indent=2))
        else:
            render(resp, agent, interactive=sys.stdin.isatty())
        return 0

    print(
        "MLOps Copilot - ask about datasets, experiments, pipelines or the deployment. Ctrl+C to exit."
    )
    try:
        while True:
            q = input("\nyou> ").strip()
            if q:
                render(agent.run(q), agent, interactive=True)
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
