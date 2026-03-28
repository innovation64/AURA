from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

from .core import AURAAgent
from .tools import ToolPolicy
from .core import AURAConfig
from .types import Interaction


def _parse_raw(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _interaction_to_dict(interaction: Interaction) -> dict:
    return asdict(interaction)


def main() -> None:
    parser = argparse.ArgumentParser(description="AURA environment agent")
    parser.add_argument("input", help="Raw environment input (string or JSON)")
    parser.add_argument("--query", dest="query", help="Optional user query", default=None)
    parser.add_argument("--json", dest="json_out", action="store_true", help="Output JSON")
    parser.add_argument(
        "--probe",
        dest="probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable active environment probing",
    )
    parser.add_argument(
        "--max-steps",
        dest="max_steps",
        type=int,
        default=3,
        help="Max probing steps",
    )
    parser.add_argument(
        "--allow-tool",
        dest="allow_tools",
        action="append",
        default=[],
        help="Tool allowlist pattern (repeatable, supports *)",
    )
    parser.add_argument(
        "--deny-tool",
        dest="deny_tools",
        action="append",
        default=[],
        help="Tool denylist pattern (repeatable, supports *)",
    )
    args = parser.parse_args()

    policy = None
    if args.allow_tools or args.deny_tools:
        policy = ToolPolicy(allow=args.allow_tools or None, deny=args.deny_tools or None)
    config = AURAConfig(
        explore_enabled=args.probe,
        explore_max_steps=args.max_steps,
        tool_policy=policy,
    )
    agent = AURAAgent(config=config)
    raw_input = _parse_raw(args.input)
    interaction = agent.run(raw_input, user_query=args.query)

    if args.json_out:
        print(json.dumps(_interaction_to_dict(interaction), ensure_ascii=False))
    else:
        print(interaction.message)


if __name__ == "__main__":
    main()
