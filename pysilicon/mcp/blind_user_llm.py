"""
Blind-user LLM harness for pysilicon MCP tools.

This module simulates a new user interacting with the pysilicon MCP tools
through an LLM (OpenAI-compatible API) to test that the tools are useful
and discoverable without prior context.

The harness imports tool schemas and dispatch logic from the shared registry
(``pysilicon.mcp.registry``) so tool definitions are never duplicated.

Usage
-----
Run interactively (requires ``OPENAI_API_KEY`` in the environment)::

    python -m pysilicon.mcp.blind_user_llm

Or supply a task via CLI argument::

    python -m pysilicon.mcp.blind_user_llm \\
        --task "Show me an example of a DataList with an enum field"

Or call from Python::

    from pysilicon.mcp.blind_user_llm import run_session
    result = run_session(task="Show me a schema with memory addresses")
    print(result["final_response"])
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from openai import OpenAI

from pysilicon.mcp.registry import REGISTRY


DEFAULT_MODEL = "gpt-4.1"

SYSTEM_PROMPT = """\
You are helping a hardware engineer author pysilicon data schemas.

You are a new user with no prior knowledge of the pysilicon schema library.
Use the available tools to discover what schema examples exist and to fetch
ones that look relevant to the task.  Work through the tools step by step,
starting with listing the available examples.

pysilicon schemas define the shape of streaming data exchanged with hardware
accelerators (command headers, response headers, response footers, etc.).
The main schema types are DataList and DataArray, composed from specialised
field types (IntField, FloatField, EnumField, MemAddr, etc.).

Your goal is to help the user understand how to write a schema that meets
their needs, using the examples as a starting point.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_session(
    *,
    task: str,
    model: str = DEFAULT_MODEL,
    max_rounds: int = 10,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a simulated blind-user session for *task*.

    Parameters
    ----------
    task:
        A plain-English description of what the user wants, e.g.
        ``"Show me a schema with memory address fields"``.
    model:
        OpenAI model name to use.
    max_rounds:
        Maximum number of tool-call rounds before giving up.
    verbose:
        If ``True``, print each message and tool call to stdout.

    Returns
    -------
    dict
        ``{"messages": [...], "final_response": "<assistant text>"}``
    """
    client = _build_client()
    tool_schemas = REGISTRY.tool_schemas()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    if verbose:
        print(f"[blind_user] task: {task}")
        print(f"[blind_user] model: {model}, max_rounds: {max_rounds}")

    final_response = ""
    for round_num in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
        )

        choice = response.choices[0]
        message = choice.message

        # Append the assistant message to history
        messages.append(message.model_dump(exclude_unset=True))

        # Check for tool calls
        tool_calls = message.tool_calls or []
        if not tool_calls:
            # No tool calls — this is the final text response
            final_response = message.content or ""
            if verbose:
                print(f"[blind_user] final response received after {round_num + 1} round(s)")
            break

        # Execute each tool call and append results
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            if verbose:
                print(f"[blind_user] round {round_num + 1}: tool call → {tool_name}({arguments})")

            try:
                result = REGISTRY.dispatch(tool_name, arguments)
                result_text = json.dumps(result, indent=2)
            except Exception as exc:  # noqa: BLE001
                result_text = json.dumps({"error": str(exc)})

            if verbose:
                print(f"[blind_user] tool result ({len(result_text)} chars)")

            messages.append({
                "role": "tool",
                "content": result_text,
                "tool_call_id": tool_call.id,
            })
    else:
        final_response = (
            f"Session ended after {max_rounds} rounds without a final response."
        )
        if verbose:
            print(f"[blind_user] {final_response}")

    return {"messages": messages, "final_response": final_response}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Blind-user LLM harness for pysilicon MCP tools. "
            "Simulates a new user with no prior context interacting with the tools."
        )
    )
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Task for the simulated user. If omitted, an interactive prompt is shown."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum number of tool-call rounds (default: 10).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tool call details to stdout.",
    )
    return parser


def _build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)
    return OpenAI(api_key=api_key)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    task = args.task
    if not task:
        try:
            task = input("Enter task for the blind-user session: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            sys.exit(1)

    if not task:
        print("Error: task cannot be empty.", file=sys.stderr)
        sys.exit(1)

    result = run_session(
        task=task,
        model=args.model,
        max_rounds=args.max_rounds,
        verbose=args.verbose,
    )

    print("\n--- Final response ---")
    print(result["final_response"])


if __name__ == "__main__":
    main()
