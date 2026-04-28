"""
Headless LLM harness for pysilicon MCP tools.

This module simulates a user interacting with the pysilicon MCP tools
through an LLM (OpenAI-compatible API) to test that the tools are useful
and discoverable in standalone headless operation.

The harness imports tool schemas and dispatch logic from the shared registry
(``pysilicon.mcp.registry``) so tool definitions are never duplicated.

Usage
-----
Run interactively (requires ``OPENAI_API_KEY`` in the environment)::

    python -m pysilicon.mcp.headless

Or supply a task via CLI argument::

    python -m pysilicon.mcp.headless \
        --task "Show me an example of a DataList with an enum field"

Or call from Python::

    from pysilicon.mcp.headless import run_session
    result = run_session(task="Show me a schema with memory addresses")
    print(result["final_response"])
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

from pysilicon.mcp.file_tools import _resolve_safe, make_file_tools
from pysilicon.mcp.registry import REGISTRY
from pysilicon.mcp.schema_tools import validate_schema_from_file


DEFAULT_MODEL = "gpt-4.1"

SYSTEM_PROMPT = """\
You are helping a hardware engineer author pysilicon code.

You are a new user with no prior knowledge of the pysilicon functions.
Use the available tools to discover what examples exist and to fetch
ones that look relevant to the task. Work through the tools step by step,
starting with the pysilicon vocabulary glossary and semantic example search.

pysilicon schemas define the shape of streaming data exchanged with hardware
accelerators (command headers, response headers, response footers, etc.).
The main schema types are DataList and DataArray, composed from specialised
field types (IntField, FloatField, EnumField, MemAddr, etc.).

Your goal is to help the user understand how to write pysilicon code that meets
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
    output_path: str | os.PathLike[str] | None = None,
    mode: str | None = None,
    work_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run a simulated headless session for *task*."""
    client = _build_client()
    tool_schemas, dispatch_tool = _build_tool_runtime(
        mode=mode,
        work_dir=work_dir,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    if verbose:
        print(f"[headless] task: {task}")
        print(f"[headless] model: {model}, max_rounds: {max_rounds}")

    final_response = ""
    api_calls: list[dict[str, Any]] = []
    token_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for round_num in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
        )

        usage = _usage_to_dict(response.usage)
        api_call = {
            "round": round_num + 1,
            "usage": usage,
            "tool_calls": [],
        }
        api_calls.append(api_call)
        for key in token_totals:
            token_totals[key] += usage[key]

        choice = response.choices[0]
        message = choice.message
        messages.append(message.model_dump(exclude_unset=True))

        tool_calls = message.tool_calls or []
        if not tool_calls:
            final_response = message.content or ""
            if verbose:
                print(f"[headless] final response received after {round_num + 1} round(s)")
            break

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            if verbose:
                print(f"[headless] round {round_num + 1}: tool call -> {tool_name}({arguments})")

            try:
                result = dispatch_tool(tool_name, arguments)
                result_text = json.dumps(result, indent=2)
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc)}
                result_text = json.dumps(result, indent=2)

            api_call["tool_calls"].append(
                {
                    "name": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "result_text": result_text,
                }
            )

            if verbose:
                print(f"[headless] tool result ({len(result_text)} chars)")

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
            print(f"[headless] {final_response}")

    transcript_path = None
    if output_path is not None:
        transcript_path = _write_session_report(
            output_path=Path(output_path),
            task=task,
            model=model,
            max_rounds=max_rounds,
            final_response=final_response,
            api_calls=api_calls,
            token_totals=token_totals,
        )

    return {
        "messages": messages,
        "final_response": final_response,
        "api_calls": api_calls,
        "token_totals": token_totals,
        "output_path": str(transcript_path) if transcript_path is not None else None,
    }


def _build_tool_runtime(
    *,
    mode: str | None,
    work_dir: str | os.PathLike[str] | None,
) -> tuple[list[dict[str, Any]], Any]:
    if mode not in (None, "workspace", "headless"):
        raise ValueError(
            f"mode must be None, 'workspace', or 'headless', got {mode!r}"
        )

    tool_schemas = REGISTRY.tool_schemas(profile=mode)

    if mode != "headless":
        return tool_schemas, REGISTRY.dispatch

    resolved_work_dir = Path(work_dir).resolve() if work_dir is not None else Path.cwd().resolve()
    list_files_fn, read_file_fn, write_file_fn, edit_file_fn = make_file_tools(
        resolved_work_dir
    )
    file_tool_dispatch = {
        "list_files": list_files_fn,
        "read_file": read_file_fn,
        "write_file": write_file_fn,
        "edit_file": edit_file_fn,
    }

    def validate_schema_headless(
        schema_name: str,
        input_path: str,
        output_path: str,
    ) -> dict[str, Any]:
        resolved_input = _resolve_safe(resolved_work_dir, input_path)
        resolved_output = _resolve_safe(resolved_work_dir, output_path)
        return validate_schema_from_file(
            schema_name=schema_name,
            input_path=str(resolved_input),
            output_path=str(resolved_output),
        )

    tool_schemas = [
        *tool_schemas,
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": (
                    "List files and directories under a path within the configured "
                    "work directory. Use '.' to list the work directory root."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path relative to the work directory.",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read the UTF-8 text content of a file within the configured "
                    "work directory."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to the work directory.",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Write UTF-8 text content to a file within the configured work "
                    "directory. Parent directories are created automatically and "
                    "existing files are overwritten."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Destination file path relative to the work directory.",
                        },
                        "content": {
                            "type": "string",
                            "description": "UTF-8 text to write.",
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": (
                    "Replace a unique occurrence of old_str with new_str in a file "
                    "within the configured work directory."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to the work directory.",
                        },
                        "old_str": {
                            "type": "string",
                            "description": "Exact string to replace. Must appear exactly once.",
                        },
                        "new_str": {
                            "type": "string",
                            "description": "Replacement string.",
                        },
                    },
                    "required": ["path", "old_str", "new_str"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    def dispatch_tool(name: str, arguments: dict[str, Any]) -> Any:
        if name in file_tool_dispatch:
            return file_tool_dispatch[name](**arguments)
        if name == "pysilicon_validate_schema":
            return validate_schema_headless(**arguments)
        return REGISTRY.dispatch(name, arguments)

    return tool_schemas, dispatch_tool


def _usage_to_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _write_session_report(
    *,
    output_path: Path,
    task: str,
    model: str,
    max_rounds: int,
    final_response: str,
    api_calls: list[dict[str, Any]],
    token_totals: dict[str, int],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_session_report(
            task=task,
            model=model,
            max_rounds=max_rounds,
            final_response=final_response,
            api_calls=api_calls,
            token_totals=token_totals,
        ),
        encoding="utf-8",
    )
    return output_path.resolve()


def _render_session_report(
    *,
    task: str,
    model: str,
    max_rounds: int,
    final_response: str,
    api_calls: list[dict[str, Any]],
    token_totals: dict[str, int],
) -> str:
    lines = [
        "Headless LLM session report",
        "=",
        f"Task: {task}",
        f"Model: {model}",
        f"Max rounds: {max_rounds}",
        "",
        "Token totals",
        "-",
        f"Prompt tokens: {token_totals['prompt_tokens']}",
        f"Completion tokens: {token_totals['completion_tokens']}",
        f"Total tokens: {token_totals['total_tokens']}",
        "",
        "API calls",
        "-",
    ]

    for api_call in api_calls:
        usage = api_call["usage"]
        lines.extend(
            [
                f"Round {api_call['round']}",
                f"  Prompt tokens: {usage['prompt_tokens']}",
                f"  Completion tokens: {usage['completion_tokens']}",
                f"  Total tokens: {usage['total_tokens']}",
            ]
        )

        tool_calls = api_call["tool_calls"]
        if tool_calls:
            lines.append("  Tool calls:")
            for index, tool_call in enumerate(tool_calls, start=1):
                lines.extend(
                    [
                        f"    {index}. {tool_call['name']}",
                        "       Arguments:",
                        _indent_block(json.dumps(tool_call["arguments"], indent=2), "         "),
                        "       Result:",
                        _indent_block(tool_call["result_text"], "         "),
                    ]
                )
        else:
            lines.append("  Tool calls: none")

        lines.append("")

    lines.extend(
        [
            "Final response",
            "-",
            final_response,
            "",
        ]
    )
    return "\n".join(lines)


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines() or [""])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Headless LLM harness for pysilicon MCP tools. "
            "Simulates a user exercising the tools outside an editor host."
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
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to a text file where the full session report will be written.",
    )
    parser.add_argument(
        "--mode",
        choices=("workspace", "headless"),
        default=None,
        help=(
            "Optional MCP profile to mirror. Use 'headless' to expose the same "
            "tool surface as the standalone headless MCP server."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Root directory for headless file tools. Defaults to the current "
            "working directory when --mode headless is used."
        ),
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
            task = input("Enter task for the headless session: ").strip()
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
        output_path=args.output,
        mode=args.mode,
        work_dir=args.work_dir,
    )

    print("\n--- Final response ---")
    print(result["final_response"])
    if result["output_path"]:
        print(f"\nSession report written to {result['output_path']}")


if __name__ == "__main__":
    main()