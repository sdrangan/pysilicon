#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

from pysilicon.mcp.cli_build_example_rag import build_example_rag as _build_example_rag


TEMPLATE_PACKAGE = "pysilicon.mcp.resources"
TEMPLATE_NAME = "mcp.json"
PYTHON_PLACEHOLDER = "__PYSILICON_PYTHON__"
IMPORT_PROBE = "import pysilicon.mcp.server"


def load_template() -> dict:
    template_text = resources.files(TEMPLATE_PACKAGE).joinpath(TEMPLATE_NAME).read_text(
        encoding="utf-8"
    )
    return json.loads(template_text)


def render_mcp_config(*, python_path: str, vector_store_id: str | None = None) -> str:
    config = load_template()
    servers = config.setdefault("servers", {})

    if "pysilicon" not in servers:
        copied_server = None
        if len(servers) == 1:
            copied_server = next(iter(servers.values()))

        servers["pysilicon"] = copied_server or {
            "type": "stdio",
            "args": ["-m", "pysilicon.mcp.server"],
        }

    servers["pysilicon"]["command"] = python_path

    if vector_store_id is not None:
        servers["pysilicon"].setdefault("env", {})["PYSILICON_EXAMPLES_VECTOR_STORE_ID"] = vector_store_id

    return json.dumps(config, indent=2) + "\n"


def validate_python_interpreter(python_path: str) -> None:
    try:
        completed = subprocess.run(
            [python_path, "-c", IMPORT_PROBE],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"The selected Python interpreter is not runnable: {python_path}\n{exc}"
        ) from exc

    if completed.returncode == 0:
        return

    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    details = stderr or stdout or "Unknown error while importing pysilicon.mcp.server"
    raise RuntimeError(
        "The selected Python interpreter cannot import pysilicon.mcp.server: "
        f"{python_path}\n{details}"
    )


def write_mcp_config(
    *, workspace: Path, python_path: str, force: bool = False, vector_store_id: str | None = None
) -> Path:
    vscode_dir = workspace / ".vscode"
    target_path = vscode_dir / "mcp.json"

    if target_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing MCP config: {target_path}. Use --force to replace it."
        )

    vscode_dir.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        render_mcp_config(python_path=python_path, vector_store_id=vector_store_id),
        encoding="utf-8",
    )
    return target_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a VS Code MCP configuration for pysilicon in a workspace."
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace folder where .vscode/mcp.json will be created. Defaults to the current directory.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for the MCP server. Defaults to the interpreter running this command.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing .vscode/mcp.json file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated config instead of writing it.",
    )
    parser.add_argument(
        "--build-rag",
        action="store_true",
        help=(
            "Build the OpenAI vector store for pysilicon examples and write the "
            "resulting ID into .vscode/mcp.json under servers.pysilicon.env. "
            "Requires OPENAI_API_KEY to be set in the environment."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    python_path = str(Path(args.python).resolve())

    validate_python_interpreter(python_path)

    vector_store_id: str | None = None
    if args.build_rag:
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "Error: OPENAI_API_KEY environment variable is not set.\n"
                "Set it before running --build-rag:\n"
                "  export OPENAI_API_KEY=sk-...",
                file=sys.stderr,
            )
            return 1
        vector_store_id = _build_example_rag(verbose=True)

    if args.dry_run:
        print(render_mcp_config(python_path=python_path, vector_store_id=vector_store_id), end="")
        return 0

    output_path = write_mcp_config(
        workspace=workspace,
        python_path=python_path,
        force=args.force,
        vector_store_id=vector_store_id,
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())