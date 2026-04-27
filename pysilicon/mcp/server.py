from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from pysilicon.mcp.registry import REGISTRY


def build_mcp(
    mode: str = "workspace",
    work_dir: str | os.PathLike[str] | None = None,
) -> FastMCP:
    """Create and configure a pysilicon MCP server for the given *mode*.

    Parameters
    ----------
    mode:
        ``"workspace"`` – for hosts (VS Code, Claude Code, …) that already
        provide workspace file and editing tools.  Only domain-specific
        helpers (schema drafting/validation, component glossary) are
        registered; generic file tools and the RAG search tool are **not**
        exposed.

        ``"headless"`` – for standalone execution (unit tests, CI, API
        calls, …) where the host does not supply file tools.  Exposes all
        workspace-mode tools **plus** generic file tools scoped to
        *work_dir* and the ``pysilicon_rag_search_examples`` tool.

    work_dir:
        Root directory for file tools.  **Required** when ``mode="headless"``.
        All file-tool paths are resolved relative to (and must stay within)
        this directory.

    Returns
    -------
    FastMCP
        A fully configured MCP server instance ready to be run with
        ``mcp.run(transport="stdio")``.

    Raises
    ------
    ValueError
        If *mode* is not ``"workspace"`` or ``"headless"``, or if
        ``mode="headless"`` but *work_dir* is ``None``.
    """
    if mode not in ("workspace", "headless"):
        raise ValueError(
            f"mode must be 'workspace' or 'headless', got {mode!r}"
        )
    if mode == "headless" and work_dir is None:
        raise ValueError("work_dir is required for headless mode")

    mcp_instance = FastMCP("pysilicon")
    REGISTRY.register_all(mcp_instance, profile=mode)

    if mode == "headless":
        from pysilicon.mcp.file_tools import make_file_tools

        work_root = Path(work_dir).resolve()  # type: ignore[arg-type]
        list_files_fn, read_file_fn, write_file_fn, edit_file_fn = make_file_tools(
            work_root
        )

        mcp_instance.tool(
            name="list_files",
            description=(
                "List files and directories under a path within the configured "
                "work directory. path defaults to the work directory root."
            ),
        )(list_files_fn)
        mcp_instance.tool(
            name="read_file",
            description=(
                "Read the UTF-8 text content of a file within the configured "
                "work directory."
            ),
        )(read_file_fn)
        mcp_instance.tool(
            name="write_file",
            description=(
                "Write UTF-8 text content to a file within the configured "
                "work directory. Parent directories are created automatically. "
                "Existing files are overwritten."
            ),
        )(write_file_fn)
        mcp_instance.tool(
            name="edit_file",
            description=(
                "Replace a unique occurrence of old_str with new_str in a file "
                "within the configured work directory. Fails if old_str is not "
                "found or appears more than once."
            ),
        )(edit_file_fn)

    return mcp_instance


# Module-level instance used by the stdio entrypoint and backward-compatible
# imports (e.g. ``from pysilicon.mcp.server import mcp``).
mcp = build_mcp(mode="workspace")


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
