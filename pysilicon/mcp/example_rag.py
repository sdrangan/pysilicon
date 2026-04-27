"""
OpenAI vector-store backed search over the packaged pysilicon examples corpus,
plus a helper for reading individual example files from the package.

Environment variables
---------------------
PYSILICON_EXAMPLES_VECTOR_STORE_ID
    The OpenAI vector store ID created by ``pysilicon-build-example-rag``.
    Required for ``search_schema_examples``.
OPENAI_API_KEY
    Standard OpenAI API key (used automatically by the ``openai`` library).

Tools exposed
-------------
search_schema_examples(task, keywords, k)
    Search the vector store for examples relevant to *task*.
    Returns a structured error dict when the env var is missing; never
    raises so that the MCP host can surface the message gracefully.

get_example_file(path)
    Return the full content of a packaged example file by its path
    relative to the ``pysilicon.examples`` package root.
"""
from __future__ import annotations

import os
from importlib import resources
from typing import Any

from pysilicon.mcp.cli_build_example_rag import decode_upload_filename

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_STORE_ENV = "PYSILICON_EXAMPLES_VECTOR_STORE_ID"
_EXAMPLES_PACKAGE = "pysilicon.examples"

# ---------------------------------------------------------------------------
# search_schema_examples
# ---------------------------------------------------------------------------


def search_schema_examples(
    task: str,
    keywords: list[str] | None = None,
    k: int | None = 5,
) -> dict[str, Any]:
    """Search the OpenAI-hosted vector store for schema examples.

    When ``PYSILICON_EXAMPLES_VECTOR_STORE_ID`` is not set this function
    returns a structured error dict (it does *not* raise) so callers can
    surface a helpful message without crashing.

    Parameters
    ----------
    task:
        Natural-language description of what schema you want to build.
    keywords:
        Optional list of pysilicon vocabulary keywords (e.g. ``["DataList",
        "MemAddr", "transaction_id"]``) to augment the query.
    k:
        Maximum number of matches to return (default 5, capped at 20).
        ``None`` uses the default.

    Returns
    -------
    dict
        ``summary``          – human-readable status string.
        ``normalized_query`` – the query string sent to the vector store.
        ``matches``          – list of match dicts (empty on error).
        ``error``            – present only when something went wrong.
    """
    keywords = keywords or []
    if k is None:
        k = 5
    k = max(1, min(k, 20))

    # Build a normalised query string that combines task + keywords.
    if keywords:
        normalized_query = f"{task}. Keywords: {', '.join(keywords)}"
    else:
        normalized_query = task

    vector_store_id = os.environ.get(VECTOR_STORE_ENV, "").strip()
    if not vector_store_id:
        return {
            "summary": (
                f"Vector store ID not configured. "
                f"Set the {VECTOR_STORE_ENV!r} environment variable to the ID "
                "returned by `pysilicon-build-example-rag`, then retry."
            ),
            "normalized_query": normalized_query,
            "matches": [],
            "error": (
                f"Environment variable {VECTOR_STORE_ENV!r} is missing or empty. "
                "Run `pysilicon-build-example-rag` to create the vector store and "
                f"then export {VECTOR_STORE_ENV}=<id>."
            ),
        }

    try:
        from openai import OpenAI  # local import keeps module importable without openai
    except ImportError:
        return {
            "summary": "openai package is not installed.",
            "normalized_query": normalized_query,
            "matches": [],
            "error": "Install openai: pip install openai",
        }

    try:
        client = OpenAI()
        page = client.vector_stores.search(
            vector_store_id,
            query=normalized_query,
            max_num_results=k,
        )
        raw_results = list(page)
    except Exception as exc:  # noqa: BLE001
        return {
            "summary": f"Vector store search failed: {exc}",
            "normalized_query": normalized_query,
            "matches": [],
            "error": str(exc),
        }

    matches = []
    for item in raw_results:
        match: dict[str, Any] = {}
        # Extract file path from attributes if available
        attrs = getattr(item, "attributes", None) or {}
        file_path = attrs.get("path") or getattr(item, "filename", None)
        if file_path:
            match["path"] = decode_upload_filename(file_path)
        # Extract snippet text from content blocks
        content = getattr(item, "content", None) or []
        snippet_parts = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                snippet_parts.append(text)
        match["snippet"] = "\n".join(snippet_parts)
        # Include score if available
        score = getattr(item, "score", None)
        if score is not None:
            match["score"] = score
        matches.append(match)

    return {
        "summary": (
            f"Found {len(matches)} match(es) for query: {normalized_query!r}"
        ),
        "normalized_query": normalized_query,
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# get_example_file
# ---------------------------------------------------------------------------


def get_example_file(path: str) -> dict[str, Any]:
    """Return the full content of a packaged example file.

    Parameters
    ----------
    path:
        Path relative to the ``pysilicon.examples`` package root, e.g.
        ``"poly.py"`` or ``"conv2d.py"``.

    Returns
    -------
    dict
        ``path``    – the requested path (echoed back).
        ``content`` – full UTF-8 text content of the file.

    Raises
    ------
    ValueError
        If the file does not exist in the packaged examples.
    """
    # Prevent directory traversal
    if ".." in path or path.startswith("/"):
        raise ValueError(
            f"Invalid example path {path!r}: must be a relative path with no '..' components."
        )
    try:
        content = (
            resources.files(_EXAMPLES_PACKAGE)
            .joinpath(path)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, TypeError) as exc:
        available = _list_example_paths()
        raise ValueError(
            f"Example file not found: {path!r}. "
            f"Available examples: {available}. "
            f"Original error: {exc}"
        ) from exc
    return {"path": path, "content": content}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _list_example_paths() -> list[str]:
    """Return sorted list of example file paths relative to the package root."""
    try:
        pkg = resources.files(_EXAMPLES_PACKAGE)
        paths = []
        for resource in pkg.iterdir():
            name = getattr(resource, "name", str(resource))
            if name.startswith("_") or not name.endswith(".py"):
                continue
            paths.append(name)
        return sorted(paths)
    except Exception:  # noqa: BLE001
        return []
