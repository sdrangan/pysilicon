"""
OpenAI vector-store backed search over the packaged pysilicon examples corpus.

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
"""
from __future__ import annotations

import os
from typing import Any

from pysilicon.mcp.cli_build_example_rag import decode_upload_filename

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_STORE_ENV = "PYSILICON_EXAMPLES_VECTOR_STORE_ID"

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

