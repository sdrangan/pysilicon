"""
CLI entrypoint: build or refresh the OpenAI vector store for pysilicon examples.

Usage
-----
As a console script (after ``pip install pysilicon``)::

    pysilicon-build-example-rag

Or directly::

    python -m pysilicon.mcp.cli_build_example_rag

What it does
------------
1. Enumerates Python files under the ``pysilicon.examples`` package.
2. Optionally generates an in-memory catalog markdown summary.
3. Uploads each file to OpenAI (Files API).
4. Creates a new vector store and adds all uploaded files.
5. Waits for the vector store to finish processing.
6. Prints the resulting ``vector_store_id`` and basic stats.

After running, export the printed ID::

    export PYSILICON_EXAMPLES_VECTOR_STORE_ID=<id>

and optionally persist it in your ``.env`` file or MCP server configuration.

Requirements
------------
``OPENAI_API_KEY`` must be set in the environment.
"""
from __future__ import annotations

import io
import sys
import time
from importlib import resources

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXAMPLES_PACKAGE = "pysilicon.examples"
_VECTOR_STORE_NAME = "pysilicon-examples"


def _enumerate_example_files() -> list[tuple[str, str]]:
    """Return a list of (filename, text_content) for each example .py file."""
    pkg = resources.files(_EXAMPLES_PACKAGE)
    results: list[tuple[str, str]] = []
    for resource in pkg.iterdir():
        name = getattr(resource, "name", str(resource))
        if name.startswith("_") or not name.endswith(".py"):
            continue
        try:
            content = resource.read_text(encoding="utf-8")
            results.append((name, content))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] Could not read {name}: {exc}", file=sys.stderr)
    return sorted(results, key=lambda t: t[0])


def _generate_catalog(files: list[tuple[str, str]]) -> str:
    """Generate a brief markdown catalog of the examples corpus."""
    lines = [
        "# pysilicon examples catalog",
        "",
        "This catalog describes the curated schema examples shipped with pysilicon.",
        "Each file defines DataList/DataArray schemas for an accelerator interface.",
        "",
        "## Files",
        "",
    ]
    for name, content in files:
        # Extract module docstring as description if present
        doc_line = ""
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                doc_line = stripped.strip('"\' ')
                if doc_line:
                    break
        lines.append(f"### {name}")
        if doc_line:
            lines.append(f"{doc_line}")
        lines.append("")
    return "\n".join(lines)


def _wait_for_vector_store(client: "openai.OpenAI", vs_id: str, timeout: int = 120) -> None:  # type: ignore[name-defined]
    """Poll until vector store status is 'completed' or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        vs = client.vector_stores.retrieve(vs_id)
        status = getattr(vs, "status", None)
        if status == "completed":
            return
        if status in ("expired", "failed"):
            raise RuntimeError(f"Vector store {vs_id!r} ended with status {status!r}.")
        time.sleep(3)
    raise TimeoutError(
        f"Vector store {vs_id!r} did not reach 'completed' status within {timeout}s."
    )


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def build_example_rag(*, verbose: bool = True) -> str:
    """Build (or rebuild) the OpenAI vector store for pysilicon examples.

    Returns the vector store ID string.
    """
    try:
        import openai
    except ImportError as exc:
        raise SystemExit("openai package is required: pip install openai") from exc

    client = openai.OpenAI()  # reads OPENAI_API_KEY from env

    # 1. Enumerate example files
    if verbose:
        print(f"Scanning packaged examples from '{_EXAMPLES_PACKAGE}' ...")
    files = _enumerate_example_files()
    if not files:
        raise SystemExit("No example files found in the package.")
    if verbose:
        for name, content in files:
            print(f"  {name}  ({len(content)} chars)")

    # 2. Generate catalog
    catalog_md = _generate_catalog(files)
    all_uploads: list[tuple[str, str]] = [("_CATALOG.md", catalog_md)] + files
    if verbose:
        print(f"\nPreparing {len(all_uploads)} files for upload (including catalog) ...")

    # 3. Upload files to OpenAI
    uploaded_ids: list[str] = []
    for fname, content in all_uploads:
        if verbose:
            print(f"  Uploading {fname} ...", end="", flush=True)
        encoded = content.encode("utf-8")
        file_obj = io.BytesIO(encoded)
        file_obj.name = fname  # openai uses .name as the filename
        response = client.files.create(file=file_obj, purpose="assistants")
        uploaded_ids.append(response.id)
        if verbose:
            print(f" -> {response.id}")

    # 4. Create vector store
    if verbose:
        print(f"\nCreating vector store '{_VECTOR_STORE_NAME}' ...")
    vs = client.vector_stores.create(
        name=_VECTOR_STORE_NAME,
        file_ids=uploaded_ids,
    )
    vs_id: str = vs.id
    if verbose:
        print(f"  Vector store ID: {vs_id}")
        print("  Waiting for processing to complete ...")

    # 5. Wait for processing
    _wait_for_vector_store(client, vs_id)

    # 6. Stats
    vs_final = client.vector_stores.retrieve(vs_id)
    file_counts = getattr(vs_final, "file_counts", None)
    if verbose:
        print("\n✓ Vector store ready.")
        print(f"  ID:              {vs_id}")
        if file_counts is not None:
            print(f"  Files completed: {getattr(file_counts, 'completed', '?')}")
            print(f"  Files failed:    {getattr(file_counts, 'failed', '?')}")
        print()
        print("To use this vector store, set the environment variable:")
        print(f"  export PYSILICON_EXAMPLES_VECTOR_STORE_ID={vs_id}")

    return vs_id


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    """Console-script entry point for ``pysilicon-build-example-rag``."""
    vs_id = build_example_rag(verbose=True)
    # Print just the ID on stdout last so scripts can capture it
    print(vs_id)


if __name__ == "__main__":
    main()
