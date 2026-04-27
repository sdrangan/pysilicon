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
1. Enumerates text files under the ``pysilicon.mcp.corpus`` package.
2. Optionally generates an in-memory catalog markdown summary.
3. Uploads each file to OpenAI (Files API).
4. Creates a new vector store and adds all uploaded files.
5. Waits for the vector store to finish processing.
6. Prints the resulting ``vector_store_id`` and basic stats.

After running, set the printed ID in your shell or MCP server configuration.

Requirements
------------
``OPENAI_API_KEY`` must be set in the environment.
"""
from __future__ import annotations

import io
import os
import sys
import time
from importlib import resources
from pathlib import PurePosixPath

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXAMPLES_PACKAGE = "pysilicon.mcp.corpus"
_VECTOR_STORE_NAME = "pysilicon-examples"
_VECTOR_STORE_ENV = "PYSILICON_EXAMPLES_VECTOR_STORE_ID"
_OPENAI_SUPPORTED_SUFFIXES = frozenset({".py", ".cpp", ".c", ".h", ".md"})
_PATH_ESCAPE = "__pysilicon_path__"


def _print_env_var_instructions(var_name: str, value: str) -> None:
    """Print cross-platform commands for setting an environment variable."""
    print(f"To use this vector store, set {var_name} with one of these commands:")
    print("  Unix/Linux/macOS (current shell):")
    print(f"    export {var_name}={value}")
    print("  PowerShell (current session):")
    print(f'    $env:{var_name} = "{value}"')
    print("  PowerShell (persist for future sessions):")
    print(f'    setx {var_name} "{value}"')


def _delete_vector_store_if_present(client: "openai.OpenAI", vector_store_id: str, *, verbose: bool) -> None:  # type: ignore[name-defined]
    """Best-effort deletion of an existing vector store by ID."""
    if verbose:
        print(f"Deleting previous vector store '{vector_store_id}' ...", end="", flush=True)
    try:
        response = client.vector_stores.delete(vector_store_id)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(" failed")
        print(
            f"[warn] Could not delete previous vector store {vector_store_id!r}: {exc}",
            file=sys.stderr,
        )
        return

    deleted = getattr(response, "deleted", None)
    if verbose:
        if deleted is False:
            print(" not confirmed")
        else:
            print(" done")


def _walk_corpus_files(root, prefix: str = "") -> list[tuple[str, str]]:
    """Return ``(relative_path, text)`` pairs for text files under *root*."""
    results: list[tuple[str, str]] = []
    for resource in root.iterdir():
        name = getattr(resource, "name", str(resource))
        if name.startswith("_"):
            continue
        rel_path = f"{prefix}/{name}" if prefix else name
        if resource.is_dir():
            results.extend(_walk_corpus_files(resource, rel_path))
            continue
        try:
            content = resource.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] Could not read {rel_path}: {exc}", file=sys.stderr)
            continue
        results.append((rel_path, content))
    return results


def _enumerate_example_files() -> list[tuple[str, str]]:
    """Return ``(relative_path, text_content)`` pairs for packaged corpus files."""
    pkg = resources.files(_EXAMPLES_PACKAGE)
    return sorted(_walk_corpus_files(pkg), key=lambda t: t[0])


def _generate_catalog(files: list[tuple[str, str]]) -> str:
    """Generate a brief markdown catalog of the packaged corpus."""
    lines = [
        "# pysilicon examples catalog",
        "",
        "This catalog describes the packaged pysilicon MCP corpus.",
        "It includes example code, supporting tests, and documentation snippets.",
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


def _split_uploadable_files(
    files: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Split packaged corpus files into uploadable and skipped path lists."""
    uploadable: list[tuple[str, str]] = []
    skipped: list[str] = []
    for name, content in files:
        suffix = PurePosixPath(name).suffix.lower()
        if suffix in _OPENAI_SUPPORTED_SUFFIXES:
            uploadable.append((name, content))
        else:
            skipped.append(name)
    return uploadable, skipped


def encode_upload_filename(path: str) -> str:
    """Encode a corpus-relative path into a filename safe for OpenAI upload."""
    if "/" not in path:
        return path
    return path.replace("/", _PATH_ESCAPE)


def decode_upload_filename(filename: str) -> str:
    """Decode an uploaded filename back to a corpus-relative path."""
    if _PATH_ESCAPE not in filename:
        return filename
    return filename.replace(_PATH_ESCAPE, "/")


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
    previous_vs_id = os.environ.get(_VECTOR_STORE_ENV)

    # 1. Enumerate packaged corpus files
    if verbose:
        print(f"Scanning packaged corpus files from '{_EXAMPLES_PACKAGE}' ...")
    files = _enumerate_example_files()
    if not files:
        raise SystemExit("No corpus files found in the package.")
    files, skipped_files = _split_uploadable_files(files)
    if not files:
        raise SystemExit("No uploadable corpus files found in the package.")
    if verbose:
        for name, content in files:
            print(f"  {name}  ({len(content)} chars)")
        for name in skipped_files:
            print(f"  [skip] {name} (unsupported by OpenAI Files API)")

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
        file_obj.name = encode_upload_filename(fname)
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

    if previous_vs_id and previous_vs_id != vs_id:
        _delete_vector_store_if_present(client, previous_vs_id, verbose=verbose)
        if verbose:
            print()

    if verbose:
        _print_env_var_instructions("PYSILICON_EXAMPLES_VECTOR_STORE_ID", vs_id)

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
