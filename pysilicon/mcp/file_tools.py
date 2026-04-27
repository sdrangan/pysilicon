"""Generic file tools for headless mode.

All tools returned by :func:`make_file_tools` are scoped to a single root
directory (*work_dir*).  Paths that would escape that root are rejected with
a :class:`ValueError` before any filesystem operation is attempted.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Path-safety helper
# ---------------------------------------------------------------------------


def _resolve_safe(root: Path, path: str) -> Path:
    """Resolve *path* relative to *root*, raising :class:`ValueError` if it escapes.

    Parameters
    ----------
    root:
        The allowed root directory (already resolved to an absolute path).
    path:
        A path supplied by the caller.  May be relative or absolute; if
        absolute it must already be inside *root*.

    Raises
    ------
    ValueError
        If *path* is empty or resolves to a location outside *root*.
    """
    if not path.strip():
        raise ValueError("Path must not be empty.")
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Path {path!r} escapes the allowed root {root!s}."
        ) from None
    return resolved


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_file_tools(
    work_dir: str | os.PathLike[str],
) -> tuple[Any, Any, Any, Any]:
    """Return ``(list_files, read_file, write_file, edit_file)`` scoped to *work_dir*.

    All four functions operate only within the resolved absolute path of
    *work_dir*.  Any attempt to access a path outside that root raises a
    :class:`ValueError` before any I/O occurs.

    Parameters
    ----------
    work_dir:
        The root directory for all file operations.  It does not need to
        exist yet (``write_file`` will create it on demand), but it will be
        resolved to an absolute path at call time.

    Returns
    -------
    tuple
        ``(list_files, read_file, write_file, edit_file)`` – four callables
        suitable for registration as MCP tools.
    """
    root = Path(work_dir).resolve()

    # ------------------------------------------------------------------
    # list_files
    # ------------------------------------------------------------------

    def list_files(path: str = ".") -> dict[str, Any]:
        """List files and directories under *path* within the work directory.

        Parameters
        ----------
        path:
            Directory path relative to the work directory (default: the work
            directory itself).

        Returns
        -------
        dict
            ``path``    – normalised relative path that was listed.
            ``entries`` – sorted list of entry dicts, each with ``name``,
                          ``path`` (relative to work_dir), ``type``
                          (``"file"`` or ``"directory"``), and ``size``
                          (bytes for files, ``null`` for directories).
            ``error``   – present only when the operation failed.
        """
        try:
            target = _resolve_safe(root, path)
        except ValueError as exc:
            return {"error": str(exc), "entries": []}

        if not target.exists():
            return {"error": f"Path {path!r} does not exist.", "entries": []}
        if not target.is_dir():
            return {"error": f"Path {path!r} is not a directory.", "entries": []}

        entries: list[dict[str, Any]] = []
        try:
            for child in sorted(target.iterdir()):
                entries.append(
                    {
                        "name": child.name,
                        "path": str(child.relative_to(root)),
                        "type": "directory" if child.is_dir() else "file",
                        "size": child.stat().st_size if child.is_file() else None,
                    }
                )
        except OSError as exc:
            return {"error": str(exc), "entries": []}

        return {"path": str(target.relative_to(root)), "entries": entries}

    # ------------------------------------------------------------------
    # read_file
    # ------------------------------------------------------------------

    def read_file(path: str) -> dict[str, Any]:
        """Read the UTF-8 text content of *path* within the work directory.

        Parameters
        ----------
        path:
            File path relative to the work directory.

        Returns
        -------
        dict
            ``path``    – normalised relative path.
            ``content`` – full UTF-8 text, or ``null`` on error.
            ``error``   – present only when the operation failed.
        """
        try:
            target = _resolve_safe(root, path)
        except ValueError as exc:
            return {"error": str(exc), "content": None}

        if not target.exists():
            return {"error": f"File {path!r} does not exist.", "content": None}
        if not target.is_file():
            return {"error": f"Path {path!r} is not a file.", "content": None}

        try:
            content = target.read_text(encoding="utf-8")
        except OSError as exc:
            return {"error": str(exc), "content": None}

        return {"path": str(target.relative_to(root)), "content": content}

    # ------------------------------------------------------------------
    # write_file
    # ------------------------------------------------------------------

    def write_file(path: str, content: str) -> dict[str, Any]:
        """Write *content* to *path* within the work directory.

        Parent directories are created automatically.  If the file already
        exists it is overwritten.

        Parameters
        ----------
        path:
            Destination file path relative to the work directory.
        content:
            UTF-8 text to write.

        Returns
        -------
        dict
            ``path`` – normalised relative path.
            ``ok``   – ``True`` on success.
            ``error`` – present only when the operation failed.
        """
        try:
            target = _resolve_safe(root, path)
        except ValueError as exc:
            return {"error": str(exc), "ok": False}

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return {"error": str(exc), "ok": False}

        return {"path": str(target.relative_to(root)), "ok": True}

    # ------------------------------------------------------------------
    # edit_file
    # ------------------------------------------------------------------

    def edit_file(path: str, old_str: str, new_str: str) -> dict[str, Any]:
        """Replace a unique occurrence of *old_str* with *new_str* in *path*.

        The replacement is performed only when *old_str* appears exactly once
        in the file, ensuring deterministic edits.

        Parameters
        ----------
        path:
            File path relative to the work directory.
        old_str:
            The exact string to replace.  Must appear exactly once.
        new_str:
            The replacement string.

        Returns
        -------
        dict
            ``path``  – normalised relative path.
            ``ok``    – ``True`` on success.
            ``error`` – present only when the operation failed (including
                        when *old_str* is not found or is not unique).
        """
        try:
            target = _resolve_safe(root, path)
        except ValueError as exc:
            return {"error": str(exc), "ok": False}

        if not target.exists():
            return {"error": f"File {path!r} does not exist.", "ok": False}

        try:
            original = target.read_text(encoding="utf-8")
        except OSError as exc:
            return {"error": str(exc), "ok": False}

        count = original.count(old_str)
        if count == 0:
            return {"error": f"old_str not found in {path!r}.", "ok": False}
        if count > 1:
            return {
                "error": f"old_str found {count} times in {path!r}; must be unique.",
                "ok": False,
            }

        updated = original.replace(old_str, new_str, 1)
        try:
            target.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return {"error": str(exc), "ok": False}

        return {"path": str(target.relative_to(root)), "ok": True}

    return list_files, read_file, write_file, edit_file
