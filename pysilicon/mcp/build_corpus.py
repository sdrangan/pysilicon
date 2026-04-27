"""
CLI entry point: build_corpus

Assembles a packaged corpus under ``pysilicon/mcp/corpus/`` for users who do
not have the full repository checkout but need to build the vector store
locally.

Source mappings
---------------
* ``<repo_root>/examples/``       -> corpus/examples/
* ``<repo_root>/tests/examples/`` -> corpus/tests/
* ``<repo_root>/docs/examples/``  -> corpus/docs/

Only files with these extensions are included::

    .py  .cpp  .c  .h  .hpp  .tcl  .md

Files that are useful for retrieval but not accepted directly by the OpenAI
Files API are converted during corpus build::

    .hpp -> .hpp.md
    .tcl -> .tcl.md

Git-tracked files are used when ``git`` is available (via ``git ls-files``);
otherwise the source trees are walked and the extension filter is applied.

Usage
-----
As a console script (after ``pip install -e .``)::

    build_corpus

Or directly::

    python -m pysilicon.mcp.build_corpus
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".cpp", ".c", ".h", ".hpp", ".tcl", ".md"}
)

UNSUPPORTED_UPLOAD_EXTENSIONS: frozenset[str] = frozenset({".hpp", ".tcl"})

MARKDOWN_CODE_FENCE_LANGUAGE: dict[str, str] = {
    ".hpp": "cpp",
    ".tcl": "tcl",
}

# Each tuple is (source_root_relative_to_repo, corpus_subdir_name)
SOURCE_MAPPINGS: list[tuple[str, str]] = [
    ("examples", "examples"),
    ("tests/examples", "tests"),
    ("docs/examples", "docs"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Return the repository root (the directory that contains pysilicon/)."""
    # This file lives at <repo>/pysilicon/mcp/build_corpus.py
    return Path(__file__).resolve().parents[2]


def _corpus_root() -> Path:
    """Return the corpus directory path (inside the installed package tree)."""
    return Path(__file__).resolve().parent / "corpus"


def _git_tracked_files(src: Path, repo: Path) -> list[Path] | None:
    """Return Git-tracked files under *src*, or None if Git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", str(src)],
            capture_output=True,
            text=True,
            cwd=repo,
        )
    except FileNotFoundError:
        return None  # git not installed
    if result.returncode != 0:
        return None
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        p = repo / line
        if p.is_file() and p.suffix in ALLOWED_EXTENSIONS:
            paths.append(p)
    return paths


def _walk_files(src: Path) -> list[Path]:
    """Walk *src* and return regular files with allowed extensions."""
    return [
        p
        for p in src.rglob("*")
        if p.is_file() and p.suffix in ALLOWED_EXTENSIONS
    ]


def _collect_files(src: Path, repo: Path) -> list[Path]:
    """Return files to copy from *src*, preferring Git-tracked list."""
    git_files = _git_tracked_files(src, repo)
    if git_files is not None:
        return git_files
    return _walk_files(src)


def _dest_relative_path(rel: Path) -> Path:
    """Return the destination path inside corpus for a source-relative path."""
    if rel.suffix in UNSUPPORTED_UPLOAD_EXTENSIONS:
        return rel.with_name(f"{rel.name}.md")
    return rel


def _render_converted_markdown(*, source_file: Path, repo: Path) -> str:
    """Render an unsupported source file as markdown for vector-store upload."""
    rel_source = source_file.relative_to(repo).as_posix()
    suffix = source_file.suffix.lower()
    code_lang = MARKDOWN_CODE_FENCE_LANGUAGE.get(suffix, "text")
    source_text = source_file.read_text(encoding="utf-8")
    return "\n".join(
        [
            f"# Source: {rel_source}",
            "",
            f"Original extension: `{suffix}`",
            "",
            f"```{code_lang}",
            source_text,
            "```",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def build_corpus(*, verbose: bool = True) -> Path:
    """Build the corpus directory and return its path."""
    repo = _repo_root()
    corpus = _corpus_root()

    # 1. Delete and recreate corpus directory
    if corpus.exists():
        shutil.rmtree(corpus)
        if verbose:
            print(f"Removed existing corpus: {corpus}")
    corpus.mkdir(parents=True)
    if verbose:
        print(f"Created corpus directory: {corpus}")

    manifest: list[dict[str, object]] = []

    # 2. Copy files for each source mapping
    for src_rel, dest_subdir in SOURCE_MAPPINGS:
        src = repo / src_rel
        if not src.exists():
            if verbose:
                print(f"  [skip] Source not found: {src}")
            continue

        dest_root = corpus / dest_subdir
        files = _collect_files(src, repo)

        if verbose:
            print(f"  {src_rel}/ -> corpus/{dest_subdir}/  ({len(files)} files)")

        for src_file in files:
            if not src_file.is_file():
                continue
            rel = src_file.relative_to(src)
            dest_rel = _dest_relative_path(rel)
            dest_file = dest_root / dest_rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if src_file.suffix in UNSUPPORTED_UPLOAD_EXTENSIONS:
                dest_file.write_text(
                    _render_converted_markdown(source_file=src_file, repo=repo),
                    encoding="utf-8",
                )
            else:
                shutil.copy2(src_file, dest_file)
            size = dest_file.stat().st_size
            manifest.append(
                {
                    "source": str(src_file.relative_to(repo)),
                    "destination": str(dest_file.relative_to(corpus)),
                    "size": size,
                }
            )
            if verbose:
                if src_file.suffix in UNSUPPORTED_UPLOAD_EXTENSIONS:
                    print(f"    {rel} -> {dest_rel}  ({size} bytes, converted)")
                else:
                    print(f"    {rel}  ({size} bytes)")

    # 3. Write manifest
    manifest_path = corpus / "manifest.json"
    manifest_path.write_text(
        json.dumps({"files": manifest, "total": len(manifest)}, indent=2),
        encoding="utf-8",
    )
    if verbose:
        print(f"\nWrote manifest ({len(manifest)} entries): {manifest_path}")

    return corpus


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    """Console-script entry point for ``build_corpus``."""
    corpus = build_corpus(verbose=True)
    print(f"\nCorpus ready: {corpus}")


if __name__ == "__main__":
    main()
