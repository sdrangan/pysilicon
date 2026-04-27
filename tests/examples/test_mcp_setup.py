"""
Tests for pysilicon_mcp_setup: --build-rag flag behaviour (no network calls).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from pysilicon.scripts.pysilicon_mcp_setup import (
    build_parser,
    main,
    render_mcp_config,
    write_mcp_config,
)


# ---------------------------------------------------------------------------
# render_mcp_config with vector_store_id
# ---------------------------------------------------------------------------


def test_render_mcp_config_without_vector_store_id():
    """Rendered config should not contain env key when no vector_store_id given."""
    config_str = render_mcp_config(python_path="/usr/bin/python3")
    config = json.loads(config_str)
    server = config["servers"]["pysilicon"]
    assert "env" not in server or "PYSILICON_EXAMPLES_VECTOR_STORE_ID" not in server.get("env", {})


def test_render_mcp_config_with_vector_store_id():
    """Rendered config should include PYSILICON_EXAMPLES_VECTOR_STORE_ID in env."""
    config_str = render_mcp_config(python_path="/usr/bin/python3", vector_store_id="vs_test123")
    config = json.loads(config_str)
    server = config["servers"]["pysilicon"]
    assert "env" in server
    assert server["env"]["PYSILICON_EXAMPLES_VECTOR_STORE_ID"] == "vs_test123"


# ---------------------------------------------------------------------------
# write_mcp_config with vector_store_id
# ---------------------------------------------------------------------------


def test_write_mcp_config_with_vector_store_id(tmp_path):
    """Written file should contain the vector store ID when provided."""
    output_path = write_mcp_config(
        workspace=tmp_path,
        python_path="/usr/bin/python3",
        vector_store_id="vs_written999",
    )
    config = json.loads(output_path.read_text())
    assert config["servers"]["pysilicon"]["env"]["PYSILICON_EXAMPLES_VECTOR_STORE_ID"] == "vs_written999"


# ---------------------------------------------------------------------------
# --build-rag flag: missing OPENAI_API_KEY
# ---------------------------------------------------------------------------


def test_main_build_rag_missing_api_key_exits_nonzero(tmp_path, monkeypatch, capsys):
    """main() should return 1 and print a clear error when OPENAI_API_KEY is absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("pysilicon.scripts.pysilicon_mcp_setup.validate_python_interpreter"):
        with patch("sys.argv", ["pysilicon_mcp_setup", "--workspace", str(tmp_path), "--build-rag"]):
            result = main()

    assert result == 1
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY" in captured.err


# ---------------------------------------------------------------------------
# --build-rag flag: successful build (mocked)
# ---------------------------------------------------------------------------


def test_main_build_rag_writes_vector_store_id(tmp_path, monkeypatch):
    """main() with --build-rag should call build_example_rag and persist the returned ID."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")

    with patch("pysilicon.scripts.pysilicon_mcp_setup.validate_python_interpreter"):
        with patch(
            "pysilicon.scripts.pysilicon_mcp_setup._build_example_rag",
            return_value="vs_mocked_abc",
        ):
            with patch("sys.argv", ["pysilicon_mcp_setup", "--workspace", str(tmp_path), "--build-rag"]):
                result = main()

    assert result == 0
    mcp_json = tmp_path / ".vscode" / "mcp.json"
    assert mcp_json.exists()
    config = json.loads(mcp_json.read_text())
    assert config["servers"]["pysilicon"]["env"]["PYSILICON_EXAMPLES_VECTOR_STORE_ID"] == "vs_mocked_abc"


# ---------------------------------------------------------------------------
# --build-rag + --dry-run: no file written, config printed with env
# ---------------------------------------------------------------------------


def test_main_build_rag_dry_run_does_not_write_file(tmp_path, monkeypatch, capsys):
    """--dry-run should print the config containing the env var but not write any file."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")

    with patch("pysilicon.scripts.pysilicon_mcp_setup.validate_python_interpreter"):
        with patch(
            "pysilicon.scripts.pysilicon_mcp_setup._build_example_rag",
            return_value="vs_dry_run_id",
        ):
            with patch(
                "sys.argv",
                ["pysilicon_mcp_setup", "--workspace", str(tmp_path), "--build-rag", "--dry-run"],
            ):
                result = main()

    assert result == 0
    # File must NOT have been created
    assert not (tmp_path / ".vscode" / "mcp.json").exists()
    # The printed output must contain the vector store ID
    captured = capsys.readouterr()
    config = json.loads(captured.out)
    assert config["servers"]["pysilicon"]["env"]["PYSILICON_EXAMPLES_VECTOR_STORE_ID"] == "vs_dry_run_id"


# ---------------------------------------------------------------------------
# --build-rag + --force: overwrites existing file
# ---------------------------------------------------------------------------


def test_main_build_rag_force_overwrites_existing(tmp_path, monkeypatch):
    """--force allows overwriting an existing .vscode/mcp.json."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")

    vscode_dir = tmp_path / ".vscode"
    vscode_dir.mkdir()
    existing = vscode_dir / "mcp.json"
    existing.write_text('{"servers": {}}')

    with patch("pysilicon.scripts.pysilicon_mcp_setup.validate_python_interpreter"):
        with patch(
            "pysilicon.scripts.pysilicon_mcp_setup._build_example_rag",
            return_value="vs_forced_id",
        ):
            with patch(
                "sys.argv",
                ["pysilicon_mcp_setup", "--workspace", str(tmp_path), "--build-rag", "--force"],
            ):
                result = main()

    assert result == 0
    config = json.loads(existing.read_text())
    assert config["servers"]["pysilicon"]["env"]["PYSILICON_EXAMPLES_VECTOR_STORE_ID"] == "vs_forced_id"
