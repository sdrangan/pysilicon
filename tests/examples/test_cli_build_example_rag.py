from __future__ import annotations

import sys
from types import SimpleNamespace

import pysilicon.mcp.cli_build_example_rag as rag_cli


def test_build_example_rag_deletes_previous_store_from_env(monkeypatch):
    delete_calls: list[str] = []

    class FakeFilesApi:
        def __init__(self):
            self.count = 0

        def create(self, *, file, purpose):
            assert purpose == "assistants"
            self.count += 1
            return SimpleNamespace(id=f"file_{self.count}")

    class FakeVectorStoresApi:
        def create(self, *, name, file_ids):
            assert name == "pysilicon-examples"
            assert file_ids
            return SimpleNamespace(id="vs_new")

        def retrieve(self, vector_store_id):
            assert vector_store_id == "vs_new"
            return SimpleNamespace(
                id="vs_new",
                status="completed",
                file_counts=SimpleNamespace(completed=4, failed=0),
            )

        def delete(self, vector_store_id):
            delete_calls.append(vector_store_id)
            return SimpleNamespace(id=vector_store_id, deleted=True)

    fake_client = SimpleNamespace(files=FakeFilesApi(), vector_stores=FakeVectorStoresApi())
    fake_openai = SimpleNamespace(OpenAI=lambda: fake_client)

    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("PYSILICON_EXAMPLES_VECTOR_STORE_ID", "vs_old")
    monkeypatch.setattr(rag_cli, "_enumerate_example_files", lambda: [("poly.py", "x = 1")])
    monkeypatch.setattr(rag_cli, "_generate_catalog", lambda files: "catalog")
    monkeypatch.setattr(rag_cli, "_wait_for_vector_store", lambda client, vs_id: None)

    assert rag_cli.build_example_rag(verbose=False) == "vs_new"
    assert delete_calls == ["vs_old"]


def test_print_env_var_instructions_shows_unix_and_powershell_commands(capsys):
    rag_cli._print_env_var_instructions("PYSILICON_EXAMPLES_VECTOR_STORE_ID", "vs_test_123")

    captured = capsys.readouterr()
    output = captured.out

    assert "Unix/Linux/macOS (current shell):" in output
    assert "export PYSILICON_EXAMPLES_VECTOR_STORE_ID=vs_test_123" in output
    assert "PowerShell (current session):" in output
    assert '$env:PYSILICON_EXAMPLES_VECTOR_STORE_ID = "vs_test_123"' in output
    assert "PowerShell (persist for future sessions):" in output
    assert 'setx PYSILICON_EXAMPLES_VECTOR_STORE_ID "vs_test_123"' in output