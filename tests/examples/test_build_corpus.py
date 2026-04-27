from __future__ import annotations

import json
from pathlib import Path

import pysilicon.mcp.build_corpus as build_corpus_mod


def test_build_corpus_converts_unsupported_extensions(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    corpus = repo / "pysilicon" / "mcp" / "corpus"
    examples = repo / "examples" / "demo"
    tests_examples = repo / "tests" / "examples"
    docs_examples = repo / "docs" / "examples"

    examples.mkdir(parents=True)
    tests_examples.mkdir(parents=True)
    docs_examples.mkdir(parents=True)

    (examples / "kernel.hpp").write_text("#pragma once\nint demo();\n", encoding="utf-8")
    (examples / "run.tcl").write_text("puts \"hello\"\n", encoding="utf-8")
    (examples / "driver.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    monkeypatch.setattr(build_corpus_mod, "_repo_root", lambda: repo)
    monkeypatch.setattr(build_corpus_mod, "_corpus_root", lambda: corpus)
    monkeypatch.setattr(build_corpus_mod, "_collect_files", lambda src, _repo: sorted(src.rglob("*")))

    result = build_corpus_mod.build_corpus(verbose=False)

    assert result == corpus
    assert (corpus / "examples" / "demo" / "kernel.hpp.md").exists()
    assert (corpus / "examples" / "demo" / "run.tcl.md").exists()
    assert (corpus / "examples" / "demo" / "driver.cpp").exists()

    kernel_md = (corpus / "examples" / "demo" / "kernel.hpp.md").read_text(encoding="utf-8")
    assert "# Source: examples/demo/kernel.hpp" in kernel_md
    assert "Original extension: `.hpp`" in kernel_md
    assert "```cpp" in kernel_md
    assert "#pragma once" in kernel_md

    tcl_md = (corpus / "examples" / "demo" / "run.tcl.md").read_text(encoding="utf-8")
    assert "# Source: examples/demo/run.tcl" in tcl_md
    assert "Original extension: `.tcl`" in tcl_md
    assert "```tcl" in tcl_md
    assert 'puts "hello"' in tcl_md

    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    destinations = {entry["destination"].replace("\\", "/") for entry in manifest["files"]}
    assert "examples/demo/kernel.hpp.md" in destinations
    assert "examples/demo/run.tcl.md" in destinations
    assert "examples/demo/driver.cpp" in destinations