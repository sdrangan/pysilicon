from pathlib import Path

from pysilicon.utils import toolchain
import shutil
import pytest
import subprocess


def test_find_vitis_path_prefers_env_direct_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain.platform, "system", lambda: "Windows")

    exe = tmp_path / "vitis-run.bat"
    exe.write_text("@echo off\n", encoding="utf-8")

    monkeypatch.setenv("PYSILICON_VITIS_PATH", str(exe))
    out = toolchain.find_vitis_path()

    assert out == str(exe.resolve())


def test_find_vitis_path_selects_highest_version_under_root(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain.platform, "system", lambda: "Windows")
    monkeypatch.delenv("PYSILICON_VITIS_PATH", raising=False)

    older = tmp_path / "2024.2" / "bin" / "vitis-run.bat"
    newer = tmp_path / "2025.1" / "bin" / "vitis-run.bat"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text("@echo off\n", encoding="utf-8")
    newer.write_text("@echo off\n", encoding="utf-8")

    out = toolchain.find_vitis_path(top_dir=tmp_path)

    assert out == str(newer.resolve())


def test_find_vitis_path_windows_nested_vitis_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain.platform, "system", lambda: "Windows")
    monkeypatch.delenv("PYSILICON_VITIS_PATH", raising=False)

    older = tmp_path / "2024.2" / "Vitis" / "bin" / "vitis-run.bat"
    newer = tmp_path / "2025.2" / "Vitis" / "bin" / "vitis-run.bat"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text("@echo off\n", encoding="utf-8")
    newer.write_text("@echo off\n", encoding="utf-8")

    out = toolchain.find_vitis_path(top_dir=tmp_path)

    assert out == str(newer.resolve())


def test_find_vitis_path_linux_uses_env_root(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain.platform, "system", lambda: "Linux")

    v1 = tmp_path / "2024.1" / "bin" / "vitis-run"
    v2 = tmp_path / "2024.2" / "bin" / "vitis-run"
    v1.parent.mkdir(parents=True)
    v2.parent.mkdir(parents=True)
    v1.write_text("#!/bin/sh\n", encoding="utf-8")
    v2.write_text("#!/bin/sh\n", encoding="utf-8")
    v1.chmod(0o755)
    v2.chmod(0o755)

    monkeypatch.setenv("PYSILICON_VITIS_PATH", str(tmp_path))
    out = toolchain.find_vitis_path()

    assert out == str(v2.resolve())


def test_run_vitis_hls_result_passed(monkeypatch, tmp_path):
    expected = subprocess.CompletedProcess(
        args=["vitis-run"],
        returncode=0,
        stdout="ok\n",
        stderr="",
    )

    def fake_run_vitis_hls(tcl_script, work_dir=None, args=None, capture_output=True):
        assert tcl_script == tmp_path / "run.tcl"
        assert work_dir == tmp_path
        assert args == ["alpha", "beta"]
        assert capture_output is True
        return expected

    monkeypatch.setattr(toolchain, "run_vitis_hls", fake_run_vitis_hls)

    out = toolchain.run_vitis_hls_result(
        tmp_path / "run.tcl",
        work_dir=tmp_path,
        args=["alpha", "beta"],
    )

    assert out == {
        "status": "passed",
        "stdout": "ok\n",
        "stderr": "",
        "message": None,
    }


def test_run_vitis_hls_result_subprocess_error(monkeypatch, tmp_path):
    def fake_run_vitis_hls(tcl_script, work_dir=None, args=None, capture_output=True):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["vitis-run"],
            output="step log\n",
            stderr="failed\n",
        )

    monkeypatch.setattr(toolchain, "run_vitis_hls", fake_run_vitis_hls)

    out = toolchain.run_vitis_hls_result(tmp_path / "run.tcl", work_dir=tmp_path)

    assert out["status"] == "subprocess_error"
    assert out["stdout"] == "step log\n"
    assert out["stderr"] == "failed\n"
    assert "returned non-zero exit status 1" in out["message"]


def test_run_vitis_hls_result_runtime_error(monkeypatch, tmp_path):
    def fake_run_vitis_hls(tcl_script, work_dir=None, args=None, capture_output=True):
        raise RuntimeError("Vitis installation not found.")

    monkeypatch.setattr(toolchain, "run_vitis_hls", fake_run_vitis_hls)

    out = toolchain.run_vitis_hls_result(tmp_path / "run.tcl", work_dir=tmp_path)

    assert out == {
        "status": "runtime_error",
        "stdout": None,
        "stderr": None,
        "message": "Vitis installation not found.",
    }


# Get the directory where this test file lives
TEST_DIR = Path(__file__).parent
RESOURCE_DIR = TEST_DIR / "resources"

@pytest.mark.vitis
def test_vitis_smoke_with_resources(tmp_path):
    vitis_path = toolchain.find_vitis_path()
    if not vitis_path:
        pytest.skip("Vitis installation not found; skipping smoke integration test.")

    # 1. Copy the "Golden" files to the temp workspace
    # This keeps the original resources 'clean'
    shutil.copy(RESOURCE_DIR / "smoke_test.cpp", tmp_path / "main.cpp")
    shutil.copy(RESOURCE_DIR / "smoke_test.tcl", tmp_path / "run.tcl")
    
    # 2. Run
    result = toolchain.run_vitis_hls(tmp_path / "run.tcl", work_dir=tmp_path)
    
    # 3. Assert
    assert result.returncode == 0