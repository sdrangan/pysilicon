"""
tests/poly/test_timing_capture.py — integration tests for xsim_vcd callable API.

These tests verify the Python-callable ``run_xsim_vcd`` function.  Tests
that require an actual Vivado/xsim installation are automatically skipped
when that environment is unavailable.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import the callable function under test
# ---------------------------------------------------------------------------

from pysilicon.scripts.xsim_vcd import (
    run_xsim_vcd,
    modify_tcl,
    create_vcd_batch,
)


# ---------------------------------------------------------------------------
# Platform guard: tests that need Windows are skipped elsewhere
# ---------------------------------------------------------------------------

IS_WINDOWS = os.name == "nt"
requires_windows = pytest.mark.skipif(
    not IS_WINDOWS, reason="xsim_vcd only works on Windows"
)


# ---------------------------------------------------------------------------
# Unit tests for helper functions (platform-independent)
# ---------------------------------------------------------------------------


class TestModifyTcl:
    """Tests for the TCL modifier helper — no simulator required."""

    def test_inserts_vcd_commands(self, tmp_path: Path) -> None:
        tcl_src = tmp_path / "poly.tcl"
        tcl_dst = tmp_path / "poly_vcd.tcl"
        tcl_src.write_text(
            "log_wave -r /\nrun all\nquit\n"
        )
        modify_tcl(str(tcl_src), str(tcl_dst), trace_level="*")
        content = tcl_dst.read_text()
        assert "open_vcd" in content
        assert "log_vcd *" in content

    def test_replaces_quit_with_close_vcd(self, tmp_path: Path) -> None:
        tcl_src = tmp_path / "poly.tcl"
        tcl_dst = tmp_path / "poly_vcd.tcl"
        tcl_src.write_text(
            "log_wave -r /\nrun all\nquit\n"
        )
        modify_tcl(str(tcl_src), str(tcl_dst), trace_level="*")
        content = tcl_dst.read_text()
        assert "close_vcd" in content
        # Original quit should be replaced
        lines = [l.strip() for l in content.splitlines()]
        assert "quit" in lines  # should still end with quit
        assert "close_vcd" in lines

    def test_port_trace_level(self, tmp_path: Path) -> None:
        tcl_src = tmp_path / "poly.tcl"
        tcl_dst = tmp_path / "poly_vcd.tcl"
        tcl_src.write_text("log_wave -r /\nrun all\nquit\n")
        modify_tcl(str(tcl_src), str(tcl_dst), trace_level="port")
        content = tcl_dst.read_text()
        assert "log_vcd port" in content


class TestCreateVcdBatch:
    """Tests for the batch file creator — no simulator required."""

    def test_creates_file_with_xsim_line(self, tmp_path: Path) -> None:
        original = tmp_path / "run_xsim.bat"
        new_bat = tmp_path / "run_xsim_vcd.bat"
        original.write_text(
            "@echo off\nC:\\Xilinx\\Vivado\\bin\\xsim poly.tcl --nolog\n"
        )
        create_vcd_batch("poly", str(original), str(new_bat))
        content = new_bat.read_text()
        assert "poly_vcd.tcl" in content
        assert "cd /d" in content

    def test_raises_if_no_xsim_line(self, tmp_path: Path) -> None:
        original = tmp_path / "run_xsim.bat"
        new_bat = tmp_path / "run_xsim_vcd.bat"
        original.write_text("@echo off\n:: nothing useful here\n")
        with pytest.raises(RuntimeError, match="No xsim line found"):
            create_vcd_batch("poly", str(original), str(new_bat))


# ---------------------------------------------------------------------------
# run_xsim_vcd: non-Windows raises RuntimeError
# ---------------------------------------------------------------------------


class TestRunXsimVcdNonWindows:
    @pytest.mark.skipif(IS_WINDOWS, reason="Test only meaningful on non-Windows")
    def test_raises_on_non_windows(self) -> None:
        with pytest.raises(RuntimeError, match="Windows"):
            run_xsim_vcd(top="poly")


# ---------------------------------------------------------------------------
# Integration test: requires Vivado / xsim environment
# ---------------------------------------------------------------------------

_XSIM_AVAILABLE = (
    IS_WINDOWS
    and (Path("pysilicon_poly_proj").exists() or Path("examples/poly/pysilicon_poly_proj").exists())
)

requires_xsim = pytest.mark.skipif(
    not _XSIM_AVAILABLE,
    reason="Vivado/xsim environment not available",
)


class TestRunXsimVcdIntegration:
    @requires_xsim
    def test_generates_vcd_file(self, tmp_path: Path) -> None:
        out_path = run_xsim_vcd(
            top="poly",
            comp="pysilicon_poly_proj",
            out="test_out.vcd",
            workdir=Path(__file__).resolve().parents[2] / "examples" / "poly",
        )
        assert out_path.exists(), f"Expected VCD at {out_path}"
        assert out_path.stat().st_size > 0
