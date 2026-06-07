"""Tests for ``waveflow.utils.cosimparse.CosimReportParser``.

Two fixture flavours under ``tests/utils/cosim_fixtures/``:

- ``poly_cosim.rpt`` — a real Vitis 2025.1+ report (copied from a CI run
  of the poly example) with the table layout.
- ``cosim.log``    — a hand-crafted legacy-shape log with a
  ``Total Execution Time: 110 cycles`` line; representative of what
  pre-2025 Vitis HLS emits.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from waveflow.utils.cosimparse import CosimReportParser


FIXTURE_DIR = Path(__file__).resolve().parent / "cosim_fixtures"
POLY_RPT = FIXTURE_DIR / "poly_cosim.rpt"
LEGACY_LOG = FIXTURE_DIR / "cosim.log"


def _make_sol(tmp_path: Path, fixture: Path, target_name: str) -> Path:
    """Copy ``fixture`` into a tmp ``<sol>/sim/report/`` layout."""
    sol = tmp_path / "solution1"
    report_dir = sol / "sim" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture, report_dir / target_name)
    return sol


def test_parser_reads_vitis_2025_rpt_via_sol_path(tmp_path):
    """sol_path-based lookup finds ``<sol>/sim/report/<top>_cosim.rpt``."""
    sol = _make_sol(tmp_path, POLY_RPT, "poly_cosim.rpt")
    parser = CosimReportParser(sol_path=sol, top="poly")
    assert parser.report_path.name == "poly_cosim.rpt"
    assert parser.get_transaction_cycles() == 144


def test_parser_reads_vitis_2025_rpt_without_top_hint(tmp_path):
    """Without a top hint the parser globs ``*_cosim.rpt`` and picks the first."""
    sol = _make_sol(tmp_path, POLY_RPT, "poly_cosim.rpt")
    parser = CosimReportParser(sol_path=sol)
    assert parser.report_path.name == "poly_cosim.rpt"
    assert parser.get_transaction_cycles() == 144


def test_parser_falls_back_to_legacy_cosim_log(tmp_path):
    """When only the legacy ``cosim.log`` is present the parser uses it."""
    sol = _make_sol(tmp_path, LEGACY_LOG, "cosim.log")
    parser = CosimReportParser(sol_path=sol, top="poly")
    assert parser.report_path.name == "cosim.log"
    assert parser.get_transaction_cycles() == 110


def test_parser_prefers_rpt_over_log(tmp_path):
    """When both formats exist (mid-version transitions), prefer the
    structured 2025.1+ table."""
    sol = _make_sol(tmp_path, POLY_RPT, "poly_cosim.rpt")
    shutil.copy(LEGACY_LOG, sol / "sim" / "report" / "cosim.log")
    parser = CosimReportParser(sol_path=sol, top="poly")
    assert parser.report_path.name == "poly_cosim.rpt"
    assert parser.get_transaction_cycles() == 144


def test_parser_explicit_report_path(tmp_path):
    """``report_path`` short-circuits sol_path discovery."""
    target = tmp_path / "cosim.log"
    shutil.copy(LEGACY_LOG, target)
    parser = CosimReportParser(report_path=target)
    assert parser.report_path == target
    assert parser.get_transaction_cycles() == 110


def test_parser_missing_report_lists_candidates(tmp_path):
    """A clean miss reports every candidate path the parser tried."""
    sol = tmp_path / "solution1"
    (sol / "sim" / "report").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        CosimReportParser(sol_path=sol, top="poly")
    msg = str(excinfo.value)
    assert "poly_cosim.rpt" in msg
    assert "cosim.log" in msg


def test_parser_requires_one_of_sol_or_report_path():
    """Constructor must be given at least one path source."""
    with pytest.raises(ValueError, match="sol_path or report_path"):
        CosimReportParser()
