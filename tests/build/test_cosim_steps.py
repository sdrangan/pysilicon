"""Tests for ``waveflow.build.cosim_steps``.

Exercises ``ExtractCosimTimingStep`` end-to-end against the same fixture
the parser tests use — both fixture formats land at the same
``transaction_cycles`` field in the produced JSON.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from waveflow.build.build import BuildConfig
from waveflow.build.cosim_steps import ExtractCosimTimingStep


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "utils" / "cosim_fixtures"
POLY_RPT = FIXTURE_DIR / "poly_cosim.rpt"
LEGACY_LOG = FIXTURE_DIR / "cosim.log"


def _seed_solution(tmp_path: Path, fixture: Path, target_name: str) -> Path:
    """Copy ``fixture`` into a tmp ``<sol>/sim/report/`` layout and return sol path."""
    sol = tmp_path / "waveflow_poly_proj" / "solution1"
    report_dir = sol / "sim" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture, report_dir / target_name)
    return sol


def test_extract_cosim_timing_2025_rpt(tmp_path):
    """Vitis 2025.1+ report: cycles parsed, vitis_version tag set."""
    sol = _seed_solution(tmp_path, POLY_RPT, "poly_cosim.rpt")
    step = ExtractCosimTimingStep(
        name="extract_cosim_timing",
        top="poly",
        report_dir_artifact="report_dir",
        output_path="results/cosim_timing.json",
    )
    result = step.run(BuildConfig(root_dir=tmp_path), report_dir=sol)
    out_path = result["cosim_timing"]
    assert out_path == tmp_path / "results" / "cosim_timing.json"
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["transaction_cycles"] == 144
    assert data["source"] == "cosim"
    assert data["top"] == "poly"
    assert data["vitis_version"] == "2025.1+"
    assert data["report_path"].endswith("poly_cosim.rpt")


def test_extract_cosim_timing_legacy_log(tmp_path):
    """Legacy cosim.log: same JSON shape, vitis_version tagged as pre-2025."""
    sol = _seed_solution(tmp_path, LEGACY_LOG, "cosim.log")
    step = ExtractCosimTimingStep(
        name="extract_cosim_timing",
        top="poly",
        report_dir_artifact="report_dir",
        output_path="results/cosim_timing.json",
    )
    result = step.run(BuildConfig(root_dir=tmp_path), report_dir=sol)
    data = json.loads(result["cosim_timing"].read_text(encoding="utf-8"))
    assert data["transaction_cycles"] == 110
    assert data["vitis_version"] == "pre-2025"
    assert data["report_path"].endswith("cosim.log")


def test_extract_cosim_timing_missing_report_raises(tmp_path):
    """An empty solution dir yields a FileNotFoundError from the parser
    (the step does not swallow it)."""
    sol = tmp_path / "waveflow_poly_proj" / "solution1"
    (sol / "sim" / "report").mkdir(parents=True)
    step = ExtractCosimTimingStep(
        name="extract_cosim_timing", top="poly",
        report_dir_artifact="report_dir",
    )
    with pytest.raises(FileNotFoundError):
        step.run(BuildConfig(root_dir=tmp_path), report_dir=sol)


def test_extract_cosim_timing_consumes_produces():
    """Property accessors mirror the constructor parameters."""
    step = ExtractCosimTimingStep(
        name="extract_cosim_timing", top="poly",
        report_dir_artifact="custom_report_dir",
        output_path="custom/path.json",
    )
    assert step.consumes == ["custom_report_dir"]
    assert step.produces == {"cosim_timing": Path("custom/path.json")}


# ---------------------------------------------------------------------------
# ValidateTimingStep — Phase 4
# ---------------------------------------------------------------------------

from waveflow.build.cosim_steps import ValidateTimingStep


def _write_timing_json(path: Path, cycles: int, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "transaction_cycles": cycles,
            "source": source,
            "top": "poly",
        }),
        encoding="utf-8",
    )


def test_validate_timing_pass_within_tolerance(tmp_path):
    """py and cosim cycles within tolerance → verdict.pass is True, step succeeds."""
    py_path = tmp_path / "py_timing.json"
    cs_path = tmp_path / "cosim_timing.json"
    _write_timing_json(py_path, 110, "py_sim")
    _write_timing_json(cs_path, 113, "cosim")
    step = ValidateTimingStep(
        name="validate_timing",
        py_timing_artifact="py_timing",
        cosim_timing_artifact="cosim_timing",
        tolerance_cycles=20,
        output_path="results/timing_verdict.json",
    )
    result = step.run(
        BuildConfig(root_dir=tmp_path),
        py_timing=py_path, cosim_timing=cs_path,
    )
    verdict = json.loads(result["timing_verdict"].read_text(encoding="utf-8"))
    assert verdict["pass"] is True
    assert verdict["py_cycles"] == 110
    assert verdict["cosim_cycles"] == 113
    assert verdict["delta"] == 3
    assert verdict["tolerance"] == 20


def test_validate_timing_fail_outside_tolerance(tmp_path):
    """Out-of-tolerance delta raises RuntimeError and writes a fail verdict."""
    py_path = tmp_path / "py_timing.json"
    cs_path = tmp_path / "cosim_timing.json"
    _write_timing_json(py_path, 100, "py_sim")
    _write_timing_json(cs_path, 200, "cosim")  # delta=100, tolerance=20
    step = ValidateTimingStep(
        name="validate_timing", tolerance_cycles=20,
        output_path="results/timing_verdict.json",
    )
    with pytest.raises(RuntimeError, match="exceeds tolerance"):
        step.run(
            BuildConfig(root_dir=tmp_path),
            py_timing=py_path, cosim_timing=cs_path,
        )
    # The verdict file is still written so post-mortem tooling can inspect.
    verdict = json.loads(
        (tmp_path / "results" / "timing_verdict.json").read_text(encoding="utf-8")
    )
    assert verdict["pass"] is False
    assert verdict["delta"] == 100


def test_validate_timing_at_tolerance_boundary_passes(tmp_path):
    """delta == tolerance is OK (inclusive boundary)."""
    py_path = tmp_path / "py.json"
    cs_path = tmp_path / "cs.json"
    _write_timing_json(py_path, 100, "py_sim")
    _write_timing_json(cs_path, 120, "cosim")  # delta=20
    step = ValidateTimingStep(
        name="validate_timing", tolerance_cycles=20,
    )
    # No raise = pass.
    result = step.run(
        BuildConfig(root_dir=tmp_path),
        py_timing=py_path, cosim_timing=cs_path,
    )
    verdict = json.loads(result["timing_verdict"].read_text(encoding="utf-8"))
    assert verdict["pass"] is True
    assert verdict["delta"] == 20


def test_validate_timing_artifact_renaming(tmp_path):
    """Custom artifact names plumb through to consumes / produces."""
    step = ValidateTimingStep(
        name="validate_timing",
        py_timing_artifact="py_custom",
        cosim_timing_artifact="cs_custom",
        output_path="custom/verdict.json",
    )
    assert step.consumes == ["py_custom", "cs_custom"]
    assert step.produces == {"timing_verdict": Path("custom/verdict.json")}
